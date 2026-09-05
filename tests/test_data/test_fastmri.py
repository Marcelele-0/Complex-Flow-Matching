"""Unit tests for FastMRIDataset, SENSE combination, and ESPIRiT precomputation."""

from __future__ import annotations

import sys
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from cfm.core.dataset import BaseComplexDataset
from cfm.core.registry import DATASETS
from cfm.data.fastmri import FastMRIDataset
from cfm.data.hdf5_manager import WorkerHDF5Manager
from cfm.data.masks import CartesianMaskGenerator, PoissonDiscMaskGenerator
from cfm.data.transforms import CenterCropOrPad
from cfm.utils.fft import fft2c

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

# sigpy is an optional extra (only the prep script needs it), so the suite skips
# rather than errors on a training-only install.
pytest.importorskip("sigpy", reason="sigpy is an optional extra: uv sync --extra espirit")

import prep_fastmri_espirit  # noqa: E402

compute_espirit_maps = prep_fastmri_espirit.compute_espirit_maps
process_h5_file = prep_fastmri_espirit.process_h5_file


def _write_volume(
    path: Path,
    num_slices: int,
    num_coils: int,
    h: int,
    w: int,
    rng: np.random.Generator,
    with_sens: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Write one synthetic multi-coil volume and return its maps and ground truth.

    k-space is synthesised with the *centered* forward transform, the same one
    ``sigpy`` calibrates against, so the fixture cannot hide a shift convention
    error the way an ``fft2``/``ifft2`` round trip would.
    """
    sens_maps = rng.standard_normal((num_slices, num_coils, h, w)) + 1j * rng.standard_normal(
        (num_slices, num_coils, h, w)
    )
    norm = np.sqrt(np.sum(np.abs(sens_maps) ** 2, axis=1, keepdims=True) + 1e-8)
    sens_maps = (sens_maps / norm).astype(np.complex64)

    x_gt = (
        rng.standard_normal((num_slices, 1, h, w)) + 1j * rng.standard_normal((num_slices, 1, h, w))
    ).astype(np.complex64)

    coils_img = torch.from_numpy((sens_maps * x_gt).astype(np.complex64))
    kspace = fft2c(coils_img).numpy().astype(np.complex64)

    with h5py.File(path, "w") as hf:
        hf.create_dataset("kspace", data=kspace)
        if with_sens:
            hf.create_dataset("sensitivity_maps", data=sens_maps)

    return sens_maps, x_gt


@pytest.fixture
def mock_fastmri_h5_dir(tmp_path) -> tuple[str, list[torch.Tensor]]:
    """Creates temporary dataset directory with mock fastMRI multicoil volumes."""
    data_dir = tmp_path / "mock_fastmri"
    data_dir.mkdir(parents=True)
    rng = np.random.default_rng(0)

    all_gt_slices = []
    for name, num_slices, num_coils, h, w in [
        ("file1.h5", 4, 4, 32, 32),
        ("file2.h5", 2, 4, 32, 32),
    ]:
        _, x_gt = _write_volume(data_dir / name, num_slices, num_coils, h, w, rng)
        for s in range(num_slices):
            all_gt_slices.append(torch.from_numpy(x_gt[s]))

    return str(data_dir), all_gt_slices


def test_fastmri_registry_and_subclass() -> None:
    """Verify FastMRIDataset is registered in DATASETS and inherits BaseComplexDataset."""
    assert "fastmri" in DATASETS
    assert "fast_mri" in DATASETS
    assert DATASETS.get("fastmri") is FastMRIDataset
    assert issubclass(FastMRIDataset, BaseComplexDataset)


def test_sense_combine_matches_sigpy_espirit_convention() -> None:
    """SENSE combination is aligned with the grid sigpy's ESPIRiT maps live on.

    sigpy's ``EspiritCalib`` calls ``sp.ifft`` with ``center=True``, i.e.
    ``ifftshift -> ifftn -> fftshift``, so its maps sit on the centered image grid.
    Combining them with coil images produced by ``ifft2(ifftshift(k))`` - a centered
    inverse missing its trailing shift - pairs every pixel with a sensitivity from
    half a field of view away. The phantom is deliberately off-center so that
    misalignment shows up: with the shift dropped this correlation falls to ~0.6.
    """
    import sigpy.mri.app as app

    h = w = 64
    phantom = np.zeros((h, w), dtype=np.complex64)
    phantom[10:30, 40:55] = 1.0 + 0.5j

    yy, xx = np.mgrid[0:h, 0:w]
    corners = [(0, 0), (0, w), (h, 0), (h, w)]
    sens = np.stack(
        [
            np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (0.8 * h) ** 2))
            * np.exp(1j * 0.01 * (yy + xx) * (i + 1))
            for i, (cy, cx) in enumerate(corners)
        ]
    ).astype(np.complex64)
    sens = sens / np.sqrt((np.abs(sens) ** 2).sum(0, keepdims=True))

    kspace = fft2c(torch.from_numpy((sens * phantom).astype(np.complex64)))
    maps = torch.from_numpy(
        np.asarray(
            app.EspiritCalib(kspace.numpy(), calib_width=24, show_pbar=False).run(),
            dtype=np.complex64,
        )
    )

    recon = FastMRIDataset.sense_combine(kspace, maps)[0].abs().numpy()
    truth = np.abs(phantom)

    correlation = np.corrcoef(recon.ravel(), truth.ravel())[0, 1]
    assert correlation > 0.99, f"SENSE combination misaligned with ESPIRiT maps: {correlation:.4f}"


def test_fastmri_generation_mode_identity(mock_fastmri_h5_dir) -> None:
    """FastMRIDataset in generation mode performs SENSE combination recovering ground truth."""
    data_dir, gt_slices = mock_fastmri_h5_dir
    dataset = FastMRIDataset(data_dir=data_dir, num_slices=1, mode="generation", use_cache=False)

    assert len(dataset) == len(gt_slices)
    assert len(dataset) == 6  # 4 + 2 slices

    for idx in range(len(dataset)):
        item = dataset[idx]
        assert isinstance(item, torch.Tensor)
        assert item.shape == (1, 32, 32)
        assert item.dtype == torch.complex64
        torch.testing.assert_close(item, gt_slices[idx], atol=1e-5, rtol=1e-5)


def test_fastmri_reconstruction_mode(mock_fastmri_h5_dir) -> None:
    """FastMRIDataset in reconstruction mode returns input, mask, target, and multicoil dict."""
    data_dir, _ = mock_fastmri_h5_dir
    dataset = FastMRIDataset(
        data_dir=data_dir,
        mode="reconstruction",
        acceleration=4,
        use_cache=False,
    )

    item = dataset[0]
    assert isinstance(item, dict)
    assert set(item.keys()) == {
        "input",
        "mask",
        "target",
        "masked_kspace",
        "sensitivity_maps",
    }

    assert item["input"].shape == (1, 32, 32)
    assert item["mask"].shape == (1, 32, 32)
    assert item["target"].shape == (1, 32, 32)
    assert item["masked_kspace"].shape == (4, 32, 32)
    assert item["sensitivity_maps"].shape == (4, 32, 32)

    assert item["input"].dtype == torch.complex64
    assert item["target"].dtype == torch.complex64
    assert item["masked_kspace"].dtype == torch.complex64
    assert item["sensitivity_maps"].dtype == torch.complex64
    assert item["mask"].dtype == torch.float32

    # Peak amplitude normalized to 1.0
    assert item["target"].abs().max().item() <= 1.0 + 1e-6


@pytest.mark.parametrize("crop_size", [None, 16])
def test_fastmri_masked_kspace_is_consistent_with_target(mock_fastmri_h5_dir, crop_size) -> None:
    """masked_kspace equals F(S * target) on the sampled lines, with and without a crop.

    This is what the data consistency projection assumes. It breaks if the target is
    rescaled or cropped without the k-space and maps being brought along.
    """
    data_dir, _ = mock_fastmri_h5_dir
    pre_transform = CenterCropOrPad(crop_size) if crop_size is not None else None
    dataset = FastMRIDataset(
        data_dir=data_dir,
        mode="reconstruction",
        use_cache=False,
        pre_transform=pre_transform,
    )

    item = dataset[0]
    expected = fft2c(item["sensitivity_maps"] * item["target"]) * item["mask"]
    torch.testing.assert_close(item["masked_kspace"], expected, atol=1e-5, rtol=1e-5)

    if crop_size is not None:
        assert item["target"].shape == (1, crop_size, crop_size)
        assert item["sensitivity_maps"].shape == (4, crop_size, crop_size)


def test_fastmri_post_transform_may_not_resize(mock_fastmri_h5_dir) -> None:
    """A post_transform that changes shape is rejected rather than silently desyncing."""
    data_dir, _ = mock_fastmri_h5_dir
    dataset = FastMRIDataset(
        data_dir=data_dir,
        mode="reconstruction",
        use_cache=False,
        post_transform=CenterCropOrPad(16),
    )

    with pytest.raises(ValueError, match="post_transform changed the spatial shape"):
        _ = dataset[0]


def test_fastmri_mask_generators(mock_fastmri_h5_dir) -> None:
    """FastMRIDataset correctly integrates with Poisson-Disc and Cartesian mask generators."""
    data_dir, _ = mock_fastmri_h5_dir

    # Poisson-Disc mask config
    dataset_poisson = FastMRIDataset(
        data_dir=data_dir,
        mode="reconstruction",
        mask={"type": "poisson_disc", "acceleration": 4},
        use_cache=False,
    )
    assert isinstance(dataset_poisson.mask_generator, PoissonDiscMaskGenerator)
    item_p = dataset_poisson[0]
    assert isinstance(item_p, dict)
    assert item_p["mask"].shape == (1, 32, 32)

    # Cartesian mask generator instance
    cart_gen = CartesianMaskGenerator(acceleration=4.0)
    dataset_cart = FastMRIDataset(
        data_dir=data_dir,
        mode="reconstruction",
        mask=cart_gen,
        use_cache=False,
    )
    assert isinstance(dataset_cart.mask_generator, CartesianMaskGenerator)
    item_c = dataset_cart[0]
    assert isinstance(item_c, dict)
    assert item_c["mask"].shape == (1, 32, 32)


def test_fastmri_multi_slice_window(mock_fastmri_h5_dir) -> None:
    """FastMRIDataset supports 2.5D multi-slice windows in generation mode."""
    data_dir, _ = mock_fastmri_h5_dir
    dataset = FastMRIDataset(
        data_dir=data_dir,
        num_slices=3,
        mode="generation",
        use_cache=False,
    )

    item = dataset[0]
    assert isinstance(item, torch.Tensor)
    assert item.shape == (3, 1, 32, 32)
    assert item.dtype == torch.complex64


def test_fastmri_index_cache(mock_fastmri_h5_dir, tmp_path) -> None:
    """FastMRIDataset builds and reloads persistent index cache."""
    data_dir, _ = mock_fastmri_h5_dir
    cache_dir = tmp_path / "cache"

    # First initialization: builds cache
    ds1 = FastMRIDataset(data_dir=data_dir, use_cache=True, cache_dir=cache_dir)
    assert len(ds1) == 6

    # Verify cache file exists
    cache_files = list(cache_dir.glob("fastmri_index_*.json"))
    assert len(cache_files) == 1

    # Second initialization: loads from cache
    ds2 = FastMRIDataset(data_dir=data_dir, use_cache=True, cache_dir=cache_dir)
    assert len(ds2) == 6
    assert ds2.slice_map == ds1.slice_map


def test_fastmri_index_build_does_not_retain_handles(mock_fastmri_h5_dir) -> None:
    """Indexing must not leave a handle open per volume: the full cohort is thousands."""
    data_dir, _ = mock_fastmri_h5_dir
    WorkerHDF5Manager.reset()

    FastMRIDataset(data_dir=data_dir, use_cache=False)

    manager = WorkerHDF5Manager.get_instance()
    assert manager._handles == {}


def test_fastmri_max_volumes(mock_fastmri_h5_dir) -> None:
    """max_volumes caps the cohort, which is what bounds the local debug subset."""
    data_dir, _ = mock_fastmri_h5_dir

    full = FastMRIDataset(data_dir=data_dir, use_cache=False)
    assert len(full.files) == 2

    capped = FastMRIDataset(data_dir=data_dir, use_cache=False, max_volumes=1)
    assert len(capped.files) == 1
    assert len(capped) == 4  # file1.h5 has 4 slices

    with pytest.raises(ValueError, match="max_volumes must be >= 1"):
        FastMRIDataset(data_dir=data_dir, max_volumes=0)


def test_fastmri_sidecar_sensitivity_maps(tmp_path) -> None:
    """Sensitivity maps can live in a sidecar directory instead of the source archive."""
    data_dir = tmp_path / "kspace_only"
    sens_dir = tmp_path / "maps"
    data_dir.mkdir()
    sens_dir.mkdir()
    rng = np.random.default_rng(1)

    sens_maps, x_gt = _write_volume(data_dir / "vol.h5", 2, 4, 32, 32, rng, with_sens=False)
    with h5py.File(sens_dir / "vol.h5", "w") as hf:
        hf.create_dataset("sensitivity_maps", data=sens_maps)

    dataset = FastMRIDataset(data_dir=str(data_dir), use_cache=False, sens_dir=str(sens_dir))
    torch.testing.assert_close(dataset[0], torch.from_numpy(x_gt[0]), atol=1e-5, rtol=1e-5)

    # A missing sidecar names the directory it looked in.
    missing = FastMRIDataset(
        data_dir=str(data_dir),
        use_cache=False,
        sens_dir=str(tmp_path / "absent"),
        auto_calibrate=False,
    )
    with pytest.raises(FileNotFoundError, match="No sidecar sensitivity map file"):
        _ = missing[0]


def test_fastmri_missing_sensitivity_maps_raises(tmp_path) -> None:
    """FastMRIDataset raises FileNotFoundError when sidecars are absent and auto-calib off."""
    data_dir = tmp_path / "no_sens"
    data_dir.mkdir(parents=True)
    f_path = data_dir / "nosens.h5"

    with h5py.File(f_path, "w") as hf:
        hf.create_dataset("kspace", data=np.zeros((2, 4, 16, 16), dtype=np.complex64))

    dataset = FastMRIDataset(data_dir=str(data_dir), use_cache=False, auto_calibrate=False)
    with pytest.raises(FileNotFoundError, match="No sidecar sensitivity map file"):
        _ = dataset[0]


def test_fastmri_rejects_3d_kspace(tmp_path) -> None:
    """3D k-space is ambiguous between singlecoil and single-slice multicoil."""
    data_dir = tmp_path / "singlecoil"
    data_dir.mkdir(parents=True)
    with h5py.File(data_dir / "sc.h5", "w") as hf:
        hf.create_dataset("kspace", data=np.zeros((5, 16, 16), dtype=np.complex64))

    with pytest.raises(ValueError, match="requires multi-coil 4D"):
        FastMRIDataset(data_dir=str(data_dir), use_cache=False)


def test_fastmri_validation_errors(mock_fastmri_h5_dir) -> None:
    """FastMRIDataset raises ValueError on invalid construction arguments."""
    data_dir, _ = mock_fastmri_h5_dir

    # Even num_slices
    with pytest.raises(ValueError, match="positive odd integer"):
        FastMRIDataset(data_dir=data_dir, num_slices=2)

    # Invalid mode
    with pytest.raises(ValueError, match="mode must be 'generation' or 'reconstruction'"):
        FastMRIDataset(data_dir=data_dir, mode="unknown")

    # Reconstruction mode with num_slices > 1
    with pytest.raises(ValueError, match="supports num_slices=1 only"):
        FastMRIDataset(data_dir=data_dir, mode="reconstruction", num_slices=3)

    # Acceleration < 1
    with pytest.raises(ValueError, match="acceleration must be >= 1"):
        FastMRIDataset(data_dir=data_dir, acceleration=0.5)

    # Non-existent data_dir
    with pytest.raises(FileNotFoundError):
        FastMRIDataset(data_dir="non_existent_directory_123")


def test_fastmri_dataloader_iteration(mock_fastmri_h5_dir) -> None:
    """FastMRIDataset batches cleanly in PyTorch DataLoader."""
    data_dir, _ = mock_fastmri_h5_dir
    WorkerHDF5Manager.reset()

    dataset = FastMRIDataset(data_dir=data_dir, use_cache=False)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)

    batches = list(loader)
    assert len(batches) == 3
    assert batches[0].shape == (2, 1, 32, 32)
    assert batches[0].dtype == torch.complex64


def test_fastmri_dataloader_with_workers(mock_fastmri_h5_dir) -> None:
    """Worker processes each open their own HDF5 handles, so streaming survives a fork."""
    data_dir, gt_slices = mock_fastmri_h5_dir
    WorkerHDF5Manager.reset()

    dataset = FastMRIDataset(data_dir=data_dir, use_cache=False)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=2)

    collected = torch.cat([batch for batch in loader], dim=0)
    torch.testing.assert_close(collected, torch.stack(gt_slices), atol=1e-5, rtol=1e-5)


def test_fastmri_heterogeneous_volumes_collate_with_crop(tmp_path) -> None:
    """Volumes of differing matrix size only batch once a fixed crop is applied.

    fastMRI knee volumes are 640x368 and 640x372, and brain volumes differ again, so
    without a fixed output size ``default_collate`` raises on any batch that spans
    two volumes - which shuffling guarantees.
    """
    data_dir = tmp_path / "mixed"
    data_dir.mkdir()
    rng = np.random.default_rng(2)
    _write_volume(data_dir / "a.h5", 2, 4, 32, 40, rng)
    _write_volume(data_dir / "b.h5", 2, 4, 48, 32, rng)

    uncropped = FastMRIDataset(data_dir=str(data_dir), use_cache=False)
    with pytest.raises(RuntimeError):
        list(DataLoader(uncropped, batch_size=4, shuffle=False, num_workers=0))

    cropped = FastMRIDataset(
        data_dir=str(data_dir), use_cache=False, transform=CenterCropOrPad((32, 32))
    )
    batches = list(DataLoader(cropped, batch_size=4, shuffle=False, num_workers=0))
    assert batches[0].shape == (4, 1, 32, 32)


def test_prep_fastmri_espirit_pipeline(tmp_path) -> None:
    """ESPIRiT precomputation computes and writes valid sensitivity maps to sidecar HDF5."""
    file_path = tmp_path / "raw_kspace.h5"
    out_dir = tmp_path / "sidecars"

    # Create synthetic smooth coil k-space
    img = np.zeros((1, 1, 32, 32), dtype=np.complex64)
    img[:, :, 8:24, 8:24] = 1.0
    sens = np.stack([np.ones((32, 32)) * np.exp(1j * k) for k in range(4)], axis=0) / 2.0
    sens = sens[np.newaxis, ...].astype(np.complex64)  # [1, 4, 32, 32]
    kspace = fft2c(torch.from_numpy((sens * img).astype(np.complex64))).numpy()

    with h5py.File(file_path, "w") as hf:
        hf.create_dataset("kspace", data=kspace)

    # Run precomputation to sidecar dir
    success = process_h5_file(
        file_path=file_path, output_dir=out_dir, calib_width=16, overwrite=False
    )
    assert success is True

    # Source archive remains untouched
    with h5py.File(file_path, "r") as hf:
        assert "sensitivity_maps" not in hf

    sidecar_path = out_dir / "raw_kspace.h5"
    assert sidecar_path.is_file()
    with h5py.File(sidecar_path, "r") as hf:
        assert "sensitivity_maps" in hf
        sens_maps = hf["sensitivity_maps"][:]
        assert sens_maps.shape == (1, 4, 32, 32)
        assert np.iscomplexobj(sens_maps)

    # Running again without overwrite returns False (skipped)
    skipped = process_h5_file(file_path=file_path, output_dir=out_dir, overwrite=False)
    assert skipped is False

    # Running with overwrite returns True
    overwritten = process_h5_file(
        file_path=file_path, output_dir=out_dir, calib_width=16, overwrite=True
    )
    assert overwritten is True


def test_prep_fastmri_espirit_sidecar_output(tmp_path) -> None:
    """--output_dir writes a sidecar file and leaves the source archive untouched."""
    src_dir = tmp_path / "src"
    out_dir = tmp_path / "out"
    src_dir.mkdir()
    file_path = src_dir / "vol.h5"

    img = np.zeros((2, 1, 32, 32), dtype=np.complex64)
    img[:, :, 8:24, 8:24] = 1.0
    sens = np.stack([np.ones((32, 32)) * np.exp(1j * k) for k in range(4)], axis=0) / 2.0
    sens = np.broadcast_to(sens[np.newaxis, ...], (2, 4, 32, 32)).astype(np.complex64)
    kspace = fft2c(torch.from_numpy((sens * img).astype(np.complex64))).numpy()

    with h5py.File(file_path, "w") as hf:
        hf.create_dataset("kspace", data=kspace)

    assert process_h5_file(file_path=file_path, calib_width=16, output_dir=out_dir) is True

    with h5py.File(file_path, "r") as hf:
        assert "sensitivity_maps" not in hf, "source archive must not be modified"

    with h5py.File(out_dir / "vol.h5", "r") as hf:
        assert hf["sensitivity_maps"].shape == (2, 4, 32, 32)

    # No temporary files left behind.
    assert list(out_dir.glob("*.tmp*")) == []

    # Second pass skips the existing sidecar.
    assert process_h5_file(file_path=file_path, calib_width=16, output_dir=out_dir) is False


def test_compute_espirit_maps_direct() -> None:
    """compute_espirit_maps processes 3D and 4D arrays and outputs complex64 maps."""
    # 3D slice: [num_coils, H, W]
    img = np.zeros((1, 24, 24), dtype=np.complex64)
    img[:, 6:18, 6:18] = 1.0
    sens = np.stack([np.ones((24, 24)) * np.exp(1j * k) for k in range(4)], axis=0) / 2.0
    ksp_3d = fft2c(torch.from_numpy((sens * img).astype(np.complex64))).numpy()
    maps_3d = compute_espirit_maps(ksp_3d, calib_width=12)
    assert maps_3d.shape == (4, 24, 24)
    assert maps_3d.dtype == np.complex64

    # 4D volume: [num_slices, num_coils, H, W]
    ksp_4d = np.stack([ksp_3d, ksp_3d], axis=0)
    maps_4d = compute_espirit_maps(ksp_4d, calib_width=12)
    assert maps_4d.shape == (2, 4, 24, 24)
    assert maps_4d.dtype == np.complex64
