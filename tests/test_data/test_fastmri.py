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

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import prep_fastmri_espirit  # noqa: E402

compute_espirit_maps = prep_fastmri_espirit.compute_espirit_maps
process_h5_file = prep_fastmri_espirit.process_h5_file


@pytest.fixture
def mock_fastmri_h5_dir(tmp_path) -> tuple[str, list[torch.Tensor]]:
    """Creates temporary dataset directory with mock fastMRI multicoil volumes."""
    data_dir = tmp_path / "mock_fastmri"
    data_dir.mkdir(parents=True)

    vol_configs = [
        ("file1.h5", 4, 4, 32, 32),
        ("file2.h5", 2, 4, 32, 32),
    ]

    all_gt_slices = []
    for name, num_slices, num_coils, h, w in vol_configs:
        f_path = data_dir / name

        # Synthetic sensitivity maps normalized sum_c |S_c|^2 = 1
        sens_maps = np.random.randn(num_slices, num_coils, h, w) + 1j * np.random.randn(
            num_slices, num_coils, h, w
        )
        norm = np.sqrt(np.sum(np.abs(sens_maps) ** 2, axis=1, keepdims=True) + 1e-8)
        sens_maps = (sens_maps / norm).astype(np.complex64)

        # Ground truth single-channel complex images
        x_gt = (
            np.random.randn(num_slices, 1, h, w) + 1j * np.random.randn(num_slices, 1, h, w)
        ).astype(np.complex64)

        for s in range(num_slices):
            all_gt_slices.append(torch.from_numpy(x_gt[s]))

        # Synthesize multi-coil k-space: F(S * x_gt)
        coils_img = sens_maps * x_gt
        kspace = np.fft.fftshift(
            np.fft.fft2(coils_img, axes=(-2, -1), norm="ortho"),
            axes=(-2, -1),
        ).astype(np.complex64)

        with h5py.File(f_path, "w") as hf:
            hf.create_dataset("kspace", data=kspace)
            hf.create_dataset("sensitivity_maps", data=sens_maps)

    return str(data_dir), all_gt_slices


def test_fastmri_registry_and_subclass() -> None:
    """Verify FastMRIDataset is registered in DATASETS and inherits BaseComplexDataset."""
    assert "fastmri" in DATASETS
    assert "fast_mri" in DATASETS
    assert DATASETS.get("fastmri") is FastMRIDataset
    assert issubclass(FastMRIDataset, BaseComplexDataset)


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


def test_fastmri_missing_sensitivity_maps_raises(tmp_path) -> None:
    """FastMRIDataset raises KeyError when sensitivity_maps dataset is absent."""
    data_dir = tmp_path / "no_sens"
    data_dir.mkdir(parents=True)
    f_path = data_dir / "nosens.h5"

    with h5py.File(f_path, "w") as hf:
        hf.create_dataset("kspace", data=np.zeros((2, 4, 16, 16), dtype=np.complex64))

    dataset = FastMRIDataset(data_dir=str(data_dir), use_cache=False)
    with pytest.raises(KeyError, match="missing 'sensitivity_maps'"):
        _ = dataset[0]


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


def test_prep_fastmri_espirit_pipeline(tmp_path) -> None:
    """ESPIRiT precomputation script computes and writes valid sensitivity maps to HDF5."""
    file_path = tmp_path / "raw_kspace.h5"

    # Create synthetic smooth coil k-space
    img = np.zeros((1, 1, 32, 32), dtype=np.complex64)
    img[:, :, 8:24, 8:24] = 1.0
    sens = np.stack([np.ones((32, 32)) * np.exp(1j * k) for k in range(4)], axis=0) / 2.0
    sens = sens[np.newaxis, ...].astype(np.complex64)  # [1, 4, 32, 32]
    coils = sens * img
    kspace = np.fft.fftshift(
        np.fft.fft2(coils, axes=(-2, -1), norm="ortho"),
        axes=(-2, -1),
    ).astype(np.complex64)

    with h5py.File(file_path, "w") as hf:
        hf.create_dataset("kspace", data=kspace)

    # Run precomputation
    success = process_h5_file(file_path=file_path, calib_width=16, overwrite=False)
    assert success is True

    with h5py.File(file_path, "r") as hf:
        assert "sensitivity_maps" in hf
        sens_maps = hf["sensitivity_maps"][:]
        assert sens_maps.shape == (1, 4, 32, 32)
        assert np.iscomplexobj(sens_maps)

    # Running again without overwrite returns False (skipped)
    skipped = process_h5_file(file_path=file_path, overwrite=False)
    assert skipped is False

    # Running with overwrite returns True
    overwritten = process_h5_file(file_path=file_path, calib_width=16, overwrite=True)
    assert overwritten is True


def test_compute_espirit_maps_direct() -> None:
    """compute_espirit_maps processes 3D and 4D arrays and outputs complex64 maps."""
    # 3D slice: [num_coils, H, W]
    img = np.zeros((1, 24, 24), dtype=np.complex64)
    img[:, 6:18, 6:18] = 1.0
    sens = np.stack([np.ones((24, 24)) * np.exp(1j * k) for k in range(4)], axis=0) / 2.0
    ksp_3d = np.fft.fftshift(
        np.fft.fft2(sens * img, axes=(-2, -1), norm="ortho"), axes=(-2, -1)
    ).astype(np.complex64)
    maps_3d = compute_espirit_maps(ksp_3d, calib_width=12)
    assert maps_3d.shape == (4, 24, 24)
    assert maps_3d.dtype == np.complex64

    # 4D volume: [num_slices, num_coils, H, W]
    ksp_4d = np.stack([ksp_3d, ksp_3d], axis=0)
    maps_4d = compute_espirit_maps(ksp_4d, calib_width=12)
    assert maps_4d.shape == (2, 4, 24, 24)
    assert maps_4d.dtype == np.complex64
