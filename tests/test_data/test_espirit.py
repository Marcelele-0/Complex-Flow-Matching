"""Unit tests for ESPIRiT calibration and automatic FastMRIDataset calibration."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch

from cyfm.data import (
    FastMRIDataset,
    compute_espirit_maps,
    ensure_espirit_maps,
)
from cyfm.data.espirit import process_h5_file
from cyfm.data.torch_espirit import calibrate_fastmri_file_torch, compute_espirit_torch
from cyfm.utils.fft import fft2c

pytest.importorskip("sigpy", reason="sigpy is required for ESPIRiT tests")


def _create_mock_kspace_file(
    file_path: Path,
    num_slices: int = 2,
    num_coils: int = 4,
    height: int = 24,
    width: int = 24,
) -> None:
    """Create a synthetic multi-coil k-space HDF5 file without sensitivity maps."""
    img = np.zeros((num_slices, 1, height, width), dtype=np.complex64)
    img[:, :, height // 4 : 3 * height // 4, width // 4 : 3 * width // 4] = 1.0
    sens = np.stack(
        [np.ones((height, width)) * np.exp(1j * c * np.pi / num_coils) for c in range(num_coils)],
        axis=0,
    )
    sens = np.broadcast_to(sens[np.newaxis, ...], (num_slices, num_coils, height, width))
    kspace = fft2c(torch.from_numpy((sens * img).astype(np.complex64))).numpy()

    with h5py.File(file_path, "w") as hf:
        hf.create_dataset("kspace", data=kspace)


def test_compute_espirit_maps_3d_and_4d() -> None:
    """Test compute_espirit_maps on 3D slice and 4D volume tensors."""
    # 3D: [num_coils, H, W]
    ksp_3d = np.zeros((4, 24, 24), dtype=np.complex64)
    ksp_3d[:, 6:18, 6:18] = 1.0
    maps_3d = compute_espirit_maps(ksp_3d, calib_width=12)
    assert maps_3d.shape == (4, 24, 24)
    assert maps_3d.dtype == np.complex64

    # 4D: [num_slices, num_coils, H, W]
    ksp_4d = np.stack([ksp_3d, ksp_3d], axis=0)
    maps_4d = compute_espirit_maps(ksp_4d, calib_width=12)
    assert maps_4d.shape == (2, 4, 24, 24)
    assert maps_4d.dtype == np.complex64

    # Invalid dimension raises ValueError
    with pytest.raises(ValueError, match="Expected kspace ndim 3 or 4"):
        compute_espirit_maps(np.zeros((24, 24), dtype=np.complex64))


def test_process_h5_file_sidecar_only(tmp_path: Path) -> None:
    """Test process_h5_file requires output_dir and writes sidecars without modifying source."""
    f_src = tmp_path / "vol.h5"
    out_dir = tmp_path / "sidecars"
    _create_mock_kspace_file(f_src, num_slices=2, num_coils=4, height=24, width=24)

    # Missing output_dir raises ValueError (in-place modification forbidden)
    with pytest.raises(ValueError, match="output_dir is required"):
        process_h5_file(f_src, calib_width=12)

    # Sidecar run
    success_sidecar = process_h5_file(f_src, output_dir=out_dir, calib_width=12, overwrite=False)
    assert success_sidecar is True

    # Source remains untouched
    with h5py.File(f_src, "r") as hf:
        assert "sensitivity_maps" not in hf

    # Sidecar exists
    sidecar_path = out_dir / "vol.h5"
    assert sidecar_path.is_file()
    with h5py.File(sidecar_path, "r") as hf:
        assert "sensitivity_maps" in hf
        assert hf["sensitivity_maps"].shape == (2, 4, 24, 24)

    # Re-running without overwrite skips
    assert process_h5_file(f_src, output_dir=out_dir, calib_width=12, overwrite=False) is False
    # Re-running with overwrite succeeds
    assert process_h5_file(f_src, output_dir=out_dir, calib_width=12, overwrite=True) is True


def test_ensure_espirit_maps_multiple_files(tmp_path: Path) -> None:
    """Test ensure_espirit_maps over multiple files sequentially and in parallel."""
    f1 = tmp_path / "f1.h5"
    f2 = tmp_path / "f2.h5"
    _create_mock_kspace_file(f1, num_slices=1, num_coils=4, height=20, width=20)
    _create_mock_kspace_file(f2, num_slices=1, num_coils=4, height=20, width=20)

    # Empty list
    assert ensure_espirit_maps([]) == []

    # Missing output_dir raises ValueError
    with pytest.raises(ValueError, match="output_dir is required"):
        ensure_espirit_maps([f1, f2], calib_width=12)

    # Sequential sidecar execution
    seq_dir = tmp_path / "seq_sidecars"
    out_paths = ensure_espirit_maps([f1, f2], output_dir=seq_dir, calib_width=12, num_workers=1)
    assert len(out_paths) == 2
    for p in [f1, f2]:
        with h5py.File(p, "r") as hf:
            assert "sensitivity_maps" not in hf  # Source untouched
    for p in out_paths:
        assert p.is_file()
        with h5py.File(p, "r") as hf:
            assert "sensitivity_maps" in hf

    # Parallel sidecar execution
    f3 = tmp_path / "f3.h5"
    f4 = tmp_path / "f4.h5"
    _create_mock_kspace_file(f3, num_slices=1, num_coils=4, height=20, width=20)
    _create_mock_kspace_file(f4, num_slices=1, num_coils=4, height=20, width=20)
    out_dir = tmp_path / "parallel_sidecars"

    out_paths_parallel = ensure_espirit_maps(
        [f3, f4], output_dir=out_dir, calib_width=12, num_workers=2
    )
    assert len(out_paths_parallel) == 2
    for f in [f3, f4]:
        with h5py.File(f, "r") as hf:
            assert "sensitivity_maps" not in hf  # Source untouched
        sidecar = out_dir / f.name
        assert sidecar.is_file()
        with h5py.File(sidecar, "r") as hf:
            assert "sensitivity_maps" in hf


def test_fastmri_dataset_auto_calibration_default_sens_dir(tmp_path: Path) -> None:
    """Test FastMRIDataset writes sidecars into default sens_dir when sens_dir=None."""
    data_dir = tmp_path / "raw_data"
    data_dir.mkdir()
    vol_path = data_dir / "raw_vol.h5"
    _create_mock_kspace_file(vol_path, num_slices=2, num_coils=4, height=24, width=24)

    # Verify initially missing sensitivity maps in source
    with h5py.File(vol_path, "r") as hf:
        assert "sensitivity_maps" not in hf

    # Initialize FastMRIDataset with auto_calibrate=True (default) and sens_dir=None
    dataset = FastMRIDataset(data_dir=str(data_dir), use_cache=False)

    # Source file MUST remain untouched (NEVER opened r+)
    with h5py.File(vol_path, "r") as hf:
        assert "sensitivity_maps" not in hf

    # Default sidecar dir should be data_dir.parent / f"{data_dir.name}_sens"
    expected_sens_dir = data_dir.parent / f"{data_dir.name}_sens"
    sidecar_path = expected_sens_dir / "raw_vol.h5"
    assert sidecar_path.is_file()
    with h5py.File(sidecar_path, "r") as hf:
        assert "sensitivity_maps" in hf
        assert hf["sensitivity_maps"].shape == (2, 4, 24, 24)

    # Sample retrieval must succeed
    item = dataset[0]
    assert isinstance(item, torch.Tensor)
    assert item.shape == (1, 24, 24)
    assert item.dtype == torch.complex64


def test_fastmri_dataset_auto_calibration_sidecar(tmp_path: Path) -> None:
    """Test FastMRIDataset automatically calibrates into sens_dir when configured."""
    data_dir = tmp_path / "raw_data_sidecar"
    sens_dir = tmp_path / "sidecar_maps"
    data_dir.mkdir()
    vol_path = data_dir / "raw_vol.h5"
    _create_mock_kspace_file(vol_path, num_slices=2, num_coils=4, height=24, width=24)

    # Initialize FastMRIDataset pointing at sens_dir
    dataset = FastMRIDataset(
        data_dir=str(data_dir),
        sens_dir=str(sens_dir),
        use_cache=False,
    )

    # Source file must remain unmodified
    with h5py.File(vol_path, "r") as hf:
        assert "sensitivity_maps" not in hf

    # Sidecar must have been created in sens_dir
    sidecar_path = sens_dir / "raw_vol.h5"
    assert sidecar_path.is_file()
    with h5py.File(sidecar_path, "r") as hf:
        assert "sensitivity_maps" in hf

    # Sample retrieval must succeed against the freshly written sidecar
    item = dataset[0]
    assert isinstance(item, torch.Tensor)
    assert item.is_complex()
    assert item.shape == (1, 24, 24)


@contextmanager
def _ddp(rank: int, world_size: int) -> Iterator[None]:
    """Make torch.distributed look like an initialised job of the given shape."""
    with ExitStack() as stack:
        for name, value in (
            ("is_available", True),
            ("is_initialized", True),
            ("get_rank", rank),
            ("get_world_size", world_size),
            ("get_backend", "gloo"),
        ):
            stack.enter_context(patch(f"torch.distributed.{name}", return_value=value))
        yield


def test_fastmri_dataset_auto_calibration_ddp_shards_across_ranks(tmp_path: Path) -> None:
    """Each rank calibrates its own stride, so no rank idles in a collective.

    Rank-0-only calibration parks every other rank in a collective for the whole job,
    and NCCL aborts a rank that waits past COLLECTIVE_TIMEOUT (45 min).
    """
    data_dir = tmp_path / "ddp_data"
    data_dir.mkdir()
    for i in range(4):
        _create_mock_kspace_file(
            data_dir / f"vol{i}.h5", num_slices=1, num_coils=4, height=20, width=20
        )

    seen: dict[int, list[str]] = {}
    for rank in (0, 1):
        with (
            _ddp(rank, 2),
            patch("torch.distributed.all_reduce") as mock_all_reduce,
            patch("cyfm.data.espirit.ensure_espirit_maps") as mock_ensure,
        ):
            _ = FastMRIDataset(data_dir=str(data_dir), use_cache=False)
            mock_ensure.assert_called_once()
            seen[rank] = sorted(Path(f).name for f in mock_ensure.call_args[0][0])
            mock_all_reduce.assert_called_once()

    # Disjoint shards that together cover every volume exactly once
    assert set(seen[0]) & set(seen[1]) == set()
    assert sorted(seen[0] + seen[1]) == [f"vol{i}.h5" for i in range(4)]


def test_fastmri_dataset_auto_calibration_ddp_failure_propagation(tmp_path: Path) -> None:
    """Any rank's failure must stop every rank, not just the one that failed."""
    data_dir = tmp_path / "ddp_fail_data"
    data_dir.mkdir()
    _create_mock_kspace_file(data_dir / "vol.h5", num_slices=1, num_coils=4, height=20, width=20)

    # The rank whose own shard fails raises after signalling the others.
    with (
        _ddp(0, 2),
        patch("torch.distributed.all_reduce") as mock_all_reduce,
        patch("cyfm.data.espirit.ensure_espirit_maps", side_effect=RuntimeError("Out of memory")),
    ):
        with pytest.raises(RuntimeError, match="failed on at least one rank"):
            _ = FastMRIDataset(data_dir=str(data_dir), use_cache=False)
        mock_all_reduce.assert_called_once()

    # A rank with an empty or healthy shard still stops when the reduction reports a
    # failure elsewhere, so nobody is left alone in the next collective.
    def fail_elsewhere(tensor, op=None):
        tensor.fill_(1)

    with (
        _ddp(1, 2),
        patch("torch.distributed.all_reduce", side_effect=fail_elsewhere) as mock_all_reduce,
        patch("cyfm.data.espirit.ensure_espirit_maps") as mock_ensure,
    ):
        with pytest.raises(RuntimeError, match="failed on at least one rank"):
            _ = FastMRIDataset(data_dir=str(data_dir), use_cache=False)
        mock_all_reduce.assert_called_once()
        mock_ensure.assert_not_called()  # rank 1 of 2 has no shard for a single volume


def _coil_phantom(num_coils: int = 8, size: int = 64) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a smooth-sensitivity multi-coil phantom and its k-space.

    Returns:
        Tuple (kspace [C, H, W], coil images [C, H, W], ground-truth image [H, W]).
    """
    yy, xx = np.mgrid[0:size, 0:size]
    sens = []
    for c in range(num_coils):
        ang = 2 * np.pi * c / num_coils
        cy, cx = size / 2 + 0.45 * size * np.sin(ang), size / 2 + 0.45 * size * np.cos(ang)
        bump = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (0.5 * size) ** 2)))
        sens.append(bump * np.exp(1j * 0.5 * ang))
    maps = np.asarray(sens, dtype=np.complex64)
    maps /= np.sqrt((np.abs(maps) ** 2).sum(0, keepdims=True)) + 1e-9

    img = ((yy - size / 2) ** 2 + (xx - size / 2) ** 2 < (0.35 * size) ** 2).astype(np.float32)
    img = (img * (1.0 + 0.3 * np.sin(xx / 4.0))).astype(np.complex64)

    coil_imgs = (maps * img[None]).astype(np.complex64)
    kspace = fft2c(torch.from_numpy(coil_imgs)).numpy().astype(np.complex64)
    return kspace, coil_imgs, img


def test_compute_espirit_torch_matches_sigpy() -> None:
    """The torch ESPIRiT must reproduce SigPy on data whose subspace exceeds 24 kernels.

    Regression: truncating the signal subspace to the 24 leading kernels dropped the
    correlation to 0.84 and left an 18% SENSE reconstruction error, while every
    shape/dtype assertion in this file still passed.
    """
    sp = pytest.importorskip("sigpy")
    app = pytest.importorskip("sigpy.mri.app")

    kspace, coil_imgs, truth = _coil_phantom()

    calib = sp.resize(kspace, [kspace.shape[0], 24, 24])
    mat = (
        sp.array_to_blocks(calib, [6, 6], [1, 1])
        .reshape([kspace.shape[0], -1, 36])
        .transpose([1, 0, 2])
        .reshape([-1, kspace.shape[0] * 36])
    )
    singular = np.linalg.svd(mat, full_matrices=False)[1]
    assert int((singular > 0.02 * singular.max()).sum()) > 24, "phantom must exceed the old cap"

    reference = np.asarray(
        app.EspiritCalib(
            kspace,
            calib_width=24,
            thresh=0.02,
            kernel_width=6,
            crop=0.95,
            max_iter=100,
            device=sp.Device(-1),
            show_pbar=False,
        ).run()
    )
    maps = compute_espirit_torch(
        kspace, calib_width=24, kernel_width=6, thresh=0.02, crop=0.95, max_iter=30, device="cpu"
    ).numpy()

    support = np.abs(reference).sum(0) > 1e-6
    a = np.abs(reference)[:, support].ravel()
    b = np.abs(maps)[:, support].ravel()
    correlation = np.abs(np.vdot(a, b)) / (np.linalg.norm(a) * np.linalg.norm(b))
    assert correlation > 0.999, f"correlation with sigpy {correlation:.4f}"

    # The background mask must agree too: a truncated subspace silently shrinks it.
    assert abs(int((np.abs(maps).sum(0) > 1e-6).sum()) - int(support.sum())) < 0.02 * support.size

    # What the dataset actually does with the maps.
    combined = (np.conj(maps) * coil_imgs).sum(0)
    scale = np.vdot(combined[support], truth[support]) / np.vdot(
        combined[support], combined[support]
    )
    error = np.linalg.norm((scale * combined - truth)[support]) / np.linalg.norm(truth[support])
    assert error < 0.01, f"SENSE reconstruction error {error:.4f}"


def test_compute_espirit_torch_honours_calibration_parameters(tmp_path: Path) -> None:
    """Calibration parameters must reach the GPU writer, not be silently dropped."""
    from cyfm.data.torch_espirit import calibrate_fastmri_file_torch

    src = tmp_path / "vol.h5"
    _create_mock_kspace_file(src, num_slices=1, num_coils=4, height=24, width=24)

    with patch("cyfm.data.torch_espirit.compute_espirit_torch") as mock_compute:
        mock_compute.return_value = torch.zeros(4, 24, 24, dtype=torch.complex64)
        calibrate_fastmri_file_torch(
            src_path=src,
            dest_path=tmp_path / "sens" / "vol.h5",
            device="cpu",
            calib_width=12,
            kernel_width=4,
            thresh=0.05,
            crop=0.8,
            max_iter=7,
        )
    kwargs = mock_compute.call_args.kwargs
    assert kwargs["calib_width"] == 12
    assert kwargs["kernel_width"] == 4
    assert kwargs["thresh"] == 0.05
    assert kwargs["crop"] == 0.8
    assert kwargs["max_iter"] == 7


def test_calibrate_torch_recomputes_sidecar_missing_the_key(tmp_path: Path) -> None:
    """A sidecar without the maps is not finished work and must not be skipped."""
    from cyfm.data.torch_espirit import calibrate_fastmri_file_torch

    src = tmp_path / "vol.h5"
    dest = tmp_path / "sens" / "vol.h5"
    _create_mock_kspace_file(src, num_slices=1, num_coils=4, height=24, width=24)
    dest.parent.mkdir(parents=True)
    with h5py.File(dest, "w") as hf:
        hf.create_dataset("unrelated", data=np.zeros(3))

    assert calibrate_fastmri_file_torch(src, dest, device="cpu", max_iter=5, overwrite=False)
    with h5py.File(dest, "r") as hf:
        assert "sensitivity_maps" in hf


def test_compute_espirit_torch_cpu_and_gpu() -> None:
    """Test pure PyTorch ESPIRiT calibration on CPU and GPU (if available)."""
    from cyfm.data.torch_espirit import compute_espirit_torch

    # [num_coils, H, W]
    ksp = torch.randn(4, 32, 32, dtype=torch.complex64)
    maps_cpu = compute_espirit_torch(ksp, calib_width=16, device="cpu", max_iter=10)
    assert maps_cpu.shape == (4, 32, 32)
    assert maps_cpu.dtype == torch.complex64
    assert torch.isfinite(maps_cpu).all()

    if torch.cuda.is_available():
        maps_gpu = compute_espirit_torch(ksp, calib_width=16, device="cuda", max_iter=10)
        assert maps_gpu.shape == (4, 32, 32)
        assert maps_gpu.device.type == "cuda"
        assert torch.isfinite(maps_gpu).all()


def test_calibrate_fastmri_file_torch(tmp_path: Path) -> None:
    """Test calibrate_fastmri_file_torch generates sidecar with correct shape."""
    from cyfm.data.torch_espirit import calibrate_fastmri_file_torch

    src_file = tmp_path / "mock_vol.h5"
    dest_file = tmp_path / "sens" / "mock_vol.h5"
    _create_mock_kspace_file(src_file, num_slices=2, num_coils=4, height=24, width=24)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    success = calibrate_fastmri_file_torch(
        src_path=src_file,
        dest_path=dest_file,
        device=device,
        max_iter=10,
        overwrite=True,
    )
    assert success is True
    assert dest_file.is_file()

    with h5py.File(dest_file, "r") as hf:
        assert "sensitivity_maps" in hf
        assert hf["sensitivity_maps"].shape == (2, 4, 24, 24)
        assert hf["sensitivity_maps"].dtype == np.complex64


def test_calibrating_a_subset_gives_the_same_maps_as_calibrating_everything(
    tmp_path: Path,
) -> None:
    """The property the central-slice optimisation rests on.

    ESPIRiT runs per slice from that slice's own k-space, so restricting the loop is
    not an approximation: the maps for a kept slice must come out bit-identical. If
    calibration ever gains a 3D component this test fails, which is the point -- the
    build would otherwise silently start producing different data.
    """
    rng = np.random.default_rng(0)
    kspace = (rng.normal(size=(6, 3, 24, 24)) + 1j * rng.normal(size=(6, 3, 24, 24))).astype(
        np.complex64
    )
    src = tmp_path / "vol.h5"
    with h5py.File(src, "w") as handle:
        handle.create_dataset("kspace", data=kspace)

    full = tmp_path / "full.h5"
    partial = tmp_path / "partial.h5"
    wanted = [2, 3]
    assert calibrate_fastmri_file_torch(src, full, device="cpu")
    assert calibrate_fastmri_file_torch(src, partial, device="cpu", slices=wanted)

    with h5py.File(full, "r") as a, h5py.File(partial, "r") as b:
        for s in wanted:
            np.testing.assert_array_equal(a["sensitivity_maps"][s], b["sensitivity_maps"][s])
        # And the slices that were skipped are left zeroed rather than wrong.
        assert not np.any(b["sensitivity_maps"][0])
        assert b["sensitivity_maps"].attrs["calibrated_slices"] == "2,3"


def test_a_partial_sidecar_is_not_mistaken_for_a_complete_one(tmp_path: Path) -> None:
    """Reusing a narrower sidecar would hand back zeroed maps, i.e. a black image."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "data"))
    from build_knee_pd_store import has_maps

    rng = np.random.default_rng(1)
    kspace = (rng.normal(size=(6, 3, 24, 24)) + 1j * rng.normal(size=(6, 3, 24, 24))).astype(
        np.complex64
    )
    src = tmp_path / "vol.h5"
    with h5py.File(src, "w") as handle:
        handle.create_dataset("kspace", data=kspace)
    sidecar = tmp_path / "sens.h5"
    assert calibrate_fastmri_file_torch(src, sidecar, device="cpu", slices=[2, 3])

    assert has_maps(sidecar, [2, 3])
    assert has_maps(sidecar, [3])
    assert not has_maps(sidecar, [1, 2, 3])
