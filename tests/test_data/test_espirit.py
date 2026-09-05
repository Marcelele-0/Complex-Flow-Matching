"""Unit tests for ESPIRiT calibration and automatic FastMRIDataset calibration."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch

from cfm.data import (
    FastMRIDataset,
    compute_espirit_maps,
    ensure_espirit_maps,
)
from cfm.data.espirit import process_h5_file
from cfm.utils.fft import fft2c

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


def test_process_h5_file_in_place_and_sidecar(tmp_path: Path) -> None:
    """Test process_h5_file writes maps in-place and as sidecar files."""
    f_in_place = tmp_path / "vol1.h5"
    _create_mock_kspace_file(f_in_place, num_slices=2, num_coils=4, height=24, width=24)

    # In-place run
    success = process_h5_file(f_in_place, calib_width=12, overwrite=False)
    assert success is True

    with h5py.File(f_in_place, "r") as hf:
        assert "sensitivity_maps" in hf
        assert hf["sensitivity_maps"].shape == (2, 4, 24, 24)

    # Re-running without overwrite skips
    assert process_h5_file(f_in_place, calib_width=12, overwrite=False) is False
    # Re-running with overwrite succeeds
    assert process_h5_file(f_in_place, calib_width=12, overwrite=True) is True

    # Sidecar run
    f_src = tmp_path / "vol2.h5"
    out_dir = tmp_path / "sidecars"
    _create_mock_kspace_file(f_src, num_slices=2, num_coils=4, height=24, width=24)

    success_sidecar = process_h5_file(f_src, calib_width=12, output_dir=out_dir, overwrite=False)
    assert success_sidecar is True

    # Source remains untouched
    with h5py.File(f_src, "r") as hf:
        assert "sensitivity_maps" not in hf

    # Sidecar exists
    sidecar_path = out_dir / "vol2.h5"
    assert sidecar_path.is_file()
    with h5py.File(sidecar_path, "r") as hf:
        assert "sensitivity_maps" in hf
        assert hf["sensitivity_maps"].shape == (2, 4, 24, 24)


def test_ensure_espirit_maps_multiple_files(tmp_path: Path) -> None:
    """Test ensure_espirit_maps over multiple files sequentially and in parallel."""
    f1 = tmp_path / "f1.h5"
    f2 = tmp_path / "f2.h5"
    _create_mock_kspace_file(f1, num_slices=1, num_coils=4, height=20, width=20)
    _create_mock_kspace_file(f2, num_slices=1, num_coils=4, height=20, width=20)

    # Empty list
    assert ensure_espirit_maps([]) == []

    # Sequential in-place execution
    out_paths = ensure_espirit_maps([f1, f2], calib_width=12, num_workers=1)
    assert len(out_paths) == 2
    for p in [f1, f2]:
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
        sidecar = out_dir / f.name
        assert sidecar.is_file()
        with h5py.File(sidecar, "r") as hf:
            assert "sensitivity_maps" in hf


def test_fastmri_dataset_auto_calibration_in_place(tmp_path: Path) -> None:
    """Test FastMRIDataset automatically calibrates raw HDF5 files lacking sensitivity maps."""
    data_dir = tmp_path / "raw_data"
    data_dir.mkdir()
    vol_path = data_dir / "raw_vol.h5"
    _create_mock_kspace_file(vol_path, num_slices=2, num_coils=4, height=24, width=24)

    # Verify initially missing sensitivity maps
    with h5py.File(vol_path, "r") as hf:
        assert "sensitivity_maps" not in hf

    # Initialize FastMRIDataset with auto_calibrate=True (default)
    dataset = FastMRIDataset(data_dir=str(data_dir), mode="generation", use_cache=False)

    # Sensitivity maps must now exist in-place
    with h5py.File(vol_path, "r") as hf:
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
        mode="reconstruction",
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

    # Reconstruction sample retrieval must succeed
    item = dataset[0]
    assert isinstance(item, dict)
    assert "input" in item
    assert "sensitivity_maps" in item
    assert item["sensitivity_maps"].shape == (4, 24, 24)


def test_fastmri_dataset_auto_calibration_ddp_coordination(tmp_path: Path) -> None:
    """Test DDP coordination: rank 0 calibrates while other ranks wait at barrier."""
    data_dir = tmp_path / "ddp_data"
    data_dir.mkdir()
    vol_path = data_dir / "vol.h5"
    _create_mock_kspace_file(vol_path, num_slices=1, num_coils=4, height=20, width=20)

    # Mock rank 1 in DDP: should not call ensure_espirit_maps directly, only barrier
    with (
        patch("torch.distributed.is_available", return_value=True),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_rank", return_value=1),
        patch("torch.distributed.barrier") as mock_barrier,
        patch("cfm.data.espirit.ensure_espirit_maps") as mock_ensure,
    ):
        _ = FastMRIDataset(data_dir=str(data_dir), use_cache=False)
        mock_barrier.assert_called_once()
        mock_ensure.assert_not_called()

    # Mock rank 0 in DDP: should call ensure_espirit_maps and then barrier
    with (
        patch("torch.distributed.is_available", return_value=True),
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_rank", return_value=0),
        patch("torch.distributed.barrier") as mock_barrier,
        patch("cfm.data.espirit.ensure_espirit_maps") as mock_ensure,
    ):
        _ = FastMRIDataset(data_dir=str(data_dir), use_cache=False)
        mock_ensure.assert_called_once()
        mock_barrier.assert_called_once()
