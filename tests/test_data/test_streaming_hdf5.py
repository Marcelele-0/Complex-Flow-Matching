"""Unit tests for streaming HDF5 loader, WorkerHDF5Manager, and index caching."""

from __future__ import annotations

import json
import os
import time

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from cfm.core.registry import DATASETS
from cfm.data.dataset import SKMTEADataset, _process_slice_array
from cfm.data.hdf5_manager import WorkerHDF5Manager
from cfm.data.transforms import ComplexToCylinderTransform, Compose


@pytest.fixture
def multi_volume_h5_dir(tmp_path) -> str:
    """Creates a temporary dataset directory with multiple multi-echo, multi-coil HDF5 volumes."""
    data_dir = tmp_path / "files_recon_calib-24"
    data_dir.mkdir(parents=True)

    for vol_idx in range(3):
        file_path = data_dir / f"scan_{vol_idx:03d}.h5"
        with h5py.File(file_path, "w", libver="latest") as f:
            # (Nx=6, Ny=32, Nz=32, echoes=2, coils=4)
            depth = 6
            re = np.arange(depth * 32 * 32 * 2 * 4, dtype=np.float32).reshape(depth, 32, 32, 2, 4)
            im = np.ones((depth, 32, 32, 2, 4), dtype=np.float32)
            data = re + 1j * im
            f.create_dataset("target", data=data)

    return str(tmp_path)


# ---------------------------------------------------------------------------
# 1. WorkerHDF5Manager Tests
# ---------------------------------------------------------------------------


def test_hdf5_manager_singleton_and_reset(multi_volume_h5_dir) -> None:
    WorkerHDF5Manager.reset()
    mgr1 = WorkerHDF5Manager.get_instance()
    mgr2 = WorkerHDF5Manager.get_instance()
    assert mgr1 is mgr2

    f_path = os.path.join(multi_volume_h5_dir, "files_recon_calib-24", "scan_000.h5")
    h1 = mgr1.get_handle(f_path)
    assert h1.id.valid

    WorkerHDF5Manager.reset()
    assert WorkerHDF5Manager._instance is None
    assert not h1.id.valid


def test_hdf5_manager_handles_caching_and_closing(multi_volume_h5_dir) -> None:
    WorkerHDF5Manager.reset()
    mgr = WorkerHDF5Manager.get_instance(rdcc_nbytes=2 * 1024 * 1024, rdcc_nslots=101)
    f_path = os.path.join(multi_volume_h5_dir, "files_recon_calib-24", "scan_000.h5")

    h1 = mgr.get_handle(f_path)
    h2 = mgr.get_handle(f_path)
    assert h1 is h2

    mgr.close_handle(f_path)
    assert not h1.id.valid

    # Reopening gets a new handle
    h3 = mgr.get_handle(f_path)
    assert h3.id.valid
    assert h3 is not h1
    mgr.close_all()


def test_hdf5_manager_fork_safety_pid_change(multi_volume_h5_dir) -> None:
    WorkerHDF5Manager.reset()
    mgr = WorkerHDF5Manager.get_instance()
    f_path = os.path.join(multi_volume_h5_dir, "files_recon_calib-24", "scan_000.h5")
    mgr.get_handle(f_path)
    assert len(mgr._handles) == 1

    # Simulate process fork by setting old PID
    mgr._pid = os.getpid() - 9999
    mgr._check_pid()

    # Handles should have been cleared without raising
    assert len(mgr._handles) == 0
    assert mgr._pid == os.getpid()
    mgr.close_all()


def test_hdf5_manager_context_manager(multi_volume_h5_dir) -> None:
    f_path = os.path.join(multi_volume_h5_dir, "files_recon_calib-24", "scan_000.h5")
    with WorkerHDF5Manager() as mgr:
        h = mgr.get_handle(f_path)
        assert h.id.valid
    assert not h.id.valid


# ---------------------------------------------------------------------------
# 2. Multi-Echo & Multi-Coil API Tests
# ---------------------------------------------------------------------------


def test_process_slice_array_echo_and_coil_combinations() -> None:
    # Shape: [32, 32, 2, 4]
    raw = np.arange(32 * 32 * 2 * 4, dtype=np.float32).reshape(32, 32, 2, 4) + 1j * np.ones(
        (32, 32, 2, 4), dtype=np.float32
    )

    # 1. Single echo, single coil -> [1, 32, 32]
    t1 = _process_slice_array(raw, echo_idx=0, coil_idx=0)
    assert t1.shape == (1, 32, 32)
    assert t1.dtype == torch.complex64
    assert np.isclose(t1[0, 0, 0].item(), raw[0, 0, 0, 0])

    # 2. Echo "both", coil 0 -> [2, 32, 32]
    t2 = _process_slice_array(raw, echo_idx="both", coil_idx=0)
    assert t2.shape == (2, 32, 32)
    assert np.isclose(t2[1, 0, 0].item(), raw[0, 0, 1, 0])

    # 3. Echo "average", coil 0 -> [1, 32, 32]
    t3 = _process_slice_array(raw, echo_idx="average", coil_idx=0)
    assert t3.shape == (1, 32, 32)
    expected_mean = np.mean(raw[0, 0, :, 0])
    assert np.isclose(t3[0, 0, 0].item(), expected_mean)

    # 4. Echo 0, coil "rss" -> [1, 32, 32]
    t4 = _process_slice_array(raw, echo_idx=0, coil_idx="rss")
    assert t4.shape == (1, 32, 32)
    expected_rss = np.sqrt(np.sum(np.abs(raw[0, 0, 0, :]) ** 2))
    assert np.isclose(t4[0, 0, 0].item().real, expected_rss, atol=1e-4)

    # 5. Echo 0, coil "all" -> [4, 32, 32]
    t5 = _process_slice_array(raw, echo_idx=0, coil_idx="all")
    assert t5.shape == (4, 32, 32)

    # 6. Echo "both", coil "all" -> [8, 32, 32]
    t6 = _process_slice_array(raw, echo_idx="both", coil_idx="all")
    assert t6.shape == (8, 32, 32)

    # 7. Invalid selectors raise errors
    with pytest.raises(IndexError):
        _process_slice_array(raw, echo_idx=5, coil_idx=0)

    with pytest.raises(IndexError):
        _process_slice_array(raw, echo_idx=0, coil_idx=10)

    with pytest.raises(ValueError):
        _process_slice_array(raw, echo_idx="invalid_echo", coil_idx=0)

    with pytest.raises(ValueError):
        _process_slice_array(raw, echo_idx=0, coil_idx="invalid_coil")


def test_dataset_multi_echo_multi_coil(multi_volume_h5_dir) -> None:
    dataset_rss = SKMTEADataset(
        data_dir=multi_volume_h5_dir, echo_idx="average", coil_idx="rss", use_cache=False
    )
    assert len(dataset_rss) == 18  # 3 files * 6 slices
    sample_rss = dataset_rss[0]
    assert isinstance(sample_rss, torch.Tensor)
    assert sample_rss.shape == (1, 32, 32)

    dataset_all = SKMTEADataset(
        data_dir=multi_volume_h5_dir, echo_idx="both", coil_idx="all", use_cache=False
    )
    sample_all = dataset_all[0]
    assert isinstance(sample_all, torch.Tensor)
    assert sample_all.shape == (8, 32, 32)


# ---------------------------------------------------------------------------
# 3. Persistent Index Caching Tests
# ---------------------------------------------------------------------------


def test_index_cache_save_and_load(multi_volume_h5_dir, tmp_path) -> None:
    cache_dir = tmp_path / "custom_cache"
    cache_dir.mkdir(parents=True)

    # 1. First run builds cache
    dataset1 = SKMTEADataset(data_dir=multi_volume_h5_dir, use_cache=True, cache_dir=str(cache_dir))
    assert len(dataset1) == 18

    cache_files = list(cache_dir.glob("skmtea_index_*.json"))
    assert len(cache_files) == 1

    with open(cache_files[0], encoding="utf-8") as fh:
        payload = json.load(fh)
    assert "volume_depths" in payload
    assert "slice_map" in payload
    assert len(payload["slice_map"]) == 18

    # 2. Second run loads directly from cache
    dataset2 = SKMTEADataset(data_dir=multi_volume_h5_dir, use_cache=True, cache_dir=str(cache_dir))
    assert dataset1.slice_map == dataset2.slice_map
    assert dataset1._volume_depths == dataset2._volume_depths


def test_index_cache_invalidation_on_file_modification(multi_volume_h5_dir, tmp_path) -> None:
    cache_dir = tmp_path / "invalidation_cache"
    cache_dir.mkdir(parents=True)

    dataset1 = SKMTEADataset(data_dir=multi_volume_h5_dir, use_cache=True, cache_dir=str(cache_dir))
    assert len(dataset1) == 18
    old_cache_files = list(cache_dir.glob("skmtea_index_*.json"))
    assert len(old_cache_files) == 1

    # Modify an HDF5 volume (update depth)
    WorkerHDF5Manager.reset()
    file_path = os.path.join(multi_volume_h5_dir, "files_recon_calib-24", "scan_000.h5")
    time.sleep(0.01)  # Ensure mtime changes
    with h5py.File(file_path, "a") as f:
        del f["target"]
        f.create_dataset("target", data=np.zeros((8, 32, 32, 2, 4), dtype=np.complex64))

    dataset2 = SKMTEADataset(data_dir=multi_volume_h5_dir, use_cache=True, cache_dir=str(cache_dir))
    assert len(dataset2) == 20  # 8 + 6 + 6

    new_cache_files = list(cache_dir.glob("skmtea_index_*.json"))
    assert len(new_cache_files) == 2  # New hash generated


def test_index_cache_corrupt_fallback(multi_volume_h5_dir, tmp_path) -> None:
    cache_dir = tmp_path / "corrupt_cache"
    cache_dir.mkdir(parents=True)

    dataset1 = SKMTEADataset(data_dir=multi_volume_h5_dir, use_cache=True, cache_dir=str(cache_dir))
    assert len(dataset1) == 18
    cache_file = list(cache_dir.glob("skmtea_index_*.json"))[0]

    # Corrupt the cache file
    with open(cache_file, "w", encoding="utf-8") as fh:
        fh.write("invalid json content {{{")

    # Should recover gracefully and re-scan
    dataset2 = SKMTEADataset(data_dir=multi_volume_h5_dir, use_cache=True, cache_dir=str(cache_dir))
    assert len(dataset2) == 18


# ---------------------------------------------------------------------------
# 4. 2.5D Contiguous Window Streaming Tests
# ---------------------------------------------------------------------------


def test_2_5d_window_contiguous_reading(multi_volume_h5_dir) -> None:
    pipeline = Compose([ComplexToCylinderTransform()])
    dataset = SKMTEADataset(
        data_dir=multi_volume_h5_dir,
        num_slices=3,
        transform=pipeline,
        use_cache=False,
    )
    assert len(dataset) == 18

    # Edge slice (slice 0) should use reflection [1, 0, 1]
    sample_0 = dataset[0]
    assert isinstance(sample_0, torch.Tensor)
    assert sample_0.shape == (3, 3, 32, 32)
    # Reflected boundaries at slice 0: offset -1 is slice 1, offset 0 is slice 0
    torch.testing.assert_close(sample_0[0], sample_0[2])

    # Middle slice
    sample_2 = dataset[2]
    assert isinstance(sample_2, torch.Tensor)
    assert sample_2.shape == (3, 3, 32, 32)


# ---------------------------------------------------------------------------
# 5. Multiprocessing DataLoader Verification
# ---------------------------------------------------------------------------


def test_multiprocessing_dataloader_num_workers_2(multi_volume_h5_dir, tmp_path) -> None:
    cache_dir = tmp_path / "mp_cache"
    dataset = SKMTEADataset(
        data_dir=multi_volume_h5_dir,
        use_cache=True,
        cache_dir=str(cache_dir),
    )

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2)

    total_slices = 0
    for batch in loader:
        assert isinstance(batch, torch.Tensor)
        assert batch.dim() == 4
        assert batch.shape[-2:] == (32, 32)
        total_slices += batch.shape[0]

    assert total_slices == 18


# ---------------------------------------------------------------------------
# 6. Core Registry Aliases
# ---------------------------------------------------------------------------


def test_dataset_registry_aliases() -> None:
    for name in ("skm_tea", "skmtea", "skmtea_streaming", "skmtea_full"):
        assert name in DATASETS
        assert DATASETS.contains(name)
        cls = DATASETS.get(name)
        assert cls is SKMTEADataset


def test_process_slice_array_special_dtypes_and_nans() -> None:
    # 2D array
    raw_2d = np.array([[1.0, 2.0], [3.0, np.nan]], dtype=np.float32)
    t_2d = _process_slice_array(raw_2d)
    assert t_2d.shape == (1, 2, 2)
    assert t_2d.dtype == torch.complex64
    assert t_2d[0, 1, 1] == 0.0  # NaN replaced

    # 3D array [H, W, E]
    raw_3d = np.ones((4, 4, 2), dtype=np.complex64)
    t_3d = _process_slice_array(raw_3d, echo_idx="both")
    assert t_3d.shape == (2, 4, 4)

    # Compound complex dtype [('r', '<f4'), ('i', '<f4')]
    compound_dtype = np.dtype([("r", "<f4"), ("i", "<f4")])
    raw_compound = np.zeros((4, 4, 1, 1), dtype=compound_dtype)
    raw_compound["r"] = 3.0
    raw_compound["i"] = 4.0
    t_comp = _process_slice_array(raw_compound)
    assert t_comp.shape == (1, 4, 4)
    assert torch.allclose(t_comp.real, torch.tensor(3.0))
    assert torch.allclose(t_comp.imag, torch.tensor(4.0))


def test_dataset_build_via_registry(multi_volume_h5_dir) -> None:
    dataset = DATASETS.build("skmtea_streaming", data_dir=multi_volume_h5_dir, use_cache=False)
    assert isinstance(dataset, SKMTEADataset)
    assert len(dataset) == 18


def test_dataset_reconstruction_multiprocessing(multi_volume_h5_dir) -> None:
    dataset = SKMTEADataset(
        data_dir=multi_volume_h5_dir,
        mode="reconstruction",
        echo_idx=0,
        coil_idx=0,
        use_cache=False,
    )
    loader = DataLoader(dataset, batch_size=3, num_workers=2, shuffle=False)
    batches = list(loader)
    assert len(batches) == 6
    for b in batches:
        assert "input" in b and "mask" in b and "target" in b
        assert b["input"].shape == (3, 1, 32, 32)


def test_dataset_missing_dir_raises_file_not_found(tmp_path) -> None:
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError):
        SKMTEADataset(data_dir=str(empty_dir))
