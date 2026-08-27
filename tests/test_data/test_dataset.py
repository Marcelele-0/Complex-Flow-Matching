import h5py
import numpy as np
import pytest
import torch

from cfm.data.dataset import SKMTEADataset
from cfm.data.transforms import ComplexToCylinderTransform, Compose


@pytest.fixture
def dummy_skm_tea_dir(tmp_path) -> str:
    """Creates a fake directory structure and h5 file for testing."""
    data_dir = tmp_path / "files_recon_calib-24"
    data_dir.mkdir(parents=True)

    file_path = data_dir / "dummy_scan.h5"
    with h5py.File(file_path, "w") as f:
        # According to SKM-TEA format (Nx, Ny, Nz, echoes, coils)
        dummy_target = np.random.randn(4, 32, 32, 2, 4) + 1j * np.random.randn(4, 32, 32, 2, 4)
        f.create_dataset("target", data=dummy_target)

    return str(tmp_path)


def test_skm_tea_dataset_loading(dummy_skm_tea_dir) -> None:
    """Tests initialization and loading without transformations."""
    dataset = SKMTEADataset(data_dir=dummy_skm_tea_dir)
    assert len(dataset) == 4

    sample = dataset[0]
    # Without pipeline, expect a raw complex tensor [1, H, W]
    assert sample.shape == (1, 32, 32)
    assert sample.dtype == torch.complex64


def test_skm_tea_dataset_with_transforms(dummy_skm_tea_dir) -> None:
    """Tests loading with full processing to the cylinder topology."""
    pipeline = Compose([ComplexToCylinderTransform()])
    dataset = SKMTEADataset(data_dir=dummy_skm_tea_dir, transform=pipeline)

    sample = dataset[0]
    # With pipeline, expect a cylinder [3, H, W] float32
    assert sample.shape == (3, 32, 32)
    assert sample.dtype == torch.float32


def test_skm_tea_dataset_num_slices_1_matches_default(dummy_skm_tea_dir) -> None:
    # num_slices defaults to 1; passing it explicitly must not change behavior
    # at all (same __len__, same per-index tensor, bit-for-bit).
    dataset_default = SKMTEADataset(data_dir=dummy_skm_tea_dir)
    dataset_explicit = SKMTEADataset(data_dir=dummy_skm_tea_dir, num_slices=1)

    assert len(dataset_default) == len(dataset_explicit)

    for idx in range(len(dataset_default)):
        torch.testing.assert_close(dataset_explicit[idx], dataset_default[idx])


def test_skm_tea_dataset_num_slices_3_shape(dummy_skm_tea_dir) -> None:
    """Tests that the num_slices=3 window loads with the expected [3, 1, H, W]
    raw shape and [3, 3, H, W] shape once the per-slice pipeline is applied."""
    dataset_raw = SKMTEADataset(data_dir=dummy_skm_tea_dir, num_slices=3)
    assert len(dataset_raw) == len(SKMTEADataset(data_dir=dummy_skm_tea_dir))

    sample_raw = dataset_raw[0]
    assert sample_raw.shape == (3, 1, 32, 32)
    assert sample_raw.dtype == torch.complex64

    pipeline = Compose([ComplexToCylinderTransform()])
    dataset_transformed = SKMTEADataset(
        data_dir=dummy_skm_tea_dir, transform=pipeline, num_slices=3
    )
    sample_transformed = dataset_transformed[0]
    assert sample_transformed.shape == (3, 3, 32, 32)
    assert sample_transformed.dtype == torch.float32
