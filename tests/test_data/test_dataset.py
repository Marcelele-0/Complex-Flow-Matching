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
