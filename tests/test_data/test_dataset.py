import h5py
import numpy as np
import pytest
import torch

from cfm.data.dataset import SKMTEADataset
from cfm.data.transforms import CenterCropModulo, ComplexToCylinderTransform, Compose


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


@pytest.fixture
def dummy_skm_tea_dir_large(tmp_path) -> str:
    """A volume large enough (and not a multiple of 16) that the fixed 24-line
    ACS region plus the acceleration=4 recipe, and CenterCropModulo, are both
    exercised meaningfully."""
    data_dir = tmp_path / "files_recon_calib-24"
    data_dir.mkdir(parents=True)

    file_path = data_dir / "dummy_scan.h5"
    with h5py.File(file_path, "w") as f:
        dummy_target = np.random.randn(2, 50, 140, 2, 4) + 1j * np.random.randn(2, 50, 140, 2, 4)
        f.create_dataset("target", data=dummy_target)

    return str(tmp_path)


def _reconstruction_dataset(data_dir: str, **kwargs) -> SKMTEADataset:
    return SKMTEADataset(
        data_dir=data_dir,
        mode="reconstruction",
        pre_transform=CenterCropModulo(base=16),
        post_transform=ComplexToCylinderTransform(),
        **kwargs,
    )


def test_reconstruction_mask_zeros_out_kspace(dummy_skm_tea_dir_large) -> None:
    dataset = _reconstruction_dataset(dummy_skm_tea_dir_large)
    f_path = dataset.files[0]

    with h5py.File(f_path, "r") as f:
        img_np = f["target"][0, :, :, 0, 0]
    img_complex = torch.from_numpy(np.nan_to_num(img_np)).to(torch.complex64).unsqueeze(0)
    img_complex = CenterCropModulo(base=16)(img_complex)

    mask = dataset._undersampling_mask(f_path, 0, img_complex.shape[1], img_complex.shape[2])
    y = torch.fft.fft2(img_complex, norm="ortho")
    y_under = y * mask

    assert torch.all(y_under[mask == 0] == 0)


def test_reconstruction_energy_inequality(dummy_skm_tea_dir_large) -> None:
    dataset = _reconstruction_dataset(dummy_skm_tea_dir_large)
    f_path = dataset.files[0]

    with h5py.File(f_path, "r") as f:
        img_np = f["target"][0, :, :, 0, 0]
    img_complex = torch.from_numpy(np.nan_to_num(img_np)).to(torch.complex64).unsqueeze(0)
    img_complex = CenterCropModulo(base=16)(img_complex)

    mask = dataset._undersampling_mask(f_path, 0, img_complex.shape[1], img_complex.shape[2])
    y = torch.fft.fft2(img_complex, norm="ortho")
    y_under = y * mask

    assert torch.linalg.norm(y_under) <= torch.linalg.norm(y)


def test_reconstruction_sample_shape_and_keys(dummy_skm_tea_dir_large) -> None:
    dataset = _reconstruction_dataset(dummy_skm_tea_dir_large)
    sample = dataset[0]

    assert set(sample.keys()) == {"input", "mask", "target"}
    assert sample["input"].shape == (3, 48, 128)
    assert sample["target"].shape == (3, 48, 128)
    assert sample["mask"].shape == (1, 48, 128)
    assert sample["input"].dtype == torch.float32
    assert sample["mask"].dtype == torch.float32


def test_reconstruction_mask_deterministic_across_instances(dummy_skm_tea_dir_large) -> None:
    sample_a = _reconstruction_dataset(dummy_skm_tea_dir_large)[0]
    sample_b = _reconstruction_dataset(dummy_skm_tea_dir_large)[0]

    torch.testing.assert_close(sample_a["mask"], sample_b["mask"])


def test_generation_mode_default_is_unaffected(dummy_skm_tea_dir_large) -> None:
    dataset = SKMTEADataset(data_dir=dummy_skm_tea_dir_large)
    assert isinstance(dataset[0], torch.Tensor)


def test_reconstruction_with_multi_slice_raises_value_error(dummy_skm_tea_dir_large) -> None:
    with pytest.raises(ValueError):
        SKMTEADataset(data_dir=dummy_skm_tea_dir_large, mode="reconstruction", num_slices=3)


def test_invalid_mode_raises_value_error(dummy_skm_tea_dir_large) -> None:
    with pytest.raises(ValueError):
        SKMTEADataset(data_dir=dummy_skm_tea_dir_large, mode="bogus")


def test_invalid_acceleration_raises_value_error(dummy_skm_tea_dir_large) -> None:
    with pytest.raises(ValueError):
        SKMTEADataset(data_dir=dummy_skm_tea_dir_large, mode="reconstruction", acceleration=0)
