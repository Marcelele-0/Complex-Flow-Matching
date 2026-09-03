"""Unit tests for k-space undersampling mask generators and registry integration."""

import numpy as np
import pytest
import torch

from cfm.core.registry import MASKS
from cfm.data.masks import (
    CartesianMaskGenerator,
    PoissonDiscMaskGenerator,
)

# --- Registry Tests ---


def test_masks_registry_contains_generators() -> None:
    """Verify mask registry contains cartesian and poisson_disc generators."""
    assert "cartesian" in MASKS
    assert "poisson_disc" in MASKS
    assert MASKS.get("cartesian") is CartesianMaskGenerator
    assert MASKS.get("poisson_disc") is PoissonDiscMaskGenerator


def test_masks_registry_build() -> None:
    """Verify building mask generators via MASKS.build factory."""
    cart = MASKS.build("cartesian", acceleration=4)
    assert isinstance(cart, CartesianMaskGenerator)
    assert cart.acceleration == 4.0

    pd = MASKS.build("poisson_disc", acceleration=8)
    assert isinstance(pd, PoissonDiscMaskGenerator)
    assert pd.acceleration == 8.0


# --- BaseMaskGenerator Tests ---


def test_base_mask_generator_validation() -> None:
    """Verify input validation on acceleration and shape dimensionality."""
    with pytest.raises(ValueError, match=r"acceleration must be >= 1.0"):
        CartesianMaskGenerator(acceleration=0.5)

    gen = CartesianMaskGenerator(acceleration=4)
    with pytest.raises(ValueError, match=r"shape must have at least 2 dimensions"):
        gen.generate((128,))

    # Callable invocation alias
    mask1 = gen((48, 128), seed=42)
    mask2 = gen.generate((48, 128), seed=42)
    assert torch.equal(mask1, mask2)


# --- CartesianMaskGenerator Tests ---


@pytest.mark.parametrize("accel", [4.0, 8.0, 12.0])
def test_cartesian_multi_acceleration_shapes_and_values(accel: float) -> None:
    """Verify output tensor shapes, binary values, and dtypes for 4x, 8x, 12x."""
    gen = CartesianMaskGenerator(acceleration=accel)
    h, w = 64, 128
    mask = gen((h, w), seed=123)

    assert mask.shape == (1, h, w)
    assert mask.dtype == torch.float32
    assert torch.all((mask == 0.0) | (mask == 1.0))

    # Readout lines are full lines (every column has identical row values)
    col_profile = mask[0, 0, :]
    for row in range(h):
        assert torch.equal(mask[0, row, :], col_profile)


def test_cartesian_sparsity_ordering() -> None:
    """Verify that higher acceleration yields sparser mask (12x < 8x < 4x)."""
    h, w = 128, 256
    m4 = CartesianMaskGenerator(acceleration=4)((h, w), seed=42)
    m8 = CartesianMaskGenerator(acceleration=8)((h, w), seed=42)
    m12 = CartesianMaskGenerator(acceleration=12)((h, w), seed=42)

    kept_4 = m4.sum().item()
    kept_8 = m8.sum().item()
    kept_12 = m12.sum().item()

    assert kept_12 < kept_8 < kept_4


def test_cartesian_fully_sampled_center() -> None:
    """Verify center ACS region is fully sampled."""
    w = 128
    gen = CartesianMaskGenerator(acceleration=4, num_center_lines=24)
    mask = gen((32, w), seed=42)

    center_start = w // 2 - 24 // 2
    center_end = center_start + 24
    assert torch.all(mask[0, :, center_start:center_end] == 1.0)


def test_cartesian_center_fraction() -> None:
    """Verify center_fraction configuration."""
    w = 200
    gen = CartesianMaskGenerator(acceleration=4, center_fraction=0.1)
    mask = gen((32, w), seed=42)

    center_lines = int(round(w * 0.1))  # 20 lines
    c_start = w // 2 - center_lines // 2
    c_end = c_start + center_lines
    assert torch.all(mask[0, :, c_start:c_end] == 1.0)


def test_cartesian_axis_parameter() -> None:
    """Verify undersampling along height (axis=-2) versus width (axis=-1)."""
    h, w = 128, 64
    gen_y = CartesianMaskGenerator(acceleration=4, axis=-2)
    mask_y = gen_y((h, w), seed=42)

    assert mask_y.shape == (1, h, w)
    # Along axis=-2, every row has uniform value across columns
    row_profile = mask_y[0, :, 0]
    for col in range(w):
        assert torch.equal(mask_y[0, :, col], row_profile)


def test_cartesian_determinism_and_seed() -> None:
    """Verify deterministic generation with seed and divergence across different seeds."""
    gen = CartesianMaskGenerator(acceleration=4)
    m1 = gen((64, 128), seed=999)
    m2 = gen((64, 128), seed=999)
    m3 = gen((64, 128), seed=111)

    assert torch.equal(m1, m2)
    assert not torch.equal(m1, m3)


def test_cartesian_full_sampling() -> None:
    """Verify acceleration 1.0 yields all ones."""
    gen = CartesianMaskGenerator(acceleration=1.0)
    mask = gen((48, 64))
    assert torch.all(mask == 1.0)


# --- PoissonDiscMaskGenerator Tests ---


@pytest.mark.parametrize("accel", [4.0, 8.0, 12.0])
def test_poisson_disc_multi_acceleration_shapes_and_values(accel: float) -> None:
    """Verify output tensor shapes, binary values, and dtypes for 4x, 8x, 12x."""
    gen = PoissonDiscMaskGenerator(acceleration=accel, tol=0.08)
    h, w = 64, 64
    mask = gen((h, w), seed=42)

    assert mask.shape == (1, h, w)
    assert mask.dtype == torch.float32
    assert torch.all((mask == 0.0) | (mask == 1.0))

    actual_accel = (h * w) / mask.sum().item()
    # Tolerance allows minor discretization effects on small grids
    assert abs(actual_accel - accel) / accel < 0.15


def test_poisson_disc_sparsity_ordering() -> None:
    """Verify that higher acceleration yields sparser mask (12x < 8x < 4x)."""
    h, w = 128, 128
    m4 = PoissonDiscMaskGenerator(acceleration=4)((h, w), seed=42)
    m8 = PoissonDiscMaskGenerator(acceleration=8)((h, w), seed=42)
    m12 = PoissonDiscMaskGenerator(acceleration=12)((h, w), seed=42)

    kept_4 = m4.sum().item()
    kept_8 = m8.sum().item()
    kept_12 = m12.sum().item()

    assert kept_12 < kept_8 < kept_4


def test_poisson_disc_fully_sampled_center() -> None:
    """Verify calibration center box is fully sampled."""
    h, w = 64, 64
    calib = (16, 16)
    gen = PoissonDiscMaskGenerator(acceleration=4, calib_size=calib)
    mask = gen((h, w), seed=42)

    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    c_y0 = max(0, int(np.floor(cy - calib[0] / 2.0)))
    c_y1 = min(h, int(np.ceil(cy + calib[0] / 2.0)))
    c_x0 = max(0, int(np.floor(cx - calib[1] / 2.0)))
    c_x1 = min(w, int(np.ceil(cx + calib[1] / 2.0)))

    assert torch.all(mask[0, c_y0:c_y1, c_x0:c_x1] == 1.0)


def test_poisson_disc_calib_fraction() -> None:
    """Verify calibration center specified via fraction."""
    h, w = 80, 80
    gen = PoissonDiscMaskGenerator(acceleration=4, calib_fraction=0.1)
    mask = gen((h, w), seed=42)

    ch, cw = int(round(h * 0.1)), int(round(w * 0.1))
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    c_y0 = max(0, int(np.floor(cy - ch / 2.0)))
    c_y1 = min(h, int(np.ceil(cy + ch / 2.0)))
    c_x0 = max(0, int(np.floor(cx - cw / 2.0)))
    c_x1 = min(w, int(np.ceil(cx + cw / 2.0)))

    assert torch.all(mask[0, c_y0:c_y1, c_x0:c_x1] == 1.0)


def test_poisson_disc_crop_corners() -> None:
    """Verify crop_corners=True zeroes out outer corners where normalized radius > 1.0."""
    h, w = 64, 64
    gen = PoissonDiscMaskGenerator(acceleration=4, crop_corners=True)
    mask = gen((h, w), seed=42)

    # Corners: (0, 0), (0, w-1), (h-1, 0), (h-1, w-1)
    assert mask[0, 0, 0] == 0.0
    assert mask[0, 0, w - 1] == 0.0
    assert mask[0, h - 1, 0] == 0.0
    assert mask[0, h - 1, w - 1] == 0.0


def test_poisson_disc_determinism_and_seed() -> None:
    """Verify reproducibility with fixed seed."""
    gen = PoissonDiscMaskGenerator(acceleration=4)
    m1 = gen((64, 64), seed=777)
    m2 = gen((64, 64), seed=777)
    m3 = gen((64, 64), seed=888)

    assert torch.equal(m1, m2)
    assert not torch.equal(m1, m3)


def test_poisson_disc_full_sampling() -> None:
    """Verify acceleration 1.0 yields all ones."""
    gen = PoissonDiscMaskGenerator(acceleration=1.0)
    mask = gen((32, 32))
    assert torch.all(mask == 1.0)


# --- SKMTEADataset Integration Tests ---


@pytest.fixture
def dummy_h5_dir(tmp_path) -> str:
    """Creates a temporary HDF5 file formatted for SKM-TEA."""
    import h5py

    data_dir = tmp_path / "files_recon_calib-24"
    data_dir.mkdir(parents=True)
    file_path = data_dir / "test_scan.h5"
    with h5py.File(file_path, "w") as f:
        dummy_target = np.random.randn(2, 48, 128, 1, 1) + 1j * np.random.randn(2, 48, 128, 1, 1)
        f.create_dataset("target", data=dummy_target)
    return str(tmp_path)


def test_dataset_integration_cartesian_default(dummy_h5_dir) -> None:
    """Verify SKMTEADataset defaults to Cartesian mask from registry."""
    from cfm.data.dataset import SKMTEADataset

    ds = SKMTEADataset(data_dir=dummy_h5_dir, mode="reconstruction", acceleration=4)
    assert isinstance(ds.mask_generator, CartesianMaskGenerator)
    assert ds.mask_generator.acceleration == 4.0

    sample = ds[0]
    assert isinstance(sample, dict)
    assert sample["mask"].shape == (1, 48, 128)
    assert sample["mask"].dtype == torch.float32


def test_dataset_integration_poisson_disc(dummy_h5_dir) -> None:
    """Verify SKMTEADataset can be configured with poisson_disc mask."""
    from cfm.data.dataset import SKMTEADataset

    ds = SKMTEADataset(
        data_dir=dummy_h5_dir,
        mode="reconstruction",
        mask="poisson_disc",
        acceleration=8,
    )
    assert isinstance(ds.mask_generator, PoissonDiscMaskGenerator)
    assert ds.mask_generator.acceleration == 8.0

    sample = ds[0]
    assert isinstance(sample, dict)
    assert sample["mask"].shape == (1, 48, 128)
    # Check effective acceleration is close to 8x
    actual_accel = (48 * 128) / sample["mask"].sum().item()
    assert abs(actual_accel - 8.0) / 8.0 < 0.15


def test_dataset_integration_hydra_dict_config(dummy_h5_dir) -> None:
    """Verify SKMTEADataset parses Hydra dictionary configuration."""
    from cfm.data.dataset import SKMTEADataset

    mask_cfg = {"type": "cartesian", "acceleration": 12, "num_center_lines": 8}
    ds = SKMTEADataset(
        data_dir=dummy_h5_dir,
        mode="reconstruction",
        mask=mask_cfg,
    )
    assert isinstance(ds.mask_generator, CartesianMaskGenerator)
    assert ds.acceleration == 12.0
    assert ds.mask_generator.num_center_lines == 8

    sample = ds[0]
    assert isinstance(sample, dict)
    assert sample["mask"].shape == (1, 48, 128)


def test_dataset_integration_explicit_mask_generator_instance(dummy_h5_dir) -> None:
    """Verify SKMTEADataset accepts an instantiated BaseMaskGenerator."""
    from cfm.data.dataset import SKMTEADataset

    custom_gen = CartesianMaskGenerator(acceleration=6, num_center_lines=12)
    ds = SKMTEADataset(
        data_dir=dummy_h5_dir,
        mode="reconstruction",
        mask_generator=custom_gen,
    )
    assert ds.mask_generator is custom_gen
    assert ds.acceleration == 6.0
