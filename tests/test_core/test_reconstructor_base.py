"""Tests for BaseReconstructor interface and implementations (ZeroFilled, FlowMatching)."""

import pytest
import torch

from cfm.core.reconstructor import (
    BaseReconstructor,
    FlowMatchingReconstructor,
    ZeroFilledReconstructor,
)
from cfm.manifolds.cylindrical import CylindricalManifold
from cfm.models.cylindrical_unet import CylindricalUNet


def test_cannot_instantiate_abstract_reconstructor() -> None:
    """BaseReconstructor cannot be instantiated directly without reconstruct()."""
    with pytest.raises(TypeError):
        BaseReconstructor()  # type: ignore[abstract]


def test_zero_filled_reconstructor_identity_on_fully_sampled() -> None:
    """Zero-filled reconstructor recovers ground truth exactly when mask is all ones."""
    recon = ZeroFilledReconstructor()
    b, h, w = 2, 16, 16

    # Ground truth complex image
    x_gt = torch.complex(torch.randn(b, 1, h, w), torch.randn(b, 1, h, w))
    # Forward k-space
    kspace = torch.fft.fftshift(torch.fft.fft2(x_gt, norm="ortho"), dim=(-2, -1))
    mask = torch.ones(b, 1, h, w)

    x_rec = recon.reconstruct(masked_kspace=kspace, mask=mask)

    assert x_rec.shape == (b, 1, h, w)
    assert x_rec.is_complex()
    torch.testing.assert_close(x_rec, x_gt, atol=1e-5, rtol=1e-5)


def test_zero_filled_reconstructor_with_sensitivity_maps() -> None:
    """Zero-filled reconstructor combines multi-coil images using coil sensitivity maps."""
    recon = ZeroFilledReconstructor()
    b, coils, h, w = 2, 4, 16, 16

    # Generate synthetic sensitivity maps normalized sum |S|^2 = 1
    sens_maps = torch.complex(torch.randn(b, coils, h, w), torch.randn(b, coils, h, w))
    norm = torch.sqrt((sens_maps.abs() ** 2).sum(dim=1, keepdim=True) + 1e-8)
    sens_maps = sens_maps / norm

    x_gt = torch.complex(torch.randn(b, 1, h, w), torch.randn(b, 1, h, w))
    # Coil images
    x_coils = sens_maps * x_gt
    kspace_coils = torch.fft.fftshift(torch.fft.fft2(x_coils, norm="ortho"), dim=(-2, -1))
    mask = torch.ones(b, 1, h, w)

    x_rec = recon.reconstruct(masked_kspace=kspace_coils, mask=mask, sensitivity_maps=sens_maps)

    assert x_rec.shape == (b, 1, h, w)
    torch.testing.assert_close(x_rec, x_gt, atol=1e-5, rtol=1e-5)


def test_flow_matching_reconstructor_end_to_end() -> None:
    """FlowMatchingReconstructor runs integration and enforces k-space data consistency."""
    manifold = CylindricalManifold()
    model = CylindricalUNet(
        base_channels=16,
        in_channels=manifold.state_channels,
        out_channels=manifold.velocity_channels,
    )
    recon = FlowMatchingReconstructor(model=model, manifold=manifold, data_consistency=True)

    b, h, w = 2, 16, 16
    x_gt = torch.complex(torch.randn(b, 1, h, w), torch.randn(b, 1, h, w))
    kspace_full = torch.fft.fftshift(torch.fft.fft2(x_gt, norm="ortho"), dim=(-2, -1))

    # Mask with central 8 lines
    mask = torch.zeros(b, 1, h, w)
    mask[:, :, :, 4:12] = 1.0
    masked_kspace = kspace_full * mask

    x_rec = recon.reconstruct(masked_kspace=masked_kspace, mask=mask, num_steps=2)

    assert x_rec.shape == (b, 1, h, w)
    assert x_rec.is_complex()

    # Verify exact Data Consistency: k_rec on acquired lines matches masked_kspace
    kspace_rec = torch.fft.fftshift(torch.fft.fft2(x_rec, norm="ortho"), dim=(-2, -1))
    dc_error = torch.linalg.norm((kspace_rec * mask) - masked_kspace).item()
    assert dc_error < 1e-5, f"Data consistency violated, error: {dc_error}"
