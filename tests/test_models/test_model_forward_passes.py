"""Every registered architecture honours the velocity-field shape contract."""

import torch

from cyfm.models.unet import CylindricalUNet


def test_2d_model_forward_unet() -> None:
    model = CylindricalUNet(in_channels=3, out_channels=2, base_channels=8)

    x = torch.randn(2, 3, 64, 64)
    t = torch.rand(2)

    out = model(x, t)
    assert out.shape == (2, 2, 64, 64)


def test_euclidean_unet_forward() -> None:
    # Standard UNet for Euclidean uses in_channels=2
    model = CylindricalUNet(in_channels=2, out_channels=2, base_channels=8)

    x = torch.randn(2, 2, 64, 64)
    t = torch.rand(2)

    out = model(x, t)
    assert out.shape == (2, 2, 64, 64)
