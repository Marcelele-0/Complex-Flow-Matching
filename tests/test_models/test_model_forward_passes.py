import torch

from cfm.models.cylindrical_unet import CylindricalUNet
from cfm.models.cylindrical_unet_attention import CylindricalUNetAttention
from cfm.models.cylindrical_unet_cross_slice import CylindricalUNetCrossSlice


def test_2d_model_forward_unet() -> None:
    model = CylindricalUNet(in_channels=3, out_channels=2, base_channels=8)

    x = torch.randn(2, 3, 64, 64)
    t = torch.rand(2)

    out = model(x, t)
    assert out.shape == (2, 2, 64, 64)


def test_2d_model_forward_attention() -> None:
    model = CylindricalUNetAttention(
        in_channels=3, out_channels=2, base_channels=8, channel_mults=[1, 2]
    )

    x = torch.randn(2, 3, 64, 64)
    t = torch.rand(2)

    out = model(x, t)
    assert out.shape == (2, 2, 64, 64)


def test_2_5d_model_forward() -> None:
    model = CylindricalUNetCrossSlice(
        in_channels=3, base_channels=8, channel_mults=[1, 2], attn_heads=2
    )

    # [B, S, C, H, W] where S=3
    x_window = torch.randn(2, 3, 3, 64, 64)
    t = torch.rand(2)

    out = model(x_window, t)
    # Output is for the center slice: [B, 2, H, W]
    assert out.shape == (2, 2, 64, 64)


def test_euclidean_unet_forward() -> None:
    # Standard UNet for Euclidean uses in_channels=2
    model = CylindricalUNet(in_channels=2, out_channels=2, base_channels=8)

    x = torch.randn(2, 2, 64, 64)
    t = torch.rand(2)

    out = model(x, t)
    assert out.shape == (2, 2, 64, 64)
