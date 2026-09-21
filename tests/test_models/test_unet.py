"""The paper's U-Net: its shapes, its time conditioning and its bounded head."""

import torch

from cyfm.models.unet import CylindricalUNet


def test_unet_output_shape() -> None:
    """Checks if U-Net returns the correct output shape [B, 2, H, W]."""
    b, channels, h, w = 2, 3, 32, 32
    x = torch.randn(b, channels, h, w)

    # Time tensor must be 1D, containing one scalar per element in batch
    t = torch.rand(b)

    # Small capacity (base_channels=16) for test speed
    model = CylindricalUNet(base_channels=16)

    out = model(x, t)

    assert out.shape == (b, 2, h, w), f"Expected shape {(b, 2, h, w)}, got {out.shape}"
    assert out.dtype == torch.float32


def test_unet_time_conditioning() -> None:
    """Ensures the network actually changes behavior under the influence of time."""
    b, channels, h, w = 1, 3, 16, 16
    x = torch.randn(b, channels, h, w)

    t_0 = torch.zeros(b)
    t_1 = torch.ones(b)

    model = CylindricalUNet(base_channels=16)
    model.eval()

    with torch.no_grad():
        out_0 = model(x, t_0)
        out_1 = model(x, t_1)

    difference = torch.abs(out_0 - out_1).sum()
    assert difference > 1e-5, "Network is ignoring the time embedding!"


def test_unet_variable_resolution() -> None:
    """Checks tolerance for different input resolutions (e.g., without hardcoding dimensions)."""
    model = CylindricalUNet(base_channels=16)
    t = torch.tensor([0.5])

    # Larger resolution: 64x64
    x_64 = torch.randn(1, 3, 64, 64)
    out_64 = model(x_64, t)

    assert out_64.shape == (1, 2, 64, 64), "Failed on 64x64 resolution"
