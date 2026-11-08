"""The paper's U-Net: its shapes, its time conditioning and its bounded head."""

import pytest
import torch

from cyfm.models.unet import AttentionBlock, CylindricalUNet


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


def test_unet_three_level_downsampling() -> None:
    """A size divisible by 8 but not 16 exercises all three levels and their skips.

    24 -> 12 -> 6 -> 3 at the bottleneck. If a decoder skip did not line up with
    its upsampled feature the concatenation would raise, so a clean round trip to
    the input shape is the check.
    """
    model = CylindricalUNet(base_channels=16)
    out = model(torch.randn(1, 3, 24, 24), torch.tensor([0.3]))
    assert out.shape == (1, 2, 24, 24)


def test_unet_attention_toggle() -> None:
    """The default carries bottleneck attention; it can be turned off from config."""
    assert CylindricalUNet(base_channels=16).bottleneck_attn is not None

    without = CylindricalUNet(base_channels=16, use_attention=False)
    assert without.bottleneck_attn is None
    # The rest of the trunk still runs unchanged with attention off.
    out = without(torch.randn(1, 3, 32, 32), torch.rand(1))
    assert out.shape == (1, 2, 32, 32)


def test_attention_block_changes_its_input() -> None:
    """The attention block is active, not a shape-preserving no-op."""
    torch.manual_seed(0)
    attn = AttentionBlock(channels=64)
    x = torch.randn(2, 64, 8, 8)
    out = attn(x)
    assert out.shape == x.shape
    assert not torch.allclose(out, x)


def test_attention_block_rejects_channels_indivisible_by_heads() -> None:
    with pytest.raises(ValueError, match="divisible by num_heads"):
        AttentionBlock(channels=12, num_heads=8)


def test_unet_rejects_base_channels_not_multiple_of_8() -> None:
    """GroupNorm(8, C) and the attention heads both need widths divisible by 8."""
    with pytest.raises(ValueError, match="multiple of 8"):
        CylindricalUNet(base_channels=12)


def test_unet_rejects_input_indivisible_by_depth() -> None:
    """Three levels need H, W divisible by 8; a bad size fails with a clear message."""
    model = CylindricalUNet(base_channels=16)  # num_levels=3 -> stride 8
    with pytest.raises(ValueError, match="divisible by 2"):
        model(torch.randn(1, 3, 20, 20), torch.rand(1))  # 20 % 8 != 0


def test_unet_default_parameter_count_is_pinned() -> None:
    """Lock the shipped size so a topology change is a deliberate, visible edit.

    15.38M params at base_channels=64 (3 levels + bottleneck attention). The two
    geometries differ only in init_conv's input width (3 vs 2 channels), which is
    the 576-parameter gap between the counts below.
    """
    cyl = CylindricalUNet(in_channels=3, base_channels=64)
    euc = CylindricalUNet(in_channels=2, base_channels=64)
    assert sum(p.numel() for p in cyl.parameters()) == 15_375_746
    assert sum(p.numel() for p in euc.parameters()) == 15_375_170
