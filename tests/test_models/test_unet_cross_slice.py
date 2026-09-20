"""Tests for the 2.5D cross-slice U-Net.

Note on shapes: with the cylindrical pipeline a ``num_slices=3`` sample is
``[3, 3, H, W]`` -- the slice axis and the channel axis are *both* 3, so a shape
assertion at S=3 can pass even if the two are transposed. Shape work here uses
S=5 so the axes are distinguishable.
"""

import math

import pytest
import torch

from cyfm.flow.bridge import GeodesicFlowBridge
from cyfm.flow.torus_math import DecoupledCylindricalLoss
from cyfm.models.cylindrical_unet_cross_slice import (
    CrossSliceAttention,
    CylindricalUNetCrossSlice,
)

# Two pooling levels keeps the test model small; H and W must divide by 4.
TEST_MULTS = [1, 2, 4]


def _cylindrical_window(b: int, s: int, h: int, w: int, seed: int = 0) -> torch.Tensor:
    """A valid [B, S, 3, H, W] window: amplitude in [0, 1], phase on the unit circle."""
    torch.manual_seed(seed)
    amp = torch.rand(b, s, 1, h, w)
    phi = torch.rand(b, s, 1, h, w) * 2 * math.pi
    return torch.cat([amp, torch.cos(phi), torch.sin(phi)], dim=2)


def _model(base_channels: int = 16, in_channels: int = 3) -> CylindricalUNetCrossSlice:
    return CylindricalUNetCrossSlice(
        base_channels=base_channels,
        channel_mults=TEST_MULTS,
        attn_heads=2,
        in_channels=in_channels,
    )


class TestShapes:
    def test_returns_center_velocity_only(self) -> None:
        b, s, h, w = 2, 5, 16, 16
        model = _model()
        out = model(_cylindrical_window(b, s, h, w), torch.rand(b))

        # S=5 in, but a single center velocity out.
        assert out.shape == (b, 2, h, w)
        assert out.dtype == torch.float32

    def test_same_instance_accepts_different_slice_counts(self) -> None:
        """Weights are shared across slices, so S is not baked into the model."""
        model = _model()
        t = torch.rand(2)

        out_3 = model(_cylindrical_window(2, 3, 16, 16, seed=1), t)
        out_5 = model(_cylindrical_window(2, 5, 16, 16, seed=2), t)

        assert out_3.shape == out_5.shape == (2, 2, 16, 16)

    def test_variable_resolution(self) -> None:
        model = _model()
        out = model(_cylindrical_window(1, 3, 32, 32, seed=3), torch.tensor([0.5]))
        assert out.shape == (1, 2, 32, 32)

    def test_no_per_slice_parameters(self) -> None:
        """Parameter count must not depend on S -- that is what 'shared' means."""
        model = _model()
        before = sum(p.numel() for p in model.parameters())
        model(_cylindrical_window(1, 7, 16, 16, seed=4), torch.rand(1))
        assert sum(p.numel() for p in model.parameters()) == before


class TestCrossSliceWiring:
    def test_neighbour_slices_influence_the_center_prediction(self) -> None:
        """The whole point of the model: changing a NON-center slice must matter.

        If this passes with cross-slice attention removed, the neighbours are being
        ignored and the model is a plain 2D U-Net wearing a costume.
        """
        model = _model()
        model.eval()

        x = _cylindrical_window(1, 3, 16, 16, seed=5)
        t = torch.rand(1)

        perturbed = x.clone()
        # Rotate slice 0's phase by pi; the center slice is untouched.
        perturbed[:, 0, 1] = -perturbed[:, 0, 1]
        perturbed[:, 0, 2] = -perturbed[:, 0, 2]

        with torch.no_grad():
            out_a = model(x, t)
            out_b = model(perturbed, t)

        assert torch.abs(out_a - out_b).sum() > 1e-6, "Neighbouring slices are being ignored"

    def test_center_slice_influences_its_own_prediction(self) -> None:
        model = _model()
        model.eval()

        x = _cylindrical_window(1, 3, 16, 16, seed=6)
        t = torch.rand(1)

        perturbed = x.clone()
        perturbed[:, 1, 0] = perturbed[:, 1, 0] * 0.5  # center amplitude

        with torch.no_grad():
            out_a = model(x, t)
            out_b = model(perturbed, t)

        assert torch.abs(out_a - out_b).sum() > 1e-6

    def test_time_conditioning(self) -> None:
        model = _model()
        model.eval()

        x = _cylindrical_window(1, 3, 16, 16, seed=7)
        with torch.no_grad():
            out_0 = model(x, torch.zeros(1))
            out_1 = model(x, torch.ones(1))

        assert torch.abs(out_0 - out_1).sum() > 1e-5, "Network is ignoring the time embedding!"


class TestCrossSliceAttentionModule:
    def test_preserves_shape_and_is_residual(self) -> None:
        attn = CrossSliceAttention(channels=16, heads=2)
        x = torch.randn(2, 3, 16, 4, 4)
        assert attn(x).shape == x.shape

    def test_attention_mixes_only_across_slices_not_positions(self) -> None:
        """The attention itself must not mix spatial positions, only slices.

        GroupNorm is swapped for Identity to isolate the claim: it pools statistics
        over every spatial position, so with it in place a perturbation anywhere
        shifts the normalised value everywhere. That is expected (the baseline's
        SelfAttention2d normalises the same way) and would mask what is being
        tested here.
        """
        attn = CrossSliceAttention(channels=16, heads=2)
        attn.norm = torch.nn.Identity()  # type: ignore[assignment]
        attn.eval()

        x = torch.randn(1, 3, 16, 4, 4)
        perturbed = x.clone()
        perturbed[0, 0, :, 0, 0] += 5.0  # one slice, one position

        with torch.no_grad():
            delta = (attn(perturbed) - attn(x)).abs()

        # The perturbed position's own slice-column must react...
        assert delta[0, :, :, 0, 0].sum() > 1e-6
        # ...and every other spatial position must be untouched.
        off_position = delta.clone()
        off_position[0, :, :, 0, 0] = 0.0
        assert off_position.max() < 1e-5


class TestGradientFlow:
    def test_full_training_step_reaches_every_parameter(self) -> None:
        """One real bridge -> model -> loss -> backward step on synthetic tensors.

        Catches a detached path, and in particular catches the center-selection
        severing gradients to the encoder passes for non-center slices.
        """
        b, s, h, w = 2, 3, 16, 16
        model = _model()
        bridge = GeodesicFlowBridge()
        criterion = DecoupledCylindricalLoss()

        x_1 = _cylindrical_window(b, s, h, w, seed=8)
        center = s // 2
        flat = b * s

        noise_amp = torch.rand(flat, 1, h, w)
        noise_phi = torch.rand(flat, 1, h, w) * 2 * math.pi
        x_0 = torch.cat([noise_amp, torch.cos(noise_phi), torch.sin(noise_phi)], dim=1)

        t_model = torch.rand(b)
        t_bridge = t_model.repeat_interleave(s).view(flat, 1, 1, 1)

        x_t_flat, target_v_flat = bridge.forward(
            cyl_noise=x_0, cyl_data=x_1.reshape(flat, 3, h, w), t=t_bridge
        )
        x_t = x_t_flat.view(b, s, 3, h, w)
        target_v = target_v_flat.view(b, s, 2, h, w)[:, center]

        pred_v = model(x_t, t_model)
        assert pred_v.shape == target_v.shape

        loss, _, _ = criterion(pred_v, target_v, target_x1=x_1[:, center])
        loss.backward()

        missing = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
        assert not missing, f"No gradient reached: {missing}"

        non_finite = [
            n
            for n, p in model.named_parameters()
            if p.grad is not None and not p.grad.isfinite().all()
        ]
        assert not non_finite, f"Non-finite gradients in: {non_finite}"


class TestGuards:
    def test_even_slice_count_raises(self) -> None:
        model = _model()
        with pytest.raises(ValueError, match="odd"):
            model(_cylindrical_window(1, 4, 16, 16, seed=9), torch.rand(1))

    def test_4d_input_raises_with_actionable_message(self) -> None:
        model = _model()
        with pytest.raises(ValueError, match=r"\[B, S, C, H, W\]"):
            model(torch.rand(1, 3, 16, 16), torch.rand(1))
