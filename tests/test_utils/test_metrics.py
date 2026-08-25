import math

import pytest
import torch

from cfm.utils.metrics import (
    circular_phase_error,
    peak_signal_noise_ratio,
    shortest_angular_difference,
    structural_similarity,
)


def _random_complex(batch: int, h: int, w: int, seed: int = 0) -> torch.Tensor:
    """Builds a random complex [B, 1, H, W] tensor with amplitude in [0, 1]."""
    torch.manual_seed(seed)
    amp = torch.rand(batch, 1, h, w)
    phi = torch.rand(batch, 1, h, w) * 2 * math.pi - math.pi
    return torch.polar(amp, phi)


class TestIdentity:
    """A tensor scored against itself must report a perfect match."""

    def test_psnr_identity_is_inf(self) -> None:
        z = _random_complex(4, 16, 16, seed=1)
        psnr = peak_signal_noise_ratio(z, z)
        assert torch.isinf(psnr)
        assert psnr > 0

    def test_ssim_identity_is_one(self) -> None:
        z = _random_complex(4, 16, 16, seed=2)
        ssim = structural_similarity(z, z)
        assert torch.allclose(ssim, torch.tensor(1.0), atol=1e-5)

    def test_circular_phase_error_identity_is_zero(self) -> None:
        z = _random_complex(4, 16, 16, seed=3)
        error = circular_phase_error(z, z)
        assert torch.allclose(error, torch.tensor(0.0), atol=1e-6)


class TestPhaseWrap:
    """Core behaviour under test: angular distance must wrap, never raw-subtract."""

    def test_wrap_near_boundary_gives_short_arc(self) -> None:
        pred = torch.tensor([2 * math.pi - 0.01])
        target = torch.tensor([0.01])
        diff = shortest_angular_difference(pred, target)
        assert torch.allclose(diff.abs(), torch.tensor(0.02), atol=1e-4)
        assert diff.abs() < 1.0  # sanity: nowhere near the naive ~6.26 rad error

    def test_wrap_near_boundary_via_circular_phase_error(self) -> None:
        pred = torch.full((1, 1, 2, 2), 2 * math.pi - 0.01)
        target = torch.full((1, 1, 2, 2), 0.01)
        error = circular_phase_error(pred, target)
        assert torch.allclose(error, torch.tensor(0.02), atol=1e-4)

    def test_antipodal_angles_give_exactly_pi(self) -> None:
        pred = torch.tensor([0.0])
        target = torch.tensor([math.pi])
        diff = shortest_angular_difference(pred, target)
        assert torch.allclose(diff.abs(), torch.tensor(math.pi), atol=1e-6)

    def test_error_never_exceeds_pi_for_wide_range_angles(self) -> None:
        torch.manual_seed(4)
        pred = (torch.rand(10_000) - 0.5) * 40  # well outside [-pi, pi]
        target = (torch.rand(10_000) - 0.5) * 40
        diff = shortest_angular_difference(pred, target)
        assert torch.all(diff.abs() <= math.pi + 1e-6)

    def test_shortest_angular_difference_always_in_range(self) -> None:
        torch.manual_seed(5)
        pred = (torch.rand(10_000) - 0.5) * 100
        target = (torch.rand(10_000) - 0.5) * 100
        diff = shortest_angular_difference(pred, target)
        assert torch.all(diff >= -math.pi - 1e-6)
        assert torch.all(diff <= math.pi + 1e-6)


class TestAmplitudeMasking:
    def _pred_target_with_noisy_background(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Foreground phases match exactly; background phase is maximally wrong."""
        h, w = 4, 4
        amp = torch.zeros(1, 1, h, w)
        amp[:, :, 1:3, 1:3] = 1.0  # 2x2 tissue block, rest is background (amp=0)

        target_phi = torch.zeros(1, 1, h, w)
        pred_phi = torch.zeros(1, 1, h, w)
        # Background pixels get maximal phase error; foreground stays a perfect match.
        pred_phi[amp == 0] = math.pi

        target = torch.polar(amp, target_phi)
        pred = torch.polar(amp, pred_phi)
        return pred, target

    def test_mask_threshold_excludes_zero_amplitude_background(self) -> None:
        pred, target = self._pred_target_with_noisy_background()
        error = circular_phase_error(pred, target, mask_threshold=0.05)
        assert torch.allclose(error, torch.tensor(0.0), atol=1e-6)

    def test_without_mask_background_error_dominates(self) -> None:
        pred, target = self._pred_target_with_noisy_background()
        error = circular_phase_error(pred, target)
        assert error > 1.0  # background pi-error pollutes the unmasked average

    def test_explicit_amplitude_mask_matches_threshold_mask(self) -> None:
        pred, target = self._pred_target_with_noisy_background()
        explicit_mask = torch.abs(target) > 0
        error_explicit = circular_phase_error(pred, target, amplitude_mask=explicit_mask)
        error_threshold = circular_phase_error(pred, target, mask_threshold=0.05)
        assert torch.allclose(error_explicit, error_threshold, atol=1e-6)

    def test_empty_mask_yields_nan(self) -> None:
        pred, target = self._pred_target_with_noisy_background()
        empty_mask = torch.zeros_like(torch.abs(target), dtype=torch.bool)
        error = circular_phase_error(pred, target, amplitude_mask=empty_mask)
        assert torch.isnan(error)


class TestReduction:
    def test_reduction_none_returns_per_sample_shape(self) -> None:
        z = _random_complex(5, 16, 16, seed=6)
        noise = _random_complex(5, 16, 16, seed=7)

        assert peak_signal_noise_ratio(z, noise, reduction="none").shape == (5,)
        assert structural_similarity(z, noise, reduction="none").shape == (5,)
        assert circular_phase_error(z, noise, reduction="none").shape == (5,)

    def test_reduction_sum_matches_manual_sum(self) -> None:
        z = _random_complex(5, 16, 16, seed=8)
        noise = _random_complex(5, 16, 16, seed=9)
        per_sample = peak_signal_noise_ratio(z, noise, reduction="none")
        total = peak_signal_noise_ratio(z, noise, reduction="sum")
        assert torch.allclose(total, per_sample.sum())


class TestInputPaths:
    """Both the complex-tensor path and the pre-decomposed real-tensor path must agree."""

    def test_psnr_complex_and_decomposed_paths_agree(self) -> None:
        z = _random_complex(3, 16, 16, seed=10)
        noise = _random_complex(3, 16, 16, seed=11)

        psnr_complex = peak_signal_noise_ratio(z, noise)
        psnr_decomposed = peak_signal_noise_ratio(torch.abs(z), torch.abs(noise))
        assert torch.allclose(psnr_complex, psnr_decomposed)

    def test_ssim_complex_and_decomposed_paths_agree(self) -> None:
        z = _random_complex(3, 16, 16, seed=12)
        noise = _random_complex(3, 16, 16, seed=13)

        ssim_complex = structural_similarity(z, noise)
        ssim_decomposed = structural_similarity(torch.abs(z), torch.abs(noise))
        assert torch.allclose(ssim_complex, ssim_decomposed)

    def test_phase_error_complex_and_decomposed_paths_agree(self) -> None:
        z = _random_complex(3, 16, 16, seed=14)
        noise = _random_complex(3, 16, 16, seed=15)

        error_complex = circular_phase_error(z, noise)
        error_decomposed = circular_phase_error(torch.angle(z), torch.angle(noise))
        assert torch.allclose(error_complex, error_decomposed)


class TestValueErrorGuards:
    def test_mismatched_shapes_raise(self) -> None:
        z = _random_complex(2, 16, 16, seed=16)
        other = _random_complex(2, 8, 8, seed=17)
        with pytest.raises(ValueError):
            peak_signal_noise_ratio(z, other)

    def test_even_window_size_raises(self) -> None:
        z = _random_complex(2, 16, 16, seed=18)
        with pytest.raises(ValueError):
            structural_similarity(z, z, window_size=10)

    def test_amplitude_mask_and_mask_threshold_together_raise(self) -> None:
        z = _random_complex(2, 16, 16, seed=19)
        mask = torch.ones(2, 1, 16, 16, dtype=torch.bool)
        with pytest.raises(ValueError):
            circular_phase_error(z, z, amplitude_mask=mask, mask_threshold=0.05)

    def test_mask_threshold_with_non_complex_target_raises(self) -> None:
        phase = torch.zeros(2, 1, 16, 16)
        with pytest.raises(ValueError):
            circular_phase_error(phase, phase, mask_threshold=0.05)

    def test_unknown_reduction_raises(self) -> None:
        z = _random_complex(2, 16, 16, seed=20)
        with pytest.raises(ValueError):
            peak_signal_noise_ratio(z, z, reduction="bogus")
