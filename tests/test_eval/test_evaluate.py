"""Tests for the reconstruction evaluation path.

Everything here runs with a dummy model and synthetic tensors (no checkpoint, no
HDF5), which is what makes the eval path verifiable on a machine that has neither.
"""

import math

import pytest
import torch

from cfm.evaluate import (
    MetricAccumulator,
    compute_batch_metrics,
    format_summary_table,
    integrate_from_t,
    reconstruct_batch,
    sample_cylindrical_noise,
    select_indices,
)
from cfm.flow.bridge import GeodesicFlowBridge
from cfm.flow.solver import CylindricalODESolver


def _cylindrical_state(batch: int, h: int, w: int, seed: int = 0) -> torch.Tensor:
    """A valid state on the cylinder: amplitude in [0, 1], phase on the unit circle."""
    torch.manual_seed(seed)
    amp = torch.rand(batch, 1, h, w)
    phi = torch.rand(batch, 1, h, w) * 2 * math.pi
    return torch.cat([amp, torch.cos(phi), torch.sin(phi)], dim=1)


def _state_and_time_model(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Dummy velocity field that depends on both the state and the time."""
    t_map = t.view(-1, 1, 1, 1).expand_as(x[:, 0:1])
    v_amp = x[:, 0:1] * 0.3 + t_map
    v_phi = torch.sin(x[:, 1:2]) * 0.7 - t_map
    return torch.cat([v_amp, v_phi], dim=1)


class TestIntegration:
    @pytest.mark.parametrize("num_steps", [3, 4, 5, 7])
    def test_integrate_from_zero_reproduces_solver_sample(self, num_steps: int) -> None:
        """At t_start=0 the helper must match the trusted solver bit-for-bit.

        This pins the Heun scheme, the Euler final step, the step size and the time
        schedule to CylindricalODESolver.sample in a single assertion.
        """
        solver = CylindricalODESolver(num_steps=num_steps)
        x_0 = _cylindrical_state(2, 8, 8, seed=1)

        ours = integrate_from_t(_state_and_time_model, solver, x_0, t_start=0.0)
        theirs = solver.sample(_state_and_time_model, x_0)

        assert torch.equal(ours, theirs)

    def test_integrate_from_t_feeds_absolute_times(self) -> None:
        """The model must see absolute times in [t_start, 1), not step indices."""
        seen: list[float] = []

        def recording_model(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            assert t.dim() == 1, "time must be 1-D [B]"
            assert t.shape[0] == x.shape[0]
            seen.append(float(t[0]))
            return torch.zeros(x.shape[0], 2, x.shape[2], x.shape[3])

        solver = CylindricalODESolver(num_steps=5)
        x_start = _cylindrical_state(2, 4, 4, seed=2)
        integrate_from_t(recording_model, solver, x_start, t_start=0.4)

        expected = [0.4, 0.52, 0.52, 0.64, 0.64, 0.76, 0.76, 0.88, 0.88]
        assert len(seen) == 2 * solver.num_steps - 1
        assert seen == pytest.approx(expected, abs=1e-6)
        assert all(0.4 <= t < 1.0 for t in seen)

    def test_invalid_t_start_raises(self) -> None:
        solver = CylindricalODESolver(num_steps=3)
        x_start = _cylindrical_state(1, 4, 4, seed=3)
        with pytest.raises(ValueError):
            integrate_from_t(_state_and_time_model, solver, x_start, t_start=1.5)

    def test_noise_is_on_the_cylinder(self) -> None:
        """Sampled noise must match train.py's distribution and stay on the manifold."""
        noise = sample_cylindrical_noise(4, 8, 8, torch.device("cpu"))
        assert noise.shape == (4, 3, 8, 8)
        assert torch.all(noise[:, 0:1] >= 0.0) and torch.all(noise[:, 0:1] <= 1.0)
        radius_sq = noise[:, 1:2] ** 2 + noise[:, 2:3] ** 2
        assert torch.allclose(radius_sq, torch.ones_like(radius_sq), atol=1e-6)


class TestEndToEndReconstruction:
    def test_t_start_one_is_a_perfect_reconstruction(self) -> None:
        """t_start=1 makes the bridge return the target, so metrics must be ~perfect.

        A deliberately nonsense model proves the no-op is genuine rather than
        accidental: with dt=0 its velocity cannot move the state.
        """

        def nonsense_model(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return torch.randn(x.shape[0], 2, x.shape[2], x.shape[3]) * 10.0

        solver = CylindricalODESolver(num_steps=5)
        bridge = GeodesicFlowBridge()
        x_1 = _cylindrical_state(3, 16, 16, seed=4)

        pred = reconstruct_batch(nonsense_model, solver, bridge, x_1, t_start=1.0)
        metrics = compute_batch_metrics(pred, x_1, mask_threshold=0.05)

        # NOT +inf: cylinder_to_complex rebuilds re = m*cos(phi), and the bridge's
        # atan2 -> +u_phi -> cos/sin roundtrip perturbs phi by ~1e-7, so abs() is
        # not bit-exact. Measured ~154-156 dB.
        assert torch.all(metrics["psnr_db"] > 100.0)
        assert torch.allclose(metrics["ssim"], torch.ones_like(metrics["ssim"]), atol=1e-4)
        assert torch.all(metrics["phase_error_rad"] < 1e-5)

    def test_unmasked_phase_error_is_reported_alongside(self) -> None:
        solver = CylindricalODESolver(num_steps=2)
        bridge = GeodesicFlowBridge()
        x_1 = _cylindrical_state(2, 16, 16, seed=5)

        pred = reconstruct_batch(_state_and_time_model, solver, bridge, x_1, t_start=0.5)
        metrics = compute_batch_metrics(pred, x_1, mask_threshold=0.05)

        assert "phase_error_rad_unmasked" in metrics
        assert metrics["phase_error_rad_unmasked"].shape == (2,)

        # With no threshold the masked entry IS the unmasked one, so no extra key.
        unmasked_only = compute_batch_metrics(pred, x_1, mask_threshold=None)
        assert "phase_error_rad_unmasked" not in unmasked_only

    def test_all_air_slice_is_unscored_not_zero(self) -> None:
        """An all-air slice must come back NaN and be counted, never averaged in."""

        def nonsense_model(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return torch.randn(x.shape[0], 2, x.shape[2], x.shape[3]) * 10.0

        solver = CylindricalODESolver(num_steps=3)
        bridge = GeodesicFlowBridge()
        x_1 = _cylindrical_state(3, 16, 16, seed=6)
        x_1[1, 0] = 0.0  # middle slice is pure air: zero amplitude everywhere

        pred = reconstruct_batch(nonsense_model, solver, bridge, x_1, t_start=1.0)
        metrics = compute_batch_metrics(pred, x_1, mask_threshold=0.05)

        assert torch.isnan(metrics["phase_error_rad"][1])
        assert torch.isposinf(metrics["psnr_db"][1])

        phase_acc = MetricAccumulator("phase_error_rad")
        phase_acc.update(metrics["phase_error_rad"], ["a", "b", "c"])
        phase_summary = phase_acc.summary()

        psnr_acc = MetricAccumulator("psnr_db")
        psnr_acc.update(metrics["psnr_db"], ["a", "b", "c"])
        psnr_summary = psnr_acc.summary()

        assert phase_summary.scored == 2 and phase_summary.nan == 1
        assert psnr_summary.scored == 2 and psnr_summary.pos_inf == 1
        # The unscorable sample must not poison the reported number.
        assert math.isfinite(phase_summary.mean)
        assert math.isfinite(psnr_summary.mean)
        assert phase_summary.unscored_ids == ("b",)


class TestMetricAccumulator:
    def test_partitions_nan_and_inf(self) -> None:
        acc = MetricAccumulator("psnr_db")
        acc.update(torch.tensor([1.0, float("nan")]))
        acc.update(torch.tensor([float("inf"), 3.0]))

        summary = acc.summary()
        assert summary.total == 4
        assert summary.scored == 2
        assert summary.nan == 1
        assert summary.pos_inf == 1
        assert summary.mean == pytest.approx(2.0)
        assert summary.minimum == pytest.approx(1.0)
        assert summary.maximum == pytest.approx(3.0)

    def test_all_unscored_yields_nan_not_zero(self) -> None:
        acc = MetricAccumulator("phase_error_rad")
        acc.update(torch.tensor([float("nan"), float("nan")]))

        summary = acc.summary()
        assert summary.scored == 0
        assert math.isnan(summary.mean)

        table = format_summary_table([summary])
        assert "n/a" in table
        assert "WARNING" in table

    def test_rejects_non_1d_values(self) -> None:
        acc = MetricAccumulator("ssim")
        with pytest.raises(ValueError):
            acc.update(torch.zeros(2, 3))

    def test_rejects_mismatched_sample_ids(self) -> None:
        acc = MetricAccumulator("ssim")
        with pytest.raises(ValueError):
            acc.update(torch.zeros(3), ["only", "two"])

    def test_clean_run_has_no_legend(self) -> None:
        acc = MetricAccumulator("ssim")
        acc.update(torch.tensor([0.9, 0.8]))

        table = format_summary_table([acc.summary()])
        assert "nan  =" not in table
        assert "WARNING" not in table
        assert "2/2" in table


class TestSelectIndices:
    SLICE_MAP = [("/d/MTR_030.h5", i) for i in range(5)] + [("/d/MTR_184.h5", i) for i in range(3)]

    def test_filters_to_requested_file(self) -> None:
        assert select_indices(self.SLICE_MAP, {"MTR_184.h5"}) == [5, 6, 7]

    def test_none_keeps_everything(self) -> None:
        assert select_indices(self.SLICE_MAP, None) == list(range(8))

    def test_no_match_raises_naming_both_sides(self) -> None:
        with pytest.raises(ValueError, match="No slices matched") as exc:
            select_indices(self.SLICE_MAP, {"MTR_999.h5"})
        message = str(exc.value)
        assert "MTR_999.h5" in message
        assert "MTR_030.h5" in message

    def test_max_samples_strides_rather_than_truncating(self) -> None:
        slice_map = [("/d/a.h5", i) for i in range(10)]
        # Striding gives a spread across the volume; truncation would give [0, 1, 2],
        # three near-identical neighbouring slices from one end of the knee.
        assert select_indices(slice_map, None, max_samples=3) == [0, 3, 6]

    def test_max_samples_larger_than_population_is_a_noop(self) -> None:
        slice_map = [("/d/a.h5", i) for i in range(3)]
        assert select_indices(slice_map, None, max_samples=10) == [0, 1, 2]
