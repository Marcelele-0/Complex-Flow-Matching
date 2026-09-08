"""Tests for the reconstruction evaluation path.

Everything here runs with a dummy model and synthetic tensors (no checkpoint, no
HDF5), which is what makes the eval path verifiable on a machine that has neither.
"""

import csv
import math

import pytest
import torch

from cfm.data.splits import select_indices
from cfm.evaluate import (
    MetricAccumulator,
    compute_batch_metrics,
    format_summary_table,
    integrate_from_t,
    reconstruct_batch,
    write_eval_records,
)
from cfm.flow.solver import CylindricalODESolver
from cfm.manifolds import CylindricalManifold, EuclideanManifold
from cfm.manifolds.cylindrical import sample_cylindrical_noise


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
        cpu = torch.device("cpu")

        # The manifold must delegate to the module-level draw, not reimplement it:
        # the Euclidean 'matched' prior's fairness guarantee rests on that one
        # function's RNG call order.
        via_manifold = CylindricalManifold().sample_noise(
            4, 8, 8, cpu, torch.Generator(device="cpu").manual_seed(7)
        )
        direct = sample_cylindrical_noise(
            4, 8, 8, cpu, torch.Generator(device="cpu").manual_seed(7)
        )
        assert torch.equal(via_manifold, direct)

        noise = sample_cylindrical_noise(4, 8, 8, cpu)
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

        manifold = CylindricalManifold()
        solver = manifold.make_solver(num_steps=5)
        x_1 = _cylindrical_state(3, 16, 16, seed=4)

        pred = reconstruct_batch(nonsense_model, manifold, solver, x_1, t_start=1.0)
        metrics = compute_batch_metrics(manifold, pred, x_1, mask_threshold=0.05)

        # NOT +inf: cylinder_to_complex rebuilds re = m*cos(phi), and the bridge's
        # atan2 -> +u_phi -> cos/sin roundtrip perturbs phi by ~1e-7, so abs() is
        # not bit-exact. Measured ~154-156 dB.
        assert torch.all(metrics["psnr_db"] > 100.0)
        assert torch.allclose(metrics["ssim"], torch.ones_like(metrics["ssim"]), atol=1e-4)
        assert torch.all(metrics["phase_error_rad"] < 1e-5)

    def test_unmasked_phase_error_is_reported_alongside(self) -> None:
        manifold = CylindricalManifold()
        solver = manifold.make_solver(num_steps=2)
        x_1 = _cylindrical_state(2, 16, 16, seed=5)

        pred = reconstruct_batch(_state_and_time_model, manifold, solver, x_1, t_start=0.5)
        metrics = compute_batch_metrics(manifold, pred, x_1, mask_threshold=0.05)

        assert "phase_error_rad_unmasked" in metrics
        assert metrics["phase_error_rad_unmasked"].shape == (2,)

        # With no threshold the masked entry IS the unmasked one, so no extra key.
        unmasked_only = compute_batch_metrics(manifold, pred, x_1, mask_threshold=None)
        assert "phase_error_rad_unmasked" not in unmasked_only

    def test_dc_error_appears_only_when_a_sampling_mask_is_given(self) -> None:
        manifold = CylindricalManifold()
        solver = manifold.make_solver(num_steps=2)
        x_1 = _cylindrical_state(2, 16, 16, seed=11)
        pred = reconstruct_batch(_state_and_time_model, manifold, solver, x_1, t_start=0.5)

        without = compute_batch_metrics(manifold, pred, x_1, mask_threshold=0.05)
        assert "data_consistency_error" not in without

        mask = torch.ones(1, 1, 16, 16)
        with_mask = compute_batch_metrics(
            manifold, pred, x_1, mask_threshold=0.05, sampling_mask=mask
        )

        # The key name is what MetricAccumulator registers, so it reaches the
        # summary table, metrics.json and W&B without further wiring.
        assert with_mask["data_consistency_error"].shape == (2,)
        assert torch.all(with_mask["data_consistency_error"] >= 0)

    def test_dc_error_is_near_zero_for_a_perfect_reconstruction(self) -> None:
        """At t_start=1 the bridge returns the target, so k-space should match."""

        def nonsense_model(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return torch.randn(x.shape[0], 2, x.shape[2], x.shape[3]) * 10.0

        manifold = CylindricalManifold()
        solver = manifold.make_solver(num_steps=3)
        x_1 = _cylindrical_state(2, 16, 16, seed=12)

        pred = reconstruct_batch(nonsense_model, manifold, solver, x_1, t_start=1.0)
        mask = torch.ones(1, 1, 16, 16)
        metrics = compute_batch_metrics(
            manifold, pred, x_1, mask_threshold=0.05, sampling_mask=mask
        )

        assert torch.all(metrics["data_consistency_error"] < 1e-8)

    def test_all_air_slice_is_unscored_not_zero(self) -> None:
        """An all-air slice must come back NaN and be counted, never averaged in."""

        def nonsense_model(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return torch.randn(x.shape[0], 2, x.shape[2], x.shape[3]) * 10.0

        manifold = CylindricalManifold()
        solver = manifold.make_solver(num_steps=3)
        x_1 = _cylindrical_state(3, 16, 16, seed=6)
        x_1[1, 0] = 0.0  # middle slice is pure air: zero amplitude everywhere

        pred = reconstruct_batch(nonsense_model, manifold, solver, x_1, t_start=1.0)
        metrics = compute_batch_metrics(manifold, pred, x_1, mask_threshold=0.05)

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

        # The same slice is unscorable for phase (no tissue to compare) but a
        # perfect match for amplitude. Those must not share a label.
        assert phase_summary.unscored_ids == ("b",)
        assert phase_summary.perfect_ids == ()
        assert psnr_summary.perfect_ids == ("b",)
        assert psnr_summary.unscored_ids == ()


class TestEuclideanReconstruction:
    """The same eval path, run on the flat geometry.

    The point is not that these numbers are good, but that the machinery above -
    the bridge, the integration, the complex-domain scoring, the accumulator -
    is genuinely geometry-agnostic and produces a comparable table for both arms.
    """

    @staticmethod
    def _euclidean_state(batch: int, h: int, w: int, seed: int = 0) -> torch.Tensor:
        torch.manual_seed(seed)
        amp = torch.rand(batch, 1, h, w)
        phi = torch.rand(batch, 1, h, w) * 2 * math.pi
        return torch.cat([amp * torch.cos(phi), amp * torch.sin(phi)], dim=1)

    def test_t_start_one_is_a_perfect_reconstruction(self) -> None:
        """t_start=1 makes the bridge return the target, so metrics must be ~perfect.

        NOT +inf: the bridge evaluates x_0 + t*(x_1 - x_0), which at t=1 is x_1
        only up to float32 rounding, so abs() is not bit-exact. Measured ~156 dB,
        the same order as the cylindrical path - for a different reason there
        (an atan2 -> cos/sin roundtrip), but the same practical floor.
        """

        def nonsense_model(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return torch.randn(x.shape[0], 2, x.shape[2], x.shape[3]) * 10.0

        manifold = EuclideanManifold()
        solver = manifold.make_solver(num_steps=5)
        x_1 = self._euclidean_state(3, 16, 16, seed=7)

        pred = reconstruct_batch(nonsense_model, manifold, solver, x_1, t_start=1.0)
        metrics = compute_batch_metrics(manifold, pred, x_1, mask_threshold=0.05)

        assert torch.all(metrics["psnr_db"] > 100.0)
        assert torch.allclose(metrics["ssim"], torch.ones_like(metrics["ssim"]), atol=1e-4)
        assert torch.all(metrics["phase_error_rad"] < 1e-5)

    def test_produces_the_same_metric_keys_as_the_cylindrical_path(self) -> None:
        """A comparison table needs both columns to have the same rows."""

        def zero_model(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.shape[0], 2, x.shape[2], x.shape[3])

        euc = EuclideanManifold()
        euc_metrics = compute_batch_metrics(
            euc,
            reconstruct_batch(
                zero_model,
                euc,
                euc.make_solver(2),
                self._euclidean_state(2, 16, 16, seed=8),
                t_start=0.5,
            ),
            self._euclidean_state(2, 16, 16, seed=8),
            mask_threshold=0.05,
        )

        cyl = CylindricalManifold()
        cyl_metrics = compute_batch_metrics(
            cyl,
            reconstruct_batch(
                zero_model,
                cyl,
                cyl.make_solver(2),
                _cylindrical_state(2, 16, 16, seed=8),
                t_start=0.5,
            ),
            _cylindrical_state(2, 16, 16, seed=8),
            mask_threshold=0.05,
        )

        assert euc_metrics.keys() == cyl_metrics.keys()
        assert all(v.shape == (2,) for v in euc_metrics.values())

    def test_integration_runs_from_an_arbitrary_t_start(self) -> None:
        """integrate_from_t must drive the Euclidean solver as readily as the cylindrical one."""
        seen: list[float] = []

        def recording_model(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            seen.append(float(t[0]))
            return torch.zeros(x.shape[0], 2, x.shape[2], x.shape[3])

        manifold = EuclideanManifold()
        x_start = self._euclidean_state(1, 8, 8, seed=9)
        out = integrate_from_t(recording_model, manifold.make_solver(5), x_start, t_start=0.4)

        expected = [0.4, 0.52, 0.52, 0.64, 0.64, 0.76, 0.76, 0.88, 0.88]
        assert seen == pytest.approx(expected, abs=1e-6)
        # A zero field must leave a flat state untouched: no clamp, no projection.
        assert torch.equal(out, x_start)


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
        # Index 1 is NaN (unscorable), index 2 is +inf (perfect). Separate lists.
        assert summary.unscored_ids == ("#1",)
        assert summary.perfect_ids == ("#2",)

    def test_perfect_and_unscored_are_labelled_separately(self) -> None:
        """A +inf sample is a perfect score, so it must not read as a failure."""
        acc = MetricAccumulator("psnr_db")
        acc.update(torch.tensor([30.0, float("inf"), float("nan")]), ["ok", "exact", "air"])

        table = format_summary_table([acc.summary()])
        perfect_line = next(line for line in table.splitlines() if line.startswith("perfect"))
        unscored_line = next(line for line in table.splitlines() if line.startswith("unscored"))

        assert "exact" in perfect_line and "air" not in perfect_line
        assert "air" in unscored_line and "exact" not in unscored_line
        assert "not a failure" in table

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


def test_write_eval_records_round_trips(tmp_path) -> None:
    acc = MetricAccumulator("psnr_db")
    acc.update(torch.tensor([30.5, 32.1]), ["file1.h5[0]", "file1.h5[1]"])
    accumulators = {"psnr_db": acc}
    csv_path = write_eval_records(
        tmp_path,
        accumulators,
        manifold="cylindrical",
        model="TestModel",
        split="test",
        seed=42,
        t_start=0.5,
    )
    assert csv_path.exists()

    with open(csv_path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        rows = list(reader)

    expected_fields = [
        "sample_id",
        "file",
        "slice_idx",
        "manifold",
        "model",
        "split",
        "seed",
        "t_start",
        "psnr_db",
    ]
    assert fieldnames == expected_fields
    assert len(rows) == 2

    assert rows[0]["sample_id"] == "file1.h5[0]"
    assert rows[0]["file"] == "file1.h5"
    assert rows[0]["slice_idx"] == "0"
    assert rows[0]["manifold"] == "cylindrical"
    assert rows[0]["model"] == "TestModel"
    assert rows[0]["split"] == "test"
    assert rows[0]["seed"] == "42"
    assert rows[0]["t_start"] == "0.5"
    assert float(rows[0]["psnr_db"]) == pytest.approx(30.5)

    assert rows[1]["sample_id"] == "file1.h5[1]"
    assert rows[1]["file"] == "file1.h5"
    assert rows[1]["slice_idx"] == "1"
    assert float(rows[1]["psnr_db"]) == pytest.approx(32.1)


def test_write_eval_records_duplicate_sample_id_raises(tmp_path) -> None:
    acc = MetricAccumulator("psnr_db")
    acc.update(torch.tensor([30.5]), ["file1.h5[0]"])
    acc.update(torch.tensor([32.1]), ["file1.h5[0]"])
    accumulators = {"psnr_db": acc}

    with pytest.raises(ValueError, match="duplicate basenames under data_dir"):
        write_eval_records(tmp_path, accumulators)


def test_metric_accumulator_records() -> None:
    acc = MetricAccumulator("psnr_db")
    acc.update(torch.tensor([1.0, 2.0]), ["s1", "s2"])
    records = acc.records()
    assert len(records) == 2
    assert records[0] == ("s1", 1.0)
    assert records[1] == ("s2", 2.0)


class TestConditionalBridge:
    """The 'aliased' endpoint: the measurement is the initial condition.

    Under this endpoint the ground truth must not reach the start state at all,
    which is the whole point of the key. These tests pin that property, since a
    leak would inflate every metric the gate experiment reports.
    """

    def test_start_state_is_the_alias_verbatim(self) -> None:
        """At t=0 the solver must receive x_alias itself, untouched by noise."""
        seen: list[torch.Tensor] = []

        def recording_model(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            seen.append(x.clone())
            return torch.zeros(x.shape[0], 2, x.shape[2], x.shape[3])

        manifold = CylindricalManifold()
        x_1 = _cylindrical_state(2, 8, 8, seed=3)
        x_alias = _cylindrical_state(2, 8, 8, seed=9)

        reconstruct_batch(
            recording_model,
            manifold,
            manifold.make_solver(2),
            x_1,
            t_start=0.0,
            x_alias=x_alias,
            bridge_endpoint="aliased",
        )

        assert torch.equal(seen[0], x_alias)

    def test_target_never_enters_the_start_state(self) -> None:
        """Changing the target alone must not change the reconstruction.

        With a zero velocity field the output is exactly the start state, so this
        is a direct test that x_1 contributes nothing but its shape.
        """

        def zero_model(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            return torch.zeros(x.shape[0], 2, x.shape[2], x.shape[3])

        manifold = CylindricalManifold()
        x_alias = _cylindrical_state(2, 8, 8, seed=9)
        solver = manifold.make_solver(2)

        first = reconstruct_batch(
            zero_model,
            manifold,
            solver,
            _cylindrical_state(2, 8, 8, seed=3),
            t_start=0.0,
            x_alias=x_alias,
            bridge_endpoint="aliased",
        )
        second = reconstruct_batch(
            zero_model,
            manifold,
            solver,
            _cylindrical_state(2, 8, 8, seed=44),
            t_start=0.0,
            x_alias=x_alias,
            bridge_endpoint="aliased",
        )

        assert torch.equal(first, second)

    def test_is_deterministic_without_a_generator(self) -> None:
        """No noise is drawn, so two calls must agree bit-for-bit."""

        manifold = CylindricalManifold()
        x_1 = _cylindrical_state(2, 8, 8, seed=3)
        x_alias = _cylindrical_state(2, 8, 8, seed=9)
        solver = manifold.make_solver(3)

        runs = [
            reconstruct_batch(
                _state_and_time_model,
                manifold,
                solver,
                x_1,
                t_start=0.0,
                x_alias=x_alias,
                bridge_endpoint="aliased",
            )
            for _ in range(2)
        ]

        assert torch.equal(runs[0], runs[1])

    def test_rejects_a_non_zero_t_start(self) -> None:
        """t_start > 0 would mix the target back in; it must raise, not report."""
        manifold = CylindricalManifold()

        with pytest.raises(ValueError, match="requires t_start=0.0"):
            reconstruct_batch(
                _state_and_time_model,
                manifold,
                manifold.make_solver(2),
                _cylindrical_state(2, 8, 8, seed=3),
                t_start=0.5,
                x_alias=_cylindrical_state(2, 8, 8, seed=9),
                bridge_endpoint="aliased",
            )

    def test_rejects_a_missing_alias(self) -> None:
        """Without a measurement there is no initial condition to start from."""
        manifold = CylindricalManifold()

        with pytest.raises(ValueError, match="needs x_alias"):
            reconstruct_batch(
                _state_and_time_model,
                manifold,
                manifold.make_solver(2),
                _cylindrical_state(2, 8, 8, seed=3),
                t_start=0.0,
                bridge_endpoint="aliased",
            )

    def test_rejects_an_unknown_endpoint(self) -> None:
        """A typo must not fall through to the noise path."""
        manifold = CylindricalManifold()

        with pytest.raises(ValueError, match="bridge_endpoint must be one of"):
            reconstruct_batch(
                _state_and_time_model,
                manifold,
                manifold.make_solver(2),
                _cylindrical_state(2, 8, 8, seed=3),
                t_start=0.0,
                bridge_endpoint="zero_filled",
            )

    def test_noise_endpoint_is_unchanged(self) -> None:
        """The default path must be bit-for-bit what it was before the key existed."""
        manifold = CylindricalManifold()
        x_1 = _cylindrical_state(2, 8, 8, seed=3)
        solver = manifold.make_solver(3)

        torch.manual_seed(0)
        explicit = reconstruct_batch(
            _state_and_time_model, manifold, solver, x_1, t_start=0.5, bridge_endpoint="noise"
        )
        torch.manual_seed(0)
        default = reconstruct_batch(_state_and_time_model, manifold, solver, x_1, t_start=0.5)

        assert torch.equal(explicit, default)
