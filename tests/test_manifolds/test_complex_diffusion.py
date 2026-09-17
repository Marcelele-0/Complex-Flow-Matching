"""Unit tests for ComplexDiffusionManifold and PredictorCorrectorSolver.

Covers:
- VE-SDE forward process and noise schedule.
- Denoising score matching loss with likelihood weighting.
- Manifold interface contract (BaseManifold compliance).
- Predictor-Corrector SDE solver and data consistency.
"""

from __future__ import annotations

import math
import pathlib
from typing import cast

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from cfm.core.manifold import BaseManifold
from cfm.core.registry import MANIFOLDS, SOLVERS
from cfm.core.solver import BaseSDESolver
from cfm.flow.diffusion_solver import (
    PredictorCorrectorSolver,
    heun_evaluations,
    plan_within_budget,
)
from cfm.manifolds import ComplexDiffusionManifold, build_manifold
from cfm.manifolds.complex_diffusion import DEFAULT_SIGMA_MAX_320, calibrate_sigma_max

CPU = torch.device("cpu")


def _dummy_complex_slice(b: int = 2, h: int = 16, w: int = 16, seed: int = 42) -> torch.Tensor:
    """Generate synthetic complex MRI slice."""
    gen = torch.Generator().manual_seed(seed)
    real = torch.randn(b, 1, h, w, generator=gen)
    imag = torch.randn(b, 1, h, w, generator=gen)
    return torch.complex(real, imag)


class TestRegistryAndInitialization:
    """Tests registration, initialization, and configuration parsing."""

    def test_registered_in_manifolds(self) -> None:
        assert "complex_diffusion" in MANIFOLDS
        assert issubclass(cast(type, MANIFOLDS.get("complex_diffusion")), BaseManifold)

    def test_registered_in_solvers(self) -> None:
        assert "pc_diffusion" in SOLVERS
        assert "complex_diffusion" in SOLVERS
        assert issubclass(cast(type, SOLVERS.get("pc_diffusion")), BaseSDESolver)

    def test_build_manifold_defaults(self) -> None:
        """build_manifold must not carry defaults of its own.

        This is the test that catches a second default drifting from the
        constructor's. sigma_max above all: it is calibrated to the data, and a
        builder that re-types it silently wins whenever a config omits the key --
        training a prior too narrow, or too wide, to forget the data.
        """
        cfg = OmegaConf.create({"manifold": {"name": "complex_diffusion"}})
        manifold = build_manifold(cfg)
        assert isinstance(manifold, ComplexDiffusionManifold)
        assert manifold.sigma_min == 0.01
        assert manifold.sigma_max == DEFAULT_SIGMA_MAX_320
        assert manifold.eps == 1e-5
        assert manifold.likelihood_weighting is True
        assert manifold.snr == 0.16
        assert manifold.corrector_steps == 1
        assert manifold.num_steps == 50
        assert manifold.nfe_mode == "heun"

    def test_build_manifold_custom_config(self) -> None:
        cfg = OmegaConf.create(
            {
                "manifold": {
                    "name": "complex_diffusion",
                    "sigma_min": 0.05,
                    "sigma_max": 200.0,
                    "eps": 1e-4,
                    "likelihood_weighting": False,
                    "snr": 0.2,
                    "corrector_steps": 2,
                    "num_steps": 25,
                }
            }
        )
        manifold = build_manifold(cfg)
        assert isinstance(manifold, ComplexDiffusionManifold)
        assert manifold.sigma_min == 0.05
        assert manifold.sigma_max == 200.0
        assert manifold.eps == 1e-4
        assert manifold.likelihood_weighting is False
        assert manifold.snr == 0.2
        assert manifold.corrector_steps == 2
        assert manifold.num_steps == 25

    def test_fail_fast_on_invalid_parameters(self) -> None:
        with pytest.raises(ValueError, match="sigma_min must be positive"):
            ComplexDiffusionManifold(sigma_min=-0.01)

        with pytest.raises(ValueError, match="sigma_max must be greater than sigma_min"):
            ComplexDiffusionManifold(sigma_min=10.0, sigma_max=1.0)

        with pytest.raises(ValueError, match="eps must be in"):
            ComplexDiffusionManifold(eps=-1.0)

        with pytest.raises(ValueError, match="snr must be non-negative"):
            ComplexDiffusionManifold(snr=-0.1)

        with pytest.raises(ValueError, match="corrector_steps must be non-negative"):
            ComplexDiffusionManifold(corrector_steps=-1)

        with pytest.raises(ValueError, match="num_steps must be positive"):
            ComplexDiffusionManifold(num_steps=0)

        with pytest.raises(ValueError, match="Unexpected keyword arguments"):
            ComplexDiffusionManifold(invalid_param=123)

    def test_channel_counts_and_chainable_to(self) -> None:
        manifold = ComplexDiffusionManifold()
        assert manifold.state_channels == 2
        assert manifold.velocity_channels == 2
        assert manifold.to(CPU) is manifold


class TestSigmaScheduleAndSDE:
    """Tests VE-SDE noise schedule and diffusion coefficient calculations."""

    def test_sigma_endpoints(self) -> None:
        manifold = ComplexDiffusionManifold(sigma_min=0.01, sigma_max=378.0)
        sig_0 = manifold.sigma(0.0).item()
        sig_1 = manifold.sigma(1.0).item()
        assert sig_0 == pytest.approx(0.01, rel=1e-5)
        assert sig_1 == pytest.approx(378.0, rel=1e-4)

    def test_sigma_midpoint(self) -> None:
        manifold = ComplexDiffusionManifold(sigma_min=0.01, sigma_max=378.0)
        # Geometric mean: sqrt(0.01 * 378) = sqrt(3.78)
        expected = math.sqrt(0.01 * 378.0)
        sig_half = manifold.sigma(0.5).item()
        assert sig_half == pytest.approx(expected, rel=1e-4)

    def test_sigma_monotonically_increasing(self) -> None:
        manifold = ComplexDiffusionManifold()
        t = torch.linspace(0.0, 1.0, 100)
        sigmas = manifold.sigma(t)
        diffs = sigmas[1:] - sigmas[:-1]
        assert torch.all(diffs > 0)

    def test_diffusion_coefficient(self) -> None:
        manifold = ComplexDiffusionManifold(sigma_min=0.01, sigma_max=378.0)
        log_ratio = math.log(378.0 / 0.01)
        expected_g0 = 0.01 * math.sqrt(2.0 * log_ratio)
        expected_g1 = 378.0 * math.sqrt(2.0 * log_ratio)

        g0 = manifold.diffusion(0.0).item()
        g1 = manifold.diffusion(1.0).item()
        assert g0 == pytest.approx(expected_g0, rel=1e-4)
        assert g1 == pytest.approx(expected_g1, rel=1e-4)


class TestForwardSDEProcess:
    """Tests VE-SDE forward perturbation x(t) = x(0) + sigma(t) * z."""

    def test_forward_process_exact_formula(self) -> None:
        manifold = ComplexDiffusionManifold(sigma_min=0.01, sigma_max=378.0)
        x_0 = torch.ones(2, 2, 8, 8)
        z = torch.ones(2, 2, 8, 8) * 0.5
        t = torch.tensor([0.0, 1.0]).view(2, 1, 1, 1)

        x_t, returned_z = manifold.forward_process(x_0, t, z=z)
        assert torch.equal(returned_z, z)

        # At t=0: x(0) + sigma_min * z = 1 + 0.01 * 0.5 = 1.005
        expected_0 = 1.0 + 0.01 * 0.5
        # At t=1: x(0) + sigma_max * z = 1 + 378.0 * 0.5 = 190.0
        expected_1 = 1.0 + 378.0 * 0.5

        assert torch.allclose(x_t[0], torch.tensor(expected_0), atol=1e-5)
        assert torch.allclose(x_t[1], torch.tensor(expected_1), atol=1e-3)

    def test_forward_process_random_noise_and_shapes(self) -> None:
        manifold = ComplexDiffusionManifold()
        x_0 = torch.randn(3, 2, 16, 16)
        t = torch.rand(3)

        x_t, z = manifold.forward_process(x_0, t)
        assert x_t.shape == x_0.shape
        assert z.shape == x_0.shape

    def test_forward_process_determinism_with_generator(self) -> None:
        manifold = ComplexDiffusionManifold()
        x_0 = torch.randn(2, 2, 8, 8)
        t = torch.tensor([0.3, 0.7])

        gen1 = torch.Generator().manual_seed(12345)
        gen2 = torch.Generator().manual_seed(12345)

        x_t1, z1 = manifold.forward_process(x_0, t, generator=gen1)
        x_t2, z2 = manifold.forward_process(x_0, t, generator=gen2)

        assert torch.equal(z1, z2)
        assert torch.equal(x_t1, x_t2)

    def test_marginal_prob(self) -> None:
        manifold = ComplexDiffusionManifold()
        x_0 = torch.randn(2, 2, 8, 8)
        t = torch.tensor([0.2, 0.8]).view(2, 1, 1, 1)

        mean, std = manifold.marginal_prob(x_0, t)
        assert torch.equal(mean, x_0)
        assert std.shape == (2, 1, 1, 1)
        assert std[0].item() == pytest.approx(manifold.sigma(0.2).item(), rel=1e-5)
        assert std[1].item() == pytest.approx(manifold.sigma(0.8).item(), rel=1e-5)

    def test_sample_noise_prior(self) -> None:
        manifold = ComplexDiffusionManifold(sigma_max=378.0)
        gen = torch.Generator().manual_seed(42)
        noise = manifold.sample_noise(100, 16, 16, CPU, generator=gen)
        assert noise.shape == (100, 2, 16, 16)
        # Sample standard deviation should be close to sigma_max = 378
        sample_std = noise.std().item()
        assert sample_std == pytest.approx(378.0, rel=0.05)


class TestTargetScoreAndLoss:
    """Tests conditional score target and Denoising Score Matching loss."""

    def test_target_score(self) -> None:
        manifold = ComplexDiffusionManifold(sigma_min=0.01, sigma_max=378.0)
        z = torch.ones(2, 2, 4, 4) * 2.0
        t = torch.tensor([0.0, 1.0]).view(2, 1, 1, 1)

        target = manifold.target_score(z, t)
        # Target score: -z / sigma(t)
        # At t=0: -2.0 / 0.01 = -200.0
        # At t=1: -2.0 / 378.0
        assert torch.allclose(target[0], torch.tensor(-200.0), atol=1e-4)
        assert torch.allclose(target[1], torch.tensor(-2.0 / 378.0), atol=1e-4)

    def test_loss_zero_when_pred_matches_target(self) -> None:
        manifold = ComplexDiffusionManifold()
        target = torch.randn(2, 2, 8, 8)
        t = torch.rand(2, 1, 1, 1)

        total, components = manifold.loss(target, target, t=t)
        assert total.item() == pytest.approx(0.0, abs=1e-7)
        assert components["dsm"].item() == pytest.approx(0.0, abs=1e-7)

    def test_loss_likelihood_weighting(self) -> None:
        manifold = ComplexDiffusionManifold(sigma_min=0.01, sigma_max=378.0)
        pred = torch.zeros(1, 2, 4, 4)
        target = torch.ones(1, 2, 4, 4)
        t = torch.tensor([0.5]).view(1, 1, 1, 1)

        # Expected error: diff = pred - target = -1
        # g(t)^2 = 2 * log(378/0.01) * sigma(0.5)^2
        sig_half = manifold.sigma(0.5).item()
        log_ratio = math.log(378.0 / 0.01)
        expected_g2 = 2.0 * log_ratio * (sig_half**2)
        expected_loss = expected_g2 * 1.0  # since diff^2 = 1.0 everywhere

        total, components = manifold.loss(pred, target, t=t, likelihood_weighting=True)
        assert total.item() == pytest.approx(expected_loss, rel=1e-4)
        assert "dsm" in components

    def test_loss_unweighted(self) -> None:
        """Unweighted means the raw residual, which is what the flag name says.

        The recovered version applied sigma(t)^2 in this branch and still called
        it unweighted. The branch is an ablation, so it should be the thing it is
        named after; the weighted path above is unchanged and is what trains.
        """
        manifold = ComplexDiffusionManifold(sigma_min=0.01, sigma_max=378.0)
        pred = torch.zeros(1, 2, 4, 4)
        target = torch.ones(1, 2, 4, 4)
        t = torch.tensor([0.5]).view(1, 1, 1, 1)

        total, _ = manifold.loss(pred, target, t=t, likelihood_weighting=False)
        assert total.item() == pytest.approx(1.0, rel=1e-6)
        # And it does not depend on t, which is exactly what "unweighted" means.
        other, _ = manifold.loss(
            pred, target, t=torch.tensor([0.9]).view(1, 1, 1, 1), likelihood_weighting=False
        )
        assert other.item() == pytest.approx(1.0, rel=1e-6)

    def test_loss_backward_gradients(self) -> None:
        manifold = ComplexDiffusionManifold()
        pred_param = nn.Parameter(torch.zeros(2, 2, 8, 8))
        target = torch.randn(2, 2, 8, 8)
        t = torch.rand(2, 1, 1, 1)

        loss, _ = manifold.loss(pred_param, target, t=t)
        loss.backward()

        assert pred_param.grad is not None
        assert torch.all(torch.isfinite(pred_param.grad))

    def test_score_matching_loss_helper(self) -> None:
        manifold = ComplexDiffusionManifold()
        x_0 = torch.randn(2, 2, 8, 8)

        class DummyModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.net = nn.Conv2d(2, 2, kernel_size=1)

            def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                del t
                return self.net(x)

        model = DummyModel()
        loss, components = manifold.score_matching_loss(model, x_0)
        assert loss.dim() == 0
        assert "dsm" in components
        loss.backward()
        assert model.net.weight.grad is not None


class TestManifoldInterface:
    """Tests compliance with BaseManifold abstract interface."""

    def test_to_and_from_complex_roundtrip(self) -> None:
        manifold = ComplexDiffusionManifold()
        z_gt = _dummy_complex_slice(b=2, h=16, w=16)
        state = manifold.from_complex(z_gt)
        assert state.shape == (2, 2, 16, 16)

        z_rec = manifold.to_complex(state)
        assert torch.allclose(z_rec.real, z_gt.real, atol=1e-6)
        assert torch.allclose(z_rec.imag, z_gt.imag, atol=1e-6)

    def test_exp_and_log_maps(self) -> None:
        manifold = ComplexDiffusionManifold()
        x = torch.randn(2, 2, 8, 8)
        v = torch.randn(2, 2, 8, 8)

        # Exponential map: x + v
        assert torch.equal(manifold.exp_map(x, v), x + v)

        # Logarithmic map: x_1 - x_0
        x_1 = torch.randn(2, 2, 8, 8)
        assert torch.equal(manifold.log_map(x, x_1), x_1 - x)

    def test_metric_tensor(self) -> None:
        manifold = ComplexDiffusionManifold()
        x = torch.randn(3, 2, 12, 12)
        g = manifold.metric_tensor(x)
        assert g.shape == (3, 2, 12, 12)
        assert torch.all(g == 1.0)

    def test_transforms_single_slice(self) -> None:
        """Crops to a multiple of crop_base, and peak-normalises before cropping.

        The recovered version padded to a fixed 320 here. It was changed to the
        shared pipeline because Table 5 scores this arm against the Cartesian flow
        arm: a transform of its own would preprocess the two differently and move
        every absolute number in the row without failing anything.
        """
        manifold = ComplexDiffusionManifold()
        z = _dummy_complex_slice(b=1, h=40, w=40).squeeze(0)
        tf = manifold.build_transform(crop_base=16)
        state = tf(z)
        assert state.shape == (2, 32, 32)
        z_out = manifold.to_complex(state.unsqueeze(0))
        # Normalisation happens before the crop, so the surviving peak is at most
        # one and is only exactly one when the peak pixel was not cropped away.
        assert z_out.abs().max().item() <= 1.0 + 1e-5

    def test_transforms_window(self) -> None:
        manifold = ComplexDiffusionManifold()
        slice_tf, window_tf = manifold.build_window_transforms(crop_base=16)
        slices = [_dummy_complex_slice(b=1, h=40, w=40).squeeze(0) * (i + 1) for i in range(3)]
        stack = torch.stack([slice_tf(s) for s in slices])
        out = window_tf(stack)
        assert out.shape == (3, 2, 32, 32)
        assert manifold.to_complex(out).abs().max().item() <= 1.0 + 1e-5

    def test_representation_is_shared_with_the_cartesian_arm(self) -> None:
        """The property the side-by-side comparison rests on, asserted not assumed.

        Both arms inherit FlatComplexRepresentation, so the pipelines are the same
        object graph; this pins it, because a divergence here is silent.
        """
        cfg = OmegaConf.create({"manifold": {"name": "euclidean"}, "training": {"loss": {}}})
        euclidean = build_manifold(cfg)
        diffusion = ComplexDiffusionManifold()

        field = _dummy_complex_slice(b=1, h=40, w=40).squeeze(0) * 3.0
        assert torch.equal(
            euclidean.build_transform(crop_base=16)(field.clone()),
            diffusion.build_transform(crop_base=16)(field.clone()),
        )

    def test_bridge_and_geodesic(self) -> None:
        manifold = ComplexDiffusionManifold()
        x_0 = manifold.sample_noise(2, 8, 8, CPU)
        x_1 = torch.randn(2, 2, 8, 8)
        t = torch.tensor([0.4, 0.6]).view(2, 1, 1, 1)

        x_t, target_v = manifold.bridge(x_0, x_1, t)
        assert x_t.shape == (2, 2, 8, 8)
        assert target_v.shape == (2, 2, 8, 8)

        # geodesic_path and target_velocity match bridge outputs
        geo_x = manifold.geodesic_path(x_0, x_1, t)
        assert torch.allclose(geo_x, x_t, atol=1e-6)

        target_u = manifold.target_velocity(x_0, x_1, t)
        assert torch.allclose(target_u, target_v, atol=1e-6)


class TestPredictorCorrectorSolver:
    """Tests Euler-Maruyama predictor, Langevin corrector, and data consistency."""

    def test_solver_construction_and_validation(self) -> None:
        solver = PredictorCorrectorSolver(
            num_steps=20,
            sigma_min=0.01,
            sigma_max=378.0,
            snr=0.16,
            m_steps=1,
        )
        assert solver.num_steps == 20
        assert solver.snr == 0.16
        assert solver.m_steps == 1

        with pytest.raises(ValueError, match="Unexpected keyword arguments"):
            PredictorCorrectorSolver(bad_arg=42)

    def test_corrector_step(self) -> None:
        solver = PredictorCorrectorSolver(num_steps=10, snr=0.16, m_steps=1)
        x = torch.randn(2, 2, 8, 8)
        t = torch.full((2,), 0.5)

        def dummy_model(x_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
            del t_in
            return -x_in  # Simple restoring score

        x_next, x_mean = solver.corrector_step(dummy_model, x, t)
        assert x_next.shape == x.shape
        assert x_mean.shape == x.shape

    def test_predictor_step(self) -> None:
        solver = PredictorCorrectorSolver(num_steps=10)
        x = torch.randn(2, 2, 8, 8)
        t = torch.full((2,), 0.5)
        dt = 0.1

        def dummy_model(x_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
            del t_in
            return -x_in

        # Stochastic predictor
        x_next, x_mean = solver.predictor_step(dummy_model, x, t, dt, probability_flow=False)
        assert x_next.shape == x.shape
        assert x_mean.shape == x.shape

        # Probability flow ODE predictor (no stochastic increment)
        x_ode, x_ode_mean = solver.predictor_step(dummy_model, x, t, dt, probability_flow=True)
        assert torch.equal(x_ode, x_ode_mean)

    def test_step_base_sde_solver_interface(self) -> None:
        solver = PredictorCorrectorSolver()
        x_t = torch.ones(2, 2, 4, 4)
        drift = torch.ones(2, 2, 4, 4) * 2.0
        dt = 0.5
        noise = torch.ones(2, 2, 4, 4) * 3.0

        # Step without noise: 1 + 2 * 0.5 = 2.0
        assert torch.allclose(solver.step(x_t, drift, dt), torch.tensor(2.0))

        # Step with noise: 1 + 2 * 0.5 + 3 * sqrt(0.5)
        expected = 2.0 + 3.0 * math.sqrt(0.5)
        assert torch.allclose(solver.step(x_t, drift, dt, noise=noise), torch.tensor(expected))

    def test_sample_end_to_end(self) -> None:
        solver = PredictorCorrectorSolver(num_steps=3, m_steps=1)
        noise = torch.randn(2, 2, 8, 8)

        def dummy_model(x_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
            del t_in
            return torch.zeros_like(x_in)

        out = solver.sample(dummy_model, noise)
        assert out.shape == noise.shape


class TestUnconditionalSampling:
    """The acceptance criterion of this baseline, not a detail.

    Table 5 compares unconditional generators. An arm that saw a measurement would
    be solving an easier problem and winning for that reason, so the measurement
    route was deleted rather than defaulted off -- these tests pin that it stays
    deleted.
    """

    def test_sampling_runs_from_config_alone_with_no_measurement(self) -> None:
        """Manifold and solver built from config, sampling from noise only."""
        cfg = OmegaConf.create(
            {"manifold": {"name": "complex_diffusion", "sigma_max": 5.0, "num_steps": 3}}
        )
        manifold = build_manifold(cfg)
        solver = manifold.make_solver(3)

        def score(state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            del t
            return -state

        prior = manifold.sample_noise(2, 8, 8, CPU)
        out = solver.sample(score, prior)
        assert out.shape == prior.shape
        assert torch.isfinite(out).all()

    def test_sampler_refuses_a_measurement(self) -> None:
        """Deleted, not defaulted: passing one is an error, not a silent no-op."""
        solver = PredictorCorrectorSolver(num_steps=2, m_steps=0)

        def score(state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            del t
            return torch.zeros_like(state)

        with pytest.raises(ValueError, match="Unexpected keyword arguments"):
            solver.sample(score, torch.randn(1, 2, 8, 8), masked_kspace=torch.randn(1, 1, 8, 8))

    def test_no_measurement_argument_survives_on_the_sampling_path(self) -> None:
        """A reader should not have to trust the two tests above."""
        import inspect

        from cfm.flow import diffusion_solver

        source = inspect.getsource(diffusion_solver)
        for banned in ("masked_kspace", "sensitivity_maps", "data_consistency"):
            assert banned not in source, banned


class TestMatchedNFEAccounting:
    """What the two Table 5 diffusion rows mean.

    Heun spends 2n-1 model calls and this sampler spends 1+M per step, so quoting
    a step count for both would compare different budgets.
    """

    def test_heun_budget_matches_the_documented_counts(self) -> None:
        assert [heun_evaluations(k) for k in (1, 2, 4, 8, 100)] == [1, 3, 7, 15, 199]

    def test_heun_solver_reads_its_count_off_the_same_function(self) -> None:
        """One rule, not a copy on each side of the comparison.

        The flow arm this baseline is matched against is the one that must agree:
        a second copy of 2n-1 is how a "matched NFE" column stops being matched.
        """
        from cfm.flow.euclidean_solver import EuclideanODESolver

        for steps in (1, 2, 4, 8, 100):
            assert EuclideanODESolver(num_steps=steps).evaluations == heun_evaluations(steps)

    @pytest.mark.parametrize("steps", [1, 2, 4, 8, 100])
    def test_matched_mode_never_overspends_the_flow_arms_budget(self, steps: int) -> None:
        """``steps=1`` is the case that matters and the easiest to leave out:
        Heun's budget there is a single call, which one corrected step would double.
        """
        manifold = ComplexDiffusionManifold(corrector_steps=1, nfe_mode="heun")
        assert manifold.make_solver(steps).evaluations <= heun_evaluations(steps)

    def test_matched_mode_drops_the_corrector_before_the_step(self) -> None:
        """At a budget of one call there is room for a predictor and nothing else."""
        solver = ComplexDiffusionManifold(corrector_steps=1, nfe_mode="heun").make_solver(1)
        assert (solver.num_steps, solver.m_steps) == (1, 0)
        assert solver.evaluations == 1

    def test_native_mode_takes_the_step_count_literally(self) -> None:
        """The other regime of the table: the sampler as it is actually used."""
        manifold = ComplexDiffusionManifold(corrector_steps=1, nfe_mode="native")
        assert manifold.make_solver(100).num_steps == 100

    def test_plan_within_budget_accounts_for_the_corrector(self) -> None:
        assert plan_within_budget(100, corrector_steps=0) == (100, 0)
        assert plan_within_budget(100, corrector_steps=1) == (50, 1)
        # Too tight for a corrected step: the correction is what gives way.
        assert plan_within_budget(1, corrector_steps=3) == (1, 0)
        assert plan_within_budget(3, corrector_steps=5) == (1, 2)

    def test_nfe_mode_is_validated(self) -> None:
        with pytest.raises(ValueError, match="nfe_mode must be one of"):
            ComplexDiffusionManifold(nfe_mode="whatever")


class TestPaperIntegration:
    """Properties Table 5 reads off this arm."""

    def test_arm_declares_it_does_not_predict_a_velocity(self) -> None:
        """What keeps straightness and the angular probe off this row."""
        assert ComplexDiffusionManifold().predicts_velocity is False

    def test_calibrate_sigma_max_measures_the_widest_separation(self) -> None:
        fields = torch.zeros(3, 2, 4, 4)
        fields[1] = 1.0
        # Two fields differing by 1.0 in every one of 2*4*4 entries.
        assert calibrate_sigma_max(fields) == pytest.approx(math.sqrt(32.0))

    def test_calibrate_sigma_max_needs_two_fields(self) -> None:
        with pytest.raises(ValueError, match="at least two"):
            calibrate_sigma_max(torch.zeros(1, 2, 4, 4))

    def test_sigma_max_default_matches_the_calibration_rule(self) -> None:
        """The default has to be what calibrate_sigma_max would return, near enough.

        A default contradicting the rule the same module documents is how the
        prior ends up narrower than the data it has to forget. The recovered
        default was 378.0, which belongs to data of a different scale.
        """
        manifold = ComplexDiffusionManifold()
        pipeline = manifold.build_transform(crop_base=16)
        torch.manual_seed(0)
        fields = torch.stack(
            [pipeline(torch.randn(1, 320, 320, dtype=torch.complex64) * 3) for _ in range(8)]
        )
        measured = calibrate_sigma_max(fields)
        assert 0.5 * measured < manifold.sigma_max < 2.0 * measured

    def test_sampler_is_reproducible_under_a_fixed_generator(self) -> None:
        """The sampler draws noise, so a seeded run has to be repeatable."""
        manifold = ComplexDiffusionManifold(corrector_steps=1)
        solver = manifold.make_solver(4)

        def toward_origin(state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            del t
            return -state

        prior = manifold.sample_noise(2, 8, 8, CPU)
        first = solver.sample(toward_origin, prior, generator=torch.Generator().manual_seed(7))
        second = solver.sample(toward_origin, prior, generator=torch.Generator().manual_seed(7))
        assert torch.equal(first, second)

    def test_composes_with_the_table5_experiment(self) -> None:
        """The issue's acceptance criterion, asserted rather than run by hand."""
        from hydra import compose, initialize_config_dir

        conf = str(pathlib.Path(__file__).resolve().parents[2] / "conf")
        with initialize_config_dir(config_dir=conf, version_base="1.3"):
            cfg = compose(
                config_name="config",
                overrides=["+experiment=table5_fastmri", "manifold=complex_diffusion"],
            )
        manifold = build_manifold(cfg)
        assert manifold.name == "complex_diffusion"
        assert manifold.sigma_max == DEFAULT_SIGMA_MAX_320
        assert manifold.predicts_velocity is False
