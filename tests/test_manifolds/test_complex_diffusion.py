"""Unit tests for ComplexDiffusionManifold and PredictorCorrectorSolver.

Covers:
- VE-SDE forward process and noise schedule.
- Denoising score matching loss with likelihood weighting.
- Manifold interface contract (BaseManifold compliance).
- Predictor-Corrector SDE solver and data consistency.
- DiffusionReconstructor integration.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from cfm.core.manifold import BaseManifold
from cfm.core.registry import MANIFOLDS, RECONSTRUCTORS, SOLVERS
from cfm.core.solver import BaseSDESolver
from cfm.manifolds import ComplexDiffusionManifold, build_manifold
from cfm.solvers.diffusion_solver import PredictorCorrectorSolver, apply_data_consistency
from cfm.utils.fft import fft2c

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
        assert issubclass(MANIFOLDS.get("complex_diffusion"), BaseManifold)

    def test_registered_in_solvers(self) -> None:
        assert "pc_diffusion" in SOLVERS
        assert "complex_diffusion" in SOLVERS
        assert issubclass(SOLVERS.get("pc_diffusion"), BaseSDESolver)

    def test_build_manifold_defaults(self) -> None:
        cfg = OmegaConf.create({"manifold": {"name": "complex_diffusion"}})
        manifold = build_manifold(cfg)
        assert isinstance(manifold, ComplexDiffusionManifold)
        assert manifold.sigma_min == 0.01
        assert manifold.sigma_max == 378.0
        assert manifold.eps == 1e-5
        assert manifold.likelihood_weighting is True
        assert manifold.snr == 0.16
        assert manifold.corrector_steps == 1
        assert manifold.num_steps == 50

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
        manifold = ComplexDiffusionManifold(sigma_min=0.01, sigma_max=378.0)
        pred = torch.zeros(1, 2, 4, 4)
        target = torch.ones(1, 2, 4, 4)
        t = torch.tensor([0.5]).view(1, 1, 1, 1)

        # With likelihood_weighting=False: error weighted by sigma(t)^2
        sig_half = manifold.sigma(0.5).item()
        expected_loss = (sig_half**2) * 1.0

        total, _ = manifold.loss(pred, target, t=t, likelihood_weighting=False)
        assert total.item() == pytest.approx(expected_loss, rel=1e-4)

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
        manifold = ComplexDiffusionManifold()
        z = _dummy_complex_slice(b=1, h=40, w=40).squeeze(0)
        tf = manifold.build_transform(crop_base=16)
        state = tf(z)
        assert state.shape == (2, 320, 320)
        z_out = manifold.to_complex(state.unsqueeze(0))
        modulus = z_out.abs()
        assert modulus.max().item() == pytest.approx(1.0, abs=1e-5)

    def test_transforms_window(self) -> None:
        manifold = ComplexDiffusionManifold()
        slice_tf, window_tf = manifold.build_window_transforms(crop_base=16)
        slices = [_dummy_complex_slice(b=1, h=40, w=40).squeeze(0) * (i + 1) for i in range(3)]
        stack = torch.stack([slice_tf(s) for s in slices])
        out = window_tf(stack)
        assert out.shape == (3, 2, 320, 320)
        modulus = manifold.to_complex(out).abs()
        assert modulus.max().item() == pytest.approx(1.0, abs=1e-5)

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

    def test_data_consistency_single_coil(self) -> None:
        # Create true complex image and undersampled k-space
        z_true = _dummy_complex_slice(b=2, h=16, w=16)
        k_full = fft2c(z_true)

        mask = torch.zeros(2, 1, 16, 16)
        mask[:, :, :, 6:10] = 1.0  # Center lines sampled
        k_masked = k_full * mask

        # Corrupted state
        z_corrupted = _dummy_complex_slice(b=2, h=16, w=16, seed=999)
        x_corrupted = torch.cat([z_corrupted.real, z_corrupted.imag], dim=1)

        x_dc = apply_data_consistency(x_corrupted, masked_kspace=k_masked, mask=mask)
        z_dc = torch.complex(x_dc[:, 0:1], x_dc[:, 1:2])
        k_dc = fft2c(z_dc)

        # On mask: must exactly match k_masked
        torch.testing.assert_close(k_dc * mask, k_masked * mask, atol=1e-5, rtol=1e-5)
        # Off mask: must match corrupted state
        k_corrupted = fft2c(z_corrupted)
        torch.testing.assert_close(
            k_dc * (1.0 - mask), k_corrupted * (1.0 - mask), atol=1e-5, rtol=1e-5
        )

    def test_data_consistency_multi_coil(self) -> None:
        b, coils, h, w = 2, 4, 16, 16
        sens = torch.complex(torch.randn(b, coils, h, w), torch.randn(b, coils, h, w))
        sens = sens / torch.sqrt((sens.abs() ** 2).sum(dim=1, keepdim=True) + 1e-8)

        z_true = _dummy_complex_slice(b=b, h=h, w=w)
        k_coils = fft2c(sens * z_true)
        mask = torch.zeros(b, 1, h, w)
        mask[:, :, :, 4:8] = 1.0
        k_masked = k_coils * mask

        z_corr = _dummy_complex_slice(b=b, h=h, w=w, seed=123)
        x_corr = torch.cat([z_corr.real, z_corr.imag], dim=1)

        x_dc = apply_data_consistency(
            x_corr, masked_kspace=k_masked, mask=mask, sensitivity_maps=sens
        )
        assert x_dc.shape == (b, 2, h, w)

    def test_sample_end_to_end(self) -> None:
        solver = PredictorCorrectorSolver(num_steps=3, m_steps=1)
        noise = torch.randn(2, 2, 8, 8)

        def dummy_model(x_in: torch.Tensor, t_in: torch.Tensor) -> torch.Tensor:
            del t_in
            return torch.zeros_like(x_in)

        out = solver.sample(dummy_model, noise)
        assert out.shape == noise.shape


class TestReconstructorIntegration:
    """Tests DiffusionReconstructor and RECONSTRUCTORS registry integration."""

    def test_diffusion_reconstructor_registered(self) -> None:
        assert "diffusion" in RECONSTRUCTORS
        assert "pc_diffusion" in RECONSTRUCTORS

    def test_reconstruct_single_coil(self) -> None:
        class DummyScoreModel(nn.Module):
            def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
                del t
                return torch.zeros_like(x)

        model = DummyScoreModel()
        reconstructor = RECONSTRUCTORS.build("diffusion", model=model, num_steps=2)

        z_gt = _dummy_complex_slice(b=1, h=16, w=16)
        k_full = fft2c(z_gt)
        mask = torch.zeros(1, 1, 16, 16)
        mask[:, :, :, 6:10] = 1.0
        k_masked = k_full * mask

        z_rec = reconstructor.reconstruct(k_masked, mask)
        assert z_rec.shape == (1, 1, 16, 16)
        assert torch.is_complex(z_rec)
        # On the mask, measured k-space is preserved (up to float32 FFT roundtrip precision)
        k_rec = fft2c(z_rec)
        torch.testing.assert_close(k_rec * mask, k_masked * mask, atol=1e-3, rtol=1e-3)
