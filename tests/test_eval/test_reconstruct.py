"""Tests for the single-slice reconstruction and visualization module."""

import math
import matplotlib.pyplot as plt
import pytest
import torch

from cfm.flow.bridge import GeodesicFlowBridge
from cfm.flow.solver import CylindricalODESolver
from cfm.reconstruct import (
    compute_slice_metrics,
    reconstruct_slice,
    render_diagnostic_figure,
)


def _dummy_model(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Simple velocity field returning 2 channels: (v_amp, v_phi)."""
    return torch.zeros(x.shape[0], 2, x.shape[2], x.shape[3], device=x.device)


def _synthetic_cylindrical_slice(h: int = 32, w: int = 32) -> torch.Tensor:
    torch.manual_seed(0)
    amp = torch.rand(1, 1, h, w)
    phi = torch.rand(1, 1, h, w) * 2 * math.pi
    return torch.cat([amp, torch.cos(phi), torch.sin(phi)], dim=1)


class TestReconstructLogic:
    def test_reconstruct_slice_shape_and_dtype(self) -> None:
        x_1 = _synthetic_cylindrical_slice(16, 16)
        solver = CylindricalODESolver(num_steps=3)
        bridge = GeodesicFlowBridge()

        out, x_corr = reconstruct_slice(
            model=_dummy_model,
            solver=solver,
            bridge=bridge,
            x_1=x_1,
            t_start=0.5,
        )

        assert out.shape == (1, 3, 16, 16)
        assert x_corr.shape == (1, 3, 16, 16)
        assert torch.isfinite(out).all()
        # Cos^2 + Sin^2 on unit circle
        cos_val = out[0, 1]
        sin_val = out[0, 2]
        r2 = cos_val**2 + sin_val**2
        torch.testing.assert_close(r2, torch.ones_like(r2), atol=1e-5, rtol=1e-5)

    def test_compute_slice_metrics_exact_match(self) -> None:
        c = torch.complex(torch.ones(1, 16, 16) * 0.8, torch.zeros(1, 16, 16))
        metrics = compute_slice_metrics(c, c, mask_threshold=0.05)

        assert math.isinf(metrics["psnr_db"]) or metrics["psnr_db"] > 100
        assert pytest.approx(1.0, abs=1e-4) == metrics["ssim"]
        assert pytest.approx(0.0, abs=1e-4) == metrics["phase_error_rad"]

    def test_compute_slice_metrics_perturbation(self) -> None:
        target = torch.complex(torch.ones(1, 16, 16) * 0.8, torch.zeros(1, 16, 16))
        pred = torch.complex(torch.ones(1, 16, 16) * 0.7, torch.zeros(1, 16, 16))
        metrics = compute_slice_metrics(pred, target, mask_threshold=0.05)

        assert metrics["psnr_db"] > 0
        assert 0.0 < metrics["ssim"] <= 1.0
        assert metrics["phase_error_rad"] >= 0.0

    def test_render_figures_produce_valid_plots(self) -> None:
        c_corr = torch.complex(torch.rand(1, 16, 16), torch.rand(1, 16, 16))
        c1 = torch.complex(torch.rand(1, 16, 16), torch.rand(1, 16, 16))
        c2 = torch.complex(torch.rand(1, 16, 16), torch.rand(1, 16, 16))
        metrics = {"psnr_db": 30.5, "ssim": 0.92, "phase_error_rad": 0.15}

        diag_fig = render_diagnostic_figure(c_corr, c1, c2, metrics, "test_slice", t_start=0.5)
        assert isinstance(diag_fig, plt.Figure)
        plt.close(diag_fig)
