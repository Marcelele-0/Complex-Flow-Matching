"""The cylindrical geometry, R^+ x S^1 - the repository's original formulation.

This module adds no mathematics. It is an adapter that presents the existing
:class:`~cfm.flow.bridge.GeodesicFlowBridge`,
:class:`~cfm.flow.solver.CylindricalODESolver` and
:class:`~cfm.flow.torus_math.DecoupledCylindricalLoss` through the
:class:`~cfm.core.manifold.BaseManifold` interface, so that introducing a second
geometry could not change the behaviour of the first.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch

from cfm.core.manifold import BaseManifold
from cfm.core.registry import MANIFOLDS
from cfm.data.transforms import (
    AmplitudeNormalize,
    CenterCropModulo,
    ComplexToCylinderTransform,
    Compose,
    WindowAmplitudeNormalize,
)
from cfm.flow.bridge import GeodesicFlowBridge
from cfm.flow.solver import CylindricalODESolver
from cfm.flow.torus_math import DecoupledCylindricalLoss
from cfm.utils.complex_ops import cylinder_to_complex


def sample_cylindrical_noise(
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw ``x_0`` on the cylinder: amplitude ``U[0, 1]``, phase ``U[0, 2*pi)``.

    Any other distribution puts the model off its training manifold at ``t=0``.

    The two ``torch.rand`` calls and their order are load-bearing: the Euclidean
    manifold's ``matched`` prior repeats them exactly, so one generator and one
    seed produce the *same* complex noise field on both geometries, up to float32
    round-off in the ``atan2`` round trip out of this representation (~1e-7).

    Args:
        batch: Number of samples.
        height: Spatial height.
        width: Spatial width.
        device: Device to allocate on.
        generator: Optional RNG for reproducibility.

    Returns:
        Cylindrical noise of shape ``[B, 3, H, W]``.
    """
    amp = torch.rand(batch, 1, height, width, device=device, generator=generator)
    phi = torch.rand(batch, 1, height, width, device=device, generator=generator) * 2 * math.pi
    return torch.cat([amp, torch.cos(phi), torch.sin(phi)], dim=1)


@MANIFOLDS.register("cylindrical")
class CylindricalManifold(BaseManifold):
    """Amplitude on the half-line, phase on the circle, carried as ``[m, cos, sin]``.

    Args:
        amp_loss_type: ``"l1"``, ``"l2"`` or ``"mse"`` for the amplitude term.
        phase_loss_type: As above, plus ``"cosine"`` for the wrapping variant.
        lambda_phase: Weight on the phase term.
        lambda_hf: Weight on the high-frequency k-space penalty. ``0.0`` disables it.
        hf_boost_factor: Radial slope of that penalty.
    """

    name = "cylindrical"
    state_channels = 3
    velocity_channels = 2

    def __init__(
        self,
        amp_loss_type: str = "l1",
        phase_loss_type: str = "l1",
        lambda_phase: float = 1.0,
        lambda_hf: float = 0.0,
        hf_boost_factor: float = 4.0,
    ) -> None:
        self._bridge = GeodesicFlowBridge()
        self._loss = DecoupledCylindricalLoss(
            amp_loss_type=amp_loss_type,
            phase_loss_type=phase_loss_type,
            lambda_phase=lambda_phase,
            lambda_hf=lambda_hf,
            hf_boost_factor=hf_boost_factor,
        )

    def to(self, device: torch.device) -> CylindricalManifold:
        self._loss = self._loss.to(device)
        return self

    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Riemannian exp map on cylindrical product manifold R+ x S^1."""
        m_t = x[:, 0:1, :, :]
        px_t = x[:, 1:2, :, :]
        py_t = x[:, 2:3, :, :]

        v_m = v[:, 0:1, :, :]
        v_phi = v[:, 1:2, :, :]

        m_next = torch.clamp(m_t + v_m, min=0.0)
        phi_t = torch.atan2(py_t, px_t)
        phi_next = phi_t + v_phi
        px_next = torch.cos(phi_next)
        py_next = torch.sin(phi_next)
        return torch.cat([m_next, px_next, py_next], dim=1)

    def log_map(self, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        """Riemannian log map on cylindrical product manifold R+ x S^1."""
        m_0 = x_0[:, 0:1, :, :]
        m_1 = x_1[:, 0:1, :, :]
        u_m = m_1 - m_0

        phi_0 = torch.atan2(x_0[:, 2:3, :, :], x_0[:, 1:2, :, :])
        phi_1 = torch.atan2(x_1[:, 2:3, :, :], x_1[:, 1:2, :, :])
        diff = phi_1 - phi_0
        u_phi = torch.atan2(torch.sin(diff), torch.cos(diff))
        return torch.cat([u_m, u_phi], dim=1)

    def geodesic_path(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Geodesic curve x_t = (1-t)x_0 + t x_1 on R+ x S^1."""
        m_0 = x_0[:, 0:1, :, :]
        m_1 = x_1[:, 0:1, :, :]
        m_t = (1.0 - t) * m_0 + t * m_1

        phi_0 = torch.atan2(x_0[:, 2:3, :, :], x_0[:, 1:2, :, :])
        phi_1 = torch.atan2(x_1[:, 2:3, :, :], x_1[:, 1:2, :, :])
        diff = phi_1 - phi_0
        u_phi = torch.atan2(torch.sin(diff), torch.cos(diff))
        phi_t = phi_0 + t * u_phi

        px_t = torch.cos(phi_t)
        py_t = torch.sin(phi_t)
        return torch.cat([m_t, px_t, py_t], dim=1)

    def target_velocity(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor | None = None
    ) -> torch.Tensor:
        del t
        return self.log_map(x_0, x_1)

    def metric_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Decoupled product metric g = dm^2 + d_theta^2."""
        return torch.ones(
            x.shape[0],
            self.velocity_channels,
            x.shape[2],
            x.shape[3],
            device=x.device,
            dtype=x.dtype,
        )

    def build_transform(self, crop_base: int = 16) -> Callable[[torch.Tensor], torch.Tensor]:
        return Compose(
            [ComplexToCylinderTransform(), AmplitudeNormalize(), CenterCropModulo(base=crop_base)]
        )

    def build_window_transforms(
        self, crop_base: int = 16
    ) -> tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
        # AmplitudeNormalize would divide each slice by its own peak; the window
        # variant divides the whole stack by one, keeping inter-slice brightness.
        return (
            Compose([ComplexToCylinderTransform()]),
            Compose([WindowAmplitudeNormalize(), CenterCropModulo(base=crop_base)]),
        )

    def sample_noise(
        self,
        batch: int,
        height: int,
        width: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        return sample_cylindrical_noise(batch, height, width, device, generator)

    def bridge(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._bridge.forward(cyl_noise=x_0, cyl_data=x_1, t=t)

    def loss(
        self,
        pred_v: torch.Tensor,
        target_v: torch.Tensor,
        target_x1: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total, loss_amp, loss_phi, loss_hf = self._loss(pred_v, target_v, target_x1)
        # Keys chosen to reproduce the pre-refactor W&B series names exactly
        # (step_loss_amp / step_loss_phi / step_loss_hf), so historical runs stay
        # comparable with new ones.
        return total, {"amp": loss_amp, "phi": loss_phi, "hf": loss_hf}

    def make_solver(self, num_steps: int) -> CylindricalODESolver:
        return CylindricalODESolver(num_steps=num_steps)

    def to_complex(self, state: torch.Tensor) -> torch.Tensor:
        return cylinder_to_complex(state)

    def from_complex(self, z: torch.Tensor) -> torch.Tensor:
        amp = torch.abs(z)
        phi = torch.angle(z)
        return torch.cat([amp, torch.cos(phi), torch.sin(phi)], dim=1)
