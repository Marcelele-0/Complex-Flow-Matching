"""The cylindrical geometry, R^+ x S^1 - the repository's original formulation.

This module adds no mathematics. It is an adapter that presents the existing
:class:`~cfm.flow.bridge.GeodesicFlowBridge`,
:class:`~cfm.flow.solver.CylindricalODESolver` and
:class:`~cfm.flow.torus_math.DecoupledCylindricalLoss` through the
:class:`~cfm.manifolds.base.Manifold` interface, so that introducing a second
geometry could not change the behaviour of the first.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

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
from cfm.manifolds.base import Manifold
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


class CylindricalManifold(Manifold):
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
