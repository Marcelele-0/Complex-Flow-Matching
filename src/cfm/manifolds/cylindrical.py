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
from cfm.utils.complex_ops import cylinder_to_complex, wrap_to_pi
from cfm.utils.random_fields import smooth_standard_normals


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


def sample_cylindrical_noise_concentrated(
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    spread: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Cylindrical prior whose phase is concentrated rather than uniform.

    A uniform phase prior makes the geodesic bridge's angular target
    unlearnable. With ``phi_0`` uniform and independent of the data, the state
    ``phi_t = phi_0 + t * u_phi (mod 2 pi)`` is itself nearly uniform whatever
    ``u_phi`` was, so the best predictor of ``u_phi`` from ``(x_t, t)`` is close
    to a constant. Measured on this repository's synthetic target: the
    cylindrical arm's angular loss reaches 1.538 against 1.571 for predicting
    zero, i.e. it explains 2% of the phase, while its *amplitude* -- which
    interpolates linearly and does not wrap -- trains to parity with the
    Euclidean arm. The failure is the wrap, not the network.

    Concentrating ``phi_0`` restores the correlation: with ``phi_0`` near a known
    value, ``phi_t`` is close to ``t * u_phi`` and the target becomes recoverable
    from the state. The prior is a free choice in flow matching, so this costs
    nothing in principle -- but it does make the prior anisotropic on the circle,
    which is a modelling decision and is stated rather than hidden.

    Uses a wrapped normal, whose ``spread -> infinity`` limit is the uniform
    prior above; the two are one family, not two cases.

    Args:
        batch: Number of samples.
        height: Spatial height.
        width: Spatial width.
        device: Device to allocate on.
        spread: Standard deviation of the unwrapped normal, in radians. Must be
            positive; beyond about ``2 pi`` the result is uniform to numerical
            precision.
        generator: Optional RNG for reproducibility.

    Returns:
        Cylindrical noise of shape ``[B, 3, H, W]``.

    Raises:
        ValueError: If ``spread`` is not positive.
    """
    if spread <= 0.0:
        raise ValueError(f"spread must be positive, got {spread}")
    amp = torch.rand(batch, 1, height, width, device=device, generator=generator)
    phi = spread * torch.randn(batch, 1, height, width, device=device, generator=generator)
    return torch.cat([amp, torch.cos(phi), torch.sin(phi)], dim=1)


def sample_cylindrical_noise_correlated(
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    correlation_length: float,
    spread: float | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Cylindrical prior with spatial correlation and an unchanged pointwise law.

    A white prior is what drowns minibatch optimal transport at field scale: every
    coefficient is an independent draw, so two samples differ in as many
    directions as there are pixels and all pairwise costs concentrate around one
    value. Measured on this repository's smooth synthetic target (64x64,
    correlation length 4), the OT cost reduction is 2.7% under a white prior and
    21.8% under a prior smoothed at length 16 -- the level a 64-dimensional
    problem reaches. The prior's effective dimension is what falls.

    Both coordinates are built from smoothed standard normal latents: the modulus
    as ``Phi(z_a)``, uniform on ``[0, 1]`` exactly as in
    :func:`sample_cylindrical_noise`; the phase as ``2 pi Phi(z_p)``, uniform, or
    as ``spread * z_p``, a wrapped normal, when ``spread`` is set. Every pointwise
    marginal is therefore the default prior's, and only the spatial dependence is
    new -- which is what lets this be compared against the white prior at all.

    Args:
        batch: Number of samples.
        height: Spatial height.
        width: Spatial width.
        device: Device to allocate on.
        correlation_length: Gaussian smoothing sigma in pixels, positive.
        spread: Optional wrapped-normal phase spread in radians; ``None`` keeps
            the phase uniform.
        generator: Optional RNG for reproducibility.

    Returns:
        Cylindrical noise of shape ``[B, 3, H, W]``.

    Raises:
        ValueError: If ``correlation_length`` is not positive.
    """
    if correlation_length <= 0.0:
        raise ValueError(f"correlation_length must be positive, got {correlation_length}")
    shape = (batch, 1, height, width)
    amplitude_latent = torch.randn(shape, device=device, generator=generator)
    phase_latent = torch.randn(shape, device=device, generator=generator)
    amplitude_latent = smooth_standard_normals(amplitude_latent, correlation_length)
    phase_latent = smooth_standard_normals(phase_latent, correlation_length)

    amp = torch.special.ndtr(amplitude_latent)
    if spread is None:
        phi = 2 * math.pi * torch.special.ndtr(phase_latent)
    else:
        phi = spread * phase_latent
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
        phase_weight: Angular weight of the tangent inner product.
        phase_spread: Standard deviation, in radians, of a wrapped-normal phase
            prior. ``None`` keeps the uniform prior, which is the
            ``phase_spread -> infinity`` limit of the same family. See
            :func:`sample_cylindrical_noise_concentrated` for why this knob
            exists.
        spatial_correlation: Gaussian smoothing sigma, in pixels, applied to the
            prior's latents. ``None`` keeps the prior white. The pointwise law is
            unchanged either way; see :func:`sample_cylindrical_noise_correlated`.
        phase_amplitude_weighting: Weight the phase error by the clean amplitude.
            See :class:`cfm.flow.torus_math.DecoupledCylindricalLoss`.
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
        phase_weight: float = 1.0,
        phase_spread: float | None = None,
        spatial_correlation: float | None = None,
        phase_amplitude_weighting: bool = True,
    ) -> None:
        if phase_weight < 0.0:
            raise ValueError(f"phase_weight must be non-negative, got {phase_weight}")
        if phase_spread is not None and phase_spread <= 0.0:
            raise ValueError(f"phase_spread must be positive or null, got {phase_spread}")
        self.phase_weight = float(phase_weight)
        self.phase_spread = None if phase_spread is None else float(phase_spread)
        if spatial_correlation is not None and spatial_correlation <= 0.0:
            raise ValueError(
                f"spatial_correlation must be positive or null, got {spatial_correlation}"
            )
        self.spatial_correlation = (
            None if spatial_correlation is None else float(spatial_correlation)
        )
        self._bridge = GeodesicFlowBridge()
        self._loss = DecoupledCylindricalLoss(
            amp_loss_type=amp_loss_type,
            phase_loss_type=phase_loss_type,
            lambda_phase=lambda_phase,
            lambda_hf=lambda_hf,
            hf_boost_factor=hf_boost_factor,
            phase_amplitude_weighting=phase_amplitude_weighting,
        )

    @property
    def tangent_weights(self) -> torch.Tensor:
        """Amplitude at unit weight, phase at ``phase_weight``.

        ``1.0`` is the flat product metric ``ds^2 = dA^2 + dtheta^2``, which is
        not the pullback of the plane's metric -- that would carry ``A^2
        dtheta^2``. Dropping the ``A^2`` is the point of the cylinder, since it
        stops phase error being attenuated by amplitude, but it also leaves the
        angular term roughly ten times the amplitude term in scale. Exposing the
        weight is what keeps that a stated choice rather than a hidden one.
        """
        return torch.tensor([1.0, self.phase_weight])

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
        u_phi = wrap_to_pi(phi_1 - phi_0)
        return torch.cat([u_m, u_phi], dim=1)

    def geodesic_path(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Geodesic curve x_t = (1-t)x_0 + t x_1 on R+ x S^1."""
        m_0 = x_0[:, 0:1, :, :]
        m_1 = x_1[:, 0:1, :, :]
        m_t = (1.0 - t) * m_0 + t * m_1

        phi_0 = torch.atan2(x_0[:, 2:3, :, :], x_0[:, 1:2, :, :])
        phi_1 = torch.atan2(x_1[:, 2:3, :, :], x_1[:, 1:2, :, :])
        u_phi = wrap_to_pi(phi_1 - phi_0)
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
        if self.spatial_correlation is not None:
            return sample_cylindrical_noise_correlated(
                batch,
                height,
                width,
                device,
                self.spatial_correlation,
                self.phase_spread,
                generator,
            )
        if self.phase_spread is None:
            return sample_cylindrical_noise(batch, height, width, device, generator)
        return sample_cylindrical_noise_concentrated(
            batch, height, width, device, self.phase_spread, generator
        )

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
