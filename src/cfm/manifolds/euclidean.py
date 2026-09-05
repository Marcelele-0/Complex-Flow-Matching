"""The flat geometry, R^2 - the standard flow-matching baseline.

Complex pixels are carried as ``(Re z, Im z)`` and treated as ordinary vectors:
a straight-line bridge, an unweighted regression loss, and an Euler update with
no projection of any kind. Strictly no phase wrapping, no modulo arithmetic and
no continuous trigonometric embedding appears anywhere on this path.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import torch

from cfm.core.manifold import BaseManifold
from cfm.core.registry import MANIFOLDS
from cfm.data.transforms import (
    CenterCropModulo,
    ComplexToEuclideanTransform,
    Compose,
    EuclideanNormalize,
    WindowEuclideanNormalize,
)
from cfm.flow.euclidean_bridge import LinearFlowBridge
from cfm.flow.euclidean_math import EuclideanVelocityLoss
from cfm.flow.euclidean_solver import EuclideanODESolver
from cfm.utils.complex_ops import euclidean_to_complex

# "uniform" first: it is the default, being the only one that is both free of
# trigonometry and distributionally identical to the cylindrical prior.
NOISE_PRIORS = ("uniform", "gaussian", "matched")

# Guards the degenerate case where a Gaussian direction underflows to the origin.
# Probability ~0 in exact arithmetic; the clamp only ever shrinks a modulus, so a
# clamped pixel stays inside the unit disc rather than blowing up.
_DIRECTION_EPS = 1e-12


def sample_uniform_noise(
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw the cylindrical prior's *distribution* in ``(Re, Im)``, without trigonometry.

    The cylindrical arm draws modulus ``~U[0, 1]``, argument ``~U[0, 2*pi)``.
    Reaching that in ``(Re, Im)`` normally costs a ``cos``/``sin`` pair, which the
    spec rules out. An isotropic 2D Gaussian is rotationally symmetric, so
    ``g / ||g||`` is uniform on the unit circle; scaling by ``a ~ U[0, 1]`` gives
    modulus ``a`` and a uniform argument using only ``rand`` and ``randn``. Both
    arms therefore start from the same law with no trigonometry on this path.

    Distribution-matched, not sample-matched: unlike ``matched`` it does not
    replay the cylindrical RNG stream, so one seed gives the same law, not the
    same field.

    Args:
        batch: Number of samples.
        height: Spatial height.
        width: Spatial width.
        device: Device to allocate on.
        generator: Optional RNG for reproducibility.

    Returns:
        Euclidean noise of shape ``[B, 2, H, W]`` with modulus ``~U[0, 1]``.
    """
    amp = torch.rand(batch, 1, height, width, device=device, generator=generator)
    direction = torch.randn(batch, 2, height, width, device=device, generator=generator)
    norm = torch.linalg.vector_norm(direction, dim=1, keepdim=True).clamp_min(_DIRECTION_EPS)
    return amp * direction / norm


def sample_matched_noise(
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw the cylindrical prior and express it in ``(Re, Im)``.

    Repeats :func:`~cfm.manifolds.cylindrical.sample_cylindrical_noise`'s two
    ``torch.rand`` calls in the same order, so one seed yields the *same* complex
    noise field on both geometries, to float32 round-off (~1e-7 via the ``atan2``
    round trip). Nothing in a result can then be blamed on the prior.

    Args:
        batch: Number of samples.
        height: Spatial height.
        width: Spatial width.
        device: Device to allocate on.
        generator: Optional RNG for reproducibility.

    Returns:
        Euclidean noise of shape ``[B, 2, H, W]`` with modulus in ``[0, 1]``.
    """
    amp = torch.rand(batch, 1, height, width, device=device, generator=generator)
    phi = torch.rand(batch, 1, height, width, device=device, generator=generator) * 2 * math.pi
    return torch.cat([amp * torch.cos(phi), amp * torch.sin(phi)], dim=1)


def sample_gaussian_noise(
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw the textbook flow-matching prior, ``N(0, I)`` per channel.

    ``torch.randn`` only, and what standard flow matching does. Unlike the other
    two priors it is *not* confined to the unit disc normalised data occupies, so
    the model transports from a genuinely different support.

    Args:
        batch: Number of samples.
        height: Spatial height.
        width: Spatial width.
        device: Device to allocate on.
        generator: Optional RNG for reproducibility.

    Returns:
        Euclidean noise of shape ``[B, 2, H, W]``.
    """
    return torch.randn(batch, 2, height, width, device=device, generator=generator)


@MANIFOLDS.register("euclidean")
class EuclideanManifold(BaseManifold):
    """Complex pixels as flat 2-vectors, with no manifold structure imposed.

    Args:
        noise_prior: One of :data:`NOISE_PRIORS`. ``"uniform"`` (default) is the
            cylindrical prior's law without trigonometry; ``"gaussian"`` is the
            textbook ``N(0, I)``; ``"matched"`` replays the cylindrical RNG
            stream for a sample-paired run. See the ``sample_*`` functions above.
        loss_type: ``"l1"``, ``"l2"`` or ``"mse"`` on the velocity.
        lambda_hf: Weight on the high-frequency k-space penalty. ``0.0`` disables it.
        hf_boost_factor: Radial slope of that penalty.

    Raises:
        ValueError: If ``noise_prior`` is not one of :data:`NOISE_PRIORS`.
    """

    name = "euclidean"
    state_channels = 2
    velocity_channels = 2

    def __init__(
        self,
        noise_prior: str = "uniform",
        loss_type: str = "l1",
        lambda_hf: float = 0.0,
        hf_boost_factor: float = 4.0,
    ) -> None:
        if noise_prior not in NOISE_PRIORS:
            raise ValueError(
                f"noise_prior must be one of {sorted(NOISE_PRIORS)}, got {noise_prior!r}"
            )

        self.noise_prior = noise_prior
        self._bridge = LinearFlowBridge()
        self._loss = EuclideanVelocityLoss(
            loss_type=loss_type,
            lambda_hf=lambda_hf,
            hf_boost_factor=hf_boost_factor,
        )

    def to(self, device: torch.device) -> EuclideanManifold:
        self._loss = self._loss.to(device)
        return self

    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exponential map on Euclidean space R^2 is vector addition."""
        return x + v

    def log_map(self, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        """Logarithmic map on Euclidean space R^2 is vector difference."""
        return x_1 - x_0

    def geodesic_path(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Straight-line geodesic interpolation in Euclidean space."""
        return (1.0 - t) * x_0 + t * x_1

    def target_velocity(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor | None = None
    ) -> torch.Tensor:
        del t
        return x_1 - x_0

    def metric_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Flat Euclidean metric tensor."""
        return torch.ones(
            x.shape[0],
            self.velocity_channels,
            x.shape[2],
            x.shape[3],
            device=x.device,
            dtype=x.dtype,
        )

    def build_transform(self, crop_base: int = 16) -> Callable[[torch.Tensor], torch.Tensor]:
        # Normalise before cropping, exactly as the cylindrical pipeline does, so
        # both divide by a peak modulus taken over the same uncropped slice.
        return Compose(
            [ComplexToEuclideanTransform(), EuclideanNormalize(), CenterCropModulo(base=crop_base)]
        )

    def build_window_transforms(
        self, crop_base: int = 16
    ) -> tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
        # Same split as the cylindrical arm, and the same peak: one scalar over the
        # whole window, so both geometries hand the 2.5D model the same signal.
        return (
            Compose([ComplexToEuclideanTransform()]),
            Compose([WindowEuclideanNormalize(), CenterCropModulo(base=crop_base)]),
        )

    def sample_noise(
        self,
        batch: int,
        height: int,
        width: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if self.noise_prior == "matched":
            return sample_matched_noise(batch, height, width, device, generator)
        if self.noise_prior == "gaussian":
            return sample_gaussian_noise(batch, height, width, device, generator)
        return sample_uniform_noise(batch, height, width, device, generator)

    def bridge(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._bridge.forward(euc_noise=x_0, euc_data=x_1, t=t)

    def loss(
        self,
        pred_v: torch.Tensor,
        target_v: torch.Tensor,
        target_x1: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total, loss_vel, loss_hf = self._loss(pred_v, target_v, target_x1)
        return total, {"vel": loss_vel, "hf": loss_hf}

    def make_solver(self, num_steps: int) -> EuclideanODESolver:
        return EuclideanODESolver(num_steps=num_steps)

    def to_complex(self, state: torch.Tensor) -> torch.Tensor:
        return euclidean_to_complex(state)

    def from_complex(self, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([z.real, z.imag], dim=1)
