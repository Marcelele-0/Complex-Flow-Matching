"""Complex Diffusion Manifold using Variance-Exploding SDE (VE-SDE).

Models complex MRI in Cartesian R^2 with a continuous VE-SDE forward process,
denoising score matching with likelihood weighting, and Predictor-Corrector sampling.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch

from cfm.core.manifold import BaseManifold
from cfm.core.registry import MANIFOLDS
from cfm.data.transforms import (
    CenterCropOrPad,
    ComplexToEuclideanTransform,
    Compose,
    EuclideanNormalize,
    WindowEuclideanNormalize,
)
from cfm.solvers.diffusion_solver import PredictorCorrectorSolver
from cfm.utils.complex_ops import euclidean_to_complex


def _match_shape(val: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Broadcast scalar or 1D tensor to match target tensor dimensions."""
    val = val.to(target.device)
    if val.dim() == target.dim():
        return val
    if val.dim() == 0:
        return val
    if val.dim() == 1 and val.shape[0] == target.shape[0]:
        return val.view(val.shape[0], *([1] * (target.dim() - 1)))
    return val


@MANIFOLDS.register("complex_diffusion")
class ComplexDiffusionManifold(BaseManifold):
    """Complex Diffusion on R^2 via Variance-Exploding SDE (VE-SDE).

    Forward process: x(t) = x(0) + sigma(t) * z, where z ~ N(0, I).
    Noise schedule: sigma(t) = sigma_min * (sigma_max / sigma_min)^t.
    Loss: Denoising score matching with likelihood weighting g(t)^2.

    Args:
        sigma_min: Smallest noise scale at t=0. Default 0.01.
        sigma_max: Largest noise scale at t=1. Default 378.0.
        eps: Smallest positive time threshold. Default 1e-5.
        likelihood_weighting: Whether to use likelihood weighting g(t)^2. Default True.
        snr: Signal-to-noise ratio parameter r for Langevin Dynamics. Default 0.16.
        corrector_steps: Number of Langevin corrector steps M. Default 1.
        num_steps: Default number of integration steps. Default 50.
        **kwargs: Unrecognized arguments; triggers fail-fast error.

    Raises:
        ValueError: If parameters are invalid or unexpected kwargs passed.
    """

    name = "complex_diffusion"
    state_channels = 2
    velocity_channels = 2

    def __init__(
        self,
        sigma_min: float = 0.01,
        sigma_max: float = 378.0,
        eps: float = 1e-5,
        likelihood_weighting: bool = True,
        snr: float = 0.16,
        corrector_steps: int = 1,
        num_steps: int = 50,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            raise ValueError(f"Unexpected keyword arguments for ComplexDiffusionManifold: {kwargs}")
        if sigma_min <= 0:
            raise ValueError(f"sigma_min must be positive, got {sigma_min}")
        if sigma_max <= sigma_min:
            raise ValueError(
                f"sigma_max must be greater than sigma_min, got {sigma_max} <= {sigma_min}"
            )
        if eps <= 0 or eps >= 1.0:
            raise ValueError(f"eps must be in (0, 1), got {eps}")
        if snr < 0:
            raise ValueError(f"snr must be non-negative, got {snr}")
        if corrector_steps < 0:
            raise ValueError(f"corrector_steps must be non-negative, got {corrector_steps}")
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")

        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.eps = float(eps)
        self.likelihood_weighting = bool(likelihood_weighting)
        self.snr = float(snr)
        self.corrector_steps = int(corrector_steps)
        self.num_steps = int(num_steps)
        self.log_ratio = math.log(self.sigma_max / self.sigma_min)

        self._last_t: torch.Tensor | None = None
        self._last_z: torch.Tensor | None = None

    def to(self, device: torch.device) -> ComplexDiffusionManifold:
        """Move internal buffers to device (chainable)."""
        del device
        return self

    def sigma(self, t: float | torch.Tensor) -> torch.Tensor:
        """Compute noise standard deviation sigma(t) = sigma_min * (sigma_max / sigma_min)^t.

        Args:
            t: Time scalar or tensor in [0, 1].

        Returns:
            Noise scale sigma(t).
        """
        if isinstance(t, (int, float)):
            val = self.sigma_min * (self.sigma_max / self.sigma_min) ** float(t)
            return torch.tensor(val, dtype=torch.float32)
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def diffusion(self, t: float | torch.Tensor) -> torch.Tensor:
        """Compute diffusion coefficient g(t) = sigma(t) * sqrt(2 * log(sigma_max / sigma_min)).

        Args:
            t: Time scalar or tensor.

        Returns:
            Diffusion coefficient g(t).
        """
        sig = self.sigma(t)
        return sig * math.sqrt(2.0 * self.log_ratio)

    def marginal_prob(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and standard deviation of transition kernel p_t(x_t | x_0).

        Args:
            x_0: Clean state [B, C, H, W].
            t: Time tensor [B] or [B, 1, 1, 1].

        Returns:
            Tuple of (mean, std).
        """
        mean = x_0
        std = _match_shape(self.sigma(t), x_0)
        return mean, std

    def forward_process(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward VE-SDE perturbation: x(t) = x(0) + sigma(t) * z.

        Args:
            x_0: Clean state [B, 2, H, W].
            t: Time tensor [B] or [B, 1, 1, 1] in [0, 1].
            z: Optional standard Gaussian noise [B, 2, H, W].
            generator: Optional RNG generator.

        Returns:
            Tuple of (x_t, z).
        """
        if z is None:
            z = torch.randn(x_0.shape, device=x_0.device, dtype=x_0.dtype, generator=generator)
        std = _match_shape(self.sigma(t), x_0)
        x_t = x_0 + std * z
        return x_t, z

    # Aliases for convenience
    forward_sde = forward_process
    perturb = forward_process

    def target_score(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute target conditional score grad_{x_t} log p_t(x_t | x_0) = -z / sigma(t).

        Args:
            z: Standard normal noise tensor [B, 2, H, W].
            t: Time tensor [B] or [B, 1, 1, 1].

        Returns:
            Target score tensor [B, 2, H, W].
        """
        std = _match_shape(self.sigma(t), z)
        return -z / std

    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exponential map on flat R^2 is vector addition."""
        return x + v

    def log_map(self, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        """Logarithmic map on flat R^2 is vector difference."""
        return x_1 - x_0

    def geodesic_path(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Interpolate state along VE-SDE forward path."""
        x_t, _ = self.bridge(x_0, x_1, t)
        return x_t

    def target_velocity(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute target velocity / score field."""
        if t is None:
            t = torch.ones(x_0.shape[0], 1, 1, 1, device=x_0.device, dtype=x_0.dtype)
        _, u_t = self.bridge(x_0, x_1, t)
        return u_t

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

    def sample_noise(
        self,
        batch: int,
        height: int,
        width: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw sample from prior distribution at t=1: N(0, sigma_max^2 * I).

        Args:
            batch: Batch size.
            height: Image height.
            width: Image width.
            device: Allocation device.
            generator: Optional RNG generator.

        Returns:
            Noise tensor [B, 2, H, W].
        """
        return (
            torch.randn(
                batch,
                self.state_channels,
                height,
                width,
                device=device,
                generator=generator,
            )
            * self.sigma_max
        )

    def bridge(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute perturbed state x_t and target score from prior noise x_0 and data x_1.

        Args:
            x_0: Prior noise state [B, 2, H, W].
            x_1: Clean data state [B, 2, H, W].
            t: Time tensor [B, 1, 1, 1] in [0, 1].

        Returns:
            Tuple of (x_t, target_score).
        """
        # CFM convention: x_0 is noise scaled by sigma_max.
        z = x_0 / self.sigma_max

        # CFM convention: t=0 is noise, t=1 is data.
        # VE-SDE convention: t=0 is data, t=1 is noise.
        t_internal = 1.0 - t

        std = _match_shape(self.sigma(t_internal), x_1)
        x_t = x_1 + std * z
        u_t = -z / std

        self._last_t = t_internal
        self._last_z = z
        return x_t, u_t

    def loss(
        self,
        pred_v: torch.Tensor,
        target_v: torch.Tensor,
        target_x1: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        likelihood_weighting: bool | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute Denoising Score Matching loss with likelihood weighting.

        Loss = E [ g(t)^2 * || pred_score - target_score ||^2 ].

        Args:
            pred_v: Predicted score field [B, 2, H, W].
            target_v: Target score field [B, 2, H, W].
            target_x1: Optional clean data state [B, 2, H, W].
            t: Optional time tensor [B] or [B, 1, 1, 1].
            likelihood_weighting: Override for likelihood weighting flag.

        Returns:
            Tuple of (total_loss, {"dsm": total_loss}).
        """
        del target_x1
        time_t = (1.0 - t) if t is not None else self._last_t
        use_lw = (
            self.likelihood_weighting
            if likelihood_weighting is None
            else bool(likelihood_weighting)
        )

        diff = pred_v - target_v

        if time_t is not None:
            time_tensor = _match_shape(time_t, pred_v)
            std = _match_shape(self.sigma(time_tensor), pred_v)
            if use_lw:
                g2 = 2.0 * self.log_ratio * (std**2)
                err = g2 * (diff**2)
            else:
                err = (std**2) * (diff**2)
        else:
            # Unweighted fallback if t is unspecified
            err = diff**2

        total_loss = torch.mean(err)
        return total_loss, {"dsm": total_loss}

    def score_matching_loss(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_0: torch.Tensor,
        t: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        eps: float | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Convenience end-to-end forward perturbation and DSM loss computation.

        Args:
            model: Neural score network (x, t) -> score.
            x_0: Clean data state [B, 2, H, W].
            t: Optional time tensor [B].
            z: Optional noise tensor [B, 2, H, W].
            eps: Smallest time step. Defaults to self.eps.
            generator: Optional RNG generator.

        Returns:
            Tuple of (total_loss, {"dsm": total_loss}).
        """
        b = x_0.shape[0]
        min_eps = self.eps if eps is None else float(eps)
        if t is None:
            t = (
                torch.rand(b, device=x_0.device, dtype=x_0.dtype, generator=generator)
                * (1.0 - min_eps)
                + min_eps
            )
        t_model = t if t.dim() == 1 else t.view(b)
        x_t, noise_z = self.forward_process(x_0, t, z=z, generator=generator)
        pred_score = model(x_t, t_model)
        target = self.target_score(noise_z, t)
        return self.loss(pred_score, target, target_x1=x_0, t=t)

    def make_solver(self, num_steps: int) -> PredictorCorrectorSolver:
        """Construct the default Predictor-Corrector solver for this manifold."""
        return PredictorCorrectorSolver(
            num_steps=num_steps,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            snr=self.snr,
            m_steps=self.corrector_steps,
            eps=self.eps,
            manifold=self,
        )

    def to_complex(self, state: torch.Tensor) -> torch.Tensor:
        """Map 2-channel real state [B, 2, H, W] to complex tensor [B, 1, H, W]."""
        return euclidean_to_complex(state)

    def from_complex(self, z: torch.Tensor) -> torch.Tensor:
        """Map complex tensor [B, 1, H, W] to 2-channel real state [B, 2, H, W]."""
        return torch.cat([z.real, z.imag], dim=1)

    def build_transform(self, crop_base: int = 16) -> Callable[[torch.Tensor], torch.Tensor]:
        """Build data transform pipeline for single slices."""
        return Compose([ComplexToEuclideanTransform(), EuclideanNormalize(), CenterCropOrPad(320)])

    def build_window_transforms(
        self, crop_base: int = 16
    ) -> tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
        """Build data transform pipeline for 2.5D multi-slice windows."""
        return (
            Compose([ComplexToEuclideanTransform()]),
            Compose([WindowEuclideanNormalize(), CenterCropOrPad(320)]),
        )
