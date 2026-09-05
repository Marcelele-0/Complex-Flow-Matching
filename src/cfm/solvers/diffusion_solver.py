"""Predictor-Corrector SDE solver for Complex Diffusion in MRI reconstruction.

Implements Algorithm 2 (single-coil Fourier CS) and Algorithm 3 (multi-coil
parallel imaging) with Euler-Maruyama predictor, Langevin dynamics corrector,
and data consistency projections.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch

from cfm.core.registry import SOLVERS
from cfm.core.solver import BaseSDESolver
from cfm.utils.fft import fft2c, ifft2c


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


def apply_data_consistency(
    x: torch.Tensor,
    masked_kspace: torch.Tensor,
    mask: torch.Tensor,
    sensitivity_maps: torch.Tensor | None = None,
) -> torch.Tensor:
    """Project image-domain state onto k-space data fidelity constraint.

    Supports Algorithm 2 (single-coil) and Algorithm 3 (multi-coil parallel imaging).

    Args:
        x: Current state, either 2-channel real [B, 2, H, W] or complex [B, 1, H, W].
        masked_kspace: Undersampled k-space [B, 1, H, W] or [B, coils, H, W].
        mask: Binary sampling mask [B, 1, H, W] or broadcastable.
        sensitivity_maps: Optional coil sensitivities [B, coils, H, W] for Algorithm 3.

    Returns:
        Data-consistent state matching the format of x.
    """
    is_two_channel = x.shape[1] == 2 and not torch.is_complex(x)
    if is_two_channel:
        z = torch.complex(x[:, 0:1], x[:, 1:2])
    else:
        z = x

    if sensitivity_maps is not None:
        # Algorithm 3: Multi-coil parallel imaging data consistency
        k_coils = fft2c(sensitivity_maps * z)
        k_dc = mask * masked_kspace + (1.0 - mask) * k_coils
        z_dc = (sensitivity_maps.conj() * ifft2c(k_dc)).sum(dim=1, keepdim=True)
    else:
        # Algorithm 2: Single-coil data consistency
        k_pred = fft2c(z)
        k_dc = mask * masked_kspace + (1.0 - mask) * k_pred
        z_dc = ifft2c(k_dc)

    if is_two_channel:
        return torch.cat([z_dc.real, z_dc.imag], dim=1)
    return z_dc


@SOLVERS.register("pc_diffusion")
@SOLVERS.register("complex_diffusion")
class PredictorCorrectorSolver(BaseSDESolver):
    """Predictor-Corrector SDE solver for complex diffusion with data consistency.

    Predictor: Euler-Maruyama discretization of the reverse-time SDE.
    Corrector: Annealed Langevin Dynamics with fixed SNR r (default 0.16).
    Data Consistency: k-space data projection (Algorithms 2 and 3).

    Args:
        num_steps: Number of integration time steps from t=1 to t=eps.
        sigma_min: Smallest noise standard deviation at t=0.
        sigma_max: Largest noise standard deviation at t=1.
        snr: Signal-to-noise ratio parameter r for Langevin Dynamics.
        m_steps: Number of Langevin corrector steps per predictor step (M).
        probability_flow: Whether to use the probability flow ODE instead of SDE.
        denoise: Whether to apply final one-step denoising using x_mean.
        eps: Small positive scalar ending time for reverse integration.
        manifold: Optional reference to associated manifold geometry.
        **kwargs: Unrecognized arguments; triggers fail-fast error.

    Raises:
        ValueError: If any arguments are negative or out of bounds, or unexpected kwargs passed.
    """

    def __init__(
        self,
        num_steps: int = 50,
        sigma_min: float = 0.01,
        sigma_max: float = 378.0,
        snr: float = 0.16,
        m_steps: int = 1,
        probability_flow: bool = False,
        denoise: bool = True,
        eps: float = 1e-5,
        manifold: Any = None,
        **kwargs: Any,
    ) -> None:
        if kwargs:
            raise ValueError(f"Unexpected keyword arguments for PredictorCorrectorSolver: {kwargs}")
        if sigma_min <= 0:
            raise ValueError(f"sigma_min must be positive, got {sigma_min}")
        if sigma_max <= sigma_min:
            raise ValueError(
                f"sigma_max must be greater than sigma_min, got {sigma_max} <= {sigma_min}"
            )
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if snr < 0:
            raise ValueError(f"snr must be non-negative, got {snr}")
        if m_steps < 0:
            raise ValueError(f"m_steps must be non-negative, got {m_steps}")
        if eps <= 0 or eps >= 1.0:
            raise ValueError(f"eps must be in (0, 1), got {eps}")

        super().__init__(num_steps=num_steps, sigma_min=sigma_min, sigma_max=sigma_max)
        self.snr = float(snr)
        self.m_steps = int(m_steps)
        self.probability_flow = bool(probability_flow)
        self.denoise = bool(denoise)
        self.eps = float(eps)
        self.manifold = manifold
        self.log_ratio = math.log(self.sigma_max / self.sigma_min)

    def sigma(self, t: float | torch.Tensor) -> torch.Tensor:
        """Compute noise scale sigma(t) = sigma_min * (sigma_max / sigma_min)^t.

        Args:
            t: Time scalar or tensor in [0, 1].

        Returns:
            Noise standard deviation tensor.
        """
        if isinstance(t, int | float):
            val = self.sigma_min * (self.sigma_max / self.sigma_min) ** float(t)
            return torch.tensor(val, dtype=torch.float32)
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def diffusion(self, t: float | torch.Tensor) -> torch.Tensor:
        """Compute VE-SDE diffusion coefficient g(t).

        Args:
            t: Time scalar or tensor.

        Returns:
            Diffusion coefficient g(t) = sigma(t) * sqrt(2 * log(sigma_max / sigma_min)).
        """
        sig = self.sigma(t)
        return sig * math.sqrt(2.0 * self.log_ratio)

    def corrector_step(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        t: torch.Tensor,
        snr: float | None = None,
        m_steps: int | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform Langevin Dynamics corrector updates.

        Args:
            model: Score-based model (x, t) -> score.
            x: Current state [B, C, H, W].
            t: Time tensor [B].
            snr: Signal-to-noise ratio parameter r. Defaults to self.snr.
            m_steps: Number of Langevin steps. Defaults to self.m_steps.
            generator: Optional RNG generator.

        Returns:
            Tuple of (x, x_mean).
        """
        target_snr = self.snr if snr is None else float(snr)
        steps = self.m_steps if m_steps is None else int(m_steps)
        x_mean = x

        for _ in range(steps):
            grad = model(x, 1.0 - t)
            noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            if grad_norm < 1e-7:
                step_size = 0.0
                x_mean = x
            else:
                step_size = 2.0 * ((target_snr * noise_norm) / grad_norm) ** 2
                x_mean = x + step_size * grad
                x = x_mean + math.sqrt(2.0 * step_size) * noise

        return x, x_mean

    def predictor_step(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x: torch.Tensor,
        t: torch.Tensor,
        dt: float,
        probability_flow: bool | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform Euler-Maruyama predictor update on the reverse-time SDE.

        Args:
            model: Score-based model (x, t) -> score.
            x: Current state [B, C, H, W].
            t: Time tensor [B].
            dt: Positive time step backward in time.
            probability_flow: Whether to suppress stochastic increment.
            generator: Optional RNG generator.

        Returns:
            Tuple of (x, x_mean).
        """
        score = model(x, 1.0 - t)
        g = _match_shape(self.diffusion(t), x)
        g2 = g**2
        is_pf = self.probability_flow if probability_flow is None else bool(probability_flow)
        mult = 0.5 if is_pf else 1.0

        drift = g2 * score * mult
        x_mean = x + drift * dt

        if not is_pf and dt > 0:
            noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
            x = x_mean + torch.sqrt(g2 * dt) * noise
        else:
            x = x_mean

        return x, x_mean

    def step(
        self,
        x_t: torch.Tensor,
        drift: torch.Tensor,
        dt: float,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Advance state x_t under drift and diffusion with Brownian increment.

        Args:
            x_t: Current state [B, C, H, W].
            drift: Drift vector field [B, V, H, W].
            dt: Time step size.
            noise: Optional Brownian motion noise.

        Returns:
            Advanced state [B, C, H, W].
        """
        x_next = x_t + drift * dt
        if noise is not None:
            x_next = x_next + noise * math.sqrt(abs(dt))
        return x_next

    def data_consistency(
        self,
        x: torch.Tensor,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        sensitivity_maps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply k-space data fidelity projection."""
        return apply_data_consistency(
            x, masked_kspace=masked_kspace, mask=mask, sensitivity_maps=sensitivity_maps
        )

    @torch.no_grad()
    def sample(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        noise: torch.Tensor,
        *,
        masked_kspace: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        sensitivity_maps: torch.Tensor | None = None,
        data_consistency_fn: Callable[[torch.Tensor], torch.Tensor] | None = None,
        denoise: bool | None = None,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Integrate reverse-time SDE trajectory from t=1 to t=eps with data consistency.

        Args:
            model: Neural score network (x, t) -> score.
            noise: Initial noise sample at t=1 [B, C, H, W].
            masked_kspace: Optional undersampled k-space measurements.
            mask: Optional sampling mask.
            sensitivity_maps: Optional coil sensitivities for multi-coil imaging.
            data_consistency_fn: Optional custom data consistency projection callback.
            denoise: Whether to apply final one-step denoising.
            generator: Optional RNG generator.
            **kwargs: Fail-fast check for unexpected arguments.

        Returns:
            Reconstructed state [B, C, H, W].
        """
        if kwargs:
            raise ValueError(f"Unexpected keyword arguments for sample: {kwargs}")

        should_denoise = self.denoise if denoise is None else bool(denoise)
        b = noise.shape[0]
        device = noise.device

        def apply_dc(curr_x: torch.Tensor) -> torch.Tensor:
            if data_consistency_fn is not None:
                return data_consistency_fn(curr_x)
            if masked_kspace is not None and mask is not None:
                return self.data_consistency(
                    curr_x,
                    masked_kspace=masked_kspace,
                    mask=mask,
                    sensitivity_maps=sensitivity_maps,
                )
            return curr_x

        # Initial state at t=1 projected to data consistency
        x = apply_dc(noise)
        x_mean = x

        timesteps = torch.linspace(1.0, self.eps, self.num_steps, device=device)

        for i in range(self.num_steps):
            t_val = timesteps[i]
            t_vec = torch.full((b,), float(t_val), device=device, dtype=torch.float32)
            dt = (
                float(timesteps[i] - timesteps[i + 1])
                if i < self.num_steps - 1
                else float(self.eps)
            )

            # Corrector step (Langevin dynamics)
            if self.m_steps > 0:
                x, x_mean = self.corrector_step(model, x, t_vec, generator=generator)
                x = apply_dc(x)
                x_mean = apply_dc(x_mean)

            # Predictor step (Euler-Maruyama)
            x, x_mean = self.predictor_step(model, x, t_vec, dt, generator=generator)
            x = apply_dc(x)
            x_mean = apply_dc(x_mean)

        return x_mean if should_denoise else x
