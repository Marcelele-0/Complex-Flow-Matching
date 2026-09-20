"""Predictor-Corrector SDE solver for the unconditional complex-diffusion baseline.

Euler-Maruyama predictor and Langevin-dynamics corrector for a variance-exploding
SDE, after Song et al. (2021).

**Unconditional, structurally.** This sampler draws from the prior alone. It takes
no measurement, no sampling mask and no sensitivity maps, and there is no
data-consistency projection anywhere in the path. That is the acceptance criterion
of the baseline, not a default: Table 5 compares unconditional generators, and an
arm that saw a measurement would be solving an easier problem and winning for that
reason. The reconstruction route this solver once carried was deleted rather than
switched off, so a later caller cannot turn it back on by passing an argument.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch

from cyfm.core.registry import SOLVERS
from cyfm.core.solver import BaseSDESolver


def heun_evaluations(num_steps: int) -> int:
    """Function evaluations Heun spends on ``num_steps`` steps.

    Two per step, less the corrector the final step skips: 1, 3, 7, 15, 199 for
    the step counts the tables report. This is the budget the diffusion arm is
    matched against, and ``HeunODESolver.evaluations`` calls it, so both sides of
    the table read the count off one function.

    Args:
        num_steps: Solver steps.

    Returns:
        Number of model evaluations.

    Raises:
        ValueError: If ``num_steps`` is not positive.
    """
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    return 2 * num_steps - 1


def plan_within_budget(evaluations: int, corrector_steps: int) -> tuple[int, int]:
    """Steps and corrections whose combined cost fits ``evaluations`` model calls.

    A step costs ``1 + corrector_steps`` calls, so at a tight budget the
    correction is what gets dropped: one predictor step is the least the sampler
    can do. Returning the correction count as well is what keeps the match exact
    at ``evaluations = 1``, where any correction would overspend.

    Integer division can leave up to ``corrector_steps`` calls unspent, so the
    diffusion arm is matched to *at most* the flow arms' budget rather than
    exactly it. That understates the arm slightly at the tightest columns; it is
    a known limitation, tracked separately, and it errs against this baseline
    rather than in its favour.

    Args:
        evaluations: Budget in model evaluations.
        corrector_steps: Langevin corrections per step requested.

    Returns:
        Tuple of steps to run and corrections per step, costing at most
        ``evaluations`` calls in total.

    Raises:
        ValueError: If ``evaluations`` is not positive or ``corrector_steps`` is
            negative.
    """
    if evaluations <= 0:
        raise ValueError(f"evaluations must be positive, got {evaluations}")
    if corrector_steps < 0:
        raise ValueError(f"corrector_steps must be non-negative, got {corrector_steps}")

    per_step = 1 + corrector_steps
    if evaluations >= per_step:
        return evaluations // per_step, corrector_steps
    # Not even one corrected step fits: keep the predictor and spend what is left
    # on corrections, which at a budget of one means none at all.
    return 1, evaluations - 1


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


@SOLVERS.register("pc_diffusion")
@SOLVERS.register("complex_diffusion")
class PredictorCorrectorSolver(BaseSDESolver):
    """Predictor-Corrector SDE solver for the unconditional complex-diffusion arm.

    Predictor: Euler-Maruyama discretization of the reverse-time SDE.
    Corrector: Annealed Langevin Dynamics with fixed SNR r (default 0.16).

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

    @property
    def evaluations(self) -> int:
        """Model calls one :meth:`sample` makes, the number the table compares.

        One predictor call per step plus one per Langevin correction. Heun spends
        ``2n-1``, so quoting ``num_steps`` for both arms would compare different
        budgets; the evaluation sweep records this instead.
        """
        return self.num_steps * (1 + self.m_steps)

    @torch.no_grad()
    def sample(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        noise: torch.Tensor,
        generator: torch.Generator | None = None,
        *,
        denoise: bool | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Integrate the reverse-time SDE from the prior at t=1 to t=eps.

        Draws from the prior alone. There is no measurement argument and no data
        consistency: see the module docstring for why that is the baseline's
        acceptance criterion rather than a default.

        Args:
            model: Neural score network (x, t) -> score.
            noise: Prior draw at t=1 [B, C, H, W].
            generator: Optional RNG, so a reported sample is reproducible.
            denoise: Whether to return the final mean rather than the noisy state.
            **kwargs: Fail-fast check for unexpected arguments.

        Returns:
            The generated state [B, C, H, W].
        """
        if kwargs:
            raise ValueError(f"Unexpected keyword arguments for sample: {kwargs}")

        should_denoise = self.denoise if denoise is None else bool(denoise)
        b = noise.shape[0]
        device = noise.device

        x = noise
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

            # Predictor step (Euler-Maruyama)
            x, x_mean = self.predictor_step(model, x, t_vec, dt, generator=generator)

        return x_mean if should_denoise else x
