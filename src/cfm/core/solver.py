"""Base ODE and SDE solvers for Riemannian Flow Matching and Diffusion."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import torch


class BaseODESolver(ABC):
    """Abstract base class for ordinary differential equation (ODE) solvers."""

    def __init__(self, num_steps: int = 50) -> None:
        self.num_steps = num_steps

    @abstractmethod
    def step(self, x_t: torch.Tensor, v_t: torch.Tensor, dt: float) -> torch.Tensor:
        """Advance the manifold state x_t by dt under velocity field v_t.

        Args:
            x_t: Current state [B, C, H, W].
            v_t: Tangent velocity [B, V, H, W].
            dt: Time step size.

        Returns:
            Updated state [B, C, H, W] on the manifold.
        """

    @abstractmethod
    def sample(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Integrate trajectory from t=0 (noise) to t=1 (data).

        Args:
            model: Neural network parameterizing velocity field (x, t) -> v.
            noise: Initial state at t=0 [B, C, H, W].

        Returns:
            Reconstructed state at t=1 [B, C, H, W].
        """


class BaseSDESolver(ABC):
    """Abstract base class for stochastic differential equation (SDE) solvers.

    Handles drift v(x_t, t) and diffusion g(t) for Langevin/Score/Diffusion dynamics.
    """

    def __init__(
        self,
        num_steps: int = 50,
        sigma_min: float = 1e-4,
        sigma_max: float = 1.0,
    ) -> None:
        self.num_steps = num_steps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    @abstractmethod
    def diffusion(self, t: float | torch.Tensor) -> torch.Tensor:
        """Compute diffusion coefficient g(t)."""

    @abstractmethod
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
            noise: Optional Gaussian noise sample for Brownian motion.

        Returns:
            Next state [B, C, H, W].
        """

    @abstractmethod
    def sample(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        noise: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Integrate SDE trajectory from t=0 to t=1."""
