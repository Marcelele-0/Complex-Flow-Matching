"""Base ODE solver for Riemannian Flow Matching."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

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
