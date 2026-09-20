"""Flat-space ODE solver for Euclidean flow matching baseline."""

from __future__ import annotations

import torch

from cyfm.core.registry import SOLVERS
from cyfm.flow.solver import HeunODESolver


@SOLVERS.register("euclidean")
@SOLVERS.register("euclidean_heun")
@SOLVERS.register("euclidean_ode")
class EuclideanODESolver(HeunODESolver):
    """2nd-order Heun ODE solver in flat Euclidean space R^2."""

    def step(self, x_t: torch.Tensor, v_t: torch.Tensor, dt: float) -> torch.Tensor:
        """Perform a single Euler step in flat Euclidean space.

        Args:
            x_t: Current state [B, 2, H, W] (real, imag).
            v_t: Predicted velocity [B, 2, H, W] (v_re, v_im).
            dt: Time step size scalar.

        Returns:
            Next state [B, 2, H, W] in R^2.

        Raises:
            ValueError: If state and velocity channel dimensions mismatch.
        """
        if x_t.shape[1] != v_t.shape[1]:
            raise ValueError(
                f"Euclidean state and velocity must share a channel count, got "
                f"x_t: {x_t.shape[1]} channels, v_t: {v_t.shape[1]} channels"
            )

        return x_t + v_t * dt
