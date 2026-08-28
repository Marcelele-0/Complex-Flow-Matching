"""Flat-space ODE solver: the Euclidean counterpart of ``CylindricalODESolver``."""

from __future__ import annotations

import torch

from cfm.flow.solver import HeunODESolver


class EuclideanODESolver(HeunODESolver):
    """Integrates the velocity field in flat R^2 with a plain Euler update.

    ``x_{t+dt} = x_t + v(x_t, t) * dt``, on the same Heun schedule the
    cylindrical solver uses.

    The absence is the point: no amplitude clamp, no ``atan2`` -> add ->
    ``(cos, sin)`` re-projection, no modulo, no trigonometric embedding. A state
    may acquire negative modulus or leave the region a normalised complex image
    occupies, and nothing pulls it back. That repair is what the cylindrical
    formulation contributes, so adding any of it here would hand the baseline the
    advantage being measured. Do not "fix" this.
    """

    def step(self, x_t: torch.Tensor, v_t: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Performs a single Euler step in R^2. No projection, no constraint.

        Args:
            x_t (torch.Tensor): Current state [B, 2, H, W] (Re, Im)
            v_t (torch.Tensor): Predicted velocity [B, 2, H, W] (v_re, v_im)
            dt (float): Time step size

        Returns:
            torch.Tensor: Next state [B, 2, H, W]

        Raises:
            ValueError: If state and velocity channel counts disagree - a wiring
                bug that broadcasting would otherwise hide.
        """
        if x_t.shape[1] != v_t.shape[1]:
            raise ValueError(
                f"Euclidean state and velocity must share a channel count, got "
                f"x_t: {x_t.shape[1]} channels, v_t: {v_t.shape[1]} channels"
            )

        return x_t + v_t * dt
