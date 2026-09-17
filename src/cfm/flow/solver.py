"""ODE numerical solvers for probability trajectory integration."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable

import torch

from cfm.core.registry import SOLVERS
from cfm.core.solver import BaseODESolver


class HeunODESolver(BaseODESolver):
    """2nd-order predictor-corrector Heun ODE integrator."""

    def __init__(self, num_steps: int = 50) -> None:
        super().__init__(num_steps=num_steps)

    @abstractmethod
    def step(self, x_t: torch.Tensor, v_t: torch.Tensor, dt: float) -> torch.Tensor:
        """Advance state x_t by dt under velocity field v_t.

        Args:
            x_t: State tensor [B, C, H, W].
            v_t: Velocity tensor [B, 2, H, W].
            dt: Time step scalar.

        Returns:
            Advanced state tensor [B, C, H, W].
        """

    @property
    def evaluations(self) -> int:
        """Two calls per step, less the corrector the final step skips.

        Delegates so the count lives in one place: the diffusion arm matches its
        budget against :func:`~cfm.flow.diffusion_solver.heun_evaluations`, and a
        second copy of the rule here is how that match goes quietly wrong.
        """
        from cfm.flow.diffusion_solver import heun_evaluations

        return heun_evaluations(self.num_steps)

    @torch.no_grad()
    def sample(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        noise: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Integrate trajectory from t=0 to t=1 using Heun's method.

        Args:
            model: Neural velocity field callable (x, t) -> v.
            noise: Initial noise state at t=0 [B, C, H, W].
            generator: Unused; this integrator is deterministic given ``noise``.
                Accepted so every sampler presents one interface.

        Returns:
            Reconstructed state at t=1 [B, C, H, W].
        """
        del generator
        device = noise.device
        b = noise.shape[0]
        x_t = noise
        dt = 1.0 / self.num_steps

        for i in range(self.num_steps):
            t_val = i / self.num_steps
            t_next_val = (i + 1) / self.num_steps

            t_tensor = torch.full((b,), t_val, device=device, dtype=torch.float32)
            t_next_tensor = torch.full((b,), t_next_val, device=device, dtype=torch.float32)

            # Predictor step
            v_t = model(x_t, t_tensor)

            if i == self.num_steps - 1:
                x_t = self.step(x_t, v_t, dt)
                break

            x_pred = self.step(x_t, v_t, dt)
            v_next = model(x_pred, t_next_tensor)
            v_avg = 0.5 * (v_t + v_next)

            # Corrector step
            x_t = self.step(x_t, v_avg, dt)

        return x_t


@SOLVERS.register("cylindrical")
@SOLVERS.register("cylindrical_heun")
@SOLVERS.register("cylindrical_ode")
class CylindricalODESolver(HeunODESolver):
    """2nd-order Heun ODE solver on decoupled cylindrical manifold R+ x S^1."""

    def step(self, x_t: torch.Tensor, v_t: torch.Tensor, dt: float) -> torch.Tensor:
        """Advance cylindrical state by dt with manifold projection.

        Args:
            x_t: State tensor [B, 3, H, W] (m, cos(phi), sin(phi)).
            v_t: Tangent velocity [B, 2, H, W] (v_m, v_phi).
            dt: Time step scalar.

        Returns:
            Next state [B, 3, H, W] projected onto R+ x S^1.
        """
        m_t = x_t[:, 0:1, :, :]
        px_t = x_t[:, 1:2, :, :]
        py_t = x_t[:, 2:3, :, :]

        v_m = v_t[:, 0:1, :, :]
        v_phi = v_t[:, 1:2, :, :]

        # Amplitude clamp >= 0
        m_next = torch.clamp(m_t + v_m * dt, min=0.0)

        # Phase update and S^1 reprojection
        phi_t = torch.atan2(py_t, px_t)
        phi_next = phi_t + v_phi * dt

        px_next = torch.cos(phi_next)
        py_next = torch.sin(phi_next)

        return torch.cat([m_next, px_next, py_next], dim=1)
