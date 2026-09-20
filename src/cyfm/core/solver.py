"""Base solvers for Riemannian Flow Matching, and what the sweep asks of one."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

import torch


@runtime_checkable
class Sampler(Protocol):
    """All the evaluation sweep requires: a prior in, a sample out.

    Deliberately weaker than :class:`BaseODESolver`. A flow arm integrates a
    velocity with a fixed ``dt`` and so has a meaningful ``step``; the diffusion
    baseline's predictor moves between noise levels and has no velocity and no
    ``dt`` to take. Typing the manifold's factory to this protocol lets both be
    returned without pretending the second is an ODE solver.
    """

    num_steps: int

    @property
    def evaluations(self) -> int:
        """Model calls one :meth:`sample` makes.

        Required, not optional. Probing for it and falling back to Heun's count
        would report a wrong cost for any sampler that forgot to declare one,
        silently, which is the mislabelling the reported figure exists to stop.
        """
        ...

    def sample(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        noise: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Integrate from the prior at t=0 to data at t=1.

        ``generator`` is accepted by every sampler so the caller can seed one
        without knowing which it holds; a deterministic solver ignores it.
        """
        ...


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

    Separate from :class:`BaseODESolver` rather than a subclass of it: an SDE step
    consumes a Brownian increment and a noise level, where an ODE step consumes a
    velocity and a ``dt``. Both satisfy :class:`Sampler`, which is all the
    evaluation sweep asks of either.
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
