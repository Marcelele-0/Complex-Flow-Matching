"""Base solvers for Riemannian Flow Matching, and what the sweep asks of one.

The :class:`~cyfm.core.protocols.Sampler` protocol these classes satisfy now lives
in :mod:`cyfm.core.protocols` with the other structural contracts; it is
re-exported here because that is where the solvers look for it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from cyfm.core.protocols import Sampler, VelocityField

__all__ = ["BaseODESolver", "BaseSDESolver", "Sampler"]


class BaseODESolver(ABC):
    """Abstract base class for ordinary differential equation (ODE) solvers.

    Declares every member of :class:`~cyfm.core.protocols.Sampler`, so an ODE
    solver satisfies that protocol by virtue of this class rather than by each
    subclass happening to add the missing pieces. It did not before: ``sample``
    omitted ``generator`` and nothing here declared ``evaluations``, so a new
    subclass could be accepted by the type checker and then fail at the call site
    in the evaluation sweep.
    """

    def __init__(self, num_steps: int = 50) -> None:
        self.num_steps = num_steps

    @property
    @abstractmethod
    def evaluations(self) -> int:
        """Model calls one :meth:`sample` makes.

        Abstract rather than defaulting to Heun's ``2n-1``: the reported cost of a
        few-step method is the number this returns, and a default would let a
        solver with a different budget report someone else's.
        """

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
        model: VelocityField,
        noise: torch.Tensor,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Integrate trajectory from t=0 (noise) to t=1 (data).

        Args:
            model: Velocity field (x, t) -> v.
            noise: Initial state at t=0 [B, C, H, W].
            generator: Optional RNG. Declared for every sampler so the caller can
                seed one without knowing which it holds; a deterministic
                integrator ignores it.

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

    @property
    @abstractmethod
    def evaluations(self) -> int:
        """Model calls one :meth:`sample` makes; see :class:`BaseODESolver`."""

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
        model: VelocityField,
        noise: torch.Tensor,
        generator: torch.Generator | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Integrate SDE trajectory from t=0 to t=1.

        Args:
            model: The field the predictor calls; a score, for this baseline.
            noise: Prior draw [B, C, H, W].
            generator: Optional RNG, so a reported sample is reproducible.
            **kwargs: Implementation-specific options.
        """
