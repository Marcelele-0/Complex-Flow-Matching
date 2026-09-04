"""Base manifold definition for Riemannian / Lie-group Complex Flow Matching."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch

from cfm.core.solver import BaseODESolver, BaseSDESolver


class BaseManifold(ABC):
    """Abstract base class defining Riemannian manifold geometry for flow matching.

    Attributes:
        name: Unique string identifier of the geometry.
        state_channels: Channels consumed by the state representation
            (e.g. 3 on S^1 x R+, 2 on R^2).
        velocity_channels: Channels emitted as tangent vectors (2 for MRI velocity fields).
    """

    name: str
    state_channels: int
    velocity_channels: int = 2

    def to(self, device: torch.device) -> BaseManifold:
        """Move any internal modules / buffers to device and return self."""
        return self

    @abstractmethod
    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Riemannian exponential map: exp_x(v) advancing point x along tangent vector v.

        Args:
            x: Base point on manifold [B, state_channels, H, W].
            v: Tangent vector [B, velocity_channels, H, W].

        Returns:
            End point on manifold [B, state_channels, H, W].
        """

    @abstractmethod
    def log_map(self, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        """Riemannian logarithmic map: log_{x_0}(x_1) returning initial tangent velocity.

        Args:
            x_0: Starting point [B, state_channels, H, W].
            x_1: Target point [B, state_channels, H, W].

        Returns:
            Tangent vector at x_0 [B, velocity_channels, H, W].
        """

    @abstractmethod
    def geodesic_path(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Geodesic interpolation curve gamma(t) from x_0 at t=0 to x_1 at t=1.

        Args:
            x_0: Noise state [B, state_channels, H, W].
            x_1: Clean data state [B, state_channels, H, W].
            t: Time tensor [B, 1, 1, 1] in [0, 1].

        Returns:
            Interpolated state [B, state_channels, H, W].
        """

    @abstractmethod
    def target_velocity(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Target tangent velocity field u_t(x_t | x_0, x_1).

        Args:
            x_0: Noise state [B, state_channels, H, W].
            x_1: Clean data state [B, state_channels, H, W].
            t: Optional time tensor [B, 1, 1, 1].

        Returns:
            Target velocity field [B, velocity_channels, H, W].
        """

    @abstractmethod
    def metric_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Riemannian metric tensor g(x) at point x.

        Args:
            x: State tensor [B, state_channels, H, W].

        Returns:
            Metric tensor representation [B, velocity_channels, H, W].
        """

    @abstractmethod
    def sample_noise(
        self,
        batch: int,
        height: int,
        width: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Sample prior distribution x_0 at t=0.

        Args:
            batch: Batch size.
            height: Image height.
            width: Image width.
            device: Device to place tensor on.
            generator: Optional RNG generator.

        Returns:
            Noise tensor [B, state_channels, H, W].
        """

    def bridge(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Interpolate between noise x_0 and data x_1, returning state x_t and velocity u_t.

        Default implementation computes geodesic_path and target_velocity.

        Args:
            x_0: Prior noise state [B, state_channels, H, W].
            x_1: Clean data state [B, state_channels, H, W].
            t: Time tensor [B, 1, 1, 1] in [0, 1].

        Returns:
            Tuple (x_t, u_t).
        """
        x_t = self.geodesic_path(x_0, x_1, t)
        u_t = self.target_velocity(x_0, x_1, t)
        return x_t, u_t

    @abstractmethod
    def loss(
        self,
        pred_v: torch.Tensor,
        target_v: torch.Tensor,
        target_x1: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute matching loss between predicted and target velocity fields.

        Args:
            pred_v: Model output [B, velocity_channels, H, W].
            target_v: Bridge target [B, velocity_channels, H, W].
            target_x1: Clean data state [B, state_channels, H, W].

        Returns:
            Tuple (total_loss, components_dict).
        """

    @abstractmethod
    def make_solver(self, num_steps: int) -> BaseODESolver | BaseSDESolver:
        """Construct the ODE or SDE solver for this manifold geometry."""

    @abstractmethod
    def to_complex(self, state: torch.Tensor) -> torch.Tensor:
        """Map manifold state back to complex tensor [B, 1, H, W]."""

    @abstractmethod
    def from_complex(self, z: torch.Tensor) -> torch.Tensor:
        """Map complex tensor [B, 1, H, W] to manifold state [B, state_channels, H, W]."""

    @abstractmethod
    def build_transform(self, crop_base: int = 16) -> Callable[[torch.Tensor], torch.Tensor]:
        """Build data preprocessing transform for single slices."""

    @abstractmethod
    def build_window_transforms(
        self, crop_base: int = 16
    ) -> tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
        """Build 2.5D slice and window preprocessing transforms."""
