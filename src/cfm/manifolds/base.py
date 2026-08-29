"""The interface every geometry implements, and the only thing the entry points know.

``train.py``, ``generate.py`` and ``evaluate.py`` are written against
:class:`Manifold` and nothing else. Everything that is *not* a member below -
the dataset, the U-Net trunk, the optimizer, the Heun schedule, the metrics, the
accumulator, the checkpoint plumbing - is therefore shared by construction and
cannot drift between the two arms of the comparison. That is the property the
side-by-side experiment depends on: any difference in the reported numbers has
to come from one of the seven methods here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable

import torch

from cfm.flow.solver import HeunODESolver


class Manifold(ABC):
    """A geometry's answer to the six questions the pipeline asks.

    Attributes:
        name: Identifier used in configs, log lines and ``metrics.json``.
        state_channels: Channels the U-Net consumes, i.e. the width of the
            representation. 3 on the cylinder, 2 in the plane.
        velocity_channels: Channels the U-Net emits. 2 for both geometries, so
            the architecture differs by exactly one convolution's input width.
    """

    name: str
    state_channels: int
    velocity_channels: int = 2

    def to(self, device: torch.device) -> Manifold:
        """Move any owned modules onto ``device`` and return self.

        The default is a no-op because the loss modules carry no parameters or
        buffers; it exists so entry points can call it unconditionally.

        Args:
            device: Target device.

        Returns:
            This manifold, for chaining.
        """
        return self

    @abstractmethod
    def build_transform(self, crop_base: int = 16) -> Callable[[torch.Tensor], torch.Tensor]:
        """Return the data pipeline mapping a complex slice to a model input.

        Args:
            crop_base: Spatial dimensions are cropped to a multiple of this, so
                the U-Net's pooling levels divide evenly.

        Returns:
            A callable taking ``[1, H, W]`` complex and returning
            ``[state_channels, H', W']`` real.
        """

    @abstractmethod
    def build_window_transforms(
        self, crop_base: int = 16
    ) -> tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
        """Return the 2.5D pipeline pair: per-slice first, then per-window.

        :meth:`build_transform` normalises inside the per-slice stage, which is
        wrong for a slice window: each slice would be divided by its own peak, and
        the relative brightness between neighbours - the very signal cross-slice
        attention exists to read - would be destroyed. Normalisation therefore
        moves behind the stack, and this returns the two halves separately.

        Args:
            crop_base: As :meth:`build_transform`.

        Returns:
            ``(slice_transform, window_transform)``. The first maps a ``[1, H, W]``
            complex slice to ``[state_channels, H, W]``; the second takes the
            stacked ``[S, state_channels, H, W]`` window and returns it normalised
            and cropped.
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
        """Draw ``x_0``, the prior the flow starts from at ``t=0``.

        Args:
            batch: Number of samples.
            height: Spatial height.
            width: Spatial width.
            device: Device to allocate on.
            generator: Optional RNG for reproducibility.

        Returns:
            Noise of shape ``[batch, state_channels, height, width]``.
        """

    @abstractmethod
    def bridge(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Interpolate between noise and data, and give the target velocity.

        Args:
            x_0: Noise state at ``t=0``, ``[B, state_channels, H, W]``.
            x_1: Clean data at ``t=1``, ``[B, state_channels, H, W]``.
            t: Time, ``[B, 1, 1, 1]`` in ``[0, 1]``.

        Returns:
            ``(x_t, u)``: the state at ``t`` and the target velocity
            ``[B, velocity_channels, H, W]``.
        """

    @abstractmethod
    def loss(
        self,
        pred_v: torch.Tensor,
        target_v: torch.Tensor,
        target_x1: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Score a predicted velocity against the bridge's target.

        Args:
            pred_v: Model output, ``[B, velocity_channels, H, W]``.
            target_v: Bridge target, same shape.
            target_x1: Clean data, for geometries that weight by it.

        Returns:
            ``(total, components)``. ``components`` is an ordered mapping from a
            short name to a scalar tensor; entry points log it generically as
            ``step_loss_{name}``, so a geometry's breakdown reaches W&B without
            any geometry-specific code in the training loop.
        """

    @abstractmethod
    def make_solver(self, num_steps: int) -> HeunODESolver:
        """Build the ODE solver for this geometry.

        Args:
            num_steps: Integration steps over the interval.

        Returns:
            A solver whose ``step`` respects (or deliberately ignores) this
            geometry's constraints, on the shared Heun schedule.
        """

    @abstractmethod
    def to_complex(self, state: torch.Tensor) -> torch.Tensor:
        """Map a state back to the complex domain, where all scoring happens.

        Args:
            state: ``[B, state_channels, H, W]``.

        Returns:
            ``[B, 1, H, W]`` complex.
        """

    @abstractmethod
    def from_complex(self, z: torch.Tensor) -> torch.Tensor:
        """Map a normalized complex tensor back to the manifold's representation.

        Args:
            z: ``[B, 1, H, W]`` complex.

        Returns:
            ``[B, state_channels, H, W]`` manifold state.
        """
