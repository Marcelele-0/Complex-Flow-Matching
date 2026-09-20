"""The flat R^2 representation, shared by every arm that carries ``(Re z, Im z)``.

Two arms of the paper hold complex pixels as plain 2-vectors: the Cartesian flow
baseline and the complex-diffusion baseline. They differ in the *process* -- one
regresses a velocity along a straight bridge, the other a score along a
variance-exploding SDE -- and in nothing else.

The representation is therefore factored out here rather than written twice.
``manifolds/base.py`` states the property the side-by-side experiment rests on:
anything that is not a member of the interface is shared by construction and
cannot drift between arms. A second copy of ``build_transform`` would be exactly
such a drift risk, and a silent one: a normalisation that differed between two
arms would move every absolute number in the table without failing a test.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from cyfm.data.transforms import (
    CenterCropModulo,
    ComplexToEuclideanTransform,
    Compose,
    EuclideanNormalize,
    WindowEuclideanNormalize,
)
from cyfm.utils.complex_ops import euclidean_to_complex


class FlatComplexRepresentation:
    """Complex pixels as ``(Re z, Im z)`` on flat R^2, with no process attached.

    Mix in ahead of :class:`~cyfm.core.manifold.BaseManifold`; the subclass still
    owns ``sample_noise``, ``bridge``, ``loss`` and ``make_solver``, which are
    what actually distinguish one arm from another.
    """

    velocity_channels: int

    def exp_map(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Exponential map on Euclidean space R^2 is vector addition."""
        return x + v

    def log_map(self, x_0: torch.Tensor, x_1: torch.Tensor) -> torch.Tensor:
        """Logarithmic map on Euclidean space R^2 is vector difference."""
        return x_1 - x_0

    def metric_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Flat Euclidean metric tensor."""
        return torch.ones(
            x.shape[0],
            self.velocity_channels,
            x.shape[2],
            x.shape[3],
            device=x.device,
            dtype=x.dtype,
        )

    def build_transform(self, crop_base: int = 16) -> Callable[[torch.Tensor], torch.Tensor]:
        # Normalise before cropping, exactly as the cylindrical pipeline does, so
        # every arm divides by a peak modulus taken over the same uncropped slice.
        return Compose(
            [ComplexToEuclideanTransform(), EuclideanNormalize(), CenterCropModulo(base=crop_base)]
        )

    def build_window_transforms(
        self, crop_base: int = 16
    ) -> tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
        # Same split as the cylindrical arm, and the same peak: one scalar over the
        # whole window, so every arm hands the 2.5D model the same signal.
        return (
            Compose([ComplexToEuclideanTransform()]),
            Compose([WindowEuclideanNormalize(), CenterCropModulo(base=crop_base)]),
        )

    def to_complex(self, state: torch.Tensor) -> torch.Tensor:
        return euclidean_to_complex(state)

    def from_complex(self, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([z.real, z.imag], dim=1)
