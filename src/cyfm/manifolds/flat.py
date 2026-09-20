"""The flat R^2 representation, shared by every arm that carries ``(Re z, Im z)``.

Two arms of the paper hold complex pixels as plain 2-vectors: the Cartesian flow
baseline and the complex-diffusion baseline. They differ in the *process* -- one
regresses a velocity along a straight bridge, the other a score along a
variance-exploding SDE -- and in nothing else.

The representation is therefore factored out here rather than written twice.
:class:`~cyfm.core.manifold.BaseManifold` states the property the side-by-side
experiment rests on: anything that is not a member of the interface is shared by
construction and cannot drift between arms.

The preprocessing pipeline used to be built here, and that was the same drift
risk one level down -- a normalisation differing between two arms would move
every absolute number in the table without failing a test. It is now declared
(``representation``) and composed once in
:func:`cyfm.data.transforms.slice_transform`, so there is no second copy to
disagree with.
"""

from __future__ import annotations

import torch

from cyfm.core.manifold import Representation
from cyfm.utils.complex_ops import euclidean_to_complex


class FlatComplexRepresentation:
    """Complex pixels as ``(Re z, Im z)`` on flat R^2, with no process attached.

    Mix in ahead of :class:`~cyfm.core.manifold.BaseManifold`; the subclass still
    owns ``sample_noise``, ``bridge``, ``loss`` and ``make_solver``, which are
    what actually distinguish one arm from another.
    """

    velocity_channels: int
    representation = Representation.PLANE

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

    def to_complex(self, state: torch.Tensor) -> torch.Tensor:
        return euclidean_to_complex(state)

    def from_complex(self, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([z.real, z.imag], dim=1)
