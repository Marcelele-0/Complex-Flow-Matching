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

# Guards the induced angular velocity against a literal division by zero; the
# divergence it is meant to expose happens well above this.
_ANGULAR_FLOOR = 1e-12


class FlatComplexRepresentation:
    """Complex pixels as ``(Re z, Im z)`` on flat R^2, with no process attached.

    Mix in ahead of :class:`~cyfm.core.manifold.BaseManifold`; the subclass still
    owns ``sample_noise``, ``bridge``, ``loss`` and ``make_solver``, which are
    what actually distinguish one arm from another.
    """

    velocity_channels: int
    representation = Representation.PLANE
    # Defined for this representation, which is not the same as reported for
    # every arm that carries it: the score-based baseline mixes this in and is
    # still excluded, by predicts_velocity.
    reports_induced_angular_velocity = True

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

    def induced_angular_velocity(
        self, state: torch.Tensor, velocity: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``theta_dot = (x v_y - y v_x) / A^2``, which diverges as ``A -> 0``.

        The Cartesian target ``z_1 - z_0`` is bounded whenever its endpoints
        are; what is unbounded is the rate at which the path turns, and that is
        what a coarse integrator has to resolve. This is the measurement of it.
        """
        real, imag = state[:, 0], state[:, 1]
        squared = (real * real + imag * imag).clamp_min(_ANGULAR_FLOOR)
        return squared.sqrt(), (real * velocity[:, 1] - imag * velocity[:, 0]) / squared

    def to_complex(self, state: torch.Tensor) -> torch.Tensor:
        return euclidean_to_complex(state)

    def from_complex(self, z: torch.Tensor) -> torch.Tensor:
        return torch.cat([z.real, z.imag], dim=1)
