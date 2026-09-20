"""Minibatch couplings: which prior sample is paired with which datum.

Flow matching regresses a velocity field onto the displacement between paired
endpoints. Which pairs are formed is a free choice, and it is the choice this
project's contribution lives in: pairing at random leaves trajectories that cross,
so the field has to average conflicting velocities and the resulting curvature is
what forces a solver to take many small steps. A coupling that minimises total
displacement removes the crossings.

Why this module contains no factorised solver
---------------------------------------------
The obvious cheap coupling on a product manifold is to solve each coordinate
separately -- amplitude on the line, phase on the circle -- and pair the two
solutions. It is tempting because both one-dimensional problems have exact
solvers and the circular one runs in ``O(n log n)``. It is also wrong, and
measurably so.

Two separate solutions are two *different* permutations of the target batch, so
the endpoint handed to the bridge takes its amplitude from one datum and its
phase from another. That composite need not be a datum at all. Measured on a
synthetic target whose marginals are held fixed (``scripts/coupling_gate.py``):
both marginals of the coupled batch come out bit-identical to the data's, while
the circular-linear correlation collapses to ~0.04 regardless of the data's,
which ran to 0.87. The error does not shrink with batch size -- 0.384, 0.360,
0.366, 0.364 at n = 64, 256, 1024, 4096 -- so it is a bias, not sampling noise:
the construction converges to the product of the marginals rather than to the
data. It also reports a transport cost *below* the true optimum, invariant in the
data's dependence, which is the signature of scoring a plan that is not feasible.

Equality between the factorised and joint costs needs both measures to be
products of their marginals. Real k-space and real spectrograms are not.

So the coupling here is a single joint assignment over the batch, computed in the
manifold's own metric through its ``log_map``. Every prior sample is matched to a
genuine datum, so the endpoint distribution is the data distribution exactly. The
cost is a linear assignment on an ``n x n`` matrix, which at the batch sizes
training actually uses is nothing: measured at 0.06-0.08 ms for batches of 16 to
64, and 25 ms even at n = 1024. The ``O(n log n)`` the factorised route bought
was never needed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

from cyfm.core.manifold import BaseManifold
from cyfm.core.registry import COUPLINGS

__all__ = ["BaseCoupling", "IndependentCoupling", "OptimalTransportCoupling", "build_coupling"]

# Above this the dense assignment stops being free and the caller should say so
# deliberately rather than discover it as a stall.
_MAX_BATCH = 4096


class BaseCoupling(ABC):
    """Reorders a batch of data endpoints against a batch of prior samples."""

    @abstractmethod
    def __call__(
        self, prior: torch.Tensor, data: torch.Tensor, manifold: BaseManifold
    ) -> torch.Tensor:
        """Return the data batch reordered to pair with the prior batch.

        Args:
            prior: Prior states ``[B, state_channels, H, W]``.
            data: Data states ``[B, state_channels, H, W]``.
            manifold: The geometry whose metric the pairing is judged in.

        Returns:
            The data batch, permuted along the batch dimension.
        """


@COUPLINGS.register("independent")
@COUPLINGS.register("none")
class IndependentCoupling(BaseCoupling):
    """Pair whatever the dataloader happened to draw together.

    The baseline, and what flow matching does by default. Trajectories cross, so
    the regression target at a shared point is an average of conflicting
    velocities and the learned field is curved.
    """

    def __call__(
        self, prior: torch.Tensor, data: torch.Tensor, manifold: BaseManifold
    ) -> torch.Tensor:
        """Return the data batch untouched.

        Args:
            prior: Prior states, unused.
            data: Data states.
            manifold: The geometry, unused.

        Returns:
            ``data``.
        """
        del prior, manifold
        return data


@COUPLINGS.register("optimal_transport")
@COUPLINGS.register("ot")
class OptimalTransportCoupling(BaseCoupling):
    """Exact minibatch optimal transport in the manifold's own metric.

    The cost between two samples is the squared geodesic displacement summed over
    coefficients, weighted per tangent channel by
    :attr:`~cyfm.core.manifold.BaseManifold.tangent_weights`. Both geometries go
    through the same code, so a comparison between them varies the geometry and
    not the quality of the solver each one was given.

    Args:
        max_batch: Refuse batches larger than this rather than building a cost
            matrix that will not fit.

    Raises:
        ValueError: If ``max_batch`` is not positive.
    """

    def __init__(self, max_batch: int = _MAX_BATCH) -> None:
        if max_batch < 1:
            raise ValueError(f"max_batch must be positive, got {max_batch}")
        self.max_batch = int(max_batch)

    def cost_matrix(
        self, prior: torch.Tensor, data: torch.Tensor, manifold: BaseManifold
    ) -> torch.Tensor:
        """Pairwise squared geodesic cost between every prior sample and every datum.

        Built one row at a time. The dense form would be
        ``[B, B, channels, H, W]``, which at a batch of 32 and 320x320 fields is
        already hundreds of megabytes; a row is a factor of ``B`` smaller and the
        loop costs nothing beside the assignment.

        Args:
            prior: Prior states ``[B, state_channels, H, W]``.
            data: Data states ``[B, state_channels, H, W]``.
            manifold: The geometry supplying ``log_map`` and the tangent weights.

        Returns:
            Cost matrix ``[B, B]``.

        Raises:
            ValueError: If the batches disagree in shape or exceed ``max_batch``.
        """
        if prior.shape != data.shape:
            raise ValueError(
                f"prior and data must share a shape, got {tuple(prior.shape)} "
                f"and {tuple(data.shape)}"
            )
        batch = prior.shape[0]
        if batch > self.max_batch:
            raise ValueError(
                f"batch of {batch} exceeds max_batch={self.max_batch}; a dense "
                f"assignment at that size is no longer free"
            )

        weights = manifold.tangent_weights.to(prior.device, prior.dtype)
        cost = torch.empty(batch, batch, device=prior.device, dtype=prior.dtype)
        for row in range(batch):
            displacement = manifold.log_map(prior[row : row + 1].expand_as(data), data)
            weighted = displacement.square() * weights.reshape(1, -1, 1, 1)
            cost[row] = weighted.flatten(1).mean(1)
        return cost

    def __call__(
        self, prior: torch.Tensor, data: torch.Tensor, manifold: BaseManifold
    ) -> torch.Tensor:
        """Reorder the data batch onto its optimal partner in the prior batch.

        Args:
            prior: Prior states ``[B, state_channels, H, W]``.
            data: Data states ``[B, state_channels, H, W]``.
            manifold: The geometry the cost is measured in.

        Returns:
            The data batch, permuted so that sample ``i`` partners ``prior[i]``.
        """
        cost = self.cost_matrix(prior, data, manifold)
        rows, columns = linear_sum_assignment(cost.detach().cpu().numpy().astype(np.float64))
        # scipy hands back numpy indices, which land on the CPU; the permutation has
        # to live where the data does or the scatter below crosses devices.
        permutation = torch.empty(cost.shape[0], dtype=torch.long, device=data.device)
        permutation[torch.as_tensor(rows, dtype=torch.long, device=data.device)] = torch.as_tensor(
            columns, dtype=torch.long, device=data.device
        )
        return data[permutation]


def build_coupling(name: str, **kwargs: object) -> BaseCoupling:
    """Resolve a coupling by name through the registry.

    Args:
        name: Registry key, e.g. ``"independent"`` or ``"ot"``.
        **kwargs: Forwarded to the coupling's constructor.

    Returns:
        The configured coupling.

    Raises:
        ValueError: If ``name`` is not registered.
    """
    if not COUPLINGS.contains(name):
        raise ValueError(f"Unknown coupling {name!r}. Available: {COUPLINGS.list()}.")
    coupling: BaseCoupling = COUPLINGS.build(name, **kwargs)
    return coupling
