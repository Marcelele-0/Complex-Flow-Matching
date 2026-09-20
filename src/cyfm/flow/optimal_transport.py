"""Exact optimal transport primitives on the line, the circle, and the cylinder.

These are the building blocks a coupling needs, kept separate from any decision
about *which* coupling the project should use. Three solvers are provided and
they differ in what they are allowed to assume:

* :func:`sorted_transport_permutation` -- exact 1D transport on the line. The
  optimal plan between two equal-weight empirical measures is rank matching.
* :func:`circular_transport_permutation` -- exact 1D transport on ``S^1``. The
  optimal plan is a *cyclic shift* of the sorted orders, so the solver searches
  the ``n`` shifts rather than the ``n!`` permutations. This is the structure
  Delon, Salomon and Sobolevski exploit to reach ``O(n log n)``; the search here
  is ``O(n^2)`` and exact, which is the right trade at the sizes a minibatch
  coupling actually sees.
* :func:`cylinder_transport_permutation` -- exact transport on ``R+ x S^1``
  under the product cost, solved as a linear assignment on the full cost matrix.
  No factorisation, so no independence assumption.

The distinction between the first two and the third is the point. Running the
first two *separately* and pairing their outputs is not a transport plan between
the joint distributions: it produces, for each source point, an amplitude taken
from one target sample and a phase taken from another. That composite point need
not lie in the target set at all. Equality between the factorised cost and the
joint cost,

    min_pi Int (c_A + c_theta) dpi
        = min_{pi_A} Int c_A dpi_A  +  min_{pi_theta} Int c_theta dpi_theta

holds only when both measures are products of their marginals; otherwise the
left side is strictly larger, and the factorised value is a lower bound that no
feasible plan attains. :func:`transport_cost` reports either side so the gap can
be measured rather than assumed.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

__all__ = [
    "cylinder_cost_matrix",
    "cylinder_transport_permutation",
    "euclidean_transport_permutation",
    "shortest_angular_diff",
    "sliced_wasserstein2",
    "sorted_transport_permutation",
    "transport_cost",
    "wasserstein2_cylinder",
]

# Cap on the cost matrix a dense assignment is allowed to build, in entries.
# 4096^2 float32 is 64 MB, which is a reasonable ceiling for a minibatch-sized
# coupling and small enough to fail loudly rather than swap.
_MAX_DENSE_ENTRIES = 4096 * 4096

# Guards a normalisation against a zero-length random direction, which has
# probability zero but is cheap to exclude.
_DIRECTION_FLOOR = 1e-12

# Shifts evaluated at once by the circular solver. Bounds peak memory at
# _SHIFT_CHUNK * n rather than n^2.
_SHIFT_CHUNK = 256


def shortest_angular_diff(phi_start: torch.Tensor, phi_end: torch.Tensor) -> torch.Tensor:
    """Signed shortest angular displacement, in ``[-pi, pi]``.

    Matches the convention of :meth:`cyfm.flow.bridge.GeodesicFlowBridge.get_shortest_angular_diff`
    so a coupling and the bridge it feeds measure the same angle.

    Args:
        phi_start: Angles in radians, any shape broadcastable with ``phi_end``.
        phi_end: Angles in radians.

    Returns:
        ``phi_end - phi_start`` wrapped into ``[-pi, pi]``.
    """
    diff = phi_end - phi_start
    return (diff + torch.pi) % (2 * torch.pi) - torch.pi


def _validate_pair(source: torch.Tensor, target: torch.Tensor, name: str) -> int:
    """Check two 1D tensors of equal length and return that length."""
    if source.ndim != 1 or target.ndim != 1:
        raise ValueError(f"{name} inputs must be 1D")
    if source.numel() != target.numel():
        raise ValueError(
            f"{name} requires equal sample counts, got {source.numel()} and {target.numel()}"
        )
    if source.numel() == 0:
        raise ValueError(f"{name} requires at least one sample")
    return int(source.numel())


def sorted_transport_permutation(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Exact 1D optimal transport on the line, for equal-weight empirical measures.

    For any convex cost of the displacement the optimal plan is monotone, so
    matching the two sorted orders is optimal and costs one sort per side.

    Args:
        source: Source samples, shape ``[n]``.
        target: Target samples, shape ``[n]``.

    Returns:
        Permutation of shape ``[n]``: source ``i`` is matched to target
        ``perm[i]``.

    Raises:
        ValueError: If the inputs are not 1D of equal, non-zero length.
    """
    n = _validate_pair(source, target, "sorted_transport_permutation")
    order_source = torch.argsort(source)
    order_target = torch.argsort(target)
    perm = torch.empty(n, dtype=torch.long, device=source.device)
    perm[order_source] = order_target
    return perm


def circular_transport_permutation(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Exact 1D optimal transport on ``S^1`` under squared geodesic cost.

    On the circle a monotone plan is only defined up to where the circle is cut,
    so the optimum is a cyclic shift of the sorted orders rather than the
    identity shift. All ``n`` shifts are evaluated, which makes the result exact
    without appealing to the ``O(n log n)`` search of Delon et al.

    Args:
        source: Source angles in radians, shape ``[n]``.
        target: Target angles in radians, shape ``[n]``.

    Returns:
        Permutation of shape ``[n]``: source ``i`` is matched to target
        ``perm[i]``.

    Raises:
        ValueError: If the inputs are not 1D of equal, non-zero length.
    """
    n = _validate_pair(source, target, "circular_transport_permutation")
    order_source = torch.argsort(source)
    order_target = torch.argsort(target)
    source_sorted = source[order_source]
    target_sorted = target[order_target]

    positions = torch.arange(n, device=source.device)
    best_shift = 0
    best_cost = math.inf
    for start in range(0, n, _SHIFT_CHUNK):
        shifts = torch.arange(start, min(start + _SHIFT_CHUNK, n), device=source.device)
        rolled = target_sorted[(positions.unsqueeze(0) + shifts.unsqueeze(1)) % n]
        costs = shortest_angular_diff(source_sorted.unsqueeze(0), rolled).square().sum(dim=1)
        local = int(torch.argmin(costs))
        if float(costs[local]) < best_cost:
            best_cost = float(costs[local])
            best_shift = int(shifts[local])

    perm = torch.empty(n, dtype=torch.long, device=source.device)
    perm[order_source] = order_target[(positions + best_shift) % n]
    return perm


def cylinder_cost_matrix(
    source_amplitude: torch.Tensor,
    source_phase: torch.Tensor,
    target_amplitude: torch.Tensor,
    target_phase: torch.Tensor,
    phase_weight: float = 1.0,
) -> torch.Tensor:
    """Pairwise squared geodesic cost on ``R+ x S^1``.

    The product metric ``ds^2 = dA^2 + w * dtheta^2`` is not the pullback of the
    Euclidean metric of the plane, which would carry ``A^2 dtheta^2``. Dropping
    that factor is the whole point of the cylinder -- phase error stops being
    attenuated by amplitude -- but it also means ``phase_weight`` is a modelling
    choice and not a physical constant. It is exposed rather than hidden.

    Args:
        source_amplitude: Shape ``[n]``, non-negative.
        source_phase: Shape ``[n]``, radians.
        target_amplitude: Shape ``[m]``, non-negative.
        target_phase: Shape ``[m]``, radians.
        phase_weight: Weight ``w`` on the angular term. ``1.0`` is the flat
            product metric.

    Returns:
        Cost matrix of shape ``[n, m]``.

    Raises:
        ValueError: If shapes are inconsistent, ``phase_weight`` is negative, or
            the matrix would exceed the dense-assignment budget.
    """
    if source_amplitude.shape != source_phase.shape:
        raise ValueError("source amplitude and phase must have the same shape")
    if target_amplitude.shape != target_phase.shape:
        raise ValueError("target amplitude and phase must have the same shape")
    if source_amplitude.ndim != 1 or target_amplitude.ndim != 1:
        raise ValueError("cylinder_cost_matrix inputs must be 1D")
    if phase_weight < 0.0:
        raise ValueError(f"phase_weight must be non-negative, got {phase_weight}")

    entries = source_amplitude.numel() * target_amplitude.numel()
    if entries > _MAX_DENSE_ENTRIES:
        raise ValueError(
            f"cost matrix of {entries} entries exceeds the dense budget of "
            f"{_MAX_DENSE_ENTRIES}; reduce the batch or use a factorised solver "
            f"and accept its assumption"
        )

    amplitude_gap = source_amplitude.unsqueeze(1) - target_amplitude.unsqueeze(0)
    angular_gap = shortest_angular_diff(source_phase.unsqueeze(1), target_phase.unsqueeze(0))
    return amplitude_gap.square() + phase_weight * angular_gap.square()


def cylinder_transport_permutation(
    source_amplitude: torch.Tensor,
    source_phase: torch.Tensor,
    target_amplitude: torch.Tensor,
    target_phase: torch.Tensor,
    phase_weight: float = 1.0,
) -> torch.Tensor:
    """Exact joint optimal transport on ``R+ x S^1``, with no factorisation.

    Solved as a linear assignment on the dense cost matrix. At minibatch sizes
    this is cheap -- a few hundred by a few hundred -- and unlike a factorised
    solver every source point is matched to a genuine target *sample*, so the
    coupled endpoint distribution is the target distribution exactly.

    Args:
        source_amplitude: Shape ``[n]``, non-negative.
        source_phase: Shape ``[n]``, radians.
        target_amplitude: Shape ``[n]``, non-negative.
        target_phase: Shape ``[n]``, radians.
        phase_weight: Weight on the angular term of the product metric.

    Returns:
        Permutation of shape ``[n]``: source ``i`` is matched to target
        ``perm[i]``.

    Raises:
        ValueError: If the two sides differ in size, or the cost matrix exceeds
            the dense budget.
    """
    _validate_pair(source_amplitude, target_amplitude, "cylinder_transport_permutation")
    cost = cylinder_cost_matrix(
        source_amplitude, source_phase, target_amplitude, target_phase, phase_weight
    )
    rows, cols = linear_sum_assignment(cost.detach().cpu().numpy().astype(np.float64))
    device = source_amplitude.device
    perm = torch.empty(cost.shape[0], dtype=torch.long, device=device)
    perm[torch.as_tensor(rows, dtype=torch.long, device=device)] = torch.as_tensor(
        cols, dtype=torch.long, device=device
    )
    return perm


def euclidean_transport_permutation(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Exact minibatch optimal transport in the plane, under squared chordal cost.

    The Euclidean counterpart of :func:`cylinder_transport_permutation`, so a
    comparison between the two geometries varies the geometry and nothing else:
    each arm receives the optimal coupling *in its own metric*, computed by the
    same dense assignment.

    Args:
        source: Complex samples, shape ``[n]``.
        target: Complex samples, shape ``[n]``.

    Returns:
        Permutation of shape ``[n]``: source ``i`` is matched to target
        ``perm[i]``.

    Raises:
        ValueError: If the inputs are not 1D of equal, non-zero length, or the
            cost matrix would exceed the dense budget.
    """
    n = _validate_pair(source, target, "euclidean_transport_permutation")
    entries = n * n
    if entries > _MAX_DENSE_ENTRIES:
        raise ValueError(
            f"cost matrix of {entries} entries exceeds the dense budget of "
            f"{_MAX_DENSE_ENTRIES}; reduce the batch"
        )

    cost = (source.unsqueeze(1) - target.unsqueeze(0)).abs().square()
    rows, cols = linear_sum_assignment(cost.detach().cpu().numpy().astype(np.float64))
    device = source.device
    perm = torch.empty(n, dtype=torch.long, device=device)
    perm[torch.as_tensor(rows, dtype=torch.long, device=device)] = torch.as_tensor(
        cols, dtype=torch.long, device=device
    )
    return perm


def transport_cost(
    source_amplitude: torch.Tensor,
    source_phase: torch.Tensor,
    target_amplitude: torch.Tensor,
    target_phase: torch.Tensor,
    phase_weight: float = 1.0,
) -> torch.Tensor:
    """Mean squared cylinder cost between two aligned point clouds.

    The inputs are taken as already paired, element by element. Pass a target
    reordered by a permutation to score that permutation.

    Args:
        source_amplitude: Shape ``[n]``.
        source_phase: Shape ``[n]``, radians.
        target_amplitude: Shape ``[n]``, aligned with the source.
        target_phase: Shape ``[n]``, radians, aligned with the source.
        phase_weight: Weight on the angular term of the product metric.

    Returns:
        Scalar tensor, the mean per-sample cost.

    Raises:
        ValueError: If the inputs are not 1D of equal, non-zero length or
            ``phase_weight`` is negative.
    """
    _validate_pair(source_amplitude, target_amplitude, "transport_cost")
    _validate_pair(source_phase, target_phase, "transport_cost")
    if phase_weight < 0.0:
        raise ValueError(f"phase_weight must be non-negative, got {phase_weight}")

    amplitude_gap = source_amplitude - target_amplitude
    angular_gap = shortest_angular_diff(source_phase, target_phase)
    return (amplitude_gap.square() + phase_weight * angular_gap.square()).mean()


def sliced_wasserstein2(
    first: torch.Tensor,
    second: torch.Tensor,
    num_projections: int = 256,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sliced 2-Wasserstein distance between two equal-weight point clouds.

    Projects both clouds onto random directions and averages the exact
    one-dimensional transport cost over them. Unlike the dense assignment in
    :func:`wasserstein2_cylinder` this costs a sort rather than a cubic solve, so
    it stays usable at the sample counts a distributional claim actually needs --
    the exact estimator's finite-sample floor is what made an earlier comparison
    of trained arms unreadable.

    Slicing needs a vector space, so this is for clouds in ``R^d``: pass complex
    samples as ``(re, im)``. Do **not** pass ``(amplitude, phase)`` -- a linear
    projection of an angle is meaningless across the branch cut, and the circular
    marginal has an exact solver of its own in
    :func:`circular_transport_permutation`.

    Args:
        first: Point cloud of shape ``[n, d]``.
        second: Point cloud of shape ``[n, d]``.
        num_projections: Number of random directions to average over.
        generator: Optional RNG, so a reported number is reproducible.

    Returns:
        Scalar tensor.

    Raises:
        ValueError: If the clouds are not 2D of matching shape, or
            ``num_projections`` is not positive.
    """
    if first.ndim != 2 or second.ndim != 2:
        raise ValueError("sliced_wasserstein2 expects [n, d] clouds")
    if first.shape != second.shape:
        raise ValueError(
            f"clouds must match in shape, got {tuple(first.shape)} and {tuple(second.shape)}"
        )
    if first.shape[0] == 0:
        raise ValueError("clouds must hold at least one point")
    if num_projections < 1:
        raise ValueError(f"num_projections must be positive, got {num_projections}")

    dimension = first.shape[1]
    directions = torch.randn(
        dimension, num_projections, generator=generator, dtype=first.dtype, device=first.device
    )
    directions = directions / directions.norm(dim=0, keepdim=True).clamp_min(_DIRECTION_FLOOR)

    projected_first = (first @ directions).sort(dim=0).values
    projected_second = (second @ directions).sort(dim=0).values
    return (projected_first - projected_second).square().mean().clamp_min(0.0).sqrt()


def wasserstein2_cylinder(
    first_amplitude: torch.Tensor,
    first_phase: torch.Tensor,
    second_amplitude: torch.Tensor,
    second_phase: torch.Tensor,
    phase_weight: float = 1.0,
) -> torch.Tensor:
    """Exact 2-Wasserstein distance between two equal-weight clouds on the cylinder.

    Unlike a comparison of marginals this sees the joint distribution, which is
    what a factorised coupling damages while leaving every marginal intact.

    Args:
        first_amplitude: Shape ``[n]``.
        first_phase: Shape ``[n]``, radians.
        second_amplitude: Shape ``[n]``.
        second_phase: Shape ``[n]``, radians.
        phase_weight: Weight on the angular term of the product metric.

    Returns:
        Scalar tensor, the 2-Wasserstein distance.

    Raises:
        ValueError: If the two clouds differ in size or exceed the dense budget.
    """
    perm = cylinder_transport_permutation(
        first_amplitude, first_phase, second_amplitude, second_phase, phase_weight
    )
    squared = transport_cost(
        first_amplitude,
        first_phase,
        second_amplitude[perm],
        second_phase[perm],
        phase_weight,
    )
    return squared.clamp_min(0.0).sqrt()
