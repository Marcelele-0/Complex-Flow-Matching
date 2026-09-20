"""How many coefficients each estimator may look at, and how it draws them.

Two of the estimators here are superlinear in the number of points, so each one
needs a ceiling, and the two ceilings are different by three orders of magnitude
for a reason worth stating once rather than at each use.

The constants were shared by :mod:`cyfm.metrics.distributional` and
:mod:`cyfm.metrics.spatial` before the split; they live here so neither imports a
private name from the other.
"""

from __future__ import annotations

import torch

# The circular solver searches n cyclic shifts, so its cost is quadratic. Beyond
# this the marginal metrics are computed on a random subsample, which is stated
# in the report rather than done silently.
MAX_EXACT_COEFFICIENTS = 8192

# Slicing costs a sort per projection, so it can afford far more points; this is
# where its resolution advantage over the exact estimator comes from.
MAX_SLICED_COEFFICIENTS = 65536

__all__ = ["MAX_EXACT_COEFFICIENTS", "MAX_SLICED_COEFFICIENTS", "subsample"]


def subsample(values: torch.Tensor, limit: int, generator: torch.Generator) -> torch.Tensor:
    """Take at most ``limit`` entries, uniformly and reproducibly.

    Args:
        values: Values of shape ``[n]``.
        limit: Maximum number to keep.
        generator: RNG, so a reported number can be reproduced.

    Returns:
        Either ``values`` unchanged or a random subset of size ``limit``.
    """
    if values.numel() <= limit:
        return values
    picks = torch.randperm(values.numel(), generator=generator)[:limit]
    return values[picks]
