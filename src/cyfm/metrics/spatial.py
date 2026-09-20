"""Metrics that see where a coefficient sat in the field.

Everything in :mod:`cyfm.metrics.distributional` pools coefficients, so a model
that gets the pooled cloud exactly right while scrambling the field across space
scores perfectly there. These are the two numbers that notice.

* **The spatial gap**: lag-one autocorrelation of the amplitude field.
* **The phase coherence gap**: the same lag, on the circle, restricted to pairs
  whose phase exists. Phase is undefined at zero amplitude, so the reference
  fixes the share of pairs that count and the generated batch is scored on the
  same share of its own strongest pairs -- otherwise a model that put amplitude
  in the wrong places would be compared on a different set of pixels.
"""

from __future__ import annotations

import torch

from cyfm.metrics.budgets import MAX_SLICED_COEFFICIENTS, subsample

__all__ = ["measured_pair_fraction", "phase_lag_one", "spatial_lag_one"]


def measured_pair_fraction(fields: torch.Tensor) -> float:
    """Share of horizontally adjacent pairs whose phase was actually measured.

    A pair counts when neither end has amplitude exactly zero. ``atan2(0, 0)`` returns
    ``0.0``, a placeholder rather than a measurement, so a pair touching one carries no
    phase information. On the knee cohort this keeps about 84% of pairs.

    Args:
        fields: Complex fields of shape ``[B, 1, H, W]``.

    Returns:
        The fraction in ``[0, 1]``.

    Raises:
        ValueError: If ``fields`` is not a 4D complex tensor with width above 1.
    """
    left, right = _adjacent_pairs(fields)
    return float((torch.minimum(left.abs(), right.abs()) > 0).float().mean())


def _adjacent_pairs(fields: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and split a batch into horizontally adjacent pairs."""
    if fields.ndim != 4 or not fields.is_complex():
        raise ValueError(f"expected complex [B, 1, H, W], got shape {tuple(fields.shape)}")
    if fields.shape[-1] < 2:
        raise ValueError("lag-one coherence needs a width of at least two")
    return fields[..., :-1], fields[..., 1:]


def phase_lag_one(fields: torch.Tensor, keep_fraction: float | None = None) -> float:
    """Circular coherence of horizontally adjacent phases, averaged over fields.

    Nothing else in this module sees the *spatial* arrangement of phase. The pooled
    metrics compare marginals, ``w2_phase_circular`` compares a marginal, and
    :func:`spatial_lag_one` reads amplitudes. A model that reproduced a phase field
    exactly but shifted its marginal, and one that emitted spatially white phase with
    the right marginal, are indistinguishable without this.

    The statistic is the mean resultant of the phase increment,
    ``E[cos(theta_{i+1} - theta_i)]``: one for a field whose phase varies smoothly,
    zero for phase that is spatially independent, negative for alternation. No wrapping
    is needed because the cosine already identifies angles modulo ``2 pi``.

    **The support must be matched between the two batches, or the comparison is biased.**
    Real MRI carries about 17% of coefficients that are exactly zero, and dropping the
    pairs that touch them measures the reference on roughly 84% of its pairs. A network
    never emits an exact zero, so scoring it on all of its pairs scores it partly on air,
    where phase has no physical meaning -- and air does not contribute equally for the two
    geometries. A single Euler step in ``(Re, Im)`` leaves smoothly blurred values there,
    whose ``atan2`` is smooth and earns coherence from nothing, while a cylindrical arm
    carries the prior's uncorrelated phase. So pass the reference's
    :func:`measured_pair_fraction` as ``keep_fraction`` and the generated batch is scored
    on its own strongest pairs in exactly that share.

    Choosing a fixed amplitude threshold instead would not work: the reference coherence
    climbs monotonically with any such cut (0.913 at zero, 0.954 at 0.01, 0.982 at 0.02,
    0.998 at 0.2) and never reaches a plateau, so whoever picks the threshold picks the
    answer. A fraction taken from the data has no such freedom.

    Args:
        fields: Complex fields of shape ``[B, 1, H, W]``.
        keep_fraction: Share of the strongest pairs to score, matching another batch's
            measured support. ``None`` drops only the pairs touching an exact zero.

    Returns:
        The coherence, or ``nan`` if no pair survives.

    Raises:
        ValueError: If ``fields`` is not a 4D complex tensor with width above 1, or
            ``keep_fraction`` is outside ``(0, 1]``.
    """
    left, right = _adjacent_pairs(fields)
    strength = torch.minimum(left.abs(), right.abs())

    if keep_fraction is None:
        keep = strength > 0
    else:
        if not 0.0 < keep_fraction <= 1.0:
            raise ValueError(f"keep_fraction must be in (0, 1], got {keep_fraction}")
        flat = strength.reshape(-1)
        if flat.numel() > MAX_SLICED_COEFFICIENTS:
            # A generator of its own, fixed at 0, rather than one threaded in from
            # the caller. That makes this metric reproducible independently of the
            # evaluation seed, which is what a *reference* statistic has to be: the
            # reference's measured share must not move when the run's seed moves.
            flat = subsample(flat, MAX_SLICED_COEFFICIENTS, torch.Generator().manual_seed(0))
        cut = torch.quantile(flat.float(), 1.0 - keep_fraction)
        keep = strength >= cut

    if not bool(keep.any()):
        return float("nan")
    return float((right.angle()[keep] - left.angle()[keep]).cos().mean())


def spatial_lag_one(fields: torch.Tensor) -> float:
    """Correlation between horizontally adjacent amplitudes, averaged over fields.

    The pooled metrics cannot see spatial structure at all, so this is what
    separates a model that has learned a field from one that has learned its
    histogram.

    Args:
        fields: Complex fields of shape ``[B, 1, H, W]``.

    Returns:
        The correlation, or ``0.0`` for a degenerate field with no variation.

    Raises:
        ValueError: If ``fields`` is not a 4D complex tensor with width above 1.
    """
    if fields.ndim != 4 or not fields.is_complex():
        raise ValueError(f"expected complex [B, 1, H, W], got shape {tuple(fields.shape)}")
    if fields.shape[-1] < 2:
        raise ValueError("lag-one correlation needs a width of at least two")

    amplitude = fields.abs()
    left = amplitude[..., :-1].reshape(-1)
    right = amplitude[..., 1:].reshape(-1)
    left = left - left.mean()
    right = right - right.mean()
    denominator = left.norm() * right.norm()
    if float(denominator) <= 0.0:
        return 0.0
    return float((left * right).sum() / denominator)
