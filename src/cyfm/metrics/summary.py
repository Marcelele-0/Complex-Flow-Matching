"""The one call the evaluation sweep makes, and everything it returns.

Three families in one dictionary: the pooled comparisons of
:mod:`cyfm.metrics.distributional`, the two spatial gaps of
:mod:`cyfm.metrics.spatial`, and the spectral gap of
:mod:`cyfm.metrics.spectral`. Every key is computed for every dataset -- there is
no per-domain selection here and never was.

That is worth stating because the archives look otherwise. The synthetic
evaluations in ``docs/reproduce/paper_results/`` carry 20 numeric keys and the
knee evaluations 28, and the difference is chronological, not a design: the
synthetic sweep ran on 2026-09-11, while ``radial_spectrum_gap`` was written on
09-17 and ``phase_lag_one`` on 09-18. Re-running the synthetic grid today would
produce the knee's key set. Building a metric selection per domain around that
gap would have frozen an accident into the architecture.

The function was called ``distributional_metrics`` while it also returned
spatial and spectral keys, which is how that misreading starts.
"""

from __future__ import annotations

import torch

from cyfm.flow.transport import (
    circular_transport_permutation,
    shortest_angular_diff,
    sliced_wasserstein2,
    sorted_transport_permutation,
)
from cyfm.metrics.budgets import MAX_EXACT_COEFFICIENTS, MAX_SLICED_COEFFICIENTS, subsample
from cyfm.metrics.distributional import circular_linear_correlation
from cyfm.metrics.spatial import measured_pair_fraction, phase_lag_one, spatial_lag_one
from cyfm.metrics.spectral import radial_spectrum_gap

__all__ = ["generative_metrics"]


def generative_metrics(
    generated: torch.Tensor,
    reference: torch.Tensor,
    num_projections: int,
    generator: torch.Generator,
) -> dict[str, float]:
    """Compare a generated field batch with a reference batch.

    Args:
        generated: Complex fields of shape ``[B, 1, H, W]``.
        reference: Complex fields of shape ``[B', 1, H, W]``.
        num_projections: Directions averaged over by the sliced estimator.
        generator: RNG for the projections and any subsampling.

    Returns:
        Metric name to value: ``sliced_w2_complex``, ``w2_amplitude``,
        ``w2_phase_circular``, the generated and reference dependence and their
        gap, the generated and reference lag-one correlation and their gap, and
        ``radial_spectrum_gap``.

    Raises:
        ValueError: If either batch is not a complex 4D tensor, or is narrower
            than four samples on an axis, which the radial spectrum needs to
            average over a ring.
    """
    if generated.ndim != 4 or reference.ndim != 4:
        raise ValueError("both batches must be [B, 1, H, W]")
    if not (generated.is_complex() and reference.is_complex()):
        raise ValueError("both batches must be complex")

    flat_generated = generated.reshape(-1)
    flat_reference = reference.reshape(-1)

    # Slicing needs equal cloud sizes; the smaller side sets the budget.
    sliced_budget = min(flat_generated.numel(), flat_reference.numel(), MAX_SLICED_COEFFICIENTS)
    left = subsample(flat_generated, sliced_budget, generator)
    right = subsample(flat_reference, sliced_budget, generator)
    sliced = float(
        sliced_wasserstein2(
            torch.stack([left.real, left.imag], dim=1),
            torch.stack([right.real, right.imag], dim=1),
            num_projections=num_projections,
            generator=generator,
        )
    )

    exact_budget = min(flat_generated.numel(), flat_reference.numel(), MAX_EXACT_COEFFICIENTS)
    left = subsample(flat_generated, exact_budget, generator)
    right = subsample(flat_reference, exact_budget, generator)

    amplitude_perm = sorted_transport_permutation(left.abs(), right.abs())
    amplitude_w2 = float(
        (left.abs() - right.abs()[amplitude_perm]).square().mean().clamp_min(0.0).sqrt()
    )

    phase_perm = circular_transport_permutation(left.angle(), right.angle())
    phase_w2 = float(
        shortest_angular_diff(left.angle(), right.angle()[phase_perm])
        .square()
        .mean()
        .clamp_min(0.0)
        .sqrt()
    )

    generated_dependence = float(circular_linear_correlation(left.abs(), left.angle()))
    reference_dependence = float(circular_linear_correlation(right.abs(), right.angle()))

    # Bound once: each call re-materialises two shifted views of the amplitude
    # field, which at the fastMRI scale is millions of elements per sweep row.
    generated_lag_one = spatial_lag_one(generated)
    reference_lag_one = spatial_lag_one(reference)
    # Matched support: the reference defines the share of pairs whose phase exists, and
    # the generated batch is scored on the same share of its own strongest pairs.
    measured_share = measured_pair_fraction(reference)
    reference_phase_lag = phase_lag_one(reference)
    generated_phase_lag = phase_lag_one(generated, keep_fraction=measured_share)

    return {
        "sliced_w2_complex": sliced,
        "w2_amplitude": amplitude_w2,
        "w2_phase_circular": phase_w2,
        "dependence_generated": generated_dependence,
        "dependence_reference": reference_dependence,
        "dependence_gap": abs(generated_dependence - reference_dependence),
        "spatial_lag1_generated": generated_lag_one,
        "spatial_lag1_reference": reference_lag_one,
        "spatial_lag1_gap": abs(generated_lag_one - reference_lag_one),
        "phase_lag1_measured_share": measured_share,
        "phase_lag1_generated": generated_phase_lag,
        "phase_lag1_reference": reference_phase_lag,
        "phase_lag1_gap": abs(generated_phase_lag - reference_phase_lag),
        "radial_spectrum_gap": radial_spectrum_gap(generated, reference),
    }
