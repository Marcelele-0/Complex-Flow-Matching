"""Metrics for generated complex fields, grouped by what they can see.

* :mod:`~cyfm.metrics.distributional` pools coefficients and compares two clouds
  of complex numbers; it is blind to where a coefficient sat.
* :mod:`~cyfm.metrics.spatial` sees the nearest neighbour, in amplitude and in
  phase.
* :mod:`~cyfm.metrics.spectral` sees every scale at once.
* :mod:`~cyfm.metrics.geometry` scores the path rather than the endpoint, and is
  the only group gated on a property of the geometry.
* :mod:`~cyfm.metrics.summary` composes the first three into the single call the
  evaluation sweep makes.

This was one 436-line module under ``utils/``.
"""

from __future__ import annotations

from cyfm.metrics.budgets import (
    MAX_EXACT_COEFFICIENTS,
    MAX_SLICED_COEFFICIENTS,
    subsample,
)
from cyfm.metrics.distributional import circular_linear_correlation
from cyfm.metrics.geometry import AngularVelocityProbe, straightness
from cyfm.metrics.spatial import measured_pair_fraction, phase_lag_one, spatial_lag_one
from cyfm.metrics.spectral import radial_power_spectrum, radial_spectrum_gap
from cyfm.metrics.summary import generative_metrics

__all__ = [
    "MAX_EXACT_COEFFICIENTS",
    "MAX_SLICED_COEFFICIENTS",
    "AngularVelocityProbe",
    "circular_linear_correlation",
    "generative_metrics",
    "measured_pair_fraction",
    "phase_lag_one",
    "radial_power_spectrum",
    "radial_spectrum_gap",
    "spatial_lag_one",
    "straightness",
    "subsample",
]
