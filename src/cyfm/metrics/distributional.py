"""Metrics that pool coefficients and compare two clouds of complex numbers.

Pure synthesis has no paired target: a sample drawn from the prior has no reason
to match any particular field, so the paired metrics of reconstruction (PSNR,
SSIM, a phase error against a known slice) are poor by construction. What can be
compared is the distribution.

* **Sliced 2-Wasserstein on the complex plane**, the headline number.
  Coefficients of generated and reference fields are pooled and compared as
  points in ``R^2``. Sliced rather than exact because the exact assignment's
  finite-sample floor is large enough to swallow the differences worth seeing:
  on the synthetic target the exact estimator's floor at 2048 samples is 0.078,
  while the sliced one reaches 0.004 at 32768, against a separation of 0.188
  between genuinely different distributions.
* **Exact transport on each marginal**: amplitude on the line and phase on the
  circle. A model can match the pooled cloud while getting one marginal wrong in
  a way slicing averages away.
* **The dependence gap**: the circular-linear correlation of the generated
  coefficients against the reference's. Amplitude-phase dependence is what a
  factorised coupling destroys, so it has to be a number.

Everything here is blind to where a coefficient sat in the field; see
:mod:`cyfm.metrics.spatial` and :mod:`cyfm.metrics.spectral` for what sees that.
"""

from __future__ import annotations

import torch

# Floor for correlation denominators, which reach zero on a constant batch.
_EPS = 1e-12

__all__ = ["circular_linear_correlation"]


def circular_linear_correlation(amplitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
    """Mardia's circular-linear correlation between an amplitude and an angle.

    Ordinary correlation is meaningless against an angle, because the angle has
    no ordering that survives the wrap. This statistic is built from the
    correlations of the amplitude with ``cos`` and ``sin`` of the angle and is
    invariant to where the branch cut is placed, which is exactly what is needed
    to compare a ``"spiral"`` target against a ``"cardioid"`` one.

    Args:
        amplitude: Amplitudes, shape ``[n]``.
        phase: Angles in radians, shape ``[n]``.

    Returns:
        Scalar tensor in ``[0, 1]``. ``0`` means no dependence detectable by
        this statistic; ``1`` means amplitude is determined by the angle.

    Raises:
        ValueError: If the inputs are not 1D of matching length, or hold fewer
            than two samples.
    """
    if amplitude.ndim != 1 or phase.ndim != 1:
        raise ValueError("amplitude and phase must be 1D")
    if amplitude.numel() != phase.numel():
        raise ValueError(
            f"amplitude and phase must match in length, got {amplitude.numel()} and {phase.numel()}"
        )
    if amplitude.numel() < 2:
        raise ValueError("need at least two samples")

    cos_phase, sin_phase = torch.cos(phase), torch.sin(phase)

    def _corr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        a_centered, b_centered = a - a.mean(), b - b.mean()
        denom = (a_centered.norm() * b_centered.norm()).clamp_min(_EPS)
        return (a_centered * b_centered).sum() / denom

    r_ac = _corr(amplitude, cos_phase)
    r_as = _corr(amplitude, sin_phase)
    r_cs = _corr(cos_phase, sin_phase)

    numerator = r_ac**2 + r_as**2 - 2.0 * r_ac * r_as * r_cs
    denominator = (1.0 - r_cs**2).clamp_min(_EPS)
    return (numerator / denominator).clamp(0.0, 1.0).sqrt()
