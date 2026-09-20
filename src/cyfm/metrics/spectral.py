"""The azimuthally averaged power spectrum, and the gap between two of them.

Lag-one autocorrelation sees only the nearest neighbour, so a model can match it
and still put its energy at the wrong scales. The radial spectrum is the summary
that notices: it says how much power sits at each spatial frequency, averaged
over direction, which is the structure a smooth prior is supposed to reproduce.
"""

from __future__ import annotations

import torch

__all__ = ["radial_power_spectrum", "radial_spectrum_gap"]


def radial_power_spectrum(fields: torch.Tensor, num_bins: int | None = None) -> torch.Tensor:
    """Azimuthally averaged power spectrum of the amplitude field.

    Args:
        fields: Complex fields of shape ``[B, 1, H, W]``.
        num_bins: Radial bins between zero and Nyquist. Defaults to half the
            shorter axis, the resolution the grid actually supports.

    Returns:
        Mean power per radial bin, of shape ``[num_bins]``.

    Raises:
        ValueError: If ``fields`` is not a 4D complex tensor, either axis is
            shorter than four samples, or ``num_bins`` is below two.
    """
    if fields.ndim != 4 or not fields.is_complex():
        raise ValueError(f"expected complex [B, 1, H, W], got shape {tuple(fields.shape)}")
    height, width = fields.shape[-2], fields.shape[-1]
    if min(height, width) < 4:
        raise ValueError("a radial spectrum needs at least four samples on each axis")

    bins = min(height, width) // 2 if num_bins is None else int(num_bins)
    if bins < 2:
        raise ValueError(f"num_bins must be at least two, got {bins}")

    power = torch.fft.fft2(fields.abs(), norm="ortho").abs().square().mean(dim=(0, 1))

    frequency_y = torch.fft.fftfreq(height, device=fields.device).unsqueeze(1)
    frequency_x = torch.fft.fftfreq(width, device=fields.device).unsqueeze(0)
    radius = torch.sqrt(frequency_y * frequency_y + frequency_x * frequency_x)

    # Past Nyquist only the corners of the grid contribute, so those radii are
    # sampled along some directions and not others. Dropping them keeps every bin
    # an average over a full ring; folding them into the last bin would not.
    inside = radius <= 0.5
    index = (radius[inside] / 0.5 * bins).long().clamp_(max=bins - 1)
    values = power[inside]

    total = torch.zeros(bins, device=fields.device, dtype=values.dtype)
    counts = torch.zeros(bins, device=fields.device, dtype=values.dtype)
    total.scatter_add_(0, index, values)
    counts.scatter_add_(0, index, torch.ones_like(values))
    return total / counts.clamp_min(1.0)


def radial_spectrum_gap(
    generated: torch.Tensor,
    reference: torch.Tensor,
    num_bins: int | None = None,
) -> float:
    """Mean absolute log-ratio between two normalised radial power spectra.

    Reported in dex: zero is an identical spectral shape, and 1.0 means the two
    spectra differ by a factor of ten in the average bin. The comparison is made
    in the log because spectral power spans orders of magnitude across the band,
    so a linear difference would only ever describe the lowest frequencies.

    Args:
        generated: Complex fields of shape ``[B, 1, H, W]``.
        reference: Complex fields of shape ``[B', 1, H', W']``.
        num_bins: Radial bins. Defaults to half the shortest axis of either
            batch, so unequally sized batches still land on a common grid.

    Returns:
        The gap, or ``nan`` if no bin carries power in both batches -- a diverged
        sampler, an all-zero or constant field. Not ``0.0``: that is the value
        meaning *identical spectrum*, and this metric exists to catch the very
        runs that would otherwise earn it. ``spatial_lag_one`` propagates ``nan``
        for the same reason, and it survives the archive's JSON round-trip.
    """
    if num_bins is None:
        shortest = min(
            generated.shape[-2], generated.shape[-1], reference.shape[-2], reference.shape[-1]
        )
        num_bins = shortest // 2

    generated_profile = radial_power_spectrum(generated, num_bins)
    reference_profile = radial_power_spectrum(reference, num_bins)

    # Bin zero is the field's mean brightness, which w2_amplitude already covers.
    # Normalising what is left makes this a statement about spectral shape alone,
    # so it cannot restate a difference in overall power as a spatial finding.
    generated_profile = generated_profile[1:]
    reference_profile = reference_profile[1:]
    usable = (generated_profile > 0) & (reference_profile > 0)
    if not bool(usable.any()):
        return float("nan")

    generated_density = generated_profile[usable] / generated_profile[usable].sum()
    reference_density = reference_profile[usable] / reference_profile[usable].sum()
    return float((torch.log10(generated_density) - torch.log10(reference_density)).abs().mean())
