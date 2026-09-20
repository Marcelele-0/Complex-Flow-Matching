"""Spatially correlated standard normal fields.

Used on both sides of a flow: to give synthetic data spatial structure, and to
give the prior spatial structure. In both cases the correlation is imposed on
standard normal *latents* and the result is normalised to unit variance, so every
entry stays standard normal marginally. Whatever law is built from the latents
afterwards -- a uniform modulus, a uniform or wrapped-normal phase, a copula --
therefore keeps its pointwise distribution exactly, and only the dependence
between neighbouring coefficients changes.
"""

from __future__ import annotations

import math

import torch

__all__ = ["smooth_standard_normals"]


def smooth_standard_normals(noise: torch.Tensor, correlation_length: float) -> torch.Tensor:
    """Give white noise a spatial correlation length, keeping it standard normal.

    The field is filtered in the frequency domain by the transfer function of a
    Gaussian kernel of standard deviation ``correlation_length`` pixels, then
    rescaled so that every entry has unit variance *exactly*, for any length.

    Why spectral rather than a spatial convolution: a spatial kernel wider than
    the field wraps onto itself under circular padding, so one input pixel enters
    the sum through several taps and dividing by the kernel's L2 norm no longer
    gives unit variance. Measured on 64x64 fields with the earlier spatial
    implementation, the per-pixel variance was 1.057 at a length of 16 and 2.628
    at 32, which turned a uniform modulus built as ``Phi(z)`` into one with
    standard deviation 0.361 instead of 0.289 -- a changed pointwise law, which is
    exactly what the prior is not allowed to have. Sampling the continuous
    transfer function on the DFT grid is the periodised kernel, so the circular
    wrap is exact rather than truncated, and normalising by the root mean square
    of the transfer function fixes the variance regardless of aliasing: under an
    orthonormal DFT a unit white field has unit-variance spectral coefficients,
    so the output variance is the mean of the squared transfer function.

    The transfer function is laid out in the same unshifted frequency order as
    ``fft2`` returns, so no ``fftshift`` is needed: a real, even transfer function
    commutes with the shift, and the pair would be a no-op. The transform is
    orthonormal over the two spatial axes.

    Args:
        noise: Standard normal field of shape ``[..., H, W]``, real.
        correlation_length: Gaussian sigma in pixels. Non-positive returns the
            input unchanged.

    Returns:
        A real field of the same shape, standard normal marginally.
    """
    if correlation_length <= 0.0:
        return noise

    height, width = noise.shape[-2], noise.shape[-1]
    frequency_y = torch.fft.fftfreq(height, device=noise.device, dtype=noise.dtype)
    frequency_x = torch.fft.fftfreq(width, device=noise.device, dtype=noise.dtype)
    squared_frequency = frequency_y.reshape(-1, 1).square() + frequency_x.reshape(1, -1).square()

    # Continuous Fourier transform of exp(-x^2 / (2 L^2)), in cycles per pixel.
    transfer = torch.exp(-2.0 * math.pi**2 * correlation_length**2 * squared_frequency)
    # The DC term is always exp(0) = 1, so the mean below is never zero even when
    # every other frequency underflows for a length far beyond the field.
    transfer = transfer / transfer.square().mean().sqrt()

    spectrum = torch.fft.fft2(noise, dim=(-2, -1), norm="ortho")
    return torch.fft.ifft2(spectrum * transfer, dim=(-2, -1), norm="ortho").real
