"""Frequency-domain spectral loss penalty in k-space."""

from __future__ import annotations

import torch


def high_frequency_penalty(error: torch.Tensor, boost_factor: float) -> torch.Tensor:
    """Compute radially-weighted magnitude of velocity error in k-space.

    Args:
        error: Velocity error (pred - target) [B, C, H, W].
        boost_factor: Slope of the radial frequency weighting.

    Returns:
        Scalar mask-weighted mean frequency error.
    """
    fft_err = torch.fft.fftshift(torch.fft.fft2(error, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
    fft_err_mag = torch.abs(fft_err)

    h, w = error.shape[-2:]
    device = error.device
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device),
        torch.linspace(-1, 1, w, device=device),
        indexing="ij",
    )
    radius = torch.sqrt(x**2 + y**2)

    hf_mask = 1.0 + (boost_factor * radius)
    hf_mask = hf_mask.unsqueeze(0).unsqueeze(0)

    return (fft_err_mag * hf_mask).mean()
