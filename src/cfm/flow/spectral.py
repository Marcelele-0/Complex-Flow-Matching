"""Frequency-domain penalty shared by every geometry's velocity loss.

Extracted so both losses apply *bit-identical* high-frequency boosting: an HF
ablation must differ in geometry, not in how the penalty is spelled twice.
"""

from __future__ import annotations

import torch


def high_frequency_penalty(error: torch.Tensor, boost_factor: float) -> torch.Tensor:
    """Radially-weighted magnitude of the velocity error in k-space.

    Weight is 1.0 at the centre and grows linearly with normalised radius, so the
    term pushes the model toward sharp features rather than only the smooth bulk.

    Args:
        error: Velocity error ``pred - target``, shape ``[B, C, H, W]``. Channel
            layout is irrelevant: the transform is applied per channel.
        boost_factor: Slope of the radial weight. ``0.0`` reduces this to a plain
            mean k-space magnitude.

    Returns:
        Scalar tensor, the mask-weighted mean magnitude.

    Note:
        The FFT uses ``norm="ortho"``, which divides magnitudes by ``sqrt(H*W)``,
        so a given ``lambda_hf`` means the same thing at every crop size.
    """
    fft_err = torch.fft.fftshift(torch.fft.fft2(error, dim=(-2, -1), norm="ortho"), dim=(-2, -1))
    fft_err_mag = torch.abs(fft_err)

    # Build radial mask favoring high frequencies (k-space periphery)
    h, w = error.shape[-2:]
    device = error.device
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, h, device=device),
        torch.linspace(-1, 1, w, device=device),
        indexing="ij",
    )
    radius = torch.sqrt(x**2 + y**2)  # Euclidean distance from center

    # At center: mask = 1.0, at periphery: linearly increases
    hf_mask = 1.0 + (boost_factor * radius)
    hf_mask = hf_mask.unsqueeze(0).unsqueeze(0)

    return (fft_err_mag * hf_mask).mean()
