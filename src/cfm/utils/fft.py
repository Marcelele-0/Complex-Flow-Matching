"""Centered 2D Fourier transforms shared by the datasets, reconstructors and eval.

Every k-space array in this project is *centered*: DC sits at ``[H // 2, W // 2]``,
which is how fastMRI and SKM-TEA store their data and where the mask generators put
the fully sampled ACS block. Every image array is centered too, meaning the anatomy
sits in the middle of the field of view.

Moving between those two conventions needs a shift on *both* sides of the transform:
``ifftshift`` first, to put DC at index 0 where :func:`torch.fft.fft2` expects it, and
``fftshift`` afterwards, to put the origin back in the middle. Dropping the trailing
shift leaves the result rolled by half a field of view in each dimension. That error
is invisible as long as a tensor only ever makes the round trip - ``fftshift(fft2(.))``
and ``ifft2(ifftshift(.))`` are exact inverses whichever way the shifts are paired -
so it survives a round-trip unit test untouched. It stops being invisible the moment
the image-domain result is combined with a quantity that was computed elsewhere: the
ESPIRiT sensitivity maps from ``sigpy`` are on the centered grid, so pairing them with
a half-FOV-rolled coil image weights every pixel by a sensitivity from across the FOV
and destroys the coil combination. Use these helpers rather than open-coding the
shifts, so that cannot drift apart again.
"""

from __future__ import annotations

import torch

__all__ = ["fft2c", "ifft2c"]


def fft2c(x: torch.Tensor) -> torch.Tensor:
    """Centered orthonormal 2D FFT over the trailing two dimensions.

    Args:
        x: Complex image-domain tensor ``[..., H, W]`` with the origin at the center.

    Returns:
        Complex k-space tensor ``[..., H, W]`` with DC at the center.
    """
    return torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(x, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )


def ifft2c(k: torch.Tensor) -> torch.Tensor:
    """Centered orthonormal 2D inverse FFT over the trailing two dimensions.

    Args:
        k: Complex k-space tensor ``[..., H, W]`` with DC at the center.

    Returns:
        Complex image-domain tensor ``[..., H, W]`` with the origin at the center.
    """
    return torch.fft.fftshift(
        torch.fft.ifft2(torch.fft.ifftshift(k, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )
