"""Tests for the centered 2D Fourier helpers."""

from __future__ import annotations

import numpy as np
import torch

from cfm.utils.fft import fft2c, ifft2c


def test_round_trip_is_identity() -> None:
    """ifft2c inverts fft2c exactly."""
    x = torch.complex(torch.randn(2, 3, 16, 16), torch.randn(2, 3, 16, 16))
    torch.testing.assert_close(ifft2c(fft2c(x)), x, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(fft2c(ifft2c(x)), x, atol=1e-5, rtol=1e-5)


def test_dc_sits_at_the_center() -> None:
    """A constant image transforms to energy at the center bin, not at index [0, 0]."""
    h = w = 16
    k = fft2c(torch.ones(1, 1, h, w, dtype=torch.complex64))

    peak = torch.argmax(k.abs().flatten())
    assert divmod(int(peak), w) == (h // 2, w // 2)


def test_matches_sigpy_centered_convention() -> None:
    """The helpers agree with numpy's centered transform, which is sigpy's too.

    sigpy's ``fft``/``ifft`` default to ``center=True``, implemented as
    ``ifftshift -> fftn -> fftshift``. The ESPIRiT maps are produced on that grid, so
    any drift here silently misaligns coil combination.
    """
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((8, 8)) + 1j * rng.standard_normal((8, 8))).astype(np.complex64)

    expected = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x), norm="ortho"))
    actual = fft2c(torch.from_numpy(x)).numpy()

    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_shift_is_not_absorbed_by_a_round_trip() -> None:
    """An image and its half-FOV roll are distinguishable, which is why the shift matters.

    The bug this module exists to prevent is invisible to a round-trip test: pairing
    ``ifft2(ifftshift(.))`` with ``fftshift(fft2(.))`` is also self-inverse. It only
    shows up against an externally computed image-domain quantity, so this asserts
    the two conventions genuinely differ.
    """
    k = torch.complex(torch.randn(1, 1, 16, 16), torch.randn(1, 1, 16, 16))

    centered = ifft2c(k)
    missing_trailing_shift = torch.fft.ifft2(torch.fft.ifftshift(k, dim=(-2, -1)), norm="ortho")

    assert not torch.allclose(centered, missing_trailing_shift, atol=1e-4)
    torch.testing.assert_close(
        centered,
        torch.fft.fftshift(missing_trailing_shift, dim=(-2, -1)),
        atol=1e-5,
        rtol=1e-5,
    )
