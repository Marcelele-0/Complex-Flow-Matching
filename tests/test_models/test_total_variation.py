"""Tests for the total variation reconstruction baseline.

TV is the floor that appears in the literature, so a silently wrong TV row is
worse than none: it would make the method look good against a baseline that was
never actually solving the right problem. These tests pin the two things that
could go wrong quietly - the Fourier convention and the effect of the
regularisation weight - rather than only checking that the call returns.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import sigpy as sp
import torch

from cfm.core.registry import MODELS, RECONSTRUCTORS
from cfm.models.total_variation import TotalVariationReconstructor
from cfm.utils.fft import fft2c, ifft2c


def _phantom(h: int = 32, w: int = 32, seed: int = 0) -> torch.Tensor:
    """A smooth complex image: TV should reconstruct it, unlike white noise."""
    torch.manual_seed(seed)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, h), torch.linspace(-1, 1, w), indexing="ij")
    disc = ((yy**2 + xx**2) < 0.4).float()
    phase = 0.8 * xx
    return (disc * torch.exp(1j * phase)).reshape(1, 1, h, w).to(torch.complex64)


class TestFourierConvention:
    """sigpy is handed our k-space directly, so the two conventions must agree."""

    def test_sigpy_fft_matches_fft2c(self) -> None:
        x = _phantom()
        ours = fft2c(x).numpy()[0, 0]
        theirs = sp.fft(x.numpy()[0, 0], axes=(-2, -1))
        assert np.abs(ours - theirs).max() < 1e-5

    def test_sigpy_ifft_matches_ifft2c(self) -> None:
        k = fft2c(_phantom())
        ours = ifft2c(k).numpy()[0, 0]
        theirs = sp.ifft(k.numpy()[0, 0], axes=(-2, -1))
        assert np.abs(ours - theirs).max() < 1e-5


class TestTotalVariationReconstructor:
    def test_registered_under_both_registries(self) -> None:
        """evaluate.py reaches it through MODELS; the reconstructor API through RECONSTRUCTORS."""
        assert "total_variation" in MODELS
        assert "total_variation" in RECONSTRUCTORS

    def test_absorbs_generic_build_kwargs(self) -> None:
        """The registry build path passes channel counts to every architecture."""
        model = TotalVariationReconstructor(in_channels=3, out_channels=2)
        assert model.lamda == pytest.approx(0.005)

    @pytest.mark.parametrize("bad", [{"lamda": -0.1}, {"max_iter": 0}])
    def test_rejects_invalid_hyperparameters(self, bad: dict[str, Any]) -> None:
        with pytest.raises(ValueError):
            TotalVariationReconstructor(**bad)

    def test_output_shape_dtype_and_device_follow_the_input(self) -> None:
        x = _phantom()
        k = fft2c(x)
        mask = torch.ones(1, 1, 32, 32)
        out = TotalVariationReconstructor(max_iter=5).reconstruct(k, mask)
        assert out.shape == (1, 1, 32, 32)
        assert out.dtype == k.dtype
        assert out.device == k.device

    def test_recovers_a_smooth_image_from_full_sampling(self) -> None:
        """With every line measured and light regularisation, TV must beat the input error."""
        x = _phantom()
        k = fft2c(x)
        mask = torch.ones(1, 1, 32, 32)
        out = TotalVariationReconstructor(lamda=1e-4, max_iter=60).reconstruct(k, mask)
        rel = (out - x).abs().sum() / x.abs().sum()
        assert rel < 0.15

    def test_beats_zero_filling_on_an_undersampled_slice(self) -> None:
        """The point of the baseline: it must actually do better than the adjoint."""
        x = _phantom()
        mask = torch.zeros(1, 1, 32, 32)
        mask[..., ::2] = 1.0  # every other column
        mask[..., 12:20] = 1.0  # plus a centre band
        k = fft2c(x) * mask

        zf = ifft2c(k)
        tv = TotalVariationReconstructor(lamda=1e-3, max_iter=80).reconstruct(k, mask)

        err_zf = (zf - x).abs().sum()
        err_tv = (tv - x).abs().sum()
        assert err_tv < err_zf

    def test_larger_lamda_gives_a_smoother_image(self) -> None:
        """Sanity on the knob that will be swept: more weight, less total variation."""
        x = _phantom()
        k = fft2c(x)
        mask = torch.ones(1, 1, 32, 32)

        def total_variation(img: torch.Tensor) -> torch.Tensor:
            return img.diff(dim=-1).abs().sum() + img.diff(dim=-2).abs().sum()

        light = TotalVariationReconstructor(lamda=1e-4, max_iter=60).reconstruct(k, mask)
        heavy = TotalVariationReconstructor(lamda=5e-2, max_iter=60).reconstruct(k, mask)
        assert total_variation(heavy.abs()) < total_variation(light.abs())

    def test_is_reproducible_across_calls(self) -> None:
        """sigpy's power iteration starts from a random vector; seeding must fix it.

        Unseeded, two calls on the same slice differ by ~3e-4 on a unit-scale
        image. Small, but a table number should not move between runs.
        """
        x = _phantom()
        k = fft2c(x)
        mask = torch.ones(1, 1, 32, 32)
        model = TotalVariationReconstructor(lamda=1e-4, max_iter=30)
        assert torch.equal(model.reconstruct(k, mask), model.reconstruct(k, mask))

    def test_does_not_disturb_the_global_numpy_rng(self) -> None:
        """Seeding is an implementation detail and must not leak into the caller."""
        x = _phantom()
        k = fft2c(x)
        mask = torch.ones(1, 1, 32, 32)

        np.random.seed(1234)
        expected = np.random.rand(4)

        np.random.seed(1234)
        TotalVariationReconstructor(lamda=1e-4, max_iter=10).reconstruct(k, mask)
        after = np.random.rand(4)

        assert np.array_equal(expected, after)

    def test_slice_result_does_not_depend_on_batch_grouping(self) -> None:
        """A slice must score the same whether it was batched with others or not.

        Seeding per slice index rather than per call is what makes this hold, and
        it is what keeps a metric independent of ``evaluate.batch_size``.
        """
        a = _phantom()
        batch = torch.cat([a, a * 0.5], dim=0)
        mask = torch.ones(1, 1, 32, 32)

        model = TotalVariationReconstructor(lamda=1e-4, max_iter=30)
        together = model.reconstruct(fft2c(batch), mask)
        alone = model.reconstruct(fft2c(a), mask)
        assert torch.equal(together[0:1], alone)
