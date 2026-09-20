"""Tests for the spatially correlated prior.

The prior exists to lower the effective dimension minibatch OT sees at field
scale without changing what any single coefficient looks like. The tests pin
exactly that: the pointwise law is the white prior's, neighbouring coefficients
become dependent, and both geometries draw the same field from one seed -- which
is what keeps the prior from becoming a confound between them.
"""

import math

import pytest
import torch
from omegaconf import OmegaConf

from cyfm.manifolds import build_manifold
from cyfm.manifolds.cylindrical import sample_cylindrical_noise
from cyfm.utils.random_fields import smooth_standard_normals

_CPU = torch.device("cpu")


def _manifold(name: str, **keys: object) -> object:
    """Build a manifold the way the entry points do."""
    return build_manifold(
        OmegaConf.create({"manifold": {"name": name, **keys}, "training": {"loss": {}}})
    )


def _draw(manifold: object, seed: int, batch: int = 256, side: int = 48) -> torch.Tensor:
    """One prior batch as complex fields of shape ``[B, H, W]``."""
    generator = torch.Generator().manual_seed(seed)
    state = manifold.sample_noise(batch, side, side, _CPU, generator=generator)  # type: ignore[attr-defined]
    return manifold.to_complex(state).reshape(batch, side, side)  # type: ignore[attr-defined]


def _lag_one(values: torch.Tensor) -> float:
    """Correlation between horizontally adjacent entries."""
    left = values[..., :-1].reshape(-1)
    right = values[..., 1:].reshape(-1)
    return float(torch.corrcoef(torch.stack([left, right]))[0, 1])


def _circular_concentration(phase: torch.Tensor) -> float:
    """Mean resultant length: 0 for a uniform angle, 1 for a point mass."""
    return float(torch.sqrt(torch.cos(phase).mean() ** 2 + torch.sin(phase).mean() ** 2))


# --- The smoother itself ---


def test_smoothing_keeps_every_entry_standard_normal() -> None:
    """Each entry stays standard normal, or every law built on the latents drifts.

    Checked per pixel across independent fields rather than pooled over space: a
    smoothed field has far fewer effective samples than pixels, so a pooled mean
    scatters by about a standard error per handful of correlation areas, and a
    tolerance calibrated for white noise fails for a reason unrelated to the law.
    """
    noise = torch.randn(20_000, 1, 16, 16, generator=torch.Generator().manual_seed(0))
    entry = smooth_standard_normals(noise, 4.0)[:, 0, 5, 5]
    assert float(entry.mean()) == pytest.approx(0.0, abs=0.03)
    assert float(entry.var()) == pytest.approx(1.0, abs=0.03)


def test_unit_variance_survives_a_correlation_longer_than_the_field() -> None:
    """The case a spatial kernel gets wrong: it wraps onto itself and inflates variance.

    Measured on the earlier spatial implementation at 64x64 and length 32: variance
    2.628. The spectral one is exact at any length, and the field is then nearly
    constant, as a length that long demands.
    """
    noise = torch.randn(20_000, 1, 16, 16, generator=torch.Generator().manual_seed(1))
    smoothed = smooth_standard_normals(noise, 32.0)
    assert float(smoothed[:, 0, 5, 5].var()) == pytest.approx(1.0, abs=0.03)
    assert _lag_one(smoothed[:, 0]) > 0.99


def test_vanishing_length_recovers_white_noise() -> None:
    """The family is continuous at zero: a tiny length leaves the field untouched."""
    noise = torch.randn(4, 1, 16, 16, generator=torch.Generator().manual_seed(2))
    assert torch.allclose(smooth_standard_normals(noise, 1e-4), noise, atol=1e-4)


def test_non_positive_length_returns_the_input() -> None:
    """A length of zero means white noise, and nothing is computed."""
    noise = torch.randn(2, 1, 8, 8)
    assert smooth_standard_normals(noise, 0.0) is noise


# --- The pointwise law is the white prior's ---


def test_correlated_modulus_is_still_uniform_on_the_unit_interval() -> None:
    """Uniform on [0, 1]: mean 1/2, standard deviation 1/sqrt(12)."""
    modulus = _draw(_manifold("cylindrical", spatial_correlation=4.0), seed=1).abs()
    assert float(modulus.mean()) == pytest.approx(0.5, abs=0.03)
    assert float(modulus.std()) == pytest.approx(1.0 / math.sqrt(12.0), abs=0.03)


def test_correlated_phase_is_still_uniform_on_the_circle() -> None:
    """No preferred direction, and the mean |phase| of a uniform angle."""
    phase = _draw(_manifold("cylindrical", spatial_correlation=4.0), seed=2).angle()
    assert _circular_concentration(phase) < 0.05
    assert float(phase.abs().mean()) == pytest.approx(math.pi / 2, abs=0.1)


def test_default_path_is_unchanged() -> None:
    """Leaving the knob null must reproduce the white prior bit for bit."""
    manifold = _manifold("cylindrical")
    drawn = manifold.sample_noise(  # type: ignore[attr-defined]
        4, 8, 8, _CPU, generator=torch.Generator().manual_seed(3)
    )
    reference = sample_cylindrical_noise(4, 8, 8, _CPU, torch.Generator().manual_seed(3))
    assert torch.equal(drawn, reference)


# --- Neighbouring coefficients become dependent ---


@pytest.mark.parametrize("geometry", ["cylindrical", "euclidean"])
def test_only_the_correlated_prior_has_spatial_structure(geometry: str) -> None:
    """This is the one thing the knob is allowed to change."""
    white = _draw(_manifold(geometry), seed=4).abs()
    smooth = _draw(_manifold(geometry, spatial_correlation=4.0), seed=4).abs()
    assert abs(_lag_one(white)) < 0.05
    assert _lag_one(smooth) > 0.8


def test_correlation_composes_with_a_concentrated_phase() -> None:
    """Both knobs together: a narrow phase that is also spatially smooth."""
    field = _draw(_manifold("cylindrical", spatial_correlation=4.0, phase_spread=0.5), seed=5)
    assert _circular_concentration(field.angle()) > 0.5
    assert _lag_one(field.abs()) > 0.8


# --- The prior is not a confound between the arms ---


def test_both_geometries_draw_the_same_smooth_field_from_one_seed() -> None:
    """Sample-matched, not merely distribution-matched."""
    cylindrical = _draw(_manifold("cylindrical", spatial_correlation=4.0), seed=6)
    euclidean = _draw(_manifold("euclidean", spatial_correlation=4.0), seed=6)
    assert torch.allclose(cylindrical, euclidean, atol=1e-5)


# --- Validation ---


@pytest.mark.parametrize("geometry", ["cylindrical", "euclidean"])
@pytest.mark.parametrize("length", [0.0, -2.0])
def test_non_positive_correlation_is_rejected(geometry: str, length: float) -> None:
    """Null means white; zero or negative is a configuration mistake."""
    with pytest.raises(ValueError, match="spatial_correlation must be positive"):
        _manifold(geometry, spatial_correlation=length)


def test_gaussian_prior_refuses_spatial_correlation() -> None:
    """The correlated draw reproduces the unit-disc law, which N(0, I) is not."""
    with pytest.raises(ValueError, match="not defined for noise_prior='gaussian'"):
        _manifold("euclidean", noise_prior="gaussian", spatial_correlation=4.0)
