"""Unit tests for the distributional metrics in cfm.utils.metrics.

Each test pins the property the metric exists for: that it separates
distributions it should separate, and stays near zero on ones it should not.
"""

import math

import pytest
import torch

from cfm.data.toy_dataset import CylinderToyFieldDataset, CylinderToyIIDDataset
from cfm.utils.metrics import (
    distributional_metrics,
    radial_power_spectrum,
    radial_spectrum_gap,
    spatial_lag_one,
    subsample,
)


def _fields(dataset: object, count: int) -> torch.Tensor:
    """Stack the first ``count`` fields of a dataset into one batch."""
    return torch.stack([dataset[i] for i in range(count)])  # type: ignore[index]


def _generator(seed: int = 0) -> torch.Generator:
    """A CPU generator pinned to ``seed``."""
    return torch.Generator().manual_seed(seed)


# --- Spatial structure ---


def test_lag_one_separates_the_two_toy_variants() -> None:
    """This metric is the only one that can see spatial structure at all."""
    iid = _fields(CylinderToyIIDDataset(size=8, crop_size=(64, 64)), 8)
    field = _fields(CylinderToyFieldDataset(size=8, crop_size=(64, 64)), 8)

    assert abs(spatial_lag_one(iid)) < 0.1
    assert spatial_lag_one(field) > 0.8


def test_lag_one_is_zero_for_a_constant_field() -> None:
    """No variation means no correlation to report, rather than a NaN."""
    assert spatial_lag_one(torch.ones(2, 1, 8, 8, dtype=torch.complex64)) == 0.0


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        (torch.ones(2, 1, 8, 8), "expected complex"),
        (torch.ones(2, 8, 8, dtype=torch.complex64), "expected complex"),
        (torch.ones(2, 1, 8, 1, dtype=torch.complex64), "width of at least two"),
    ],
)
def test_lag_one_validation(fields: torch.Tensor, match: str) -> None:
    """A malformed batch raises rather than producing a meaningless number."""
    with pytest.raises(ValueError, match=match):
        spatial_lag_one(fields)


# --- Distributional metrics ---


def test_metrics_are_near_zero_for_two_draws_of_one_distribution() -> None:
    """The floor: a perfect model should score here, not at zero exactly."""
    first = _fields(CylinderToyIIDDataset(size=16, crop_size=(64, 64), seed=0), 16)
    second = _fields(CylinderToyIIDDataset(size=16, crop_size=(64, 64), seed=9000), 16)
    values = distributional_metrics(first, second, 128, _generator())

    assert values["sliced_w2_complex"] < 0.05
    assert values["w2_amplitude"] < 0.05
    assert values["w2_phase_circular"] < 0.1
    assert values["dependence_gap"] < 0.05
    assert values["spatial_lag1_gap"] < 0.05


def test_metrics_separate_distributions_that_differ_in_dependence() -> None:
    """Amplitude-phase dependence is the axis the datasets sweep."""
    independent = _fields(CylinderToyIIDDataset(coupling=0.0, size=16, crop_size=(64, 64)), 16)
    comonotone = _fields(CylinderToyIIDDataset(coupling=1.0, size=16, crop_size=(64, 64)), 16)
    values = distributional_metrics(independent, comonotone, 128, _generator())

    assert values["dependence_gap"] > 0.5
    assert values["sliced_w2_complex"] > 0.1


def test_marginals_survive_a_dependence_change_and_the_gap_still_fires() -> None:
    """The point of keeping a dependence metric beside the marginal ones.

    The toy holds both marginals fixed across coupling, so a report built only
    from marginal transport would call these two distributions identical.
    """
    independent = _fields(CylinderToyIIDDataset(coupling=0.0, size=16, crop_size=(64, 64)), 16)
    comonotone = _fields(CylinderToyIIDDataset(coupling=1.0, size=16, crop_size=(64, 64)), 16)
    values = distributional_metrics(independent, comonotone, 128, _generator())

    assert values["w2_amplitude"] < 0.05
    assert values["w2_phase_circular"] < 0.1
    assert values["dependence_gap"] > 0.5


def test_metrics_separate_distributions_that_differ_only_spatially() -> None:
    """And the point of keeping a spatial metric beside the pooled ones."""
    iid = _fields(CylinderToyIIDDataset(size=16, crop_size=(64, 64)), 16)
    field = _fields(CylinderToyFieldDataset(size=16, crop_size=(64, 64)), 16)
    values = distributional_metrics(iid, field, 128, _generator())

    assert values["spatial_lag1_gap"] > 0.8
    assert values["sliced_w2_complex"] < 0.05


def test_radial_spectrum_separates_white_from_correlated_fields() -> None:
    """The property the metric exists for: coarse-versus-fine balance.

    An i.i.d. field is white, so its power is spread evenly across the band; a
    correlated one is red, concentrated at low frequency. Nothing else in the
    report distinguishes those two spectra.
    """
    iid = _fields(CylinderToyIIDDataset(size=16, crop_size=(64, 64)), 16)
    field = _fields(CylinderToyFieldDataset(size=16, crop_size=(64, 64)), 16)

    assert radial_spectrum_gap(iid, field) > 0.3
    assert radial_spectrum_gap(iid, iid) == pytest.approx(0.0, abs=1e-6)


def test_radial_spectrum_floor_is_well_below_a_real_difference() -> None:
    """The number a reported gap has to be read against.

    Comparing a batch with itself is zero by construction and says nothing. Two
    independent draws of one distribution give the finite-sample floor, and a
    gap is only evidence of a spectral difference if it clears that.
    """
    first = _fields(CylinderToyIIDDataset(size=16, crop_size=(64, 64), seed=0), 16)
    second = _fields(CylinderToyIIDDataset(size=16, crop_size=(64, 64), seed=1), 16)
    correlated = _fields(CylinderToyFieldDataset(size=16, crop_size=(64, 64)), 16)

    floor = radial_spectrum_gap(first, second)
    signal = radial_spectrum_gap(first, correlated)
    assert floor < 0.1
    assert signal > 3.0 * floor


def test_radial_spectrum_ignores_a_pure_change_of_scale() -> None:
    """Overall power belongs to w2_amplitude; this metric reports shape only."""
    fields = _fields(CylinderToyFieldDataset(size=8, crop_size=(32, 32)), 8)
    assert radial_spectrum_gap(fields, fields * 4.0) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    "degenerate",
    [
        pytest.param(torch.full((4, 1, 32, 32), float("nan"), dtype=torch.complex64), id="nan"),
        pytest.param(torch.full((4, 1, 32, 32), float("inf"), dtype=torch.complex64), id="inf"),
        pytest.param(torch.zeros(4, 1, 32, 32, dtype=torch.complex64), id="zeros"),
        pytest.param(torch.full((4, 1, 32, 32), 0.5 + 0j, dtype=torch.complex64), id="constant"),
    ],
)
def test_radial_spectrum_refuses_to_score_a_diverged_sampler(degenerate: torch.Tensor) -> None:
    """A field with no usable bin must not earn the value meaning "identical".

    Each of these leaves the usable mask empty. Returning 0.0 there would put a
    perfect spectral match in metrics.json for the one run that diverged -- on
    the metric added to catch exactly that -- so the empty case is nan.
    """
    torch.manual_seed(0)
    reference = torch.randn(8, 1, 32, 32, dtype=torch.complex64)
    assert math.isnan(radial_spectrum_gap(degenerate, reference))


def test_radial_spectrum_profile_is_a_ring_average() -> None:
    """Every bin averages a full ring, so an isotropic field gives a flat profile."""
    torch.manual_seed(0)
    white = torch.randn(8, 1, 32, 32, dtype=torch.complex64)
    profile = radial_power_spectrum(white)

    assert profile.shape == (16,)
    assert bool((profile[1:] > 0).all())
    # Flat to within sampling noise: no ring carries an order of magnitude more
    # power than another, which a mis-binned or corner-folded profile would show.
    assert float(profile[1:].max() / profile[1:].min()) < 4.0


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        (torch.ones(2, 1, 8, 8), r"\[B, 1, H, W\]"),
        (torch.ones(2, 1, 3, 3, dtype=torch.complex64), "four samples"),
    ],
)
def test_radial_spectrum_validation(fields: torch.Tensor, match: str) -> None:
    """Shapes too small for a ring average fail loudly rather than return noise."""
    with pytest.raises(ValueError, match=match):
        radial_power_spectrum(fields)


def test_metrics_are_reproducible_under_a_fixed_generator() -> None:
    """A reported number has to be reproducible from the seed in the report."""
    first = _fields(CylinderToyIIDDataset(size=8, crop_size=(32, 32), seed=0), 8)
    second = _fields(CylinderToyIIDDataset(size=8, crop_size=(32, 32), seed=1), 8)

    one = distributional_metrics(first, second, 64, _generator(5))
    two = distributional_metrics(first, second, 64, _generator(5))
    assert one == two


def test_metrics_accept_unequal_batch_sizes() -> None:
    """Generated and reference counts need not match; the smaller sets the budget."""
    small = _fields(CylinderToyIIDDataset(size=4, crop_size=(32, 32), seed=0), 4)
    large = _fields(CylinderToyIIDDataset(size=12, crop_size=(32, 32), seed=1), 12)
    values = distributional_metrics(small, large, 64, _generator())
    assert math.isfinite(values["sliced_w2_complex"])


@pytest.mark.parametrize(
    ("generated", "reference", "match"),
    [
        (
            torch.ones(2, 8, 8, dtype=torch.complex64),
            torch.ones(2, 1, 8, 8, dtype=torch.complex64),
            r"\[B, 1, H, W\]",
        ),
        (torch.ones(2, 1, 8, 8), torch.ones(2, 1, 8, 8), "must be complex"),
    ],
)
def test_metrics_validation(generated: torch.Tensor, reference: torch.Tensor, match: str) -> None:
    """Shape and dtype mistakes fail here rather than deep inside a solver."""
    with pytest.raises(ValueError, match=match):
        distributional_metrics(generated, reference, 16, _generator())


# --- Subsampling ---


def test_subsample_is_a_noop_below_the_limit() -> None:
    """Under the cap nothing is dropped, so no number is silently on a subset."""
    values = torch.arange(10.0)
    assert torch.equal(subsample(values, 20, _generator()), values)


def test_subsample_takes_exactly_the_limit_above_it() -> None:
    """Above the cap the count is exact and the entries come from the input."""
    values = torch.arange(100.0)
    taken = subsample(values, 20, _generator())
    assert taken.numel() == 20
    assert set(taken.tolist()).issubset(set(values.tolist()))
