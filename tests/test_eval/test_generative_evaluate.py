"""Unit tests for the generative evaluation metrics.

These metrics replaced a paired set, so nothing about them is inherited or
already covered. Each test below pins the property the metric exists for: that it
separates distributions it should separate, and stays near zero on ones it should
not.
"""

import math

import pytest
import torch

from cfm.data.toy_dataset import CylinderToyFieldDataset, CylinderToyIIDDataset
from cfm.evaluate import (
    _subsample,
    distributional_metrics,
    format_table,
    spatial_lag_one,
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


# --- Subsampling and reporting ---


def test_subsample_is_a_noop_below_the_limit() -> None:
    """Under the cap nothing is dropped, so no number is silently on a subset."""
    values = torch.arange(10.0)
    assert torch.equal(_subsample(values, 20, _generator()), values)


def test_subsample_takes_exactly_the_limit_above_it() -> None:
    """Above the cap the count is exact and the entries come from the input."""
    values = torch.arange(100.0)
    taken = _subsample(values, 20, _generator())
    assert taken.numel() == 20
    assert set(taken.tolist()).issubset(set(values.tolist()))


def test_format_table_renders_one_row_per_step_count() -> None:
    """The sweep table is the artefact the NFE claim is read off."""
    rendered = format_table([(1, {"a": 0.5}), (100, {"a": 0.25})], ["a"])
    lines = rendered.splitlines()
    assert lines[0].split() == ["steps", "a"]
    assert lines[2].split() == ["1", "0.50000"]
    assert lines[3].split() == ["100", "0.25000"]


# --- Straightness is scored under the pairing training used ---


def _straightness_setup() -> tuple[object, object, torch.Tensor]:
    """A small cylindrical model and a data batch in its state representation."""
    from omegaconf import OmegaConf

    from cfm.manifolds import build_manifold
    from cfm.models.pointwise_mlp import PointwiseVelocityMLP

    torch.manual_seed(0)
    manifold = build_manifold(
        OmegaConf.create({"manifold": {"name": "cylindrical"}, "training": {"loss": {}}})
    )
    model = PointwiseVelocityMLP(in_channels=3, base_channels=16, depth=1).eval()
    dataset = CylinderToyIIDDataset(size=8, crop_size=(8, 8))
    data = manifold.from_complex(torch.stack([dataset[i] for i in range(8)]))
    return manifold, model, data


def test_independent_coupling_reproduces_the_uncoupled_straightness() -> None:
    """Passing the baseline coupling explicitly must not change the number."""
    from cfm.evaluate import straightness
    from cfm.flow.coupling import IndependentCoupling

    manifold, model, data = _straightness_setup()
    cpu = torch.device("cpu")
    bare = straightness(model, manifold, data, cpu, _generator(3))  # type: ignore[arg-type]
    explicit = straightness(
        model,  # type: ignore[arg-type]
        manifold,
        data,
        cpu,
        _generator(3),
        IndependentCoupling(),
    )
    assert bare == explicit


def test_ot_coupling_changes_the_pairs_that_are_scored() -> None:
    """An OT-trained model is scored on OT pairs, which are not the drawn pairs."""
    from cfm.evaluate import straightness
    from cfm.flow.coupling import OptimalTransportCoupling

    manifold, model, data = _straightness_setup()
    cpu = torch.device("cpu")
    independent = straightness(model, manifold, data, cpu, _generator(4))  # type: ignore[arg-type]
    coupled = straightness(
        model,  # type: ignore[arg-type]
        manifold,
        data,
        cpu,
        _generator(4),
        OptimalTransportCoupling(),
    )
    assert math.isfinite(coupled)
    assert coupled != independent


# --- The reference lives in the training domain ---


@pytest.mark.parametrize("geometry", ["cylindrical", "euclidean"])
def test_reference_goes_through_the_training_normalisation(geometry: str) -> None:
    """Every reference field ends up with peak modulus one, as training sees it.

    Scoring against raw fields penalised a model for the normalisation itself.
    """
    from omegaconf import OmegaConf

    from cfm.evaluate import training_pipeline
    from cfm.manifolds import build_manifold

    manifold = build_manifold(
        OmegaConf.create({"manifold": {"name": geometry}, "training": {"loss": {}}})
    )
    pipeline = training_pipeline({"crop_size": [16, 16]}, manifold)
    dataset = CylinderToyIIDDataset(size=4, crop_size=(16, 16))

    for index in range(4):
        raw = dataset[index]
        field = manifold.to_complex(pipeline(raw.clone()).unsqueeze(0))
        assert field.shape == (1, 1, 16, 16)
        assert float(field.abs().max()) == pytest.approx(1.0, abs=1e-5)
        assert float(raw.abs().max()) > 1.0
