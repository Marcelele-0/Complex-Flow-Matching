"""Unit tests for the synthetic field datasets.

The load-bearing property is that the two variants differ in spatial correlation
and in nothing else. If their pointwise laws drift apart, the pair stops being an
ablation and any architecture comparison run on it becomes uninterpretable.
"""

import pytest
import torch
from omegaconf import OmegaConf

from cyfm.core.registry import DATASETS
from cyfm.data import build_dataset
from cyfm.data.synthetic import circular_linear_correlation
from cyfm.data.toy import CylinderToyFieldDataset, CylinderToyIIDDataset

# A spatially correlated field has far fewer effective independent draws than it
# has coefficients, so its own seed-to-seed KS is an order of magnitude above the
# iid variant's. Comparing against a fixed constant would therefore fail for a
# reason that has nothing to do with the marginals; every marginal test below
# measures that floor and compares against it.
_FLOOR_MULTIPLE = 2.0


def _pooled(dataset: object, count: int = 32) -> torch.Tensor:
    """Every coefficient of the first ``count`` samples, as one complex vector."""
    return torch.cat([dataset[i].reshape(-1) for i in range(count)])  # type: ignore[index]


def _ks(first: torch.Tensor, second: torch.Tensor) -> float:
    """Two-sample Kolmogorov-Smirnov statistic."""
    a, b = first.sort().values, second.sort().values
    grid = torch.cat([a, b]).sort().values
    return float(
        (
            torch.searchsorted(a, grid, right=True).float() / a.numel()
            - torch.searchsorted(b, grid, right=True).float() / b.numel()
        )
        .abs()
        .max()
    )


def _lag_one(dataset: object, index: int = 0) -> float:
    """Correlation between horizontally adjacent amplitudes of one field."""
    field = dataset[index].reshape(dataset.height, dataset.width).abs()  # type: ignore[index,attr-defined]
    pair = torch.stack([field[:, :-1].reshape(-1), field[:, 1:].reshape(-1)])
    return float(torch.corrcoef(pair)[0, 1])


# --- Registry and config ---


def test_both_variants_are_registered() -> None:
    """Registration is what makes a dataset reachable from dataset=<name>."""
    assert DATASETS.get("cylinder_toy_iid") is CylinderToyIIDDataset
    assert DATASETS.get("cylinder_toy_field") is CylinderToyFieldDataset


@pytest.mark.parametrize("name", ["cylinder_toy_iid", "cylinder_toy_field"])
def test_shipped_config_builds(name: str) -> None:
    """The shipped config resolves through the same path the entry points use."""
    dataset = build_dataset(OmegaConf.load(f"conf/dataset/{name}.yaml"))
    sample = dataset[0]
    assert isinstance(sample, torch.Tensor)
    assert sample.is_complex()
    assert sample.shape == (1, 64, 64)


# --- The pair differs in spatial correlation and nothing else ---


def _marginal_gap_against_floor(coupling: float, part: str) -> tuple[float, float]:
    """Cross-variant KS for one marginal, beside the field variant's own floor.

    Returns:
        ``(cross_variant, within_field)``. The second is what sampling noise
        alone produces between two draws of the *same* variant, and is the only
        scale against which the first means anything.
    """
    take = (lambda z: z.abs()) if part == "amplitude" else (lambda z: z.angle())
    iid = _pooled(CylinderToyIIDDataset(coupling=coupling, size=32, crop_size=(64, 64)))
    field = _pooled(CylinderToyFieldDataset(coupling=coupling, size=32, crop_size=(64, 64)))
    other = _pooled(
        CylinderToyFieldDataset(coupling=coupling, size=32, crop_size=(64, 64), seed=10_000)
    )
    return _ks(take(iid), take(field)), _ks(take(field), take(other))


@pytest.mark.parametrize("coupling", [0.0, 1.0])
def test_variants_share_the_pointwise_amplitude_law(coupling: float) -> None:
    """Smoothing the latents must not move the amplitude marginal."""
    cross, floor = _marginal_gap_against_floor(coupling, "amplitude")
    assert cross < _FLOOR_MULTIPLE * floor


@pytest.mark.parametrize("coupling", [0.0, 1.0])
def test_variants_share_the_pointwise_phase_law(coupling: float) -> None:
    """Nor the phase marginal, which a direct smoothing would wreck at the cut."""
    cross, floor = _marginal_gap_against_floor(coupling, "phase")
    assert cross < _FLOOR_MULTIPLE * floor


def test_the_marginal_floor_is_much_tighter_for_the_iid_variant() -> None:
    """Guards the calibration itself: the floor above is a property of the field.

    If this ever stops holding, the self-calibrating tests above have gone slack
    and would pass on a genuinely shifted marginal.
    """
    first = _pooled(CylinderToyIIDDataset(size=32, crop_size=(64, 64), seed=0))
    second = _pooled(CylinderToyIIDDataset(size=32, crop_size=(64, 64), seed=10_000))
    field_first = _pooled(CylinderToyFieldDataset(size=32, crop_size=(64, 64), seed=0))
    field_second = _pooled(CylinderToyFieldDataset(size=32, crop_size=(64, 64), seed=10_000))

    assert _ks(first.abs(), second.abs()) < 0.005
    assert _ks(field_first.abs(), field_second.abs()) > 0.005


@pytest.mark.parametrize("coupling", [0.0, 0.5, 1.0])
def test_variants_share_the_pointwise_dependence(coupling: float) -> None:
    """The amplitude-phase coupling is the axis under study; it must not drift."""
    iid = _pooled(CylinderToyIIDDataset(coupling=coupling, size=32, crop_size=(64, 64)))
    field = _pooled(CylinderToyFieldDataset(coupling=coupling, size=32, crop_size=(64, 64)))
    assert float(circular_linear_correlation(iid.abs(), iid.angle())) == pytest.approx(
        float(circular_linear_correlation(field.abs(), field.angle())), abs=0.05
    )


def test_only_the_field_variant_has_spatial_structure() -> None:
    """This is the difference the pair exists to isolate."""
    assert abs(_lag_one(CylinderToyIIDDataset(size=4, crop_size=(64, 64)))) < 0.1
    assert _lag_one(CylinderToyFieldDataset(size=4, crop_size=(64, 64))) > 0.8


def test_zero_correlation_length_reproduces_the_iid_variant() -> None:
    """The control is exactly the structured variant's degenerate case."""
    iid = CylinderToyIIDDataset(coupling=0.5, size=4, crop_size=(32, 32), seed=3)
    field = CylinderToyFieldDataset(
        coupling=0.5, size=4, crop_size=(32, 32), seed=3, correlation_length=0.0
    )
    assert torch.allclose(iid[0], field[0], atol=1e-6)


def test_longer_correlation_length_smooths_further() -> None:
    """The knob is monotone, so a sweep over it is interpretable."""
    short = _lag_one(CylinderToyFieldDataset(size=4, crop_size=(64, 64), correlation_length=1.0))
    long = _lag_one(CylinderToyFieldDataset(size=4, crop_size=(64, 64), correlation_length=8.0))
    assert short < long


# --- Interface and reproducibility ---


@pytest.mark.parametrize(
    "dataset",
    [
        CylinderToyIIDDataset(size=7, crop_size=(32, 48)),
        CylinderToyFieldDataset(size=7, crop_size=(32, 48)),
    ],
)
def test_dataset_interface(dataset: object) -> None:
    """Length, shape and the slice_map the entry points read for sample names."""
    assert len(dataset) == 7  # type: ignore[arg-type]
    assert dataset[0].shape == (1, 32, 48)  # type: ignore[index]
    assert dataset.slice_map[3] == ("synthetic", 3)  # type: ignore[attr-defined]


def test_samples_are_stable_across_calls_and_vary_across_indices() -> None:
    """Sample i is a function of seed + i, so epochs and workers agree."""
    dataset = CylinderToyIIDDataset(size=4, crop_size=(16, 16), seed=11)
    assert torch.equal(dataset[2], dataset[2])
    assert not torch.equal(dataset[2], dataset[3])


def test_seed_changes_the_draw() -> None:
    """A different seed is the only held-out claim an analytic dataset supports."""
    first = CylinderToyIIDDataset(size=2, crop_size=(16, 16), seed=0)[0]
    second = CylinderToyIIDDataset(size=2, crop_size=(16, 16), seed=1)[0]
    assert not torch.equal(first, second)


def test_index_out_of_range_raises() -> None:
    """Silent wrap-around would make an epoch quietly shorter than it claims."""
    dataset = CylinderToyIIDDataset(size=3, crop_size=(16, 16))
    with pytest.raises(IndexError, match="out of range"):
        dataset[3]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"size": 0}, "size must be positive"),
        ({"crop_size": (0, 16)}, "crop_size must be two positive integers"),
        ({"crop_size": (16,)}, "crop_size must be two positive integers"),
    ],
)
def test_constructor_validation(kwargs: dict[str, object], match: str) -> None:
    """Bad configuration fails at build time, not mid-epoch."""
    with pytest.raises(ValueError, match=match):
        CylinderToyIIDDataset(**kwargs)  # type: ignore[arg-type]


def test_negative_correlation_length_raises() -> None:
    """A negative sigma has no meaning for the smoothing kernel."""
    with pytest.raises(ValueError, match="correlation_length must be non-negative"):
        CylinderToyFieldDataset(correlation_length=-1.0)


def test_correlation_length_beyond_the_field_is_handled() -> None:
    """A sigma larger than the field is legitimate: it asks for a nearly constant field.

    It first raised from torch's circular padding, and once that was clamped it
    silently broke unit variance through kernel aliasing. The spectral smoother
    handles it exactly, so the field is finite and its latents stay standard normal.
    """
    dataset = CylinderToyFieldDataset(size=2, crop_size=(16, 16), correlation_length=64.0)
    sample = dataset[0]
    assert sample.shape == (1, 16, 16)
    assert bool(torch.isfinite(sample.abs()).all())
