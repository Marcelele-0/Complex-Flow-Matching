"""Unit tests for the synthetic cylinder toy and its dependence knob.

The properties under test are the ones that make a sweep over ``coupling`` a
controlled experiment: both marginals must be invariant across the sweep, and
the observable dependence must actually move with the latent parameter. If the
first fails, a difference measured downstream could be a marginal effect; if the
second fails, the knob does nothing.
"""

import math

import pytest
import torch

from cyfm.data.synthetic import (
    COUPLING_PRESETS,
    CylinderToy,
    Structure,
    cylinder_prior,
)
from cyfm.metrics import circular_linear_correlation

# Two-sample KS at n = 40_000 per side has a noise floor near 0.006; 0.02 sits
# clear of it while still failing any real shift in a marginal.
_KS_TOLERANCE = 0.02
_SAMPLES = 40_000

STRUCTURES = ("spiral", "cardioid")


def _generator(seed: int) -> torch.Generator:
    """A CPU generator pinned to ``seed``."""
    return torch.Generator().manual_seed(seed)


def _ks_statistic(a: torch.Tensor, b: torch.Tensor) -> float:
    """Two-sample Kolmogorov-Smirnov statistic between 1D samples."""
    a_sorted, b_sorted = a.sort().values, b.sort().values
    grid = torch.cat([a_sorted, b_sorted]).sort().values
    cdf_a = torch.searchsorted(a_sorted, grid, right=True).float() / a.numel()
    cdf_b = torch.searchsorted(b_sorted, grid, right=True).float() / b.numel()
    return float((cdf_a - cdf_b).abs().max())


# --- Marginal invariance: the property that makes the sweep controlled ---


@pytest.mark.parametrize("structure", STRUCTURES)
@pytest.mark.parametrize("coupling", [0.5, 1.0])
def test_amplitude_marginal_invariant_under_coupling(structure: Structure, coupling: float) -> None:
    """Amplitude marginal must not move when only the dependence changes."""
    baseline, _ = CylinderToy(coupling=0.0, structure=structure).sample_polar(
        _SAMPLES, generator=_generator(0)
    )
    coupled, _ = CylinderToy(coupling=coupling, structure=structure).sample_polar(
        _SAMPLES, generator=_generator(1)
    )
    assert _ks_statistic(baseline, coupled) < _KS_TOLERANCE


@pytest.mark.parametrize("structure", STRUCTURES)
@pytest.mark.parametrize("coupling", [0.5, 1.0])
def test_phase_marginal_invariant_under_coupling(structure: Structure, coupling: float) -> None:
    """Phase marginal must not move when only the dependence changes."""
    _, baseline = CylinderToy(coupling=0.0, structure=structure).sample_polar(
        _SAMPLES, generator=_generator(0)
    )
    _, coupled = CylinderToy(coupling=coupling, structure=structure).sample_polar(
        _SAMPLES, generator=_generator(1)
    )
    assert _ks_statistic(baseline, coupled) < _KS_TOLERANCE


def test_ks_tolerance_is_above_the_noise_floor() -> None:
    """Guard the tolerance itself: two draws at one coupling must pass it."""
    toy = CylinderToy(coupling=0.5, structure="spiral")
    first, _ = toy.sample_polar(_SAMPLES, generator=_generator(2))
    second, _ = toy.sample_polar(_SAMPLES, generator=_generator(3))
    assert _ks_statistic(first, second) < _KS_TOLERANCE


# --- The knob must actually turn ---


@pytest.mark.parametrize("structure", STRUCTURES)
def test_observable_correlation_increases_with_coupling(structure: Structure) -> None:
    """Circular-linear correlation is near zero at 0 and strong at 1."""
    observed = []
    for coupling in (0.0, 0.5, 1.0):
        amplitude, phase = CylinderToy(coupling=coupling, structure=structure).sample_polar(
            _SAMPLES, generator=_generator(4)
        )
        observed.append(float(circular_linear_correlation(amplitude, phase)))

    assert observed[0] < 0.02
    assert observed[0] < observed[1] < observed[2]
    assert observed[2] > 0.8


def test_comonotone_spiral_makes_amplitude_a_function_of_phase() -> None:
    """At coupling 1 the spiral is deterministic: sorting by phase sorts amplitude."""
    amplitude, phase = CylinderToy(coupling=1.0, structure="spiral").sample_polar(
        20_000, generator=_generator(5)
    )
    steps = torch.diff(amplitude[torch.argsort(phase)])
    assert float((steps >= -1e-6).float().mean()) == pytest.approx(1.0)


def test_cardioid_is_mirror_symmetric_about_the_offset() -> None:
    """The periodic structure depends on |theta - offset|, so +d and -d agree."""
    amplitude, phase = CylinderToy(
        coupling=1.0, structure="cardioid", phase_offset=0.0
    ).sample_polar(200_000, generator=_generator(6))

    positive = amplitude[(phase > 1.0) & (phase < 1.1)]
    negative = amplitude[(phase < -1.0) & (phase > -1.1)]
    assert positive.numel() > 100
    assert negative.numel() > 100
    assert float(positive.mean()) == pytest.approx(float(negative.mean()), abs=0.02)


def test_spiral_and_cardioid_differ_at_the_branch_cut() -> None:
    """The two structures are genuinely different: only the spiral jumps at the cut."""
    generator_seed = 7
    spiral_amp, spiral_phase = CylinderToy(coupling=1.0, structure="spiral").sample_polar(
        200_000, generator=_generator(generator_seed)
    )
    cardioid_amp, cardioid_phase = CylinderToy(coupling=1.0, structure="cardioid").sample_polar(
        200_000, generator=_generator(generator_seed)
    )

    def _edge_gap(amplitude: torch.Tensor, phase: torch.Tensor) -> float:
        low = amplitude[phase < -math.pi + 0.2]
        high = amplitude[phase > math.pi - 0.2]
        return abs(float(low.mean()) - float(high.mean()))

    assert _edge_gap(spiral_amp, spiral_phase) > 0.3
    assert _edge_gap(cardioid_amp, cardioid_phase) < 0.05


# --- Shapes, dtypes, ranges, reproducibility ---


@pytest.mark.parametrize("structure", STRUCTURES)
def test_sample_polar_shapes_and_ranges(structure: Structure) -> None:
    """Polar samples are 1D, positive in amplitude and inside the principal branch."""
    amplitude, phase = CylinderToy(coupling=0.5, structure=structure).sample_polar(
        1024, generator=_generator(8)
    )
    assert amplitude.shape == (1024,)
    assert phase.shape == (1024,)
    assert amplitude.dtype is torch.float32
    assert bool((amplitude >= 0).all())
    assert bool((phase >= -math.pi).all() and (phase <= math.pi).all())


def test_sample_returns_complex_consistent_with_polar() -> None:
    """The complex view is exactly the polar view recombined."""
    toy = CylinderToy(coupling=0.5)
    amplitude, phase = toy.sample_polar(512, generator=_generator(9))
    complex_sample = toy.sample(512, generator=_generator(9))

    assert complex_sample.dtype is torch.complex64
    assert torch.allclose(complex_sample.abs(), amplitude, atol=1e-5)
    assert torch.allclose(complex_sample.angle(), phase, atol=1e-5)


def test_sampling_is_reproducible_and_seed_sensitive() -> None:
    """A fixed generator reproduces samples; a different one does not."""
    toy = CylinderToy(coupling=0.5)
    first = toy.sample(256, generator=_generator(11))
    repeat = toy.sample(256, generator=_generator(11))
    other = toy.sample(256, generator=_generator(12))

    assert torch.equal(first, repeat)
    assert not torch.equal(first, other)


def test_float64_grids_are_supported() -> None:
    """The grids and samples follow the requested dtype."""
    amplitude, phase = CylinderToy(coupling=0.5, dtype=torch.float64).sample_polar(
        256, generator=_generator(13)
    )
    assert amplitude.dtype is torch.float64
    assert phase.dtype is torch.float64


# --- Presets ---


def test_presets_cover_the_three_gate_settings() -> None:
    """The named presets are the values the gate experiments sweep."""
    assert COUPLING_PRESETS == {"independent": 0.0, "partial": 0.5, "comonotone": 1.0}


@pytest.mark.parametrize(("preset", "expected"), sorted(COUPLING_PRESETS.items()))
def test_from_preset_sets_the_coupling(preset: str, expected: float) -> None:
    """Presets map to their coupling and forward other arguments."""
    toy = CylinderToy.from_preset(preset, structure="cardioid")
    assert toy.coupling == expected
    assert toy.structure == "cardioid"


def test_from_preset_rejects_unknown_name() -> None:
    """An unknown preset names the valid ones rather than falling back."""
    with pytest.raises(ValueError, match="unknown preset"):
        CylinderToy.from_preset("strongly_correlated")


def test_from_preset_rejects_redundant_coupling() -> None:
    """Passing coupling alongside a preset is ambiguous and must raise."""
    with pytest.raises(ValueError, match="coupling is set by the preset"):
        CylinderToy.from_preset("partial", coupling=0.3)


# --- Prior ---


def test_prior_second_moment_matches_scale() -> None:
    """The circularly symmetric complex Gaussian has E|z|^2 = scale^2."""
    samples = cylinder_prior(_SAMPLES, scale=2.0, generator=_generator(14))
    assert samples.dtype is torch.complex64
    assert float((samples.abs() ** 2).mean()) == pytest.approx(4.0, rel=0.02)


def test_prior_phase_is_uniform_and_independent_of_amplitude() -> None:
    """The prior sits on the product-measure side, which the target need not."""
    samples = cylinder_prior(_SAMPLES, generator=_generator(15))
    phase = samples.angle()
    uniform = torch.empty(_SAMPLES).uniform_(-math.pi, math.pi, generator=_generator(16))

    assert _ks_statistic(phase, uniform) < _KS_TOLERANCE
    assert float(circular_linear_correlation(samples.abs(), phase)) < 0.02


# --- Circular-linear correlation ---


def test_circular_linear_correlation_is_one_for_a_deterministic_link() -> None:
    """A monotone deterministic link saturates the statistic."""
    phase = torch.linspace(-math.pi + 0.01, math.pi - 0.01, 8192)
    amplitude = 1.0 + 0.5 * torch.cos(phase)
    assert float(circular_linear_correlation(amplitude, phase)) > 0.99


def test_circular_linear_correlation_is_invariant_to_the_branch_cut() -> None:
    """Rotating every angle by a constant must not change the statistic."""
    amplitude, phase = CylinderToy(coupling=0.7).sample_polar(20_000, generator=_generator(17))
    rotated = torch.atan2(torch.sin(phase + 1.3), torch.cos(phase + 1.3))

    original = float(circular_linear_correlation(amplitude, phase))
    shifted = float(circular_linear_correlation(amplitude, rotated))
    assert original == pytest.approx(shifted, abs=1e-4)


@pytest.mark.parametrize(
    ("amplitude", "phase", "match"),
    [
        (torch.zeros(4, 2), torch.zeros(8), "must be 1D"),
        (torch.zeros(8), torch.zeros(6), "must match in length"),
        (torch.zeros(1), torch.zeros(1), "at least two samples"),
    ],
)
def test_circular_linear_correlation_validation(
    amplitude: torch.Tensor, phase: torch.Tensor, match: str
) -> None:
    """Malformed inputs raise rather than returning a meaningless number."""
    with pytest.raises(ValueError, match=match):
        circular_linear_correlation(amplitude, phase)


# --- Constructor and sampler validation ---


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"coupling": -0.1}, "coupling must be in"),
        ({"coupling": 1.5}, "coupling must be in"),
        ({"structure": "torus"}, "structure must be"),
        ({"grid_size": 8}, "grid_size must be >= 16"),
        ({"phase_modes": ()}, "phase_modes must not be empty"),
        ({"amplitude_modes": ()}, "amplitude_modes must not be empty"),
        ({"phase_modes": ((0.0, 4.0, 0.0),)}, "phase mode weight must be positive"),
        ({"phase_modes": ((0.0, -1.0, 1.0),)}, "phase mode concentration must be"),
        ({"amplitude_modes": ((1.0, 0.1, 0.0),)}, "amplitude mode weight must be positive"),
        ({"amplitude_modes": ((1.0, 0.0, 1.0),)}, "amplitude mode std must be positive"),
    ],
)
def test_constructor_validation(kwargs: dict[str, object], match: str) -> None:
    """Every constructor argument is checked at build time, not at sample time."""
    with pytest.raises(ValueError, match=match):
        CylinderToy(**kwargs)  # type: ignore[arg-type]


def test_sample_rejects_non_positive_n() -> None:
    """Asking for zero samples is a caller error, not an empty tensor."""
    with pytest.raises(ValueError, match="n must be positive"):
        CylinderToy().sample_polar(0)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"n": 0}, "n must be positive"), ({"n": 8, "scale": 0.0}, "scale must be positive")],
)
def test_prior_validation(kwargs: dict[str, object], match: str) -> None:
    """The prior validates its own arguments."""
    with pytest.raises(ValueError, match=match):
        cylinder_prior(**kwargs)  # type: ignore[arg-type]
