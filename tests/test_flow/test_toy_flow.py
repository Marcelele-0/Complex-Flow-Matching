"""Unit tests for the two-dimensional flow-matching scaffolding.

The properties that matter here are the confound controls the gate depends on:
the two arms must receive the same prior, the same input encoding and the same
parameter count, and the scaffolding must drive the *production* bridges and
solvers rather than a reimplementation of them.
"""

import math

import pytest
import torch

from cfm.data.synthetic import CylinderToy, cylinder_prior
from cfm.flow.bridge import GeodesicFlowBridge
from cfm.flow.euclidean_bridge import LinearFlowBridge
from cfm.flow.euclidean_solver import EuclideanODESolver
from cfm.flow.solver import CylindricalODESolver
from cfm.flow.toy_flow import (
    Geometry,
    ToyVelocityField,
    build_bridge,
    build_solver,
    from_state,
    straightness,
    to_state,
)

GEOMETRIES = ("euclidean", "cylindrical")


def _samples(n: int = 64, seed: int = 0) -> torch.Tensor:
    """A reproducible complex batch."""
    return CylinderToy(coupling=0.5).sample(n, generator=torch.Generator().manual_seed(seed))


# --- State round trip ---


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_state_round_trip_is_exact(geometry: Geometry) -> None:
    """Lifting to a state and projecting back recovers the sample."""
    samples = _samples()
    recovered = from_state(to_state(samples, geometry), geometry)
    assert torch.allclose(recovered, samples, atol=1e-6)


@pytest.mark.parametrize(("geometry", "channels"), [("euclidean", 2), ("cylindrical", 3)])
def test_state_shapes(geometry: Geometry, channels: int) -> None:
    """States are [n, C, 1, 1], which is what the production modules expect."""
    assert to_state(_samples(8), geometry).shape == (8, channels, 1, 1)


def test_cylindrical_state_lies_on_the_manifold() -> None:
    """The angular channels are a unit vector, as the solver assumes."""
    state = to_state(_samples(), "cylindrical")
    norm = state[:, 1, 0, 0] ** 2 + state[:, 2, 0, 0] ** 2
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)


@pytest.mark.parametrize(
    ("samples", "geometry", "match"),
    [
        (torch.zeros(4, 2, dtype=torch.complex64), "euclidean", "must be 1D"),
        (torch.zeros(4), "euclidean", "must be a complex tensor"),
        (torch.zeros(4, dtype=torch.complex64), "torus", "unknown geometry"),
    ],
)
def test_to_state_validation(samples: torch.Tensor, geometry: str, match: str) -> None:
    """Malformed input raises rather than producing a silently wrong state."""
    with pytest.raises(ValueError, match=match):
        to_state(samples, geometry)  # type: ignore[arg-type]


def test_from_state_rejects_a_channel_mismatch() -> None:
    """A euclidean state passed as cylindrical is caught, not reinterpreted."""
    with pytest.raises(ValueError, match="cylindrical state needs 3 channels"):
        from_state(to_state(_samples(8), "euclidean"), "cylindrical")


# --- The scaffolding drives production code ---


def test_builders_return_the_production_classes() -> None:
    """A gate must measure the method, not a reimplementation of it."""
    assert isinstance(build_bridge("euclidean"), LinearFlowBridge)
    assert isinstance(build_bridge("cylindrical"), GeodesicFlowBridge)
    assert isinstance(build_solver("euclidean", 4), EuclideanODESolver)
    assert isinstance(build_solver("cylindrical", 4), CylindricalODESolver)


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_production_bridge_accepts_the_toy_shapes(geometry: Geometry) -> None:
    """The bridges run unchanged at H = W = 1."""
    start = to_state(cylinder_prior(16, generator=torch.Generator().manual_seed(1)), geometry)
    end = to_state(_samples(16, seed=2), geometry)
    interpolated, velocity = build_bridge(geometry).forward(start, end, torch.rand(16, 1, 1, 1))

    assert interpolated.shape == start.shape
    assert velocity.shape == (16, 2, 1, 1)
    assert bool(torch.isfinite(velocity).all())


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_bridge_endpoints_are_recovered(geometry: Geometry) -> None:
    """At t = 0 and t = 1 the path is at its endpoints."""
    start = to_state(cylinder_prior(16, generator=torch.Generator().manual_seed(3)), geometry)
    end = to_state(_samples(16, seed=4), geometry)
    bridge = build_bridge(geometry)

    at_zero, _ = bridge.forward(start, end, torch.zeros(16, 1, 1, 1))
    at_one, _ = bridge.forward(start, end, torch.ones(16, 1, 1, 1))

    assert torch.allclose(from_state(at_zero, geometry), from_state(start, geometry), atol=1e-5)
    assert torch.allclose(from_state(at_one, geometry), from_state(end, geometry), atol=1e-5)


def test_cylindrical_bridge_takes_the_short_way_round() -> None:
    """The geodesic never travels more than pi in phase, unlike a chord."""
    start = to_state(torch.tensor([complex(math.cos(3.0), math.sin(3.0))]), "cylindrical")
    end = to_state(torch.tensor([complex(math.cos(-3.0), math.sin(-3.0))]), "cylindrical")
    _, velocity = build_bridge("cylindrical").forward(start, end, torch.zeros(1, 1, 1, 1))

    assert abs(float(velocity[0, 1, 0, 0])) <= math.pi + 1e-6
    assert float(velocity[0, 1, 0, 0]) == pytest.approx(2 * math.pi - 6.0, abs=1e-5)


# --- The network is symmetric across arms ---


def test_both_arms_have_identical_capacity() -> None:
    """A capacity difference would be a confound; there must not be one."""
    euclidean = ToyVelocityField("euclidean")
    cylindrical = ToyVelocityField("cylindrical")
    assert sum(p.numel() for p in euclidean.parameters()) == sum(
        p.numel() for p in cylindrical.parameters()
    )


def test_both_arms_see_the_same_encoding_of_the_same_point() -> None:
    """The shared encoding is what removes the representation confound."""
    samples = _samples(32, seed=5)
    euclidean = ToyVelocityField("euclidean").encode(to_state(samples, "euclidean"))
    cylindrical = ToyVelocityField("cylindrical").encode(to_state(samples, "cylindrical"))
    assert torch.allclose(euclidean, cylindrical, atol=1e-5)


def test_encoding_is_finite_at_the_origin() -> None:
    """The cylinder chart is singular at A = 0; the encoding must not be."""
    origin = torch.zeros(4, dtype=torch.complex64)
    encoded = ToyVelocityField("euclidean").encode(to_state(origin, "euclidean"))
    assert bool(torch.isfinite(encoded).all())


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_velocity_field_shape_and_gradients(geometry: Geometry) -> None:
    """The field outputs a tangent velocity and is trainable."""
    network = ToyVelocityField(geometry, width=32, depth=2)
    state = to_state(_samples(16, seed=6), geometry)
    velocity = network(state, torch.rand(16))

    assert velocity.shape == (16, 2, 1, 1)
    velocity.square().mean().backward()
    assert all(p.grad is not None for p in network.parameters())


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"geometry": "torus"}, "unknown geometry"),
        ({"geometry": "euclidean", "width": 0}, "width must be positive"),
        ({"geometry": "euclidean", "depth": 0}, "depth must be positive"),
    ],
)
def test_velocity_field_validation(kwargs: dict[str, object], match: str) -> None:
    """Bad configuration raises at build time."""
    with pytest.raises(ValueError, match=match):
        ToyVelocityField(**kwargs)  # type: ignore[arg-type]


# --- Solvers ---


@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize("steps", [1, 4, 32])
def test_solver_integrates_to_a_finite_state(geometry: Geometry, steps: int) -> None:
    """Integration from the prior stays on the manifold at every step count."""
    network = ToyVelocityField(geometry, width=32, depth=2)
    start = to_state(cylinder_prior(32, generator=torch.Generator().manual_seed(7)), geometry)
    end = build_solver(geometry, steps).sample(network, start)

    assert end.shape == start.shape
    assert bool(torch.isfinite(end).all())
    if geometry == "cylindrical":
        norm = end[:, 1, 0, 0] ** 2 + end[:, 2, 0, 0] ** 2
        assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)
        assert bool((end[:, 0, 0, 0] >= 0).all())


def test_solver_rejects_zero_steps() -> None:
    """A zero-step solver would silently return the prior."""
    with pytest.raises(ValueError, match="num_steps must be >= 1"):
        build_solver("euclidean", 0)


# --- Straightness ---


def test_straightness_is_zero_for_a_perfect_field() -> None:
    """No residual means a perfectly straight conditional field."""
    displacement = torch.rand(64) + 0.1
    assert float(straightness(torch.zeros(64), displacement)) == pytest.approx(0.0)


def test_straightness_is_one_when_nothing_is_explained() -> None:
    """A field that predicts nothing leaves the whole displacement unexplained."""
    displacement = torch.rand(64) + 0.1
    assert float(straightness(displacement.clone(), displacement)) == pytest.approx(1.0)


def test_straightness_validation() -> None:
    """Shape mismatch and a degenerate normaliser both raise."""
    with pytest.raises(ValueError, match="must match"):
        straightness(torch.zeros(4), torch.zeros(8))
    with pytest.raises(ValueError, match="degenerate"):
        straightness(torch.zeros(4), torch.zeros(4))
