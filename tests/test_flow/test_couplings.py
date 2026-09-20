"""Unit tests for minibatch couplings.

The property Gate A exists to enforce is here as a test: an optimal-transport
coupling must return a *permutation* of the data batch, so the endpoint
distribution is the data distribution exactly. A construction that assembles
endpoints coordinate by coordinate passes every marginal check and fails this
one.
"""

import pytest
import torch
from omegaconf import OmegaConf

from cyfm.config.adapters import manifold_from_config
from cyfm.core.manifold import BaseManifold
from cyfm.core.registry import COUPLINGS
from cyfm.data.toy import CylinderToyIIDDataset
from cyfm.flow.couplings import IndependentCoupling, OptimalTransportCoupling, build_coupling

GEOMETRIES = ("cylindrical", "euclidean")


def _manifold(name: str, **manifold_keys: float) -> BaseManifold:
    """Build a manifold the way the entry points do."""
    return manifold_from_config(
        OmegaConf.create({"manifold": {"name": name, **manifold_keys}, "training": {"loss": {}}})
    )


def _batch(
    manifold: BaseManifold, batch: int = 24, side: int = 8, coupling: float = 0.5
) -> tuple[torch.Tensor, torch.Tensor]:
    """A prior batch and a data batch in the manifold's state representation."""
    dataset = CylinderToyIIDDataset(coupling=coupling, size=batch, crop_size=(side, side))
    data = manifold.from_complex(torch.stack([dataset[i] for i in range(batch)]))  # type: ignore[attr-defined]
    prior = manifold.sample_noise(batch, side, side, torch.device("cpu"))  # type: ignore[attr-defined]
    return prior, data


def _cost(manifold: BaseManifold, prior: torch.Tensor, data: torch.Tensor) -> float:
    """Mean weighted squared geodesic displacement of a pairing."""
    weights = manifold.tangent_weights.reshape(1, -1, 1, 1)  # type: ignore[attr-defined]
    return float((manifold.log_map(prior, data).square() * weights).flatten(1).mean(1).mean())  # type: ignore[attr-defined]


# --- Registry ---


def test_both_couplings_are_registered() -> None:
    """Registration is what makes training.coupling=<name> resolve."""
    assert COUPLINGS.get("independent") is IndependentCoupling
    assert COUPLINGS.get("ot") is OptimalTransportCoupling
    assert COUPLINGS.get("optimal_transport") is OptimalTransportCoupling


def test_build_coupling_rejects_an_unknown_name() -> None:
    """An unknown name lists the valid ones rather than falling back silently."""
    with pytest.raises(ValueError, match="Unknown coupling"):
        build_coupling("sinkhorn")


# --- The Gate A property ---


@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize("coupling", [0.0, 1.0])
def test_optimal_transport_returns_a_permutation_of_the_data(
    geometry: str, coupling: float
) -> None:
    """Every endpoint is a genuine datum, so the target distribution is exact.

    This is what a factorised construction cannot do, and why one is not offered.
    """
    manifold = _manifold(geometry)
    prior, data = _batch(manifold, coupling=coupling)
    coupled = build_coupling("ot")(prior, data, manifold)

    assert torch.allclose(coupled.flatten().sort().values, data.flatten().sort().values, atol=1e-6)


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_optimal_transport_lowers_the_cost(geometry: str) -> None:
    """The assignment is optimal, so it cannot cost more than what it replaced."""
    manifold = _manifold(geometry)
    prior, data = _batch(manifold)
    coupled = build_coupling("ot")(prior, data, manifold)
    assert _cost(manifold, prior, coupled) < _cost(manifold, prior, data)


@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_optimal_transport_beats_every_shuffle(geometry: str) -> None:
    """Optimality against the alternatives, not just against the draw order."""
    manifold = _manifold(geometry)
    prior, data = _batch(manifold)
    optimal = _cost(manifold, prior, build_coupling("ot")(prior, data, manifold))

    for seed in range(5):
        shuffled = data[
            torch.randperm(data.shape[0], generator=torch.Generator().manual_seed(seed))
        ]
        assert optimal <= _cost(manifold, prior, shuffled) + 1e-6


def test_independent_coupling_is_the_identity() -> None:
    """The baseline must not quietly reorder anything."""
    manifold = _manifold("cylindrical")
    prior, data = _batch(manifold)
    assert build_coupling("independent")(prior, data, manifold) is data


# --- The metric is a choice, and it shows ---


def test_phase_weight_changes_the_assignment() -> None:
    """The angular weight is a modelling choice with a visible consequence."""
    prior, data = _batch(_manifold("cylindrical"))
    light = build_coupling("ot")(prior, data, _manifold("cylindrical", phase_weight=0.01))
    heavy = build_coupling("ot")(prior, data, _manifold("cylindrical", phase_weight=100.0))
    assert not torch.equal(light, heavy)


def test_cylindrical_tangent_weights_follow_the_config() -> None:
    """The weight reaches the geometry rather than stopping at the config."""
    assert _manifold("cylindrical", phase_weight=7.0).tangent_weights.tolist() == [1.0, 7.0]  # type: ignore[attr-defined]
    assert _manifold("euclidean").tangent_weights.tolist() == [1.0, 1.0]  # type: ignore[attr-defined]


def test_negative_phase_weight_is_rejected() -> None:
    """A negative weight would make the coupling cost non-metric."""
    with pytest.raises(ValueError, match="phase_weight must be non-negative"):
        _manifold("cylindrical", phase_weight=-1.0)


# --- Shape and budget ---


def test_cost_matrix_is_square_and_zero_on_the_diagonal_against_itself() -> None:
    """A batch coupled to itself costs nothing on the diagonal."""
    manifold = _manifold("cylindrical")
    _, data = _batch(manifold)
    cost = OptimalTransportCoupling().cost_matrix(data, data, manifold)

    assert cost.shape == (data.shape[0], data.shape[0])
    assert torch.allclose(cost.diagonal(), torch.zeros(data.shape[0]), atol=1e-6)


def test_mismatched_batches_are_rejected() -> None:
    """A shape mismatch here would silently pair the wrong things."""
    manifold = _manifold("cylindrical")
    prior, data = _batch(manifold, batch=8)
    with pytest.raises(ValueError, match="must share a shape"):
        OptimalTransportCoupling()(prior[:4], data, manifold)


def test_oversized_batch_is_refused() -> None:
    """A dense assignment stops being free above the budget, and says so."""
    manifold = _manifold("cylindrical")
    prior, data = _batch(manifold, batch=8)
    with pytest.raises(ValueError, match="exceeds max_batch"):
        OptimalTransportCoupling(max_batch=4)(prior, data, manifold)


def test_max_batch_must_be_positive() -> None:
    """A non-positive budget would refuse every batch."""
    with pytest.raises(ValueError, match="max_batch must be positive"):
        OptimalTransportCoupling(max_batch=0)
