"""Unit tests for the exact optimal transport primitives.

The load-bearing test here is :func:`test_circular_solver_is_exact`: the circular
solver searches ``n`` cyclic shifts instead of ``n!`` permutations, which is only
valid because the optimal plan on ``S^1`` has that structure. That claim is
verified against a general assignment solver rather than cited.
"""

import math

import pytest
import torch
from scipy.optimize import linear_sum_assignment

from cfm.data.synthetic import CylinderToy, cylinder_prior
from cfm.flow.optimal_transport import (
    cylinder_cost_matrix,
    cylinder_transport_permutation,
    euclidean_transport_permutation,
    shortest_angular_diff,
    sorted_transport_permutation,
    transport_cost,
    wasserstein2_cylinder,
)


def _generator(seed: int) -> torch.Generator:
    """A CPU generator pinned to ``seed``."""
    return torch.Generator().manual_seed(seed)


def _is_permutation(perm: torch.Tensor, n: int) -> bool:
    """Whether ``perm`` is a bijection of ``range(n)``."""
    return bool(torch.equal(perm.sort().values, torch.arange(n)))


def _brute_force_circular_cost(source: torch.Tensor, target: torch.Tensor) -> float:
    """Optimal circular assignment cost via a general linear assignment solver."""
    cost = shortest_angular_diff(source.unsqueeze(1), target.unsqueeze(0)).square()
    rows, cols = linear_sum_assignment(cost.numpy())
    return float(cost[rows, cols].sum())


# --- Angular difference ---


def test_shortest_angular_diff_wraps_into_the_principal_branch() -> None:
    """The result always lies in [-pi, pi], including across the cut."""
    start = torch.tensor([0.0, 3.0, -3.0, math.pi])
    end = torch.tensor([0.1, -3.0, 3.0, -math.pi])
    diff = shortest_angular_diff(start, end)

    assert bool((diff >= -math.pi - 1e-6).all())
    assert bool((diff <= math.pi + 1e-6).all())
    assert float(diff[1]) == pytest.approx(2 * math.pi - 6.0, abs=1e-5)


def test_shortest_angular_diff_is_antisymmetric_away_from_the_cut() -> None:
    """Swapping the arguments negates the displacement."""
    a = torch.tensor([0.3, -1.2, 2.0])
    b = torch.tensor([1.1, 0.4, -2.9])
    assert torch.allclose(shortest_angular_diff(a, b), -shortest_angular_diff(b, a), atol=1e-6)


# --- Line ---


def test_sorted_permutation_is_a_bijection_and_monotone() -> None:
    """Rank matching pairs the k-th smallest with the k-th smallest."""
    source = torch.randn(64, generator=_generator(0))
    target = torch.randn(64, generator=_generator(1))
    perm = sorted_transport_permutation(source, target)

    assert _is_permutation(perm, 64)
    matched = target[perm]
    ordered = matched[torch.argsort(source)]
    assert bool((torch.diff(ordered) >= -1e-6).all())


def test_sorted_permutation_beats_random_pairings() -> None:
    """The monotone plan is optimal, so no shuffle can cost less."""
    source = torch.randn(128, generator=_generator(2))
    target = torch.randn(128, generator=_generator(3))
    optimal = float((source - target[sorted_transport_permutation(source, target)]).square().sum())

    for seed in range(5):
        shuffled = target[torch.randperm(128, generator=_generator(10 + seed))]
        assert optimal <= float((source - shuffled).square().sum()) + 1e-5


# --- Circle: the structural claim ---


@pytest.mark.parametrize("seed", range(12))
def test_circular_solver_is_exact(seed: int) -> None:
    """The cyclic-shift search matches a general assignment solver exactly."""
    from cfm.flow.optimal_transport import circular_transport_permutation

    n = 14
    source = torch.empty(n).uniform_(-math.pi, math.pi, generator=_generator(seed))
    target = torch.empty(n).uniform_(-math.pi, math.pi, generator=_generator(seed + 100))

    perm = circular_transport_permutation(source, target)
    assert _is_permutation(perm, n)

    achieved = float(shortest_angular_diff(source, target[perm]).square().sum())
    assert achieved == pytest.approx(_brute_force_circular_cost(source, target), abs=1e-5)


def test_circular_solver_crosses_the_branch_cut() -> None:
    """Mass that must wrap is matched across the cut, which rank matching cannot do.

    Two clusters straddle the cut, so the target's circular order is a non-trivial
    rotation of the source's. Matching by rank on the raw values pairs each cluster
    with the wrong one; the cyclic-shift search does not.
    """
    from cfm.flow.optimal_transport import circular_transport_permutation

    source = torch.cat([torch.linspace(-3.05, -2.95, 8), torch.linspace(-0.05, 0.05, 8)])
    target = torch.cat([torch.linspace(2.95, 3.05, 8), torch.linspace(0.15, 0.25, 8)])

    circular = (
        shortest_angular_diff(source, target[circular_transport_permutation(source, target)])
        .square()
        .sum()
    )
    by_rank = (
        shortest_angular_diff(source, target[sorted_transport_permutation(source, target)])
        .square()
        .sum()
    )

    assert float(circular) < 1.5
    assert float(by_rank) > 100.0


# --- Cylinder ---


def test_cylinder_permutation_is_a_bijection() -> None:
    """Every source point is matched to exactly one genuine target sample."""
    toy = CylinderToy(coupling=0.7)
    prior = cylinder_prior(96, generator=_generator(4))
    amplitude, phase = toy.sample_polar(96, generator=_generator(5))

    perm = cylinder_transport_permutation(prior.abs(), prior.angle(), amplitude, phase)
    assert _is_permutation(perm, 96)


def test_cylinder_permutation_beats_random_pairings() -> None:
    """The assignment is optimal under the product cost."""
    toy = CylinderToy(coupling=0.5)
    prior = cylinder_prior(128, generator=_generator(6))
    amplitude, phase = toy.sample_polar(128, generator=_generator(7))

    perm = cylinder_transport_permutation(prior.abs(), prior.angle(), amplitude, phase)
    optimal = float(transport_cost(prior.abs(), prior.angle(), amplitude[perm], phase[perm]))

    for seed in range(5):
        shuffle = torch.randperm(128, generator=_generator(20 + seed))
        cost = float(transport_cost(prior.abs(), prior.angle(), amplitude[shuffle], phase[shuffle]))
        assert optimal <= cost + 1e-5


def test_phase_weight_changes_the_assignment() -> None:
    """The angular term's weight is a modelling choice with visible consequences."""
    toy = CylinderToy(coupling=0.5)
    prior = cylinder_prior(64, generator=_generator(8))
    amplitude, phase = toy.sample_polar(64, generator=_generator(9))

    light = cylinder_transport_permutation(
        prior.abs(), prior.angle(), amplitude, phase, phase_weight=0.01
    )
    heavy = cylinder_transport_permutation(
        prior.abs(), prior.angle(), amplitude, phase, phase_weight=100.0
    )
    assert not torch.equal(light, heavy)


def test_cost_matrix_shape_and_symmetry_of_the_metric() -> None:
    """The cost matrix is [n, m] and vanishes on identical points."""
    amplitude = torch.tensor([0.5, 1.0, 1.5])
    phase = torch.tensor([0.0, 1.0, -2.0])
    cost = cylinder_cost_matrix(amplitude, phase, amplitude, phase)

    assert cost.shape == (3, 3)
    assert torch.allclose(cost.diagonal(), torch.zeros(3), atol=1e-6)
    assert torch.allclose(cost, cost.T, atol=1e-6)


# --- Plane ---


def test_euclidean_permutation_is_optimal() -> None:
    """The plane's minibatch coupling beats every shuffle under chordal cost."""
    prior = cylinder_prior(128, generator=_generator(30))
    target = CylinderToy(coupling=0.5).sample(128, generator=_generator(31))

    perm = euclidean_transport_permutation(prior, target)
    assert _is_permutation(perm, 128)
    optimal = float((prior - target[perm]).abs().square().sum())

    for seed in range(5):
        shuffled = target[torch.randperm(128, generator=_generator(40 + seed))]
        assert optimal <= float((prior - shuffled).abs().square().sum()) + 1e-4


def test_euclidean_and_cylinder_couplings_disagree() -> None:
    """The two metrics pair different points, which is what the 2x2 gate varies."""
    prior = cylinder_prior(96, generator=_generator(32))
    target = CylinderToy(coupling=1.0).sample(96, generator=_generator(33))

    plane = euclidean_transport_permutation(prior, target)
    cylinder = cylinder_transport_permutation(
        prior.abs(), prior.angle(), target.abs(), target.angle()
    )
    assert not torch.equal(plane, cylinder)


def test_euclidean_permutation_validation() -> None:
    """Size mismatch and an oversized batch both raise."""
    with pytest.raises(ValueError, match="equal sample counts"):
        euclidean_transport_permutation(
            torch.zeros(8, dtype=torch.complex64), torch.zeros(6, dtype=torch.complex64)
        )
    big = torch.zeros(5000, dtype=torch.complex64)
    with pytest.raises(ValueError, match="exceeds the dense budget"):
        euclidean_transport_permutation(big, big)


# --- Wasserstein ---


def test_wasserstein_is_zero_against_a_permutation_of_itself() -> None:
    """A reordering is the same distribution, so the distance is zero."""
    toy = CylinderToy(coupling=0.6)
    amplitude, phase = toy.sample_polar(96, generator=_generator(11))
    shuffle = torch.randperm(96, generator=_generator(12))

    distance = wasserstein2_cylinder(amplitude, phase, amplitude[shuffle], phase[shuffle])
    assert float(distance) == pytest.approx(0.0, abs=1e-5)


def test_wasserstein_is_positive_between_different_clouds() -> None:
    """Distinct distributions are separated."""
    prior = cylinder_prior(96, generator=_generator(13))
    amplitude, phase = CylinderToy(coupling=0.0).sample_polar(96, generator=_generator(14))
    assert float(wasserstein2_cylinder(prior.abs(), prior.angle(), amplitude, phase)) > 0.1


# --- The factorisation gap, as a lower bound that is not attained ---


def test_factorised_cost_is_a_lower_bound_on_the_joint_cost() -> None:
    """Solving the marginals separately reports a cost no feasible plan attains.

    Both are computed on the same pair of clouds, so the inequality is the
    product-measure factorisation bound and nothing else.
    """
    from cfm.flow.optimal_transport import circular_transport_permutation

    prior = cylinder_prior(256, generator=_generator(15))
    amplitude, phase = CylinderToy(coupling=1.0).sample_polar(256, generator=_generator(16))

    amplitude_perm = sorted_transport_permutation(prior.abs(), amplitude)
    phase_perm = circular_transport_permutation(prior.angle(), phase)
    factorised = float(
        transport_cost(prior.abs(), prior.angle(), amplitude[amplitude_perm], phase[phase_perm])
    )

    joint_perm = cylinder_transport_permutation(prior.abs(), prior.angle(), amplitude, phase)
    joint = float(
        transport_cost(prior.abs(), prior.angle(), amplitude[joint_perm], phase[joint_perm])
    )

    assert factorised < joint


def test_factorised_pairing_is_not_a_permutation_of_the_target() -> None:
    """Under dependence the factorised endpoints leave the target set."""
    from cfm.flow.optimal_transport import circular_transport_permutation

    prior = cylinder_prior(256, generator=_generator(17))
    amplitude, phase = CylinderToy(coupling=1.0).sample_polar(256, generator=_generator(18))

    amplitude_perm = sorted_transport_permutation(prior.abs(), amplitude)
    phase_perm = circular_transport_permutation(prior.angle(), phase)

    assert not torch.equal(amplitude_perm, phase_perm)
    displaced = wasserstein2_cylinder(
        amplitude[amplitude_perm], phase[phase_perm], amplitude, phase
    )
    assert float(displaced) > 0.1


def _chimera_displacement(coupling: float, n: int, seed: int) -> float:
    """Distance from the factorised endpoint cloud back to the target cloud."""
    from cfm.flow.optimal_transport import circular_transport_permutation

    prior = cylinder_prior(n, generator=_generator(seed))
    amplitude, phase = CylinderToy(coupling=coupling).sample_polar(
        n, generator=_generator(seed + 1)
    )
    amplitude_perm = sorted_transport_permutation(prior.abs(), amplitude)
    phase_perm = circular_transport_permutation(prior.angle(), phase)
    return float(
        wasserstein2_cylinder(amplitude[amplitude_perm], phase[phase_perm], amplitude, phase)
    )


def test_factorised_damage_is_bias_and_not_sampling_noise() -> None:
    """Under independence the damage vanishes with n; under dependence it does not.

    This is the distinction that decides whether a factorised coupling is an
    approximation or a different distribution: sampling noise shrinks as the batch
    grows, a bias does not.
    """
    independent_small = _chimera_displacement(0.0, 64, 19)
    independent_large = _chimera_displacement(0.0, 1024, 19)
    comonotone_small = _chimera_displacement(1.0, 64, 21)
    comonotone_large = _chimera_displacement(1.0, 1024, 21)

    assert independent_large < independent_small / 1.5
    assert comonotone_large > 0.8 * comonotone_small
    assert comonotone_large > 3.0 * independent_large


# --- Validation ---


@pytest.mark.parametrize(
    ("source", "target", "match"),
    [
        (torch.zeros(4, 2), torch.zeros(8), "must be 1D"),
        (torch.zeros(8), torch.zeros(6), "equal sample counts"),
        (torch.zeros(0), torch.zeros(0), "at least one sample"),
    ],
)
def test_solver_validation(source: torch.Tensor, target: torch.Tensor, match: str) -> None:
    """Malformed inputs raise rather than silently truncating."""
    with pytest.raises(ValueError, match=match):
        sorted_transport_permutation(source, target)


def test_negative_phase_weight_is_rejected() -> None:
    """A negative weight would make the cost non-metric."""
    amplitude, phase = torch.ones(4), torch.zeros(4)
    with pytest.raises(ValueError, match="phase_weight must be non-negative"):
        cylinder_cost_matrix(amplitude, phase, amplitude, phase, phase_weight=-1.0)


def test_dense_budget_is_enforced() -> None:
    """An oversized cost matrix fails loudly instead of exhausting memory."""
    big = torch.zeros(5000)
    with pytest.raises(ValueError, match="exceeds the dense budget"):
        cylinder_cost_matrix(big, big, big, big)
