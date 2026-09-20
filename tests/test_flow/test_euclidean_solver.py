import pytest
import torch

from cyfm.flow.solvers import CylindricalODESolver, EuclideanODESolver, HeunODESolver


def test_step_is_plain_euler() -> None:
    """The update must be exactly x + v*dt, with nothing added."""
    solver = EuclideanODESolver(num_steps=10)

    x_t = torch.tensor([[[[0.5]], [[-0.25]]]])
    v_t = torch.tensor([[[[1.0]], [[2.0]]]])

    x_next = solver.step(x_t, v_t, dt=0.5)

    assert torch.equal(x_next, torch.tensor([[[[1.0]], [[0.75]]]]))


def test_step_does_not_clamp_or_reproject() -> None:
    """CRITICAL: no constraint may be imposed on the Euclidean trajectory.

    The cylindrical step clamps amplitude at 0 and forces the phase back onto
    R=1. Neither may happen here: a state that leaves the unit disc, or that
    takes a large negative value, must be carried forward untouched. Silently
    fixing it up would hand the baseline the very correction the cylindrical
    formulation is claimed to contribute.
    """
    solver = EuclideanODESolver(num_steps=10)

    x_t = torch.tensor([[[[0.1]], [[0.1]]]])
    v_t = torch.tensor([[[[-9.0]], [[9.0]]]])

    x_next = solver.step(x_t, v_t, dt=1.0)

    assert torch.equal(x_next, torch.tensor([[[[-8.9]], [[9.1]]]]))
    modulus = torch.sqrt(x_next[:, 0:1] ** 2 + x_next[:, 1:2] ** 2)
    assert torch.all(modulus > 1.0), "the state must be free to leave the unit disc"


def test_step_rejects_channel_mismatch() -> None:
    """A cylindrical state reaching this solver is a wiring bug, not a broadcast."""
    solver = EuclideanODESolver(num_steps=4)
    with pytest.raises(ValueError, match="share a channel count"):
        solver.step(torch.zeros(1, 3, 4, 4), torch.zeros(1, 2, 4, 4), dt=0.1)


def test_sample_loop_integrates_a_constant_field() -> None:
    """A constant velocity must integrate to exactly x_0 + v over [0, 1]."""
    solver = EuclideanODESolver(num_steps=8)
    x_0 = torch.rand(2, 2, 8, 8) - 0.5

    def constant_model(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.full_like(x, 0.25)

    final_state = solver.sample(constant_model, x_0)

    assert final_state.shape == x_0.shape
    assert torch.allclose(final_state, x_0 + 0.25, atol=1e-6)


def test_shares_the_heun_schedule_with_the_cylindrical_solver() -> None:
    """Both geometries must feed the model the same times, the same number of times.

    This is what makes the side-by-side a comparison of geometries rather than a
    comparison of integrators.
    """
    seen: dict[str, list[float]] = {"euclidean": [], "cylindrical": []}

    def recorder(key: str, channels: int):
        def model(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            seen[key].append(float(t[0]))
            return torch.zeros(x.shape[0], 2, x.shape[2], x.shape[3])

        return model

    EuclideanODESolver(num_steps=5).sample(recorder("euclidean", 2), torch.zeros(1, 2, 4, 4))

    cyl_state = torch.zeros(1, 3, 4, 4)
    cyl_state[:, 1] = 1.0  # a valid point on the circle
    CylindricalODESolver(num_steps=5).sample(recorder("cylindrical", 3), cyl_state)

    assert seen["euclidean"] == seen["cylindrical"]
    assert len(seen["euclidean"]) == 2 * 5 - 1


def test_both_solvers_share_the_base_class() -> None:
    """The schedule lives in one place; only `step` is allowed to differ."""
    assert issubclass(EuclideanODESolver, HeunODESolver)
    assert issubclass(CylindricalODESolver, HeunODESolver)
    assert EuclideanODESolver.sample is HeunODESolver.sample
    assert CylindricalODESolver.sample is HeunODESolver.sample
