import math
import torch

from cfm.flow.solver import CylindricalODESolver


def test_solver_step_projection() -> None:
    """Checks if a single step strictly keeps phase on the R=1 circle."""
    solver = CylindricalODESolver(num_steps=10)
    
    # Initial state: m=0.5, phi=0 -> px=1, py=0
    x_t = torch.zeros(1, 3, 2, 2)
    x_t[:, 0, :, :] = 0.5
    x_t[:, 1, :, :] = 1.0 
    
    # Velocity: +1.0 for m, +pi/2 for phi
    v_t = torch.zeros(1, 2, 2, 2)
    v_t[:, 0, :, :] = 1.0
    v_t[:, 1, :, :] = math.pi / 2.0
    
    dt = 1.0
    x_next = solver.step(x_t, v_t, dt)
    
    # Expected amplitude: 0.5 + 1.0*1.0 = 1.5
    assert torch.allclose(x_next[:, 0:1, :, :], torch.tensor(1.5))
    
    # Expected phase: 0 + pi/2 -> px=0, py=1
    assert torch.allclose(x_next[:, 1:2, :, :], torch.tensor(0.0), atol=1e-6)
    assert torch.allclose(x_next[:, 2:3, :, :], torch.tensor(1.0), atol=1e-6)
    
    # Radius must be exactly 1
    px = x_next[:, 1:2, :, :]
    py = x_next[:, 2:3, :, :]
    radius_sq = px**2 + py**2
    assert torch.allclose(radius_sq, torch.ones_like(radius_sq), atol=1e-6)


def test_solver_sample_loop() -> None:
    """Checks if the full integration loop executes with a dummy model."""
    solver = CylindricalODESolver(num_steps=4)
    noise = torch.randn(2, 3, 8, 8)
    
    # Dummy model that always outputs zero velocity
    def dummy_model(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return torch.zeros(x.shape[0], 2, x.shape[2], x.shape[3], device=x.device)
    
    final_state = solver.sample(dummy_model, noise)
    
    # With zero velocity, amplitude should remain exactly the same
    assert torch.allclose(final_state[:, 0:1], noise[:, 0:1])
    # Shape must be preserved
    assert final_state.shape == noise.shape