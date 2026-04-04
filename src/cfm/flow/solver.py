from collections.abc import Callable

import torch


class CylindricalODESolver:
    """
    Euler ODE solver tailored for the decoupled cylindrical manifold.
    Integrates the predicted velocity field over time to reconstruct
    clean MRI data from pure noise.
    """

    def __init__(self, num_steps: int = 50) -> None:
        self.num_steps = num_steps

    def step(self, x_t: torch.Tensor, v_t: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Performs a single Euler integration step and strictly projects
        the angular components back onto the unit circle (R=1).

        Args:
            x_t (torch.Tensor): Current state [B, 3, H, W]
            v_t (torch.Tensor): Predicted velocity [B, 2, H, W]
            dt (float): Time step size

        Returns:
            torch.Tensor: Next state [B, 3, H, W] safely on the manifold
        """
        # 1. Unpack state and velocity
        m_t = x_t[:, 0:1, :, :]
        px_t = x_t[:, 1:2, :, :]
        py_t = x_t[:, 2:3, :, :]

        v_m = v_t[:, 0:1, :, :]
        v_phi = v_t[:, 1:2, :, :]

        # 2. Euler step for amplitude (linear movement)
        m_next = m_t + v_m * dt

        # 3. Euler step for phase (angular movement)
        # First, recover current angle
        phi_t = torch.atan2(py_t, px_t)

        # Add angular velocity
        phi_next = phi_t + v_phi * dt

        # Reproject to Cartesian unit circle to strictly enforce R=1
        px_next = torch.cos(phi_next)
        py_next = torch.sin(phi_next)

        return torch.cat([m_next, px_next, py_next], dim=1)

    @torch.no_grad()
    def sample(
        self, model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Solves the ODE from t=0 to t=1 to generate a sample.

        Args:
            model: A callable (neural network) taking (state, time) and returning velocity.
            noise (torch.Tensor): Initial pure noise state [B, 3, H, W].

        Returns:
            torch.Tensor: Reconstructed state at t=1.
        """
        device = noise.device
        b = noise.shape[0]
        x_t = noise
        dt = 1.0 / self.num_steps

        for i in range(self.num_steps):
            # Create time tensor for the current step (shape: [B])
            # Models usually expect a 1D tensor for time embeddings
            t_val = i / self.num_steps
            t_tensor = torch.full((b,), t_val, device=device, dtype=torch.float32)

            # Get velocity from the neural network
            v_t = model(x_t, t_tensor)

            # Take a manifold-constrained step
            x_t = self.step(x_t, v_t, dt)

        return x_t
