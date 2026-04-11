from collections.abc import Callable
import torch

class CylindricalODESolver:
    """
    Heun ODE solver (2nd order) tailored for the decoupled cylindrical manifold.
    Integrates the predicted velocity field over time to reconstruct
    clean MRI data from pure noise, providing sharper reconstructions than Euler.
    """

    def __init__(self, num_steps: int = 50) -> None:
        self.num_steps = num_steps

    def step(self, x_t: torch.Tensor, v_t: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Performs a single integration step and safely projects back onto the Cylinder.

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

        # 2. Amplitude step with physical constraint (prevent negative energy)
        m_next = m_t + v_m * dt
        m_next = torch.clamp(m_next, min=0.0)

        # 3. Phase step on S^1
        phi_t = torch.atan2(py_t, px_t)
        phi_next = phi_t + v_phi * dt

        # 4. Reproject to Cartesian unit circle to strictly enforce R=1
        px_next = torch.cos(phi_next)
        py_next = torch.sin(phi_next)

        return torch.cat([m_next, px_next, py_next], dim=1)

    @torch.no_grad()
    def sample(
        self, model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor], noise: torch.Tensor
    ) -> torch.Tensor:
        """
        Solves the ODE from t=0 to t=1 using Heun's Method on a Manifold.

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
            t_val = i / self.num_steps
            t_next_val = (i + 1) / self.num_steps

            t_tensor = torch.full((b,), t_val, device=device, dtype=torch.float32)
            t_next_tensor = torch.full((b,), t_next_val, device=device, dtype=torch.float32)

            # 1. Predictor: Evaluate current vector field
            v_t = model(x_t, t_tensor)

            # Last step is just Euler to avoid overshooting t=1
            if i == self.num_steps - 1:
                x_t = self.step(x_t, v_t, dt)
                break

            # 2. Euler step into the future
            x_pred = self.step(x_t, v_t, dt)

            # 3. Evaluate future vector field
            v_next = model(x_pred, t_next_tensor)

            # 4. Corrector: Average the velocities
            v_avg = 0.5 * (v_t + v_next)

            # 5. Final step using the averaged velocity
            x_t = self.step(x_t, v_avg, dt)

        return x_t