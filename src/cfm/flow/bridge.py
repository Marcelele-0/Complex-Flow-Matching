import torch


class GeodesicFlowBridge:
    """
    Constructs the probability path (bridge) between pure noise (t=0)
    and the target complex-valued MRI data (t=1) on a cylindrical manifold.

    It computes the exact state x_t and the target vector field u_t
    required to train the Continuous Normalizing Flow model.
    """

    def __init__(self) -> None:
        # We don't need trainable parameters here, it's pure mathematics.
        pass

    def get_shortest_angular_diff(
        self, phi_start: torch.Tensor, phi_end: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculates the shortest directed angular distance between two phases.
        Result is tightly bounded in [-pi, pi].
        """
        diff = phi_end - phi_start
        # Standard trick to wrap the difference into [-pi, pi]
        diff = (diff + torch.pi) % (2 * torch.pi) - torch.pi
        return diff

    def forward(
        self, cyl_noise: torch.Tensor, cyl_data: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the interpolated state and the target velocity.

        Args:
            cyl_noise (torch.Tensor): Pure noise state at t=0 [B, 3, H, W].
            cyl_data (torch.Tensor): Clean MRI data at t=1 [B, 3, H, W].
            t (torch.Tensor): Time step embedding [B, 1, 1, 1] bounded in [0, 1].

        Returns:
            tuple:
                - cyl_t (torch.Tensor): Interpolated state at time t [B, 3, H, W].
                - target_v (torch.Tensor): Ground truth velocities [B, 2, H, W].
                  (Channel 0: v_m, Channel 1: v_phi)
        """
        # Unpack the decoupled channels
        m_0 = cyl_noise[:, 0:1, :, :]
        px_0 = cyl_noise[:, 1:2, :, :]
        py_0 = cyl_noise[:, 2:3, :, :]

        m_1 = cyl_data[:, 0:1, :, :]
        px_1 = cyl_data[:, 1:2, :, :]
        py_1 = cyl_data[:, 2:3, :, :]

        # Amplitude Dynamics (Linear Optimal Transport)
        u_m = m_1 - m_0
        m_t = m_0 + t * u_m

        # Phase Dynamics (Geodesic on S^1)
        # Recover exact angles from the unit vectors
        phi_0 = torch.atan2(py_0, px_0)
        phi_1 = torch.atan2(py_1, px_1)

        # Target angular velocity is simply the shortest path distance
        u_phi = self.get_shortest_angular_diff(phi_0, phi_1)

        # State of the phase at time t
        phi_t = phi_0 + t * u_phi

        # Reproject to Cartesian unit circle for the model input
        px_t = torch.cos(phi_t)
        py_t = torch.sin(phi_t)

        # Assemble the tensors
        cyl_t = torch.cat([m_t, px_t, py_t], dim=1)
        target_v = torch.cat([u_m, u_phi], dim=1)

        return cyl_t, target_v
