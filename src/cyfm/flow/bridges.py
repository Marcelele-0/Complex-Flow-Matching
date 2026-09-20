"""The two probability paths the geometries interpolate along.

Both are conditional flow-matching bridges between a prior sample at ``t = 0``
and a datum at ``t = 1``, and both carry a velocity that is constant along the
path. They differ only in the space they run in, which is the whole comparison:
:class:`GeodesicFlowBridge` moves the amplitude linearly and the phase along the
shorter arc of ``S^1``, so its angular target is bounded by ``pi``;
:class:`LinearFlowBridge` moves ``(Re z, Im z)`` in a straight line, whose
induced angular velocity is unbounded near the origin.
"""

from __future__ import annotations

import torch

from cyfm.utils.complex_ops import wrap_to_pi


class GeodesicFlowBridge:
    """Geodesic probability path on product manifold R+ x S^1."""

    def __init__(self) -> None:
        pass

    def get_shortest_angular_diff(
        self, phi_start: torch.Tensor, phi_end: torch.Tensor
    ) -> torch.Tensor:
        """Calculate shortest directed angular distance in (-pi, pi].

        Args:
            phi_start: Initial angle in radians [B, 1, H, W].
            phi_end: Target angle in radians [B, 1, H, W].

        Returns:
            Shortest angular displacement [B, 1, H, W] in (-pi, pi].
        """
        return wrap_to_pi(phi_end - phi_start)

    def forward(
        self, cyl_noise: torch.Tensor, cyl_data: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute geodesic interpolated state and target velocity field.

        Args:
            cyl_noise: Prior noise state [B, 3, H, W] (m, cos(phi), sin(phi)).
            cyl_data: Clean data state [B, 3, H, W] (m, cos(phi), sin(phi)).
            t: Time embedding [B, 1, 1, 1] in [0, 1].

        Returns:
            Tuple containing:
                - cyl_t: Interpolated state [B, 3, H, W] on R+ x S^1.
                - target_v: Target tangent velocity [B, 2, H, W] (u_m, u_phi).
        """
        m_0 = cyl_noise[:, 0:1, :, :]
        px_0 = cyl_noise[:, 1:2, :, :]
        py_0 = cyl_noise[:, 2:3, :, :]

        m_1 = cyl_data[:, 0:1, :, :]
        px_1 = cyl_data[:, 1:2, :, :]
        py_1 = cyl_data[:, 2:3, :, :]

        # Amplitude Dynamics (Linear OT)
        u_m = m_1 - m_0
        m_t = m_0 + t * u_m

        # Phase Dynamics (Geodesic on S^1)
        phi_0 = torch.atan2(py_0, px_0)
        phi_1 = torch.atan2(py_1, px_1)
        u_phi = self.get_shortest_angular_diff(phi_0, phi_1)
        phi_t = phi_0 + t * u_phi

        px_t = torch.cos(phi_t)
        py_t = torch.sin(phi_t)

        cyl_t = torch.cat([m_t, px_t, py_t], dim=1)
        target_v = torch.cat([u_m, u_phi], dim=1)

        return cyl_t, target_v


class LinearFlowBridge:
    """Conditional flow-matching straight-line path in flat Euclidean space R^2."""

    def __init__(self) -> None:
        pass

    def forward(
        self, euc_noise: torch.Tensor, euc_data: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute linear interpolated state and target velocity field.

        Args:
            euc_noise: Pure noise state at t=0 [B, 2, H, W] (real, imag).
            euc_data: Clean data state at t=1 [B, 2, H, W] (real, imag).
            t: Time embedding [B, 1, 1, 1] in [0, 1].

        Returns:
            Tuple containing:
                - euc_t: Interpolated state [B, 2, H, W] at time t.
                - target_v: Target velocity field [B, 2, H, W] (v_re, v_im).
        """
        target_v = euc_data - euc_noise
        euc_t = euc_noise + t * target_v

        return euc_t, target_v
