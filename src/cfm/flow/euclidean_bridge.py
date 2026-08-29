"""Straight-line probability path in flat R^2: the Euclidean baseline's bridge."""

from __future__ import annotations

import torch


class LinearFlowBridge:
    """The textbook conditional flow-matching path, the Euclidean counterpart of
    :class:`~cfm.flow.bridge.GeodesicFlowBridge`::

        x_t = (1 - t) * x_0 + t * x_1
        u_t = x_1 - x_0

    Both channels are treated identically: no decoupling into amplitude and
    phase, no shortest-angular-distance logic, no wrapping into [-pi, pi]. In
    R^2 the straight line *is* the geodesic, so this is deliberately trivial -
    it is the honest baseline the cylindrical geodesic is measured against.
    """

    def __init__(self) -> None:
        # No trainable parameters: pure mathematics, same as the geodesic bridge.
        pass

    def forward(
        self, euc_noise: torch.Tensor, euc_data: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the interpolated state and the target velocity.

        Args:
            euc_noise (torch.Tensor): Pure noise state at t=0 [B, 2, H, W].
            euc_data (torch.Tensor): Clean MRI data at t=1 [B, 2, H, W].
            t (torch.Tensor): Time step embedding [B, 1, 1, 1] bounded in [0, 1].

        Returns:
            tuple:
                - euc_t (torch.Tensor): Interpolated state at time t [B, 2, H, W].
                - target_v (torch.Tensor): Ground truth velocities [B, 2, H, W].
                  (Channel 0: v_re, Channel 1: v_im)

        Note:
            target_v is independent of t. The geodesic bridge has the same
            property per channel, so neither arm gets an easier regression target.
        """
        target_v = euc_data - euc_noise
        euc_t = euc_noise + t * target_v

        return euc_t, target_v
