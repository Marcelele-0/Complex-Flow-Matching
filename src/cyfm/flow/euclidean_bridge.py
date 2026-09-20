"""Straight-line probability path in flat R^2 Euclidean baseline."""

from __future__ import annotations

import torch


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
