"""Velocity regression loss for Euclidean flow matching baseline."""

from __future__ import annotations

import torch
import torch.nn as nn

VALID_VELOCITY_LOSSES = {"l1", "l2", "mse"}


class EuclideanVelocityLoss(nn.Module):
    """Regression loss on 2-channel Euclidean velocity field (v_re, v_im).

    Args:
        loss_type: Distance metric ('l1', 'l2', or 'mse').
    """

    def __init__(
        self,
        loss_type: str = "l1",
    ) -> None:
        super().__init__()
        if loss_type not in VALID_VELOCITY_LOSSES:
            raise ValueError(
                f"loss_type must be one of {sorted(VALID_VELOCITY_LOSSES)}, got {loss_type!r}"
            )

        self.loss_type = loss_type

        self.velocity_loss_fn = (
            nn.L1Loss(reduction="mean") if loss_type == "l1" else nn.MSELoss(reduction="mean")
        )

    def forward(
        self,
        pred_v: torch.Tensor,
        target_v: torch.Tensor,
        target_x1: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Euclidean velocity regression loss.

        Args:
            pred_v: Predicted velocity [B, 2, H, W] (v_re, v_im).
            target_v: Target velocity [B, 2, H, W] (v_re, v_im).
            target_x1: Unused clean data tensor (for interface parity).

        Returns:
            Tuple containing:
                - total_loss: Scalar total loss tensor.
                - loss_vel: Velocity regression loss scalar.

        Raises:
            ValueError: If inputs do not have exactly 2 channels.
        """
        if pred_v.shape[1] != 2 or target_v.shape[1] != 2:
            raise ValueError(
                f"Expected 2-channel velocity (v_re, v_im), got "
                f"pred_v: {pred_v.shape[1]} channels, target_v: {target_v.shape[1]} channels"
            )

        del target_x1

        loss_vel = self.velocity_loss_fn(pred_v, target_v)

        # With no auxiliary term the total IS the velocity loss. It is returned
        # twice so the caller can log the component without special-casing the
        # single-term geometry against the two-term cylindrical one.
        return loss_vel, loss_vel
