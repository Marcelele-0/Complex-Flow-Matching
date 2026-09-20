"""Velocity regression loss for Euclidean flow matching baseline."""

from __future__ import annotations

import torch
import torch.nn as nn

from cyfm.flow.spectral import high_frequency_penalty

VALID_VELOCITY_LOSSES = {"l1", "l2", "mse"}


class EuclideanVelocityLoss(nn.Module):
    """Regression loss on 2-channel Euclidean velocity field (v_re, v_im).

    Args:
        loss_type: Distance metric ('l1', 'l2', or 'mse').
        lambda_hf: High-frequency spectral penalty weight.
        hf_boost_factor: Radial slope of high-frequency weighting.
    """

    def __init__(
        self,
        loss_type: str = "l1",
        lambda_hf: float = 0.0,
        hf_boost_factor: float = 4.0,
    ) -> None:
        super().__init__()
        if loss_type not in VALID_VELOCITY_LOSSES:
            raise ValueError(
                f"loss_type must be one of {sorted(VALID_VELOCITY_LOSSES)}, got {loss_type!r}"
            )

        self.loss_type = loss_type
        self.lambda_hf = lambda_hf
        self.hf_boost_factor = hf_boost_factor

        self.velocity_loss_fn = (
            nn.L1Loss(reduction="mean") if loss_type == "l1" else nn.MSELoss(reduction="mean")
        )

    def forward(
        self,
        pred_v: torch.Tensor,
        target_v: torch.Tensor,
        target_x1: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Euclidean velocity regression loss.

        Args:
            pred_v: Predicted velocity [B, 2, H, W] (v_re, v_im).
            target_v: Target velocity [B, 2, H, W] (v_re, v_im).
            target_x1: Unused clean data tensor (for interface parity).

        Returns:
            Tuple containing:
                - total_loss: Scalar total loss tensor.
                - loss_vel: Velocity regression loss scalar.
                - loss_hf: High-frequency penalty scalar.

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

        loss_hf = torch.tensor(0.0, device=pred_v.device, dtype=pred_v.dtype)
        if self.lambda_hf > 0.0:
            loss_hf = high_frequency_penalty(pred_v - target_v, self.hf_boost_factor)

        total_loss = loss_vel + (self.lambda_hf * loss_hf)

        return total_loss, loss_vel, loss_hf
