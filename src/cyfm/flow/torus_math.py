"""Decoupled cylindrical velocity loss on R+ x S^1."""

from __future__ import annotations

import torch
import torch.nn as nn

VALID_AMP_LOSSES = {"l1", "l2", "mse"}
VALID_PHASE_LOSSES = {"l1", "l2", "mse", "cosine"}


class DecoupledCylindricalLoss(nn.Module):
    """Loss for decoupled cylindrical velocity field (v_amp, v_phase).

    Args:
        amp_loss_type: Distance metric for amplitude channel ('l1', 'l2', or 'mse').
        phase_loss_type: Metric for phase velocity ('l1', 'l2', 'mse', or 'cosine').
        lambda_phase: Weight on phase velocity loss term.
        phase_amplitude_weighting: Weight the phase error by the clean amplitude
            ``A_1 / mean(A_1)`` when ``target_x1`` is given. ``False`` makes the
            phase term an unweighted regression like the amplitude term, whose
            minimiser is then a function of ``(x_t, t)`` alone -- the weight
            depends on ``x_1``, so with it on the phase channel regresses a
            reweighted conditional statistic that the amplitude channel does not.
    """

    def __init__(
        self,
        amp_loss_type: str = "l1",
        phase_loss_type: str = "l1",
        lambda_phase: float = 1.0,
        phase_amplitude_weighting: bool = True,
    ) -> None:
        super().__init__()
        if amp_loss_type not in VALID_AMP_LOSSES:
            raise ValueError(
                f"amp_loss_type must be one of {sorted(VALID_AMP_LOSSES)}, got {amp_loss_type!r}"
            )
        if phase_loss_type not in VALID_PHASE_LOSSES:
            raise ValueError(
                f"phase_loss_type must be one of {sorted(VALID_PHASE_LOSSES)}, "
                f"got {phase_loss_type!r}"
            )

        self.phase_loss_type = phase_loss_type
        self.lambda_phase = lambda_phase
        self.phase_amplitude_weighting = phase_amplitude_weighting

        self.amp_loss_fn = (
            nn.L1Loss(reduction="mean") if amp_loss_type == "l1" else nn.MSELoss(reduction="mean")
        )
        self.phase_loss_fn = (
            nn.L1Loss(reduction="none") if phase_loss_type == "l1" else nn.MSELoss(reduction="none")
        )

    def forward(
        self,
        pred_v: torch.Tensor,
        target_v: torch.Tensor,
        target_x1: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the decoupled amplitude and phase loss.

        Args:
            pred_v: Predicted velocity [B, 2, H, W] (v_amp, v_phase).
            target_v: Target velocity [B, 2, H, W] (v_amp, v_phase).
            target_x1: Clean cylindrical state for amplitude masking [B, 3, H, W].

        Returns:
            Tuple containing:
                - total_loss: Scalar total loss tensor.
                - loss_amp: Amplitude loss scalar.
                - loss_phi: Phase loss scalar.

        Raises:
            ValueError: If inputs do not have exactly 2 channels.
        """
        if pred_v.shape[1] != 2 or target_v.shape[1] != 2:
            raise ValueError(
                f"Expected 2-channel velocity (v_amp, v_phase), got "
                f"pred_v: {pred_v.shape[1]} channels, target_v: {target_v.shape[1]} channels"
            )

        loss_amp = self.amp_loss_fn(pred_v[:, 0:1], target_v[:, 0:1])

        if self.phase_loss_type == "cosine":
            raw_phase_err = 1.0 - torch.cos(pred_v[:, 1:2] - target_v[:, 1:2])
        else:
            raw_phase_err = self.phase_loss_fn(pred_v[:, 1:2], target_v[:, 1:2])

        if target_x1 is not None and self.phase_amplitude_weighting:
            mask = target_x1[:, 0:1].detach().abs()
            mask = mask / (mask.mean() + 1e-8)
            loss_phi = (raw_phase_err * mask).mean()
        else:
            loss_phi = raw_phase_err.mean()

        total_loss = loss_amp + (self.lambda_phase * loss_phi)

        return total_loss, loss_amp, loss_phi
