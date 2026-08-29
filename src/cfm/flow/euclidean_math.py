"""Velocity loss for the flat Euclidean baseline.

The counterpart of :mod:`cfm.flow.torus_math`, deliberately much smaller: with
no manifold there is no decoupling, no angular term and no mask.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from cfm.flow.spectral import high_frequency_penalty

# "l2" and "mse" are aliases for the same squared-error loss, matching the
# vocabulary DecoupledCylindricalLoss already accepts.
VALID_VELOCITY_LOSSES = {"l1", "l2", "mse"}


class EuclideanVelocityLoss(nn.Module):
    """Plain regression loss on the 2-channel Euclidean velocity field.

    Velocity contract:
        Channel 0: v_re - real-part velocity
        Channel 1: v_im - imaginary-part velocity

    Both channels are scored by one unweighted loss. Two absences are deliberate:

    * **No per-channel weight.** ``lambda_phase`` exists on the cylinder because
      its channels have different units (amplitude vs rad/unit-time). Re and Im
      share a unit, so weighting one would be arbitrary.
    * **No amplitude mask.** The cylindrical loss masks its *phase* term because
      phase is undefined in air. A Re/Im field has no phase channel, so a mask
      here would import a cylindrical-specific correction into the baseline
      rather than remove one. ``target_x1`` is accepted and ignored only so both
      losses share a signature.

    The high-frequency k-space boost is shared verbatim with the cylindrical loss
    (:mod:`cfm.flow.spectral`), so an HF ablation means the same thing on both.
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
        """Compute total loss with optional high-frequency boosting.

        Args:
            pred_v: Predicted velocity [B, 2, H, W] (ch0: v_re, ch1: v_im)
            target_v: Target velocity from bridge [B, 2, H, W]
            target_x1: Accepted for interface parity and deliberately unused.

        Returns:
            Tuple of (total_loss, loss_vel, loss_hf)

        Raises:
            ValueError: If either velocity does not have exactly 2 channels.
        """
        # Fail loudly rather than silently broadcast a 3-channel cylindrical state.
        if pred_v.shape[1] != 2 or target_v.shape[1] != 2:
            raise ValueError(
                f"Expected 2-channel velocity (v_re, v_im), got "
                f"pred_v: {pred_v.shape[1]} channels, target_v: {target_v.shape[1]} channels"
            )

        del target_x1  # see class docstring: no masking in flat space

        loss_vel = self.velocity_loss_fn(pred_v, target_v)

        loss_hf = torch.tensor(0.0, device=pred_v.device, dtype=pred_v.dtype)
        if self.lambda_hf > 0.0:
            loss_hf = high_frequency_penalty(pred_v - target_v, self.hf_boost_factor)

        total_loss = loss_vel + (self.lambda_hf * loss_hf)

        return total_loss, loss_vel, loss_hf
