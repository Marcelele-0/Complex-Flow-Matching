"""Base loss function interface and core loss implementations for Flow Matching."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from cyfm.core.registry import LOSSES


class BaseLoss(ABC, nn.Module):
    """Abstract base class for Flow Matching and Reconstruction losses.

    All loss modules must implement forward(), returning a 2-tuple:
    (total_loss, components_dict) where total_loss is the scalar for backpropagation
    and components_dict provides named component losses for metrics and logging.
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute loss between prediction and target.

        Args:
            pred: Predicted tensor (e.g. velocity field [B, V, H, W]).
            target: Target tensor (e.g. ground truth velocity [B, V, H, W]).
            **kwargs: Optional auxiliary tensors (e.g. clean data target_x1 for masking).

        Returns:
            Tuple of:
                - total_loss (torch.Tensor): Scalar tensor for backward pass.
                - components (dict[str, torch.Tensor]): Dictionary of scalar components for logging.
        """


@LOSSES.register("cylindrical")
@LOSSES.register("decoupled_cylindrical")
class CylindricalLoss(BaseLoss):
    """Decoupled cylindrical velocity loss conforming to BaseLoss interface."""

    def __init__(
        self,
        amp_loss_type: str = "l1",
        phase_loss_type: str = "l1",
        lambda_phase: float = 1.0,
        lambda_hf: float = 0.0,
        hf_boost_factor: float = 4.0,
        phase_amplitude_weighting: bool = True,
    ) -> None:
        super().__init__()
        from cyfm.flow.torus_math import DecoupledCylindricalLoss

        self._loss = DecoupledCylindricalLoss(
            amp_loss_type=amp_loss_type,
            phase_loss_type=phase_loss_type,
            lambda_phase=lambda_phase,
            lambda_hf=lambda_hf,
            hf_boost_factor=hf_boost_factor,
            phase_amplitude_weighting=phase_amplitude_weighting,
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        target_x1: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total, loss_amp, loss_phi, loss_hf = self._loss(
            pred_v=pred, target_v=target, target_x1=target_x1
        )
        return total, {"amp": loss_amp, "phi": loss_phi, "hf": loss_hf}


@LOSSES.register("euclidean")
@LOSSES.register("euclidean_velocity")
class EuclideanLoss(BaseLoss):
    """Euclidean velocity loss conforming to BaseLoss interface."""

    def __init__(
        self,
        loss_type: str = "l1",
        lambda_hf: float = 0.0,
        hf_boost_factor: float = 4.0,
    ) -> None:
        super().__init__()
        from cyfm.flow.euclidean_math import EuclideanVelocityLoss

        self._loss = EuclideanVelocityLoss(
            loss_type=loss_type,
            lambda_hf=lambda_hf,
            hf_boost_factor=hf_boost_factor,
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        target_x1: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total, loss_vel, loss_hf = self._loss(pred_v=pred, target_v=target, target_x1=target_x1)
        return total, {"vel": loss_vel, "hf": loss_hf}
