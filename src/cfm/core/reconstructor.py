"""Base reconstructor interface and core reconstructor implementations for MRI."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.nn as nn

from cfm.core.manifold import BaseManifold
from cfm.core.registry import RECONSTRUCTORS
from cfm.core.solver import BaseODESolver
from cfm.utils.fft import fft2c, ifft2c


class BaseReconstructor(ABC, nn.Module):
    """Abstract base class for MRI reconstructors.

    Provides a standardized API across Flow Matching, Diffusion, Deep Equilibrium,
    and unrolled architectures (VarNet, MoDL, etc.).
    """

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def reconstruct(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        sensitivity_maps: torch.Tensor | None = None,
        num_steps: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Reconstruct clean complex MRI image from undersampled k-space data.

        Args:
            masked_kspace: Undersampled k-space tensor [B, 1, H, W] or [B, C, H, W] complex.
            mask: Binary sampling mask [B, 1, H, W] or [1, 1, H, W].
            sensitivity_maps: Optional coil sensitivity maps [B, C, H, W] complex.
            num_steps: Optional number of solver integration steps.

        Returns:
            Reconstructed image [B, 1, H, W] complex tensor.
        """


@RECONSTRUCTORS.register("zero_filled")
class ZeroFilledReconstructor(BaseReconstructor):
    """Zero-filled reconstruction baseline via inverse fast Fourier transform."""

    def reconstruct(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        sensitivity_maps: torch.Tensor | None = None,
        num_steps: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        del mask, num_steps
        x = ifft2c(masked_kspace)
        if sensitivity_maps is not None:
            # Coil combination: sum(S^* * x)
            x = (sensitivity_maps.conj() * x).sum(dim=1, keepdim=True)
        return x


@RECONSTRUCTORS.register("flow_matching")
class FlowMatchingReconstructor(BaseReconstructor):
    """Flow Matching MRI Reconstructor integrating probability trajectories.

    Args:
        model: Flow matching velocity prediction network.
        manifold: Manifold geometry defining prior, representation, and bridge.
        solver: Optional ODE solver. If None, built from manifold.make_solver().
        data_consistency: Whether to enforce k-space data consistency during / after sampling.
    """

    def __init__(
        self,
        model: nn.Module,
        manifold: BaseManifold,
        solver: BaseODESolver | None = None,
        data_consistency: bool = True,
    ) -> None:
        super().__init__()
        self.model = model
        self.manifold = manifold
        self.solver = solver
        self.data_consistency = data_consistency

    def reconstruct(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        sensitivity_maps: torch.Tensor | None = None,
        num_steps: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        b = masked_kspace.shape[0]
        h = masked_kspace.shape[-2]
        w = masked_kspace.shape[-1]
        device = masked_kspace.device

        steps = num_steps if num_steps is not None else 50
        solver = (
            self.solver
            if (self.solver is not None and num_steps is None)
            else self.manifold.make_solver(steps)
        )

        # Sample initial noise x_0
        noise = self.manifold.sample_noise(b, h, w, device)

        # Integrate ODE forward
        with torch.no_grad():
            x_pred_state = solver.sample(self.model, noise)

        # Map back to complex domain
        x_complex = self.manifold.to_complex(x_pred_state)

        if self.data_consistency:
            if sensitivity_maps is not None:
                # Multi-coil DC: project 1-channel image to coils via sensitivity maps
                k_pred = fft2c(sensitivity_maps * x_complex)
                k_dc = mask * masked_kspace + (1.0 - mask) * k_pred
                # SENSE combine back to 1 channel: sum(S^* * x)
                x_complex = (sensitivity_maps.conj() * ifft2c(k_dc)).sum(dim=1, keepdim=True)
            else:
                # Single-coil DC projection
                k_pred = fft2c(x_complex)
                k_dc = mask * masked_kspace + (1.0 - mask) * k_pred
                x_complex = ifft2c(k_dc)

        return x_complex


@RECONSTRUCTORS.register("diffusion")
@RECONSTRUCTORS.register("pc_diffusion")
class DiffusionReconstructor(BaseReconstructor):
    """Diffusion MRI Reconstructor using Predictor-Corrector sampling with Data Consistency.

    Args:
        model: Score prediction network (x, t) -> score.
        manifold: Optional complex diffusion manifold. If None, instantiates
            ComplexDiffusionManifold.
        solver: Optional PredictorCorrectorSolver. If None, built from manifold.make_solver().
        data_consistency: Whether to enforce k-space data consistency during / after sampling.
        num_steps: Default number of reverse integration steps.
    """

    def __init__(
        self,
        model: nn.Module,
        manifold: BaseManifold | None = None,
        solver: Any | None = None,
        data_consistency: bool = True,
        num_steps: int = 50,
    ) -> None:
        super().__init__()
        self.model = model
        self.manifold = manifold
        self.solver = solver
        self.data_consistency = data_consistency
        self.num_steps = num_steps

    def reconstruct(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        sensitivity_maps: torch.Tensor | None = None,
        num_steps: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        b = masked_kspace.shape[0]
        h = masked_kspace.shape[-2]
        w = masked_kspace.shape[-1]
        device = masked_kspace.device

        manifold = self.manifold
        if manifold is None:
            from cfm.manifolds.complex_diffusion import ComplexDiffusionManifold

            manifold = ComplexDiffusionManifold()

        steps = num_steps if num_steps is not None else self.num_steps
        solver: Any = (
            self.solver
            if (self.solver is not None and num_steps is None)
            else manifold.make_solver(steps)
        )

        noise = manifold.sample_noise(b, h, w, device)

        with torch.no_grad():
            x_pred_state = solver.sample(
                self.model,
                noise,
                masked_kspace=masked_kspace if self.data_consistency else None,
                mask=mask if self.data_consistency else None,
                sensitivity_maps=sensitivity_maps if self.data_consistency else None,
            )

        x_complex = manifold.to_complex(x_pred_state)

        if self.data_consistency:
            if sensitivity_maps is not None:
                k_pred = fft2c(sensitivity_maps * x_complex)
                k_dc = mask * masked_kspace + (1.0 - mask) * k_pred
                x_complex = (sensitivity_maps.conj() * ifft2c(k_dc)).sum(dim=1, keepdim=True)
            else:
                k_pred = fft2c(x_complex)
                k_dc = mask * masked_kspace + (1.0 - mask) * k_pred
                x_complex = ifft2c(k_dc)

        return x_complex
