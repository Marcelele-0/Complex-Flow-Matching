"""Base reconstructor interface and core reconstructor implementations for MRI."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from cfm.core.manifold import BaseManifold
from cfm.core.registry import RECONSTRUCTORS
from cfm.core.solver import BaseODESolver


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
    ) -> torch.Tensor:
        del mask, num_steps
        # IFFT from k-space to image space
        # Assuming centered k-space (fftshifted)
        x = torch.fft.ifft2(
            torch.fft.ifftshift(masked_kspace, dim=(-2, -1)),
            norm="ortho",
        )
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
    ) -> torch.Tensor:
        del sensitivity_maps
        b, _, h, w = masked_kspace.shape
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
            # Apply exact k-space data consistency projection
            k_pred = torch.fft.fftshift(torch.fft.fft2(x_complex, norm="ortho"), dim=(-2, -1))
            k_dc = mask * masked_kspace + (1.0 - mask) * k_pred
            x_complex = torch.fft.ifft2(torch.fft.ifftshift(k_dc, dim=(-2, -1)), norm="ortho")

        return x_complex
