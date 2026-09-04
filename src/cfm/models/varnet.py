"""End-to-End Variational Network (VarNet) reconstructor implementation."""

from __future__ import annotations

from typing import Any

import torch
from fastmri.models import VarNet

from cfm.core.reconstructor import BaseReconstructor
from cfm.core.registry import MODELS, RECONSTRUCTORS


@MODELS.register("varnet")
@RECONSTRUCTORS.register("varnet")
class VarNetReconstructor(BaseReconstructor):
    """End-to-end Variational Network (VarNet) reconstructor for MRI.

    Wraps ``fastmri.models.VarNet``, an unrolled physics-based deep learning architecture
    combining soft data consistency with U-Net image-space regularizers and an integrated
    sensitivity estimation network.

    Args:
        num_cascades: Number of cascades (layers) for variational network.
        sens_chans: Number of channels for sensitivity map U-Net.
        sens_pools: Number of downsampling and upsampling layers for sensitivity map U-Net.
        chans: Number of channels for cascade U-Net.
        pools: Number of downsampling and upsampling layers for cascade U-Net.
        mask_center: Whether to mask center of k-space for sensitivity map calculation.
        **kwargs: Additional unused keyword arguments for registry compatibility.
    """

    def __init__(
        self,
        num_cascades: int = 12,
        sens_chans: int = 8,
        sens_pools: int = 4,
        chans: int = 18,
        pools: int = 4,
        mask_center: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_cascades = num_cascades
        self.sens_chans = sens_chans
        self.sens_pools = sens_pools
        self.chans = chans
        self.pools = pools
        self.mask_center = mask_center

        self.varnet = VarNet(
            num_cascades=num_cascades,
            sens_chans=sens_chans,
            sens_pools=sens_pools,
            chans=chans,
            pools=pools,
            mask_center=mask_center,
        )

    def reconstruct(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        sensitivity_maps: torch.Tensor | None = None,
        num_steps: int | None = None,
        num_low_frequencies: int | None = None,
    ) -> torch.Tensor:
        """Reconstruct clean complex MRI image from undersampled k-space data.

        Adapts complex k-space [B, C, H, W] and mask [B, 1, H, W] to fastMRI VarNet
        format [B, C, H, W, 2] and [B, 1, H, W, 1] boolean. VarNet estimates coil
        sensitivity maps internally.

        Args:
            masked_kspace: Undersampled k-space tensor [B, 1, H, W] or [B, C, H, W] complex.
            mask: Binary sampling mask [B, 1, H, W] or [1, 1, H, W].
            sensitivity_maps: Optional coil sensitivity maps. VarNet learns sensitivity
                estimation internally via its SensitivityModel, so this argument is unused.
            num_steps: Optional number of solver integration steps (unused for unrolled models).
            num_low_frequencies: Optional number of center low-frequency lines.

        Returns:
            Reconstructed image [B, 1, H, W] complex tensor.
        """
        del sensitivity_maps, num_steps

        # Ensure batch dimension
        if masked_kspace.dim() == 3:
            masked_kspace = masked_kspace.unsqueeze(0)

        # Convert k-space to fastMRI format: [B, C, H, W, 2]
        if torch.is_complex(masked_kspace):
            kspace_varnet = torch.view_as_real(masked_kspace)
        elif masked_kspace.dim() == 5 and masked_kspace.shape[-1] == 2:
            kspace_varnet = masked_kspace
        else:
            raise ValueError(
                f"Expected complex tensor [B, C, H, W] or real tensor [B, C, H, W, 2], "
                f"got shape {masked_kspace.shape} with dtype {masked_kspace.dtype}"
            )

        batch_size = kspace_varnet.shape[0]

        # Adapt mask shape to [B, 1, H, W, 1] boolean tensor
        if mask.dim() == 2:
            # [H, W] -> [1, 1, H, W, 1]
            mask_varnet = mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1)
        elif mask.dim() == 3:
            # [B, H, W] -> [B, 1, H, W, 1]
            mask_varnet = mask.unsqueeze(1).unsqueeze(-1)
        elif mask.dim() == 4:
            # [B, 1, H, W] -> [B, 1, H, W, 1]
            mask_varnet = mask.unsqueeze(-1)
        elif mask.dim() == 5:
            mask_varnet = mask
        else:
            raise ValueError(f"Unexpected mask shape {mask.shape}")

        mask_varnet = mask_varnet.to(dtype=torch.bool)
        if mask_varnet.shape[0] != batch_size:
            mask_varnet = mask_varnet.expand(batch_size, -1, -1, -1, -1)

        if num_low_frequencies is None:
            # Robustly calculate num_low_frequencies for 1D and 2D masks
            # Project along H (dim=2) to get 1D mask along W
            proj = mask_varnet[0, 0, :, :, 0].any(dim=0).to(torch.int8)
            cent = proj.shape[0] // 2

            left_half = proj[:cent].flip(0)
            left_zero = int(torch.argmin(left_half).item())
            if left_half[left_zero] != 0:
                left_zero = cent

            right_half = proj[cent:]
            right_zero = int(torch.argmin(right_half).item())
            if right_half[right_zero] != 0:
                right_zero = proj.shape[0] - cent

            num_low_frequencies = int(max(2 * min(left_zero, right_zero), 1))

        # Execute VarNet forward pass
        out = self.varnet(
            masked_kspace=kspace_varnet,
            mask=mask_varnet,
            num_low_frequencies=num_low_frequencies,
        )

        # VarNet returns RSS magnitude [B, H, W]; adapt to [B, 1, H, W] complex
        if out.dim() == 3:
            out = out.unsqueeze(1)

        return out.to(torch.complex64)

    def forward(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        sensitivity_maps: torch.Tensor | None = None,
        num_steps: int | None = None,
        num_low_frequencies: int | None = None,
    ) -> torch.Tensor:
        """Forward pass delegating to reconstruct().

        Args:
            masked_kspace: Undersampled k-space tensor [B, 1, H, W] or [B, C, H, W] complex.
            mask: Binary sampling mask [B, 1, H, W] or [1, 1, H, W].
            sensitivity_maps: Optional coil sensitivity maps (unused).
            num_steps: Optional solver steps (unused).
            num_low_frequencies: Optional center low-frequency lines.

        Returns:
            Reconstructed image [B, 1, H, W] complex tensor.
        """
        return self.reconstruct(
            masked_kspace=masked_kspace,
            mask=mask,
            sensitivity_maps=sensitivity_maps,
            num_steps=num_steps,
            num_low_frequencies=num_low_frequencies,
        )
