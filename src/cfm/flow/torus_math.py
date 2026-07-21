import torch
import torch.nn as nn


class DecoupledCylindricalLoss(nn.Module):
    """Loss for cylindrical coordinates with optional high-frequency k-space boosting.

    Computes amplitude loss + phase loss + optional FFT-based high-frequency error boosting
    to enhance sharp features in k-space without compromising trajectory stability.
    """

    def __init__(
        self,
        amp_loss_type: str = "l1",
        phase_loss_type: str = "l1",
        lambda_phase: float = 1.0,
        lambda_hf: float = 0.0,
        hf_boost_factor: float = 4.0,
    ) -> None:
        super().__init__()
        self.lambda_phase = lambda_phase
        self.lambda_hf = lambda_hf
        self.hf_boost_factor = hf_boost_factor

        # Initialize base losses (using reduction='none' for phase to apply the mask)
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
        """Compute total loss with optional high-frequency boosting.

        Args:
            pred_v: Predicted velocity [B, 3, H, W]
            target_v: Target velocity from bridge [B, 3, H, W]
            target_x1: Pure data (cylindrical) for amplitude masking [B, 3, H, W]

        Returns:
            Tuple of (total_loss, loss_amp, loss_phi)
        """
        # 1. Amplitude Loss (channel 0) - always trained, ensures dark background
        loss_amp = self.amp_loss_fn(pred_v[:, 0:1], target_v[:, 0:1])

        # 2. Phase Loss (channels 1 and 2: cos/sin)
        raw_phase_err = self.phase_loss_fn(pred_v[:, 1:3], target_v[:, 1:3])

        if target_x1 is not None:
            # Create a mask from the amplitude of the clean image
            mask = target_x1[:, 0:1].detach().abs()
            # Normalize the mask to preserve the mean error scale
            mask = mask / (mask.mean() + 1e-8)
            # Attenuate the phase error where there is no tissue
            loss_phi = (raw_phase_err * mask).mean()
        else:
            loss_phi = raw_phase_err.mean()

        # === 3. HIGH-FREQUENCY K-SPACE BOOST ===
        loss_hf = torch.tensor(0.0, device=pred_v.device, dtype=pred_v.dtype)

        if self.lambda_hf > 0.0:
            # Compute error vector in spatial domain
            error_v = pred_v - target_v

            # Transform to frequency domain (FFT 2D) with low frequencies centered
            fft_err = torch.fft.fftshift(torch.fft.fft2(error_v, dim=(-2, -1)), dim=(-2, -1))
            fft_err_mag = torch.abs(fft_err)

            # Build radial mask favoring high frequencies (k-space periphery)
            b, c, h, w = pred_v.shape
            device = pred_v.device
            y, x = torch.meshgrid(
                torch.linspace(-1, 1, h, device=device),
                torch.linspace(-1, 1, w, device=device),
                indexing="ij",
            )
            radius = torch.sqrt(x**2 + y**2)  # Euclidean distance from center

            # At center: mask = 1.0, at periphery: linearly increases
            hf_mask = 1.0 + (self.hf_boost_factor * radius)
            hf_mask = hf_mask.unsqueeze(0).unsqueeze(0)

            # Apply mask and average error
            loss_hf = (fft_err_mag * hf_mask).mean()

        total_loss = loss_amp + (self.lambda_phase * loss_phi) + (self.lambda_hf * loss_hf)

        return total_loss, loss_amp, loss_phi
