import torch
import torch.nn.functional as F


class DecoupledCylindricalLoss(torch.nn.Module):
    """
    Universal loss function for flow matching on the decoupled manifold (R^+ x S^1).
    Supports multiple distance metrics for rigorous ablation studies.
    """

    def __init__(
        self,
        amp_loss_type: str = "l1",
        phase_loss_type: str = "l1",
        lambda_phase: float = 1.0,
        huber_delta: float = 1.0,
    ):
        super().__init__()
        self.amp_loss_type = amp_loss_type.lower()
        self.phase_loss_type = phase_loss_type.lower()
        self.lambda_phase = lambda_phase
        self.huber_delta = huber_delta

        # Safety checks to prevent silent fails in Hydra configs
        if self.amp_loss_type not in ["l1", "l2", "huber"]:
            raise ValueError(f"Unsupported amp_loss_type: {self.amp_loss_type}")
        if self.phase_loss_type not in ["l1", "cosine"]:
            raise ValueError(f"Unsupported phase_loss_type: {self.phase_loss_type}")

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the decoupled error between the predicted and target vector fields.

        Args:
            pred (torch.Tensor): Predicted velocities [Batch, 2, H, W]
            target (torch.Tensor): Ground Truth velocities [Batch, 2, H, W]

        Returns:
            tuple: (total_loss, loss_amp, loss_phi)
        """
        # Decouple amplitude and phase components
        pred_m = pred[:, 0:1, :, :]
        pred_phi = pred[:, 1:2, :, :]

        target_m = target[:, 0:1, :, :]
        target_phi = target[:, 1:2, :, :]

        # --- Amplitude Loss (Linear movement) ---
        if self.amp_loss_type == "l1":
            loss_amp = torch.abs(pred_m - target_m)
        elif self.amp_loss_type == "l2":
            loss_amp = (pred_m - target_m) ** 2
        elif self.amp_loss_type == "huber":
            loss_amp = F.huber_loss(pred_m, target_m, reduction="none", delta=self.huber_delta)

        # --- Phase Loss (Angular movement on S^1) ---
        if self.phase_loss_type == "l1":
            # Circular L1 distance (shortest path on Torus)
            diff_phi = torch.abs(pred_phi - target_phi)
            diff_phi = torch.remainder(diff_phi, 2 * torch.pi)
            loss_phi = torch.minimum(diff_phi, 2 * torch.pi - diff_phi)

        elif self.phase_loss_type == "cosine":
            # Cosine angular loss: 1 - cos(theta_pred - theta_target)
            # Yields 0.0 for perfect match/full rotations, and 2.0 for maximum error (pi)
            loss_phi = 1.0 - torch.cos(pred_phi - target_phi)

        # Aggregate (mean over batch and spatial dimensions)
        mean_loss_amp = loss_amp.mean()
        mean_loss_phi = loss_phi.mean()

        total_loss = mean_loss_amp + (self.lambda_phase * mean_loss_phi)

        return total_loss, mean_loss_amp, mean_loss_phi
