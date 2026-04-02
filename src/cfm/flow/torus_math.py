import torch

class DecoupledCylindricalL1Loss(torch.nn.Module):
    """
    Loss function for flow matching on the decoupled manifold (R^+ x S^1).
    It optimizes the velocity vectors using L1 distance to maintain 
    robustness against extreme noise values and outliers in MRI data.
    """
    def __init__(self, lambda_phase: float = 1.0):
        super().__init__()
        # lambda_phase balances the importance of phase error vs magnitude error
        self.lambda_phase = lambda_phase

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Calculates the decoupled error between the predicted and target vector fields.
        
        Args:
            pred (torch.Tensor): Predicted velocities [Batch, 2, H, W]
                                 Channel 0: v_m (amplitude velocity)
                                 Channel 1: v_phi (angular velocity)
            target (torch.Tensor): Ground Truth velocities [Batch, 2, H, W]
            
        Returns:
            tuple: (total_loss, loss_amp, loss_phase) - components are returned 
                   separately for easier logging (e.g., in wandb/tensorboard).
        """
        # Decouple amplitude and phase components
        pred_v_m = pred[:, 0:1, :, :]
        pred_v_phi = pred[:, 1:2, :, :]

        target_v_m = target[:, 0:1, :, :]
        target_v_phi = target[:, 1:2, :, :]

        # L1 loss for amplitude velocity (zachowujemy wymiary przestrzenne)
        loss_amp = torch.abs(pred_v_m - target_v_m)

        # Circular L1 distance for phase velocity (shortest path on S^1)
        diff_phi = torch.abs(pred_v_phi - target_v_phi)
        diff_phi = torch.remainder(diff_phi, 2 * torch.pi)
        loss_phi = torch.minimum(diff_phi, 2 * torch.pi - diff_phi)

        # Aggregate (mean over batch and spatial dimensions) na samym końcu
        mean_loss_amp = loss_amp.mean()
        mean_loss_phi = loss_phi.mean()
        
        total_loss = mean_loss_amp + (self.lambda_phase * mean_loss_phi)

        return total_loss, mean_loss_amp, mean_loss_phi