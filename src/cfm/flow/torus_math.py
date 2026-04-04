import torch
import torch.nn.functional as F


class DecoupledCylindricalLoss(torch.nn.Module):
    def __init__(self, amp_loss_type="l1", phase_loss_type="l1", lambda_phase=1.0):
        super().__init__()
        self.lambda_phase = lambda_phase
        
        # Initialize base losses (using reduction='none' for phase to apply the mask)
        self.amp_loss_fn = torch.nn.L1Loss(reduction='mean') if amp_loss_type == "l1" else torch.nn.MSELoss(reduction='mean')
        self.phase_loss_fn = torch.nn.L1Loss(reduction='none') if phase_loss_type == "l1" else torch.nn.MSELoss(reduction='none')

    def forward(self, pred_v, target_v, target_x1=None):
        """
        pred_v: [B, 3, H, W] - predicted velocity
        target_v: [B, 3, H, W] - target velocity from the bridge
        target_x1: [B, 3, H, W] - pure data (cylindrical) used as the amplitude mask
        """
        # 1. Amplitude Loss (channel 0) - always trained, ensures dark background
        loss_amp = self.amp_loss_fn(pred_v[:, 0:1], target_v[:, 0:1])

        # 2. Phase Loss (channels 1 and 2: cos/sin)
        raw_phase_err = self.phase_loss_fn(pred_v[:, 1:3], target_v[:, 1:3])

        if target_x1 is not None:
            # Create a mask from the amplitude of the clean image (target_x1 channel 0)
            # .detach() ensures that phase errors do not disrupt amplitude learning
            mask = target_x1[:, 0:1].detach().abs()
            
            # Normalize the mask to preserve the mean error scale
            mask = mask / (mask.mean() + 1e-8)
            
            # Attenuate the phase error where there is no tissue
            loss_phi = (raw_phase_err * mask).mean()
        else:
            loss_phi = raw_phase_err.mean()

        total_loss = loss_amp + self.lambda_phase * loss_phi
        
        return total_loss, loss_amp, loss_phi
