import os
import hydra
import torch
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from cfm.models.cylindrical_unet import CylindricalUNet
from cfm.utils.complex_ops import cylinder_to_complex, complex_to_cylinder

@torch.no_grad()
def euler_solver(model: torch.nn.Module, x_0: torch.Tensor, steps: int, device: torch.device) -> torch.Tensor:
    """
    Solves the ODE using the Euler method from t=0 to t=1.
    """
    dt = 1.0 / steps
    x_t = x_0.clone()
    
    for i in range(steps):
        t_val = i / steps
        t_tensor = torch.full((x_0.shape[0],), t_val, device=device)
        
        # v_pred has 2 channels: [v_amp, v_phi]
        v_pred = model(x_t, t_tensor)
        
        v_amp = v_pred[:, 0:1]
        v_phi = v_pred[:, 1:2]
        
        # Extract current p_x (cos) and p_y (sin) from x_t
        p_x = x_t[:, 1:2]
        p_y = x_t[:, 2:3]
        
        # 1. Update amplitude (regular addition)
        new_amp = x_t[:, 0:1] + v_amp * dt
        
        # 2. Update phase on the cylinder (Chain rule)
        # d(cos)/dt = -sin * dphi/dt
        # d(sin)/dt =  cos * dphi/dt
        new_px = p_x + (-p_y * v_phi) * dt
        new_py = p_y + (p_x * v_phi) * dt
        
        # Re-normalize the phase vector (so it always lies on the unit circle)
        mag = torch.sqrt(new_px**2 + new_py**2 + 1e-8)
        new_px = new_px / mag
        new_py = new_py / mag
        
        x_t = torch.cat([new_amp, new_px, new_py], dim=1)
        
    return x_t

@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting generation on: {device}")

    # 1. Load the trained model
    base_channels = cfg.get("model", {}).get("base_channels", 64)
    model = CylindricalUNet(base_channels=base_channels).to(device)
    
    checkpoint_path = "checkpoint_epoch_100.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found. Let the training finish or adjust the name.")
        
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    
    # Remove _orig_mod. prefix if it exists
    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in checkpoint.items()}
    
    model.load_state_dict(new_state_dict)
    model.eval()
    print(f"Successfully loaded weights from {checkpoint_path}")

    # 2. Generate initial noise (x_0)
    # Using the exact same crop size as in training pipeline (e.g., 320x320)
    # You can adjust these dimensions to match the output of your CenterCropModulo
    b, h, w = 1, 320, 320 
    
    noise_real = torch.randn(b, 1, h, w, device=device)
    noise_imag = torch.randn(b, 1, h, w, device=device)
    complex_noise = torch.complex(noise_real, noise_imag)
    
    # Map to cylindrical manifold [B, 3, H, W]
    x_0 = complex_to_cylinder(complex_noise)

    # 3. Solve the ODE to get the clean image representation (x_1)
    num_steps = 300
    print(f"Integrating ODE with {num_steps} steps...")
    x_1_cylindrical = euler_solver(model, x_0, steps=num_steps, device=device)

    # 4. Map back to Complex Domain
    x_1_complex = cylinder_to_complex(x_1_cylindrical)
    
    # Extract magnitude and phase for visualization
    magnitude = torch.abs(x_1_complex).squeeze().cpu().numpy()
    phase = torch.angle(x_1_complex).squeeze().cpu().numpy()

    # 5. Plot the generated MRI scan
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(magnitude, cmap="gray")
    axes[0].set_title("Generated Magnitude")
    axes[0].axis("off")
    
    im_phase = axes[1].imshow(phase, cmap="twilight")
    axes[1].set_title("Generated Phase")
    axes[1].axis("off")
    fig.colorbar(im_phase, ax=axes[1], fraction=0.046, pad=0.04)

    output_file = "generated_mri.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Saved generated scan to {output_file}")

if __name__ == "__main__":
    main()