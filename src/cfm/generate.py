import os
import math
import torch
import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig

# Project-specific imports
from cfm.models.cylindrical_unet import CylindricalUNet
from cfm.utils.complex_ops import cylinder_to_complex

# Assuming solver.py is in cfm/flow/ folder (change path if different)
from cfm.flow.solver import CylindricalODESolver

@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Inference on: {device}")

    # --- 1. Model Setup ---
    base_channels = cfg.model.get("base_channels", 64)
    model = CylindricalUNet(base_channels=base_channels).to(device)
    
    # Retrieve checkpoint directory from config or fallback to 'checkpoints'
    checkpoint_dir = cfg.get("paths", {}).get("checkpoint_dir", "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_epoch_50.pt")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} not found.")
        
    # Load state dict and handle 'torch.compile' prefixes if necessary
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    clean_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    
    model.load_state_dict(clean_state_dict)
    model.eval()
    print(f"Successfully loaded weights from {checkpoint_path}")

    # --- 2. Correct Cylindrical Noise Initialization (x_0) ---
    # We must match the training distribution: Amp in [0, 1] and Phase on circle [0, 2pi]
    # In SKM-TEA, original un-cropped slices are 512x512, modulo 16 crop retains this size.
    b, h, w = 1, 512, 512 
    
    noise_amp = torch.rand(b, 1, h, w, device=device)
    noise_phi = torch.rand(b, 1, h, w, device=device) * 2 * math.pi
    
    x_0 = torch.cat([
        noise_amp, 
        torch.cos(noise_phi), 
        torch.sin(noise_phi)
    ], dim=1)

    # --- 3. ODE Integration using generic Solver ---
    num_steps = 100 
    print(f"Integrating Cylindrical Flow with {num_steps} steps...")
    
    # Initialize the refactored solver
    ode_solver = CylindricalODESolver(num_steps=num_steps)
    
    # Run inference (sample method solves the loop from t=0 to t=1)
    x_1_cylindrical = ode_solver.sample(model, x_0)

    # --- 4. Back-Transformation to Complex Domain ---
    # Map [Amp, Cos, Sin] back to complex numbers
    x_1_complex = cylinder_to_complex(x_1_cylindrical)
    
    # Extract components for plotting
    magnitude = torch.abs(x_1_complex).squeeze().cpu().numpy()
    phase = torch.angle(x_1_complex).squeeze().cpu().numpy()

    # --- 5. Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    
    # Magnitude plot
    axes[0].imshow(magnitude, cmap="gray")
    axes[0].set_title("Generated MRI Magnitude")
    axes[0].axis("off")
    
    # Phase plot
    im_phase = axes[1].imshow(phase, cmap="twilight")
    axes[1].set_title("Generated MRI Phase")
    axes[1].axis("off")
    fig.colorbar(im_phase, ax=axes[1], fraction=0.046, pad=0.04, label="Radians")

    output_file = "generated_mri_final.png"
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Generation complete. Result saved to: {output_file}")

if __name__ == "__main__":
    main()