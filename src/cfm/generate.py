import math
import os

import hydra
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig

# Assuming solver.py is in cfm/flow/ folder (change path if different)
from cfm.flow.solver import CylindricalODESolver

# Project-specific imports
from cfm.utils.complex_ops import cylinder_to_complex
from cfm.utils.inference import (
    build_model,
    load_weights,
    reject_unsupported_sampling_model,
    resolve_checkpoint,
)


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting Inference on: {device}")

    # --- 1. Model Setup ---
    reject_unsupported_sampling_model(cfg)
    model = build_model(cfg, device)

    # Checkpoint from generate.run_name, falling back to logging.experiment_name
    checkpoint_path = resolve_checkpoint(cfg, "generate", hydra.utils.get_original_cwd())
    load_weights(model, checkpoint_path, device)

    # --- 2. Correct Cylindrical Noise Initialization (x_0) ---
    # We must match the training distribution: Amp in [0, 1] and Phase on circle [0, 2pi]
    # In SKM-TEA, original un-cropped slices are 512x512, modulo 16 crop retains this size.
    b = cfg.get("generate", {}).get("num_samples", 5)
    h, w = 512, 512

    noise_amp = torch.rand(b, 1, h, w, device=device)
    noise_phi = torch.rand(b, 1, h, w, device=device) * 2 * math.pi

    x_0 = torch.cat([noise_amp, torch.cos(noise_phi), torch.sin(noise_phi)], dim=1)

    # --- 3. ODE Integration using generic Solver ---
    num_steps = cfg.get("generate", {}).get("num_steps", 100)
    print(f"Integrating Cylindrical Flow with {num_steps} steps for {b} samples...")

    # Initialize the refactored solver
    ode_solver = CylindricalODESolver(num_steps=num_steps)

    # Run inference (sample method solves the loop from t=0 to t=1)
    with torch.no_grad():
        x_1_cylindrical = ode_solver.sample(model, x_0)

    # --- 4. Back-Transformation to Complex Domain ---
    # Map [Amp, Cos, Sin] back to complex numbers
    x_1_complex = cylinder_to_complex(x_1_cylindrical)

    # Get the proper output directory from config (Hydra CWD by default)
    output_dir = cfg.get("paths", {}).get("output_dir", ".")
    os.makedirs(output_dir, exist_ok=True)

    # --- 5. Visualization ---
    for i in range(b):
        magnitude = torch.abs(x_1_complex[i]).squeeze().cpu().numpy()
        phase = torch.angle(x_1_complex[i]).squeeze().cpu().numpy()

        fig, axes = plt.subplots(1, 2, figsize=(14, 7))

        # Magnitude plot
        axes[0].imshow(magnitude, cmap="gray")
        axes[0].set_title(f"Generated MRI Magnitude [Sample {i+1}]")
        axes[0].axis("off")

        # Phase plot
        im_phase = axes[1].imshow(phase, cmap="twilight")
        axes[1].set_title(f"Generated MRI Phase [Sample {i+1}]")
        axes[1].axis("off")
        fig.colorbar(im_phase, ax=axes[1], fraction=0.046, pad=0.04, label="Radians")

        output_file = os.path.join(output_dir, f"generated_mri_sample_{i+1}.png")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close(fig)

    print(f"Generation complete. Resulted {b} images saved to: {output_dir}")


if __name__ == "__main__":
    main()
