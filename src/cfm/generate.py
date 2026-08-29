"""Unconditional sampling from a trained flow, in whichever geometry it was trained.

The manifold supplies the prior, the ODE step and the map back to the complex
domain, so this script is identical for the cylindrical model and the Euclidean
baseline. Only ``manifold=`` changes.
"""

import os

import hydra
import matplotlib.pyplot as plt
import torch
from omegaconf import DictConfig

from cfm.manifolds import build_manifold
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

    # --- 1. Geometry and Model Setup ---
    reject_unsupported_sampling_model(cfg)

    # The manifold must match the one the checkpoint was trained under; a
    # mismatch shows up immediately as a state-dict shape error on init_conv.
    manifold = build_manifold(cfg).to(device)

    model = build_model(
        cfg,
        device,
        in_channels=manifold.state_channels,
        out_channels=manifold.velocity_channels,
    )

    # Checkpoint from generate.run_name, falling back to logging.experiment_name
    checkpoint_path = resolve_checkpoint(cfg, "generate", hydra.utils.get_original_cwd())
    load_weights(model, checkpoint_path, device)

    # --- 2. Noise Initialization (x_0) ---
    # Drawn by the manifold so it matches the training distribution exactly; any
    # other distribution puts the model off its trained support at t=0.
    b = cfg.get("generate", {}).get("num_samples", 5)
    # SKM-TEA slices are 512x512 natively and the modulo-16 crop retains that.
    image_size = cfg.get("generate", {}).get("image_size", 512)
    h, w = image_size, image_size

    x_0 = manifold.sample_noise(b, h, w, device)

    # --- 3. ODE Integration ---
    num_steps = cfg.get("generate", {}).get("num_steps", 100)
    print(f"Integrating {manifold.name} flow with {num_steps} steps for {b} samples...")

    ode_solver = manifold.make_solver(num_steps)

    # Run inference (sample method solves the loop from t=0 to t=1)
    with torch.no_grad():
        x_1_state = ode_solver.sample(model, x_0)

    # --- 4. Back-Transformation to Complex Domain ---
    x_1_complex = manifold.to_complex(x_1_state)

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
        axes[0].set_title(f"Generated MRI Magnitude [Sample {i + 1}]")
        axes[0].axis("off")

        # Phase plot
        im_phase = axes[1].imshow(phase, cmap="twilight")
        axes[1].set_title(f"Generated MRI Phase [Sample {i + 1}]")
        axes[1].axis("off")
        fig.colorbar(im_phase, ax=axes[1], fraction=0.046, pad=0.04, label="Radians")

        output_file = os.path.join(output_dir, f"generated_mri_sample_{i + 1}.png")
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        plt.close(fig)

    print(f"Generation complete. Resulted {b} images saved to: {output_dir}")


if __name__ == "__main__":
    main()
