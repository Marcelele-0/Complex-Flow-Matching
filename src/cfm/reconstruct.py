"""Visual and numerical comparison of model reconstruction vs ground truth MRI.

Loads a trained model checkpoint and a real slice from the dataset, runs the
flow reconstruction from t_start to t=1, computes quality metrics (PSNR, SSIM,
Circular Phase Error), and saves side-by-side comparative figures:
1. Diagnostic 6-panel figure (Magnitude & Phase: GT, Pred, Error).
2. Clean 3-panel strip ready for paper inclusion (Figure 2 format).
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable
from typing import Any

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig

from cfm.data import build_dataset, build_geometry_transform
from cfm.data.transforms import Compose
from cfm.evaluate import integrate_from_t
from cfm.manifolds import Manifold, build_manifold
from cfm.utils.inference import build_model, load_weights, resolve_checkpoint
from cfm.utils.metrics import (
    circular_phase_error,
    peak_signal_noise_ratio,
    shortest_angular_difference,
    structural_similarity,
)


def reconstruct_slice(
    model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    manifold: Manifold,
    solver: Any,
    x_1: torch.Tensor,
    t_start: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run flow reconstruction on a single slice from t_start to t=1.

    Args:
        model: Velocity field (state, time) -> velocity.
        solver: ODE solver.
        manifold: Manifold for noise prior and bridge.
        x_1: Target tensor [1, C, H, W].
        t_start: Where on the noise-to-data path to start from (e.g. 0.5).
        generator: Optional RNG generator.

    Returns:
        tuple (reconstructed_state, corrupted_state) at t=1 and t=t_start.
    """
    if x_1.dim() == 3:
        x_1 = x_1.unsqueeze(0)

    b, _, h, w = x_1.shape
    x_0 = manifold.sample_noise(b, h, w, x_1.device, generator)

    t = torch.full((b, 1, 1, 1), t_start, device=x_1.device, dtype=torch.float32)
    x_t, _ = manifold.bridge(x_0, x_1, t)

    x_pred = integrate_from_t(model, solver, x_t, t_start)
    return x_pred, x_t


def compute_slice_metrics(
    pred_complex: torch.Tensor,
    target_complex: torch.Tensor,
    mask_threshold: float = 0.05,
) -> dict[str, float]:
    """Compute PSNR, SSIM and Circular Phase Error for a single complex slice.

    Args:
        pred_complex: Reconstructed complex tensor [1, H, W] or [1, 1, H, W].
        target_complex: Ground truth complex tensor of same shape.
        mask_threshold: Tissue amplitude threshold for phase error masking.

    Returns:
        Dict mapping metric name to scalar float.
    """
    if pred_complex.dim() == 2:
        pred_complex = pred_complex.unsqueeze(0).unsqueeze(0)
    elif pred_complex.dim() == 3:
        pred_complex = pred_complex.unsqueeze(0)

    if target_complex.dim() == 2:
        target_complex = target_complex.unsqueeze(0).unsqueeze(0)
    elif target_complex.dim() == 3:
        target_complex = target_complex.unsqueeze(0)

    psnr = peak_signal_noise_ratio(pred_complex, target_complex, data_range=1.0, reduction="mean")
    ssim = structural_similarity(pred_complex, target_complex, data_range=1.0, reduction="mean")
    phase_err = circular_phase_error(
        pred_complex, target_complex, mask_threshold=mask_threshold, reduction="mean"
    )

    return {
        "psnr_db": float(psnr.item()),
        "ssim": float(ssim.item()),
        "phase_error_rad": float(phase_err.item()),
    }


def render_diagnostic_figure(
    corrupted_complex: torch.Tensor,
    pred_complex: torch.Tensor,
    target_complex: torch.Tensor,
    metrics: dict[str, float],
    sample_id: str,
    t_start: float,
) -> plt.Figure:
    """Create a 2x4 comparison grid.

    Row 1 (Amplitude): Corrupted -> Reconstructed -> Ground Truth -> Difference Map
    Row 2 (Phase):     Corrupted -> Reconstructed -> Ground Truth -> Geodesic Phase Error
    """
    corrupted_np = corrupted_complex.detach().cpu().squeeze().numpy()
    pred_np = pred_complex.detach().cpu().squeeze().numpy()
    target_np = target_complex.detach().cpu().squeeze().numpy()

    # Magnitudes
    corr_mag = np.abs(corrupted_np)
    pred_mag = np.abs(pred_np)
    gt_mag = np.abs(target_np)
    diff_mag = np.abs(pred_mag - gt_mag)

    # Phases
    corr_phase = np.angle(corrupted_np)
    pred_phase = np.angle(pred_np)
    gt_phase = np.angle(target_np)

    diff_angle_tensor = shortest_angular_difference(
        torch.from_numpy(pred_phase), torch.from_numpy(gt_phase)
    )
    diff_phase = torch.abs(diff_angle_tensor).numpy()

    noise_pct = (1.0 - t_start) * 100.0

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # --- Row 1: Amplitude ---
    im0 = axes[0, 0].imshow(corr_mag, cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title(
        f"1. Corrupted Mag ({noise_pct:.0f}% Noise)", fontsize=11, fontweight="bold"
    )
    axes[0, 0].axis("off")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    im1 = axes[0, 1].imshow(pred_mag, cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("2. Reconstructed Mag", fontsize=11, fontweight="bold")
    axes[0, 1].axis("off")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    im2 = axes[0, 2].imshow(gt_mag, cmap="gray", vmin=0, vmax=1)
    axes[0, 2].set_title("3. Ground Truth Mag", fontsize=11, fontweight="bold")
    axes[0, 2].axis("off")
    fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    im3 = axes[0, 3].imshow(diff_mag, cmap="magma")
    axes[0, 3].set_title("4. Mag Diff |Pred - GT|", fontsize=11, fontweight="bold")
    axes[0, 3].axis("off")
    fig.colorbar(im3, ax=axes[0, 3], fraction=0.046, pad=0.04)

    # --- Row 2: Phase ---
    im4 = axes[1, 0].imshow(corr_phase, cmap="twilight", vmin=-math.pi, vmax=math.pi)
    axes[1, 0].set_title(
        f"1. Corrupted Phase ({noise_pct:.0f}% Noise)", fontsize=11, fontweight="bold"
    )
    axes[1, 0].axis("off")
    fig.colorbar(im4, ax=axes[1, 0], fraction=0.046, pad=0.04, label="Radians")

    im5 = axes[1, 1].imshow(pred_phase, cmap="twilight", vmin=-math.pi, vmax=math.pi)
    axes[1, 1].set_title("2. Reconstructed Phase", fontsize=11, fontweight="bold")
    axes[1, 1].axis("off")
    fig.colorbar(im5, ax=axes[1, 1], fraction=0.046, pad=0.04, label="Radians")

    im6 = axes[1, 2].imshow(gt_phase, cmap="twilight", vmin=-math.pi, vmax=math.pi)
    axes[1, 2].set_title("3. Ground Truth Phase", fontsize=11, fontweight="bold")
    axes[1, 2].axis("off")
    fig.colorbar(im6, ax=axes[1, 2], fraction=0.046, pad=0.04, label="Radians")

    im7 = axes[1, 3].imshow(diff_phase, cmap="inferno", vmin=0, vmax=math.pi)
    axes[1, 3].set_title("4. Geodesic Phase Error", fontsize=11, fontweight="bold")
    axes[1, 3].axis("off")
    fig.colorbar(im7, ax=axes[1, 3], fraction=0.046, pad=0.04, label="Rad Error")

    # Global Title & Statistics Banner
    title_line1 = (
        f"Reconstruction Benchmark: {sample_id}  "
        f"(t_start={t_start:.2f}, {noise_pct:.0f}% Noise Degradation)"
    )
    title_line2 = (
        f"STATISTICS:  PSNR = {metrics['psnr_db']:.2f} dB   |   "
        f"SSIM = {metrics['ssim']:.4f}   |   "
        f"Geodesic Phase Error = {metrics['phase_error_rad']:.4f} rad"
    )
    fig.suptitle(f"{title_line1}\n{title_line2}", fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout()
    return fig


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    orig_cwd = hydra.utils.get_original_cwd()

    recon_cfg = cfg.get("reconstruct", {})
    t_start = float(recon_cfg.get("t_start", 0.5))
    num_steps = int(recon_cfg.get("num_steps", 100))
    sample_idx = int(recon_cfg.get("sample_idx", 0))
    mask_threshold = float(recon_cfg.get("mask_threshold", 0.05))
    dpi = int(recon_cfg.get("dpi", 300))
    seed = int(recon_cfg.get("seed", 42))

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    # --- 1. Locate and Load Checkpoint ---
    explicit_ckpt = recon_cfg.get("checkpoint_path")
    if explicit_ckpt:
        checkpoint_path = (
            explicit_ckpt if os.path.isabs(explicit_ckpt) else os.path.join(orig_cwd, explicit_ckpt)
        )
    else:
        checkpoint_path = resolve_checkpoint(cfg, section="reconstruct", orig_cwd=orig_cwd)

    print(f"Loading checkpoint: {checkpoint_path}")
    manifold = build_manifold(cfg).to(device)
    model = build_model(
        cfg,
        device,
        in_channels=manifold.state_channels,
        out_channels=manifold.velocity_channels,
    )
    load_weights(model, checkpoint_path, device)

    # --- 2. Load Dataset Slice ---
    data_dir = recon_cfg.get("data_dir") or cfg.get("dataset", {}).get(
        "data_dir", "data/skm-tea-mini/v1-release"
    )
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(orig_cwd, data_dir)

    dataset_cfg = cfg.get("dataset", {})
    geometry = build_geometry_transform(dataset_cfg.get("crop_size"), crop_base=16)
    pipeline = Compose([geometry, manifold.build_transform(crop_base=16)])

    dataset = build_dataset(
        dataset_cfg,
        data_dir=data_dir,
        transform=pipeline,
    )

    if sample_idx < 0 or sample_idx >= len(dataset):
        raise IndexError(
            f"sample_idx {sample_idx} is out of bounds for dataset with {len(dataset)} slices."
        )

    file_path, slice_num = dataset.slice_map[sample_idx]
    sample_id = f"{os.path.basename(file_path)}_slice_{slice_num}"
    print(f"Selected target slice: {sample_id} (index {sample_idx})")

    # Load target slice [1, C, H, W]
    sample_data = dataset[sample_idx]
    if isinstance(sample_data, dict):
        x_1 = sample_data["target"].unsqueeze(0).to(device)
    else:
        x_1 = sample_data.unsqueeze(0).to(device)

    # --- 3. Run Reconstruction ---
    solver = manifold.make_solver(num_steps)

    print(f"Running reconstruction from t_start={t_start} with {num_steps} Heun steps...")
    with torch.no_grad():
        x_pred_state, x_corrupted_state = reconstruct_slice(
            model=model,
            manifold=manifold,
            solver=solver,
            x_1=x_1,
            t_start=t_start,
            generator=generator,
        )

    # --- 4. Domain Mapping & Metrics ---
    corrupted_complex = manifold.to_complex(x_corrupted_state)
    pred_complex = manifold.to_complex(x_pred_state)
    target_complex = manifold.to_complex(x_1)

    metrics = compute_slice_metrics(
        pred_complex=pred_complex,
        target_complex=target_complex,
        mask_threshold=mask_threshold,
    )

    print("-" * 50)
    print(f"RECONSTRUCTION METRICS ({sample_id}):")
    print(f"  • PSNR:                {metrics['psnr_db']:.2f} dB")
    print(f"  • SSIM:                {metrics['ssim']:.4f}")
    print(f"  • Geodesic Phase Error:{metrics['phase_error_rad']:.4f} rad")
    print("-" * 50)

    # --- 5. Render and Save Visualizations ---
    output_dir = recon_cfg.get("output_dir")
    if not output_dir:
        output_dir = os.path.join(orig_cwd, "outputs", "reconstruction")
    elif not os.path.isabs(output_dir):
        output_dir = os.path.join(orig_cwd, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    diag_fig = render_diagnostic_figure(
        corrupted_complex=corrupted_complex,
        pred_complex=pred_complex,
        target_complex=target_complex,
        metrics=metrics,
        sample_id=sample_id,
        t_start=t_start,
    )
    diag_path = os.path.join(output_dir, f"diagnostic_{sample_id}.png")
    diag_fig.savefig(diag_path, dpi=dpi)
    plt.close(diag_fig)
    print(f"Saved diagnostic figure to: {diag_path}")


if __name__ == "__main__":
    main()
