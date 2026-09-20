# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
# ---

# %% [markdown]
# # Visual Comparison: Cylindrical Flow Matching vs Euclidean Baseline
#
# This notebook loads the trained checkpoints from our 2-patient training run on **SKM-TEA-mini** and visualizes the complex MRI reconstruction performance on:
# 1. **Seen Training Slice** (`MTR_201.h5` middle slice)
# 2. **Held-Out Test Slice** (`MTR_030.h5` middle slice)
#
# Both models are evaluated from $t_{\text{start}} = 0.5$ (50% noise degradation) over 100 Heun integration steps.

# %%
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from cyfm.evaluate import integrate_from_t
from cyfm.manifolds import CylindricalManifold, EuclideanManifold
from cyfm.models.cylindrical_unet_attention import CylindricalUNetAttention
from cyfm.utils.inference import load_weights
from cyfm.utils.metrics import (
    circular_phase_error,
    peak_signal_noise_ratio,
    structural_similarity,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

# %% [markdown]
# ## 1. Load Trained Checkpoints and Manifolds

# %%
cyl_manifold = CylindricalManifold().to(device)
euc_manifold = EuclideanManifold().to(device)

# Instantiate and load Cylindrical model
cyl_model = CylindricalUNetAttention(in_channels=3, out_channels=2).to(device)
cyl_ckpt = (
    "outputs/train/cylindrical_mini_bench/2026-08-29_18:26/checkpoints/checkpoint_epoch_25.pt"
)
load_weights(cyl_model, cyl_ckpt, device)

# Instantiate and load Euclidean baseline model
euc_model = CylindricalUNetAttention(in_channels=2, out_channels=2).to(device)
euc_ckpt = "outputs/train/euclidean_mini_bench/2026-08-29_18:27/checkpoints/checkpoint_epoch_25.pt"
load_weights(euc_model, euc_ckpt, device)

cyl_solver = cyl_manifold.make_solver(num_steps=100)
euc_solver = euc_manifold.make_solver(num_steps=100)
print("Both models and solvers successfully initialized!")


# %% [markdown]
# ## 2. Helper Functions for Loading & Reconstruction


# %%
def load_slice(file_path, slice_idx=256):
    with h5py.File(file_path, "r") as f:
        img_np = f["target"][slice_idx, :, :, 0, 0]
    img_np = np.nan_to_num(img_np)
    img_complex = torch.from_numpy(img_np).to(torch.complex64).unsqueeze(0)
    h, w = img_complex.shape[-2:]
    h_crop, w_crop = (h // 16) * 16, (w // 16) * 16
    top, left = (h - h_crop) // 2, (w - w_crop) // 2
    img_complex = img_complex[:, top : top + h_crop, left : left + w_crop]
    scale = img_complex.abs().max().clamp(min=1e-8)
    return (img_complex / scale).unsqueeze(0).to(device)


def reconstruct(gt_complex, manifold, model, solver, t_start=0.5, seed=42):
    gen = torch.Generator(device=device).manual_seed(seed)
    x_1 = manifold.from_complex(gt_complex)
    b, _, h, w = x_1.shape
    x_0 = manifold.sample_noise(b, h, w, device, gen)
    t = torch.full((b, 1, 1, 1), t_start, device=device, dtype=torch.float32)
    x_t, _ = manifold.bridge(x_0, x_1, t)
    pred_state = integrate_from_t(model, solver, x_t, t_start)
    pred_complex = manifold.to_complex(pred_state)
    corrupted_complex = manifold.to_complex(x_t)

    psnr = float(peak_signal_noise_ratio(pred_complex, gt_complex, data_range=1.0).item())
    ssim = float(structural_similarity(pred_complex, gt_complex, data_range=1.0).item())
    phase_err = float(circular_phase_error(pred_complex, gt_complex, mask_threshold=0.05).item())
    return {
        "pred": pred_complex,
        "corrupted": corrupted_complex,
        "psnr": psnr,
        "ssim": ssim,
        "phase_err": phase_err,
    }


# %% [markdown]
# ## 3. Visual Comparison: Seen Train Sample (`MTR_201.h5`)

# %%
train_file = "data/skm-tea-mini/v1-release/files_recon_calib-24/MTR_201.h5"
train_gt = load_slice(train_file, slice_idx=256)
train_cyl = reconstruct(train_gt, cyl_manifold, cyl_model, cyl_solver, t_start=0.5)
train_euc = reconstruct(train_gt, euc_manifold, euc_model, euc_solver, t_start=0.5)

print("TRAIN SAMPLE:")
print(
    f"  Cylindrical -> PSNR: {train_cyl['psnr']:.2f} dB | SSIM: {train_cyl['ssim']:.4f} | Phase Error: {train_cyl['phase_err']:.4f} rad"
)
print(
    f"  Euclidean   -> PSNR: {train_euc['psnr']:.2f} dB | SSIM: {train_euc['ssim']:.4f} | Phase Error: {train_euc['phase_err']:.4f} rad"
)

# %% [markdown]
# ## 4. Visual Comparison: Held-Out Test Sample (`MTR_030.h5`)

# %%
test_file = "data/skm-tea-mini/v1-release/files_recon_calib-24/MTR_030.h5"
test_gt = load_slice(test_file, slice_idx=256)
test_cyl = reconstruct(test_gt, cyl_manifold, cyl_model, cyl_solver, t_start=0.5)
test_euc = reconstruct(test_gt, euc_manifold, euc_model, euc_solver, t_start=0.5)

print("TEST SAMPLE (HELD-OUT):")
print(
    f"  Cylindrical -> PSNR: {test_cyl['psnr']:.2f} dB | SSIM: {test_cyl['ssim']:.4f} | Phase Error: {test_cyl['phase_err']:.4f} rad"
)
print(
    f"  Euclidean   -> PSNR: {test_euc['psnr']:.2f} dB | SSIM: {test_euc['ssim']:.4f} | Phase Error: {test_euc['phase_err']:.4f} rad"
)

# %% [markdown]
# ## 5. Paper Figure 2: Phase Boundary & Seam Artifact Comparison

# %%
gt_phase = np.angle(test_gt.squeeze().detach().cpu().numpy())
euc_phase = np.angle(test_euc["pred"].squeeze().detach().cpu().numpy())
cyl_phase = np.angle(test_cyl["pred"].squeeze().detach().cpu().numpy())

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(gt_phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
axes[0].set_title("(a) Ground Truth Phase", fontsize=14, fontweight="bold")
axes[0].axis("off")

axes[1].imshow(euc_phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
axes[1].set_title(
    "(b) Euclidean Flow ($R^2$)\nBoundary Seams & Phase Jump Artifacts",
    fontsize=14,
    fontweight="bold",
)
axes[1].axis("off")

im = axes[2].imshow(cyl_phase, cmap="twilight", vmin=-np.pi, vmax=np.pi)
axes[2].set_title(
    "(c) Cylindrical Flow (Ours)\nNative $S^1$ Geodesic Preservation",
    fontsize=14,
    fontweight="bold",
)
axes[2].axis("off")

fig.subplots_adjust(right=0.9)
cbar_ax = fig.add_axes([0.92, 0.2, 0.015, 0.6])
fig.colorbar(im, cax=cbar_ax, label="Phase (Radians)")
plt.show()
