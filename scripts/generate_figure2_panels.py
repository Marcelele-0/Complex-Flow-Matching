import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

from cfm.evaluate import integrate_from_t
from cfm.manifolds import CylindricalManifold, EuclideanManifold
from cfm.models.cylindrical_unet_attention import CylindricalUNetAttention
from cfm.utils.inference import load_weights
from cfm.utils.metrics import (
    circular_phase_error,
    peak_signal_noise_ratio,
    structural_similarity,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 1. Manifolds and models
cyl_manifold = CylindricalManifold().to(device)
euc_manifold = EuclideanManifold().to(device)

cyl_model = CylindricalUNetAttention(in_channels=3, out_channels=2).to(device)
cyl_ckpt = (
    "outputs/train/cylindrical_mini_bench/2026-08-29_18:26/checkpoints/checkpoint_epoch_25.pt"
)
load_weights(cyl_model, cyl_ckpt, device)

euc_model = CylindricalUNetAttention(in_channels=2, out_channels=2).to(device)
euc_ckpt = "outputs/train/euclidean_mini_bench/2026-08-29_18:27/checkpoints/checkpoint_epoch_25.pt"
load_weights(euc_model, euc_ckpt, device)

cyl_solver = cyl_manifold.make_solver(num_steps=100)
euc_solver = euc_manifold.make_solver(num_steps=100)

# 2. Load held-out test slice (MTR_030.h5, slice 256)
test_file = "data/skm-tea-mini/v1-release/files_recon_calib-24/MTR_030.h5"
with h5py.File(test_file, "r") as f:
    img_np = f["target"][256, :, :, 0, 0]
img_np = np.nan_to_num(img_np)
img_complex = torch.from_numpy(img_np).to(torch.complex64).unsqueeze(0)
h, w = img_complex.shape[-2:]
h_crop, w_crop = (h // 16) * 16, (w // 16) * 16
top, left = (h - h_crop) // 2, (w - w_crop) // 2
img_complex = img_complex[:, top : top + h_crop, left : left + w_crop]
scale = img_complex.abs().max().clamp(min=1e-8)
gt_complex = (img_complex / scale).unsqueeze(0).to(device)


# 3. Reconstruct
def run_recon(manifold, model, solver, t_start=0.5, seed=42):
    gen = torch.Generator(device=device).manual_seed(seed)
    x_1 = manifold.from_complex(gt_complex)
    b, _, h, w = x_1.shape
    x_0 = manifold.sample_noise(b, h, w, device, gen)
    t = torch.full((b, 1, 1, 1), t_start, device=device, dtype=torch.float32)
    x_t, _ = manifold.bridge(x_0, x_1, t)
    pred_state = integrate_from_t(model, solver, x_t, t_start)
    pred_complex = manifold.to_complex(pred_state)
    psnr = float(peak_signal_noise_ratio(pred_complex, gt_complex, data_range=1.0).item())
    ssim = float(structural_similarity(pred_complex, gt_complex, data_range=1.0).item())
    phase_err = float(circular_phase_error(pred_complex, gt_complex, mask_threshold=0.05).item())
    return pred_complex, psnr, ssim, phase_err


cyl_pred, cyl_psnr, cyl_ssim, cyl_cpe = run_recon(cyl_manifold, cyl_model, cyl_solver)
euc_pred, euc_psnr, euc_ssim, euc_cpe = run_recon(euc_manifold, euc_model, euc_solver)

print(f"Cylindrical Slice -> PSNR: {cyl_psnr:.2f} dB, SSIM: {cyl_ssim:.4f}, CPE: {cyl_cpe:.4f} rad")
print(f"Euclidean Slice   -> PSNR: {euc_psnr:.2f} dB, SSIM: {euc_ssim:.4f}, CPE: {euc_cpe:.4f} rad")

# 4. Convert to numpy arrays
gt_mag = gt_complex.abs().squeeze().detach().cpu().numpy()
gt_pha = np.angle(gt_complex.squeeze().detach().cpu().numpy())

cyl_mag = cyl_pred.abs().squeeze().detach().cpu().numpy()
cyl_pha = np.angle(cyl_pred.squeeze().detach().cpu().numpy())
cyl_mag_err = np.abs(cyl_mag - gt_mag)
cyl_pha_err = np.abs(np.arctan2(np.sin(cyl_pha - gt_pha), np.cos(cyl_pha - gt_pha)))

euc_mag = euc_pred.abs().squeeze().detach().cpu().numpy()
euc_pha = np.angle(euc_pred.squeeze().detach().cpu().numpy())
euc_mag_err = np.abs(euc_mag - gt_mag)
euc_pha_err = np.abs(np.arctan2(np.sin(euc_pha - gt_pha), np.cos(euc_pha - gt_pha)))

# Output directories
out_dir = "paper/NIPS workshop/figures/panels"
os.makedirs(out_dir, exist_ok=True)


# Helper to save raw image without any borders/text/whitespace
def save_raw(data, filename, cmap="gray", vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(4, 4), dpi=300)
    ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.axis("off")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    plt.savefig(os.path.join(out_dir, filename), pad_inches=0, bbox_inches="tight")
    plt.close()


# Save individual panels
save_raw(gt_mag, "gt_mag.png", cmap="gray", vmin=0, vmax=1)
save_raw(gt_pha, "gt_pha.png", cmap="twilight", vmin=-np.pi, vmax=np.pi)

save_raw(euc_mag, "euc_mag.png", cmap="gray", vmin=0, vmax=1)
save_raw(euc_mag_err, "euc_mag_err.png", cmap="inferno", vmin=0, vmax=0.25)
save_raw(euc_pha, "euc_pha.png", cmap="twilight", vmin=-np.pi, vmax=np.pi)
save_raw(euc_pha_err, "euc_pha_err.png", cmap="inferno", vmin=0, vmax=np.pi)

save_raw(cyl_mag, "cyl_mag.png", cmap="gray", vmin=0, vmax=1)
save_raw(cyl_mag_err, "cyl_mag_err.png", cmap="inferno", vmin=0, vmax=0.25)
save_raw(cyl_pha, "cyl_pha.png", cmap="twilight", vmin=-np.pi, vmax=np.pi)
save_raw(cyl_pha_err, "cyl_pha_err.png", cmap="inferno", vmin=0, vmax=np.pi)


# Also create glued composite pairs (Mag+Err and Pha+Err) for compact horizontal alignment:
def glue_pair(img1, img2, filename, cmap1, cmap2, vmin1, vmax1, vmin2, vmax2):
    fig, axes = plt.subplots(1, 2, figsize=(6, 3), dpi=300)
    axes[0].imshow(img1, cmap=cmap1, vmin=vmin1, vmax=vmax1)
    axes[0].axis("off")
    axes[1].imshow(img2, cmap=cmap2, vmin=vmin2, vmax=vmax2)
    axes[1].axis("off")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0.03)
    plt.margins(0, 0)
    plt.savefig(os.path.join(out_dir, filename), pad_inches=0, bbox_inches="tight")
    plt.close()


# Glued 2-panel pairs
glue_pair(gt_mag, gt_pha, "gt_pair.png", "gray", "twilight", 0, 1, -np.pi, np.pi)
glue_pair(euc_mag, euc_mag_err, "euc_mag_pair.png", "gray", "inferno", 0, 1, 0, 0.25)
glue_pair(euc_pha, euc_pha_err, "euc_pha_pair.png", "twilight", "inferno", -np.pi, np.pi, 0, np.pi)

glue_pair(cyl_mag, cyl_mag_err, "cyl_mag_pair.png", "gray", "inferno", 0, 1, 0, 0.25)
glue_pair(cyl_pha, cyl_pha_err, "cyl_pha_pair.png", "twilight", "inferno", -np.pi, np.pi, 0, np.pi)

print("All panels and pairs successfully generated in paper/NIPS workshop/figures/panels/!")
