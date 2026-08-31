import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.gridspec import GridSpec

out_dir = "paper/NIPS workshop/figures/panels"

gt_mag = plt.imread(f"{out_dir}/gt_mag.png")
gt_pha = plt.imread(f"{out_dir}/gt_pha.png")

euc_mag = plt.imread(f"{out_dir}/euc_mag.png")
euc_mag_err = plt.imread(f"{out_dir}/euc_mag_err.png")
euc_pha = plt.imread(f"{out_dir}/euc_pha.png")
euc_pha_err = plt.imread(f"{out_dir}/euc_pha_err.png")

cyl_mag = plt.imread(f"{out_dir}/cyl_mag.png")
cyl_mag_err = plt.imread(f"{out_dir}/cyl_mag_err.png")
cyl_pha = plt.imread(f"{out_dir}/cyl_pha.png")
cyl_pha_err = plt.imread(f"{out_dir}/cyl_pha_err.png")

# Width ratios with explicit micro-gaps:
# (a) Text, Mag, Pha
# Gap AB
# (b) Text, Mag, Pha, GapReconErr, MagErr, PhaErr
# Gap BC
# (c) Text, Mag, Pha, GapReconErr, MagErr, PhaErr

width_ratios = [
    # (a) Ground Truth (3 cols)
    1.45, 1.0, 1.0,
    # Gap (a) -> (b)
    0.45,
    # (b) Euclidean (6 cols: Text, Mag, Pha, mid-gap, MagErr, PhaErr)
    1.45, 1.0, 1.0, 0.15, 1.0, 1.0,
    # Gap (b) -> (c)
    0.45,
    # (c) Cylindrical (6 cols: Text, Mag, Pha, mid-gap, MagErr, PhaErr)
    1.45, 1.0, 1.0, 0.15, 1.0, 1.0,
]

# Slightly taller and larger for maximum sharpness:
fig = plt.figure(figsize=(16.5, 2.1), dpi=300)
gs = GridSpec(1, len(width_ratios), figure=fig, width_ratios=width_ratios, wspace=0.03, left=0.01, right=0.99, top=0.96, bottom=0.04)

# ==================== (a) Ground Truth ====================
ax_t0 = fig.add_subplot(gs[0, 0])
ax_t0.text(0.0, 0.65, "(a) Ground Truth", fontsize=10.5, fontweight="bold", va="center")
ax_t0.text(0.0, 0.32, "Reference Target\nFull Complex Field", fontsize=8.0, color="#444444", va="center", linespacing=1.25)
ax_t0.axis("off")

ax_a_mag = fig.add_subplot(gs[0, 1])
ax_a_mag.imshow(gt_mag)
ax_a_mag.axis("off")

ax_a_pha = fig.add_subplot(gs[0, 2])
ax_a_pha.imshow(gt_pha)
ax_a_pha.axis("off")

# Gap
fig.add_subplot(gs[0, 3]).axis("off")

# ==================== (b) Euclidean Baseline ====================
ax_t1 = fig.add_subplot(gs[0, 4])
ax_t1.text(0.0, 0.70, "(b) Euclidean ($\\mathbb{R}^2$)", fontsize=10.5, fontweight="bold", va="center")
ax_t1.text(0.0, 0.32, "PSNR: 25.35 dB\nSSIM: 0.6460\nCPE: 0.5261 rad", fontsize=8.0, color="#222222", va="center", linespacing=1.25)
ax_t1.axis("off")

# Reconstructions: Amp & Phase
ax_b_mag = fig.add_subplot(gs[0, 5])
ax_b_mag.imshow(euc_mag)
ax_b_mag.axis("off")

ax_b_pha = fig.add_subplot(gs[0, 6])
ax_b_pha.imshow(euc_pha)
ax_b_pha.axis("off")

# 15px-equivalent mid-gap between recon pair and error pair
fig.add_subplot(gs[0, 7]).axis("off")

# Error Maps: |dMag| & |dPhase|
ax_b_mag_err = fig.add_subplot(gs[0, 8])
ax_b_mag_err.imshow(euc_mag_err)
ax_b_mag_err.axis("off")

ax_b_pha_err = fig.add_subplot(gs[0, 9])
ax_b_pha_err.imshow(euc_pha_err)
ax_b_pha_err.axis("off")

# Gap
fig.add_subplot(gs[0, 10]).axis("off")

# ==================== (c) Cylindrical Flow (Ours) ====================
ax_t2 = fig.add_subplot(gs[0, 11])
ax_t2.text(0.0, 0.70, "(c) Cylindrical (Ours)", fontsize=10.5, fontweight="bold", color="#0044aa", va="center")
ax_t2.text(0.0, 0.32, "PSNR: 28.39 dB\nSSIM: 0.7624\nCPE: 0.2347 rad", fontsize=8.0, fontweight="bold", color="#003388", va="center", linespacing=1.25)
ax_t2.axis("off")

# Reconstructions: Amp & Phase
ax_c_mag = fig.add_subplot(gs[0, 12])
ax_c_mag.imshow(cyl_mag)
ax_c_mag.axis("off")

ax_c_pha = fig.add_subplot(gs[0, 13])
ax_c_pha.imshow(cyl_pha)
ax_c_pha.axis("off")

# 15px-equivalent mid-gap
fig.add_subplot(gs[0, 14]).axis("off")

# Error Maps: |dMag| & |dPhase|
ax_c_mag_err = fig.add_subplot(gs[0, 15])
ax_c_mag_err.imshow(cyl_mag_err)
ax_c_mag_err.axis("off")

ax_c_pha_err = fig.add_subplot(gs[0, 16])
ax_c_pha_err.imshow(cyl_pha_err)
ax_c_pha_err.axis("off")

out_file = "paper/NIPS workshop/figures/figure2_results.png"
plt.savefig(out_file, bbox_inches="tight", pad_inches=0.01)
plt.close()
print("Saved newly ordered and scaled Figure 2 to", out_file)
