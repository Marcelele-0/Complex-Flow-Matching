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

# Set up figure with exact 1-row layout:
# Columns:
# [0] Text A, [1] GT Mag, [2] GT Pha, [3] Gap,
# [4] Text B, [5] Euc Mag, [6] Euc Mag Err, [7] Euc Pha, [8] Euc Pha Err, [9] Gap,
# [10] Text C, [11] Cyl Mag, [12] Cyl Mag Err, [13] Cyl Pha, [14] Cyl Pha Err

width_ratios = [
    1.4, 0.9, 0.9,       # (a) Ground Truth
    0.35,                # Gap
    1.4, 0.9, 0.9, 0.9, 0.9, # (b) Euclidean
    0.35,                # Gap
    1.4, 0.9, 0.9, 0.9, 0.9  # (c) Cylindrical
]

fig = plt.figure(figsize=(15.5, 1.85), dpi=300)
gs = GridSpec(1, len(width_ratios), figure=fig, width_ratios=width_ratios, wspace=0.04, left=0.01, right=0.99, top=0.95, bottom=0.05)

# --- (a) Ground Truth ---
ax_t0 = fig.add_subplot(gs[0, 0])
ax_t0.text(0.0, 0.65, "(a) Ground Truth", fontsize=9.5, fontweight="bold", va="center")
ax_t0.text(0.0, 0.32, "Reference Target\nFull Complex Field", fontsize=7.5, color="#444444", va="center", linespacing=1.2)
ax_t0.axis("off")

ax_a1 = fig.add_subplot(gs[0, 1])
ax_a1.imshow(gt_mag)
ax_a1.axis("off")

ax_a2 = fig.add_subplot(gs[0, 2])
ax_a2.imshow(gt_pha)
ax_a2.axis("off")

# Gap
ax_gap1 = fig.add_subplot(gs[0, 3])
ax_gap1.axis("off")

# --- (b) Euclidean Baseline ---
ax_t1 = fig.add_subplot(gs[0, 4])
ax_t1.text(0.0, 0.70, "(b) Euclidean ($\\mathbb{R}^2$)", fontsize=9.5, fontweight="bold", va="center")
ax_t1.text(0.0, 0.32, "PSNR: 25.35 dB\nSSIM: 0.6460\nCPE: 0.5261 rad", fontsize=7.5, color="#222222", va="center", linespacing=1.2)
ax_t1.axis("off")

ax_b1 = fig.add_subplot(gs[0, 5])
ax_b1.imshow(euc_mag)
ax_b1.axis("off")

ax_b2 = fig.add_subplot(gs[0, 6])
ax_b2.imshow(euc_mag_err)
ax_b2.axis("off")

ax_b3 = fig.add_subplot(gs[0, 7])
ax_b3.imshow(euc_pha)
ax_b3.axis("off")

ax_b4 = fig.add_subplot(gs[0, 8])
ax_b4.imshow(euc_pha_err)
ax_b4.axis("off")

# Gap
ax_gap2 = fig.add_subplot(gs[0, 9])
ax_gap2.axis("off")

# --- (c) Cylindrical Flow (Ours) ---
ax_t2 = fig.add_subplot(gs[0, 10])
ax_t2.text(0.0, 0.70, "(c) Cylindrical (Ours)", fontsize=9.5, fontweight="bold", color="#0044aa", va="center")
ax_t2.text(0.0, 0.32, "PSNR: 28.39 dB\nSSIM: 0.7624\nCPE: 0.2347 rad", fontsize=7.5, fontweight="bold", color="#003388", va="center", linespacing=1.2)
ax_t2.axis("off")

ax_c1 = fig.add_subplot(gs[0, 11])
ax_c1.imshow(cyl_mag)
ax_c1.axis("off")

ax_c2 = fig.add_subplot(gs[0, 12])
ax_c2.imshow(cyl_mag_err)
ax_c2.axis("off")

ax_c3 = fig.add_subplot(gs[0, 13])
ax_c3.imshow(cyl_pha)
ax_c3.axis("off")

ax_c4 = fig.add_subplot(gs[0, 14])
ax_c4.imshow(cyl_pha_err)
ax_c4.axis("off")

out_file = "paper/NIPS workshop/figures/figure2_results.png"
plt.savefig(out_file, bbox_inches="tight", pad_inches=0.01)
plt.close()
print("Saved panoramic horizontal Figure 2 to", out_file)
