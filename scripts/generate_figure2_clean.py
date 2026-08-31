import os
import matplotlib.pyplot as plt
import numpy as np
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

# -------------------------------------------------------------
# Layout: 3 Horizontal Rows with Text on the Left, Images on the Right
# -------------------------------------------------------------
fig = plt.figure(figsize=(10, 4.2), dpi=300)
gs = GridSpec(3, 5, figure=fig, width_ratios=[1.8, 1, 1, 1, 1], hspace=0.15, wspace=0.04)

# Row 0: Ground Truth
ax_txt0 = fig.add_subplot(gs[0, 0])
ax_txt0.text(0.05, 0.65, "(a) Ground Truth", fontsize=11, fontweight="bold", va="center")
ax_txt0.text(0.05, 0.35, "Reference Target\nFull Complex Field", fontsize=9, color="#444444", va="center")
ax_txt0.axis("off")

ax0_0 = fig.add_subplot(gs[0, 1])
ax0_0.imshow(gt_mag)
ax0_0.set_title("Amplitude $A$", fontsize=9, pad=3)
ax0_0.axis("off")

ax0_1 = fig.add_subplot(gs[0, 2])
ax0_1.axis("off") # empty gap

ax0_2 = fig.add_subplot(gs[0, 3])
ax0_2.imshow(gt_pha)
ax0_2.set_title("Phase $\\theta$", fontsize=9, pad=3)
ax0_2.axis("off")

ax0_3 = fig.add_subplot(gs[0, 4])
ax0_3.axis("off") # empty gap

# Row 1: Euclidean Baseline
ax_txt1 = fig.add_subplot(gs[1, 0])
ax_txt1.text(0.05, 0.70, "(b) Euclidean ($\\mathbb{R}^2$)", fontsize=11, fontweight="bold", va="center")
ax_txt1.text(0.05, 0.35, "PSNR: 25.35 dB\nSSIM: 0.6460\nCPE: 0.5261 rad", fontsize=8.5, color="#222222", va="center")
ax_txt1.axis("off")

ax1_0 = fig.add_subplot(gs[1, 1])
ax1_0.imshow(euc_mag)
ax1_0.axis("off")

ax1_1 = fig.add_subplot(gs[1, 2])
ax1_1.imshow(euc_mag_err)
ax1_1.axis("off")

ax1_2 = fig.add_subplot(gs[1, 3])
ax1_2.imshow(euc_pha)
ax1_2.axis("off")

ax1_3 = fig.add_subplot(gs[1, 4])
ax1_3.imshow(euc_pha_err)
ax1_3.axis("off")

# Row 2: Cylindrical Flow (Ours)
ax_txt2 = fig.add_subplot(gs[2, 0])
ax_txt2.text(0.05, 0.70, "(c) Cylindrical (Ours)", fontsize=11, fontweight="bold", color="#0055aa", va="center")
ax_txt2.text(0.05, 0.35, "PSNR: 28.39 dB\nSSIM: 0.7624\nCPE: 0.2347 rad", fontsize=8.5, fontweight="bold", color="#003388", va="center")
ax_txt2.axis("off")

ax2_0 = fig.add_subplot(gs[2, 1])
ax2_0.imshow(cyl_mag)
ax2_0.axis("off")

ax2_1 = fig.add_subplot(gs[2, 2])
ax2_1.imshow(cyl_mag_err)
ax2_1.axis("off")

ax2_2 = fig.add_subplot(gs[2, 3])
ax2_2.imshow(cyl_pha)
ax2_2.axis("off")

ax2_3 = fig.add_subplot(gs[2, 4])
ax2_3.imshow(cyl_pha_err)
ax2_3.axis("off")

# Save cohesive tight figure
out_file = "paper/NIPS workshop/figures/figure2_results.png"
plt.savefig(out_file, bbox_inches="tight", pad_inches=0.02)
plt.close()
print("Clean figure saved to", out_file)
