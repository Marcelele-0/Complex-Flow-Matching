# Let's glue 4-panel blocks for Euclidean and Cylindrical to have perfect horizontal rows:
import os

import matplotlib.pyplot as plt

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


def save_quad(i1, i2, i3, i4, filename):
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.5), dpi=300)
    for ax, im in zip(axes, [i1, i2, i3, i4], strict=False):
        ax.imshow(im)
        ax.axis("off")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0.03)
    plt.margins(0, 0)
    plt.savefig(os.path.join(out_dir, filename), pad_inches=0, bbox_inches="tight")
    plt.close()


save_quad(euc_mag, euc_mag_err, euc_pha, euc_pha_err, "euc_quad.png")
save_quad(cyl_mag, cyl_mag_err, cyl_pha, cyl_pha_err, "cyl_quad.png")
print("Composite quad panels saved successfully!")
