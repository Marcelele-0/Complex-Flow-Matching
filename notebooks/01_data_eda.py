# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
"""Download and prepare SKM-TEA mini dataset."""

import os

DATASET_DIR = "../data/skm-tea-mini/v1-release"
DATASET_URL = "https://huggingface.co/datasets/arjundd/skm-tea-mini/resolve/main/v1-release"

if not os.path.isdir(DATASET_DIR):
    os.makedirs(DATASET_DIR, exist_ok=True)

    print("Downloading configuration files and metadata...")
    files_to_download = [
        "all_metadata.csv",
        "annotations/v1.0.0/train.json",
        "annotations/v1.0.0/val.json",
        "annotations/v1.0.0/test.json",
    ]

    for fname in files_to_download:
        out_path = f"{DATASET_DIR}/{fname}"
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        os.system(f"wget -q {DATASET_URL}/{fname} -O {out_path}")

    print("Downloading and streaming extraction (k-space & masks only)...")
    tar_files = ["files_recon_calib-24", "segmentation_masks"]

    for fname in tar_files:
        tar_url = f"{DATASET_URL}/tarball/{fname}.tar.gz"
        print(f"Processing: {fname}...")
        os.system(f"wget -c {tar_url} -O - | tar -xz -C {DATASET_DIR}/")

    print("Dataset download and extraction complete.")
else:
    print("Dataset directory already exists. Skipping download.")

# %%
"""Load and visualize k-space hybrid data from first sample."""
import glob

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Locate first available k-space file
kspace_files: list[str] = glob.glob(f"{DATASET_DIR}/files_recon_calib-24/*.h5")
if not kspace_files:
    raise FileNotFoundError("No .h5 files found. Please verify download completed.")

sample_file: str = kspace_files[0]
print(f"Analyzing: {sample_file}")
unused_variable = "this should be removed"
x = 42

with h5py.File(sample_file, "r") as f:
    # Hybrid k-space shape: (Nx, Ny, Nz, num_echoes, num_coils)
    kspace: np.ndarray = f["kspace"][...]
    target: np.ndarray = f["target"][...]

    # Select center axial slice, first echo, first coil
    slice_idx: int = kspace.shape[0] // 2
    hybrid_slice: np.ndarray = kspace[slice_idx, :, :, 0, 0]

    # Reconstruct from hybrid k-space (x, ky, kz) to image domain (x, y, z)
    # X is already in image domain, apply 2D IFFT on Y-Z axes
    image_complex: np.ndarray = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(hybrid_slice)))

    # Decouple amplitude and phase (core ICLR paper formulation)
    amplitude: np.ndarray = np.abs(image_complex)
    # Phase wraps on [-pi, pi] (manifold structure)
    phase: np.ndarray = np.angle(image_complex)
    print("test")

    # Fully sampled ground truth reference
    ground_truth: np.ndarray = target[slice_idx, :, :, 0, 0]

    # Visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(np.log(np.abs(hybrid_slice) + 1e-6), cmap="gray")
    axes[0].set_title("Hybrid K-Space (Log Magnitude)")

    axes[1].imshow(amplitude, cmap="gray")
    axes[1].set_title("Reconstructed Amplitude")

    im_phase = axes[2].imshow(phase, cmap="twilight")
    axes[2].set_title("Phase (Toroidal Manifold T^d)")
    fig.colorbar(im_phase, ax=axes[2])

    axes[3].imshow(np.abs(ground_truth), cmap="gray")
    axes[3].set_title("Reference Magnitude")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

# %%
"""Analyze undersampling artifacts and multi-coil RSS reconstruction."""
import h5py
import matplotlib.pyplot as plt
import numpy as np


def reconstruct_rss(kspace_data: np.ndarray) -> np.ndarray:
    """
    Reconstruct image from k-space using Root Sum of Squares (RSS) coil combination.

    Args:
        kspace_data: Complex k-space tensor of shape (Ny, Nz, num_coils)

    Returns:
        Reconstructed magnitude image of shape (Ny, Nz)
    """
    # Apply 2D IFFT on spatial axes for all coils
    image_complex = np.fft.ifftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace_data, axes=(0, 1)), axes=(0, 1)), axes=(0, 1)
    )
    # Root sum of squares combines coils into single magnitude image
    return np.sqrt(np.sum(np.abs(image_complex) ** 2, axis=-1))


print(f"Analyzing undersampling and multi-coil reconstruction: {sample_file}")

with h5py.File(sample_file, "r") as f:
    kspace = f["kspace"][...]

    # Center slice, first echo, all coils. Shape: (Ny, Nz, num_coils)
    slice_idx = kspace.shape[0] // 2
    hybrid_slice_multi_coil = kspace[slice_idx, :, :, 0, :]

    num_y, num_z, num_coils = hybrid_slice_multi_coil.shape

    # Create undersampling mask (Cartesian acceleration R=4)
    # Keep center k-space (low frequencies) for baseline contrast
    acceleration_factor = 4
    center_fraction = 0.08
    num_center_lines = int(num_z * center_fraction)

    mask = np.zeros(num_z, dtype=np.float32)
    center_start = num_z // 2 - num_center_lines // 2
    center_end = center_start + num_center_lines

    # Keep center lines always
    mask[center_start:center_end] = 1.0

    # Randomly sample remaining lines to achieve R=4
    remaining_lines = int(num_z / acceleration_factor) - num_center_lines
    if remaining_lines > 0:
        leftover_indices = np.concatenate(
            [np.arange(0, center_start), np.arange(center_end, num_z)]
        )
        sampled_indices = np.random.choice(leftover_indices, remaining_lines, replace=False)
        mask[sampled_indices] = 1.0

    # Expand mask to match k-space dimensions
    mask_2d = np.tile(mask, (num_y, 1))
    mask_3d = np.expand_dims(mask_2d, axis=-1)

    # Apply mask (simulate accelerated scan)
    undersampled_kspace = hybrid_slice_multi_coil * mask_3d

    # Reconstruct both versions
    fully_sampled_img = reconstruct_rss(hybrid_slice_multi_coil)
    undersampled_img = reconstruct_rss(undersampled_kspace)

    # Visualization
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))

    axes[0].imshow(mask_2d, cmap="gray")
    axes[0].set_title(f"Sampling Mask (R={acceleration_factor}×)")

    axes[1].imshow(np.log(np.abs(undersampled_kspace[:, :, 0]) + 1e-6), cmap="gray")
    axes[1].set_title("Undersampled K-Space")

    axes[2].imshow(undersampled_img, cmap="gray")
    axes[2].set_title("Zero-Filled Recon (Aliasing Artifacts!)")

    axes[3].imshow(fully_sampled_img, cmap="gray")
    axes[3].set_title("Fully Sampled Reference")

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

# %%
"""Compare multi-echo signal: S+ (echo 1) vs S- (echo 2)."""
import h5py
import matplotlib.pyplot as plt
import numpy as np


def get_rss_for_echo(kspace_data: np.ndarray, echo_idx: int) -> np.ndarray:
    """
    Extract and reconstruct specific echo using RSS coil combination.

    Args:
        kspace_data: Full k-space tensor (Nx, Ny, Nz, num_echoes, num_coils)
        echo_idx: Echo index to extract

    Returns:
        RSS-reconstructed magnitude image (Ny, Nz)
    """
    slice_idx = kspace_data.shape[0] // 2
    hybrid_data = kspace_data[slice_idx, :, :, echo_idx, :]

    # 2D IFFT for all coils
    image_complex = np.fft.ifftshift(
        np.fft.ifft2(np.fft.ifftshift(hybrid_data, axes=(0, 1)), axes=(0, 1)), axes=(0, 1)
    )
    # RSS combination
    return np.sqrt(np.sum(np.abs(image_complex) ** 2, axis=-1))


print(f"Comparing dual-echo acquisition: {sample_file}")

with h5py.File(sample_file, "r") as f:
    kspace = f["kspace"][...]

    # Reconstruct both echoes
    echo1_magnitude = get_rss_for_echo(kspace, 0)  # S+ (high SNR anatomy)
    echo2_magnitude = get_rss_for_echo(kspace, 1)  # S- (T2-weighted fluid)

    # Contrast difference reveals complementary tissue information
    contrast_diff = echo1_magnitude - echo2_magnitude

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))

    # Normalize to same scale for comparison
    vmax = np.percentile(echo1_magnitude, 99)

    axes[0].imshow(echo1_magnitude, cmap="gray", vmax=vmax)
    axes[0].set_title("Echo 1 (S+): High SNR / Anatomical Detail")

    axes[1].imshow(echo2_magnitude, cmap="gray", vmax=vmax)
    axes[1].set_title("Echo 2 (S-): T2-Weighted / Fluid Sensitivity")

    im_diff = axes[2].imshow(contrast_diff, cmap="magma")
    axes[2].set_title("Contrast Difference (Echo1 - Echo2)")
    fig.colorbar(im_diff, ax=axes[2])

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()
    plt.show()

# %%
"""Analyze magnitude and phase distributions with frequency filtering."""
import h5py
import matplotlib.pyplot as plt
import numpy as np

with h5py.File(sample_file, "r") as f:
    kspace = f["kspace"][...]
    slice_idx = kspace.shape[0] // 2
    # Single echo, single coil for simplicity
    hybrid_slice: np.ndarray = kspace[slice_idx, :, :, 0, 0]

    # Image domain conversion
    image: np.ndarray = np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(hybrid_slice)))
    magnitude: np.ndarray = np.abs(image)
    phase: np.ndarray = np.angle(image)

    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Magnitude distribution analysis
    axes[0, 0].hist(magnitude.flatten(), bins=100, color="blue", alpha=0.7)
    axes[0, 0].set_title("Magnitude Histogram")
    axes[0, 0].set_ylabel("Frequency")
    # Log scale reveals structure: air (low) vs bone (high dynamic range)
    axes[0, 0].set_yscale("log")

    # Phase distribution on toroidal manifold
    axes[0, 1].hist(phase.flatten(), bins=100, color="purple", alpha=0.7)
    axes[0, 1].set_title("Phase Histogram (Toroidal Manifold [-π, π])")
    axes[0, 1].set_ylabel("Frequency")

    # Low-pass filter: keep only center k-space (contrast)
    height, width = hybrid_slice.shape
    center_radius = 20  # radius of center square
    mask_lowpass = np.zeros((height, width))
    mask_lowpass[
        height // 2 - center_radius : height // 2 + center_radius,
        width // 2 - center_radius : width // 2 + center_radius,
    ] = 1
    lowpass_kspace = hybrid_slice * mask_lowpass
    lowpass_img = np.abs(np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(lowpass_kspace))))

    axes[1, 0].imshow(lowpass_img, cmap="gray")
    axes[1, 0].set_title(f"Low-Pass Reconstruction (Center {2*center_radius}×{2*center_radius})")
    axes[1, 0].axis("off")

    # High-pass filter: keep only periphery k-space (details/edges)
    highpass_kspace = hybrid_slice * (1 - mask_lowpass)
    highpass_img = np.abs(np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(highpass_kspace))))

    axes[1, 1].imshow(highpass_img, cmap="gray")
    axes[1, 1].set_title("High-Pass Reconstruction (Edge Details)")
    axes[1, 1].axis("off")

    plt.tight_layout()
    plt.show()

# %%
"""Interactive 3D exploration with anatomical segmentation overlay."""
import os

import h5py
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from ipywidgets import IntSlider, interact

# Define paths
RAW_MASKS_DIR = "../data/skm-tea-mini/v1-release/segmentation_masks/raw-data-track"
sample_id: str = os.path.basename(sample_file).replace(".h5", "")
nifti_mask_path: str = os.path.join(RAW_MASKS_DIR, f"{sample_id}.nii.gz")


def explore_3d_slices(data_path: str, mask_path: str) -> None:
    """
    Interactive 3D volume explorer with lazy loading and anatomical overlay.

    Uses memory-mapped proxies to efficiently handle large multi-dimensional
    datasets. Displays amplitude, phase, and segmentation mask for each slice.

    Args:
        data_path: Path to HDF5 k-space file
        mask_path: Path to NIfTI segmentation mask
    """
    # Open files as proxies (no data in memory initially)
    f_h5 = h5py.File(data_path, "r")
    target_proxy = f_h5["target"]  # Proxy to HDF5 dataset

    mask_nifti = nib.load(mask_path)
    mask_proxy = mask_nifti.dataobj  # Proxy to NIfTI data

    num_slices: int = target_proxy.shape[0]

    @interact(
        slice_idx=IntSlider(
            min=0,
            max=num_slices - 1,
            step=1,
            value=num_slices // 2,
            description="Slice X:",
        )
    )
    def show_slice(slice_idx: int) -> None:
        """Display single slice with phase, magnitude, and mask overlay."""
        # Load only single slice from disk
        # target shape: (Nx, Ny, Nz, num_echoes, num_coils)
        complex_slice: np.ndarray = target_proxy[slice_idx, :, :, 0, 0]
        magnitude: np.ndarray = np.abs(complex_slice)
        phase_data: np.ndarray = np.angle(complex_slice)

        # Load corresponding mask slice
        # NIfTI typically: (Nx, Ny, Nz)
        mask_slice: np.ndarray = np.array(mask_proxy[slice_idx, :, :])

        fig, ax = plt.subplots(1, 3, figsize=(21, 7))
        vmax: float = np.percentile(magnitude, 99)

        # Magnitude with 99th percentile normalization
        ax[0].imshow(magnitude, cmap="gray", vmax=vmax)
        ax[0].set_title(f"Magnitude - Slice {slice_idx}")

        # Phase on toroidal manifold
        ax[1].imshow(phase_data, cmap="twilight")
        ax[1].set_title("Phase (Toroidal Manifold)")

        # Magnitude with segmentation overlay
        ax[2].imshow(magnitude, cmap="gray", vmax=vmax)
        if np.any(mask_slice):
            # Masked array hides zero values, springgreen highlights tissue
            mask_overlay = np.ma.masked_where(mask_slice == 0, mask_slice)
            ax[2].imshow(mask_overlay, cmap="spring", alpha=0.5)
        ax[2].set_title("Anatomical Segmentation Overlay")

        for a in ax:
            a.axis("off")
        plt.tight_layout()
        plt.show()

    # Launch interactive explorer
    show_slice(num_slices // 2)


# Run explorer
explore_3d_slices(sample_file, nifti_mask_path)

# %%
import h5py
import matplotlib.pyplot as plt
import numpy as np

with h5py.File(sample_file, "r") as f:
    # Read magnitude for both echoes at the center slice
    # Dimensions: (slice, y, z, echo, coil)
    slice_idx = f["kspace"].shape[0] // 2

    def get_magnitude(echo_idx: int) -> np.ndarray:
        # First coil is sufficient for this statistical comparison
        data = f["kspace"][slice_idx, :, :, echo_idx, 0]
        img = np.abs(np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(data))))
        return img.flatten()

    e1 = get_magnitude(0)
    e2 = get_magnitude(1)

    # Randomly sample 10k pixels to keep plotting fast
    n_samples = min(10_000, len(e1))
    indices = np.random.choice(len(e1), n_samples, replace=False)

    plt.figure(figsize=(10, 8))
    plt.scatter(e1[indices], e2[indices], alpha=0.1, s=1, color="teal")
    plt.xlabel("Echo 1 Intensity (S+)")
    plt.ylabel("Echo 2 Intensity (S-)")
    plt.title("Joint Distribution: Echo 1 vs Echo 2")

    # y=x reference line
    lim_max = max(e1[indices].max(), e2[indices].max())
    lims = [0, lim_max]
    plt.plot(lims, lims, "r--", alpha=0.5, label="y=x (Ideal Correlation)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

print(f"Pearson Correlation: {np.corrcoef(e1, e2)[0, 1]:.4f}")
