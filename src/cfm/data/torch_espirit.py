"""Pure PyTorch implementation of the ESPIRiT algorithm for sensitivity map calibration."""

from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np
import torch


def compute_espirit_torch(
    kspace: torch.Tensor | np.ndarray,
    calib_width: int = 24,
    kernel_width: int = 6,
    thresh: float = 0.02,
    crop: float = 0.95,
    max_iter: int = 30,
    device: torch.device | str | None = None,
    kernel_chunk: int = 8,
) -> torch.Tensor:
    """Compute ESPIRiT sensitivity maps for a multi-coil k-space slice on GPU.

    Follows Uecker et al. (2014) with the same conventions as
    :class:`sigpy.mri.app.EspiritCalib`, and is verified against it in
    ``tests/test_data/test_espirit.py``.

    Args:
        kspace: [num_coils, H, W] complex tensor or numpy array.
        calib_width: Central autocalibration region (ACS) dimension.
        kernel_width: Calibration Hankel kernel size.
        thresh: Eigenvalue threshold for signal subspace selection.
        crop: Eigenvalue cropping threshold for background tissue mask (0.95).
        max_iter: Maximum number of power iterations (30 ensures convergence).
        device: Target computation device (defaults to 'cuda' if available else 'cpu').
        kernel_chunk: Kernels whose image-domain covariance is accumulated at once.
            Bounds peak memory without changing the result.

    Returns:
        [num_coils, H, W] complex64 tensor of normalized sensitivity maps.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if isinstance(kspace, np.ndarray):
        if kspace.dtype.names is not None:
            if "r" in kspace.dtype.names and "i" in kspace.dtype.names:
                kspace = kspace["r"] + 1j * kspace["i"]
            elif "real" in kspace.dtype.names and "imag" in kspace.dtype.names:
                kspace = kspace["real"] + 1j * kspace["imag"]
        if not np.iscomplexobj(kspace) and kspace.shape[-1] == 2:
            kspace = kspace[..., 0] + 1j * kspace[..., 1]
        kspace = torch.from_numpy(np.asarray(kspace, dtype=np.complex64))

    kspace = kspace.to(device=device, dtype=torch.complex64)

    if kspace.ndim != 3:
        raise ValueError(f"Expected kspace ndim 3 ([C, H, W]), got {kspace.ndim}")

    num_coils, H, W = kspace.shape

    # 1. Central autocalibration region (ACS)
    c_h, c_w = H // 2, W // 2
    calib_w = min(calib_width, H, W)
    h_start, h_end = c_h - calib_w // 2, c_h + calib_w // 2
    w_start, w_end = c_w - calib_w // 2, c_w + calib_w // 2
    calib = kspace[:, h_start:h_end, w_start:w_end]

    # 2. Extract blocks: unfold spatial dims
    patches = calib.unfold(1, kernel_width, 1).unfold(2, kernel_width, 1)
    mat = (
        patches.reshape(num_coils, -1, kernel_width * kernel_width)
        .permute(1, 0, 2)
        .reshape(-1, num_coils * kernel_width * kernel_width)
    )

    # 3. SVD on calibration matrix
    _, S, Vh = torch.linalg.svd(mat, full_matrices=False)
    mask = S > (thresh * S[0])
    Vh = Vh[mask, :]
    num_kernels = Vh.shape[0]

    if num_kernels == 0:
        return torch.zeros_like(kspace)

    # Every kernel above the threshold spans the signal subspace. Truncating the set
    # discards part of that subspace and biases both the maps and the eigenvalue used
    # for the background mask, so memory is bounded by chunking the accumulation below
    # rather than by dropping kernels.
    kernels = Vh.view(num_kernels, num_coils, kernel_width, kernel_width)

    p_h = H // 2 - kernel_width // 2
    p_w = W // 2 - kernel_width // 2
    # Scale factor matching SigPy: prod(img_shape) / kernel_width**2
    scale = (H * W) / float(kernel_width * kernel_width)

    # 4-6. Zero-pad each kernel into the image grid, transform, and accumulate the
    # image-domain covariance sum_k v_k v_k^H one chunk of kernels at a time.
    AHA = torch.zeros((H, W, num_coils, num_coils), dtype=torch.complex64, device=device)
    for start in range(0, num_kernels, max(1, kernel_chunk)):
        block = kernels[start : start + max(1, kernel_chunk)]
        padded = torch.zeros(
            (block.shape[0], num_coils, H, W), dtype=torch.complex64, device=device
        )
        padded[:, :, p_h : p_h + kernel_width, p_w : p_w + kernel_width] = block

        # 5. Centered IFFT with ortho normalization
        padded = torch.fft.ifftshift(padded, dim=(-2, -1))
        img_kernels = torch.fft.ifft2(padded, dim=(-2, -1), norm="ortho")
        img_kernels = torch.fft.fftshift(img_kernels, dim=(-2, -1))

        # [H, W, num_coils, block] -> [H, W, num_coils, num_coils]
        blk = img_kernels.permute(2, 3, 1, 0)
        AHA += blk @ blk.mH

    AHA *= scale

    # 7. Vectorized power iteration
    mps = torch.ones((H, W, num_coils, 1), dtype=torch.complex64, device=device)
    max_eig = None
    for _ in range(max_iter):
        mps = torch.matmul(AHA, mps)
        norm = torch.linalg.norm(mps, dim=-2, keepdim=True).clamp_min(1e-8)
        max_eig = norm
        mps = mps / norm

    mps = mps.squeeze(-1).permute(2, 0, 1)  # [C, H, W]
    if max_eig is not None:
        max_eig = max_eig.squeeze(-1).squeeze(-1)  # [H, W]

    # Phase normalization with first coil
    phase0 = mps[0:1]  # [1, H, W]
    mps = mps * torch.conj(phase0 / torch.abs(phase0).clamp_min(1e-8))

    # Crop background noise by eigenvalue threshold
    if crop is not None and max_eig is not None:
        mps = mps * (max_eig > crop)

    return mps


def calibrate_fastmri_file_torch(
    src_path: str | Path,
    dest_path: str | Path,
    device: str = "cuda",
    max_iter: int = 30,
    overwrite: bool = False,
    sens_key: str = "sensitivity_maps",
    calib_width: int = 24,
    kernel_width: int = 6,
    thresh: float = 0.02,
    crop: float = 0.95,
    slices: Sequence[int] | None = None,
) -> bool:
    """Calibrate fastMRI multi-coil HDF5 file on GPU.

    Args:
        src_path: Source HDF5 file with 'kspace'.
        dest_path: Destination sidecar HDF5 file to write maps to.
        device: Device to run computation on.
        max_iter: Maximum number of power iterations.
        overwrite: Overwrite existing sensitivity maps sidecar.
        sens_key: Dataset name the sensitivity maps are stored under.
        calib_width: Central autocalibration region (ACS) dimension.
        kernel_width: Calibration Hankel kernel size.
        thresh: Eigenvalue threshold for signal subspace selection.
        crop: Eigenvalue cropping threshold for the background mask.
        slices: Slice indices to calibrate; ``None`` calibrates every slice. Each
            slice is calibrated from its own k-space alone -- see the loop below --
            so restricting this changes nothing about the maps that are produced,
            it only skips work whose result would be discarded. The sidecar keeps
            the volume's full shape so callers still index by the original slice
            number, and records what was covered in the ``calibrated_slices``
            attribute, so a later run cannot mistake a partial sidecar for a
            complete one.

    Returns:
        True if successfully computed, False otherwise.
    """
    src_path = Path(src_path)
    dest_path = Path(dest_path)

    # A sidecar without the maps is not a finished sidecar: skipping on mere existence
    # would leave the KeyError at read time that calibration exists to prevent.
    if dest_path.is_file() and not overwrite:
        with h5py.File(dest_path, "r") as existing:
            if sens_key in existing:
                return False

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    tmp_path = dest_path.with_suffix(f".tmp{os.getpid()}.h5")

    try:
        with h5py.File(src_path, "r") as src, h5py.File(tmp_path, "w") as dest:
            if "kspace" not in src:
                return False

            kspace_ds = src["kspace"]
            shape = kspace_ds.shape

            if len(shape) not in (3, 4):
                raise ValueError(f"Expected kspace ndim 3 or 4, got {len(shape)}")

            sens_ds = dest.create_dataset(
                sens_key,
                shape=shape,
                dtype=np.complex64,
                chunks=True,
            )

            def calibrate(ksp: np.ndarray) -> np.ndarray:
                return (
                    compute_espirit_torch(
                        ksp,
                        calib_width=calib_width,
                        kernel_width=kernel_width,
                        thresh=thresh,
                        crop=crop,
                        max_iter=max_iter,
                        device=device,
                    )
                    .cpu()
                    .numpy()
                )

            if len(shape) == 3:
                sens_ds[:] = calibrate(kspace_ds[:])
                sens_ds.attrs["calibrated_slices"] = "all"
            else:
                wanted = range(shape[0]) if slices is None else sorted({int(i) for i in slices})
                out_of_range = [i for i in wanted if not 0 <= i < shape[0]]
                if out_of_range:
                    raise ValueError(
                        f"slices {out_of_range} outside the volume's {shape[0]} slices"
                    )
                for s in wanted:
                    sens_ds[s] = calibrate(kspace_ds[s])
                sens_ds.attrs["calibrated_slices"] = (
                    "all" if slices is None else ",".join(str(i) for i in wanted)
                )

        os.replace(tmp_path, dest_path)
        return True
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
