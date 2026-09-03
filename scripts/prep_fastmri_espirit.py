"""Precomputes ESPIRiT coil sensitivity maps for fastMRI multi-coil HDF5 files.

Iterates over .h5 files, extracts multi-coil k-space data, computes sensitivity
maps via sigpy.mri.app.EspiritCalib, and saves them under 'sensitivity_maps'.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
from pathlib import Path

import h5py
import numpy as np
import sigpy as sp
import sigpy.mri.app as app

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _extract_complex_kspace(raw_kspace: np.ndarray) -> np.ndarray:
    """Standardize raw k-space numpy array into complex64 numpy array.

    Args:
        raw_kspace: Numpy array with complex or structured dtype, or [..., 2] float.

    Returns:
        Complex array with dtype np.complex64.
    """
    if raw_kspace.dtype.names is not None:
        if "r" in raw_kspace.dtype.names and "i" in raw_kspace.dtype.names:
            raw_kspace = raw_kspace["r"] + 1j * raw_kspace["i"]
        elif "real" in raw_kspace.dtype.names and "imag" in raw_kspace.dtype.names:
            raw_kspace = raw_kspace["real"] + 1j * raw_kspace["imag"]

    if not np.iscomplexobj(raw_kspace) and raw_kspace.shape[-1] == 2:
        raw_kspace = raw_kspace[..., 0] + 1j * raw_kspace[..., 1]

    return np.asarray(raw_kspace, dtype=np.complex64)


def compute_espirit_maps(
    kspace: np.ndarray,
    calib_width: int = 24,
    thresh: float = 0.02,
    kernel_width: int = 6,
    crop: float = 0.95,
    max_iter: int = 100,
    device: int = -1,
    show_pbar: bool = False,
) -> np.ndarray:
    """Compute ESPIRiT sensitivity maps for multi-coil k-space slice or volume.

    Args:
        kspace: Array of shape [num_coils, H, W] or [num_slices, num_coils, H, W].
        calib_width: ACS autocalibration region width.
        thresh: Eigenvalue threshold for calibration matrix.
        kernel_width: Kernel width for calibration matrix.
        crop: Threshold cropping for sensitivity maps.
        max_iter: Maximum power iterations.
        device: Device index (-1 for CPU, >= 0 for CUDA device).
        show_pbar: Whether to display progress bar during calibration.

    Returns:
        Sensitivity maps array with same shape as kspace and dtype np.complex64.
    """
    ksp = _extract_complex_kspace(kspace)
    sp_dev = sp.Device(device)

    if ksp.ndim == 3:
        # Single slice: [num_coils, H, W]
        eff_calib = min(calib_width, ksp.shape[-2], ksp.shape[-1])
        eff_kernel = min(kernel_width, ksp.shape[-2], ksp.shape[-1])
        calib_app = app.EspiritCalib(
            ksp,
            calib_width=eff_calib,
            thresh=thresh,
            kernel_width=eff_kernel,
            crop=crop,
            max_iter=max_iter,
            device=sp_dev,
            show_pbar=show_pbar,
        )
        return calib_app.run().astype(np.complex64)

    if ksp.ndim == 4:
        # Multi-slice volume: [num_slices, num_coils, H, W]
        num_slices = ksp.shape[0]
        eff_calib = min(calib_width, ksp.shape[-2], ksp.shape[-1])
        eff_kernel = min(kernel_width, ksp.shape[-2], ksp.shape[-1])

        maps_slices = []
        for s in range(num_slices):
            calib_app = app.EspiritCalib(
                ksp[s],
                calib_width=eff_calib,
                thresh=thresh,
                kernel_width=eff_kernel,
                crop=crop,
                max_iter=max_iter,
                device=sp_dev,
                show_pbar=show_pbar,
            )
            maps_slices.append(calib_app.run().astype(np.complex64))
        return np.stack(maps_slices, axis=0)

    raise ValueError(f"Expected kspace ndim 3 or 4, got shape {ksp.shape}")


def process_h5_file(
    file_path: str | Path,
    calib_width: int = 24,
    thresh: float = 0.02,
    kernel_width: int = 6,
    crop: float = 0.95,
    max_iter: int = 100,
    overwrite: bool = False,
    device: int = -1,
    show_pbar: bool = False,
) -> bool:
    """Compute and write sensitivity maps into an HDF5 file.

    Args:
        file_path: Path to the target .h5 file.
        calib_width: Calibration box size.
        thresh: Eigenvalue threshold.
        kernel_width: Kernel width.
        crop: Cropping factor.
        max_iter: Maximum iterations.
        overwrite: Overwrite existing 'sensitivity_maps' if present.
        device: Device index (-1 for CPU, >= 0 for CUDA).
        show_pbar: Whether to show progress bar.

    Returns:
        True if processed and written, False if skipped.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    with h5py.File(file_path, "r+") as hf:
        if "sensitivity_maps" in hf:
            if not overwrite:
                logger.info("Skipping %s: 'sensitivity_maps' already exists.", file_path.name)
                return False
            del hf["sensitivity_maps"]

        if "kspace" not in hf:
            logger.warning("Skipping %s: no 'kspace' dataset found.", file_path.name)
            return False

        kspace = hf["kspace"][:]
        sens_maps = compute_espirit_maps(
            kspace=kspace,
            calib_width=calib_width,
            thresh=thresh,
            kernel_width=kernel_width,
            crop=crop,
            max_iter=max_iter,
            device=device,
            show_pbar=show_pbar,
        )

        hf.create_dataset("sensitivity_maps", data=sens_maps, dtype=np.complex64, chunks=True)
        logger.info("Wrote sensitivity_maps shape %s to %s", sens_maps.shape, file_path.name)
        return True


def main() -> None:
    """CLI entrypoint for computing ESPIRiT maps across fastMRI HDF5 files."""
    parser = argparse.ArgumentParser(
        description="Compute ESPIRiT sensitivity maps for fastMRI multicoil files."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to directory containing .h5 files.",
    )
    parser.add_argument(
        "--file_pattern",
        type=str,
        default="*.h5",
        help="Glob pattern matching HDF5 files (default: '*.h5').",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search for .h5 files recursively in data_dir.",
    )
    parser.add_argument(
        "--calib_width",
        type=int,
        default=24,
        help="Calibration region width (default: 24).",
    )
    parser.add_argument(
        "--thresh",
        type=float,
        default=0.02,
        help="Eigenvalue threshold for calibration matrix (default: 0.02).",
    )
    parser.add_argument(
        "--kernel_width",
        type=int,
        default=6,
        help="Kernel width (default: 6).",
    )
    parser.add_argument(
        "--crop",
        type=float,
        default=0.95,
        help="Cropping threshold (default: 0.95).",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=100,
        help="Maximum power iterations (default: 100).",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=-1,
        help="Device id (-1 for CPU, >= 0 for CUDA device).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing 'sensitivity_maps' dataset.",
    )
    parser.add_argument(
        "--show_pbar",
        action="store_true",
        help="Display progress bar during ESPIRiT calibration.",
    )

    args = parser.parse_args()

    pattern = (
        os.path.join(args.data_dir, "**", args.file_pattern)
        if args.recursive
        else os.path.join(args.data_dir, args.file_pattern)
    )
    files = sorted(glob.glob(pattern, recursive=args.recursive))

    if not files:
        logger.warning("No files found matching pattern: %s", pattern)
        return

    logger.info("Found %d files to process in %s", len(files), args.data_dir)
    num_processed = 0
    for f in files:
        success = process_h5_file(
            file_path=f,
            calib_width=args.calib_width,
            thresh=args.thresh,
            kernel_width=args.kernel_width,
            crop=args.crop,
            max_iter=args.max_iter,
            overwrite=args.overwrite,
            device=args.device,
            show_pbar=args.show_pbar,
        )
        if success:
            num_processed += 1

    logger.info(
        "ESPIRiT precomputation finished. Processed %d/%d files.",
        num_processed,
        len(files),
    )


if __name__ == "__main__":
    main()
