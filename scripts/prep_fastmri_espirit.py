"""Precomputes ESPIRiT coil sensitivity maps for fastMRI multi-coil HDF5 files.

Iterates over .h5 files, extracts multi-coil k-space data, computes sensitivity
maps via sigpy.mri.app.EspiritCalib, and saves them under 'sensitivity_maps'.

The maps are the same dtype and shape as the k-space, so they are saved as
sidecar files in ``--output_dir`` (defaulting to ``<data_dir>_sens``). Source files
are strictly read-only and never modified.

The maps land on sigpy's centered image grid, which is the convention
:mod:`cyfm.utils.fft` implements and :class:`~cyfm.data.fastmri.FastMRIDataset`
consumes. Do not introduce a transform here without changing both.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
from pathlib import Path

from cyfm.data.espirit import (
    DEFAULT_SENS_KEY,
    SENS_KEY,
    compute_espirit_maps,
    ensure_espirit_maps,
    process_h5_file,
)

__all__ = [
    "DEFAULT_SENS_KEY",
    "SENS_KEY",
    "compute_espirit_maps",
    "ensure_espirit_maps",
    "main",
    "process_h5_file",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


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
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Write sidecar map files here instead of modifying the source archive. "
            "Defaults to '<data_dir>_sens'. Source files are never modified."
        ),
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
        "--num_workers",
        type=int,
        default=1,
        help=(
            "Volumes to calibrate in parallel (default: 1). ESPIRiT on CPU is the "
            "bottleneck for the full cohort; keep this at 1 when --device selects a GPU."
        ),
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
        help=f"Overwrite existing '{SENS_KEY}' dataset.",
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

    output_dir = args.output_dir
    if output_dir is None:
        data_p = Path(args.data_dir)
        output_dir = str(data_p.parent / f"{data_p.name}_sens")
        logger.info("No --output_dir given: writing sidecars to %s", output_dir)

    try:
        ensure_espirit_maps(
            files=files,
            output_dir=output_dir,
            num_workers=args.num_workers,
            show_pbar=args.show_pbar,
            device=args.device,
            calib_width=args.calib_width,
            thresh=args.thresh,
            kernel_width=args.kernel_width,
            crop=args.crop,
            max_iter=args.max_iter,
            overwrite=args.overwrite,
            sens_key=DEFAULT_SENS_KEY,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
