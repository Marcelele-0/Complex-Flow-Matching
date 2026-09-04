"""Precomputes ESPIRiT coil sensitivity maps for fastMRI multi-coil HDF5 files.

Iterates over .h5 files, extracts multi-coil k-space data, computes sensitivity
maps via sigpy.mri.app.EspiritCalib, and saves them under 'sensitivity_maps'.

The maps are the same dtype and shape as the k-space, so writing them back into
the source archive roughly doubles its footprint and leaves half-modified files
behind if the run is interrupted. Pass ``--output_dir`` to write sidecar files
instead, and point ``dataset.sens_dir`` at that directory; in-place writing is
kept as the default only so existing local datasets keep working.

The maps land on sigpy's centered image grid, which is the convention
:mod:`cfm.utils.fft` implements and :class:`~cfm.data.fastmri.FastMRIDataset`
consumes. Do not introduce a transform here without changing both.
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import sigpy as sp
import sigpy.mri.app as app

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

SENS_KEY = "sensitivity_maps"


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

    if ksp.ndim == 3:
        return _calibrate_slice(
            ksp, calib_width, thresh, kernel_width, crop, max_iter, device, show_pbar
        )

    if ksp.ndim == 4:
        maps_slices = [
            _calibrate_slice(
                ksp[s], calib_width, thresh, kernel_width, crop, max_iter, device, show_pbar
            )
            for s in range(ksp.shape[0])
        ]
        return np.stack(maps_slices, axis=0)

    raise ValueError(f"Expected kspace ndim 3 or 4, got shape {ksp.shape}")


def _calibrate_slice(
    ksp: np.ndarray,
    calib_width: int,
    thresh: float,
    kernel_width: int,
    crop: float,
    max_iter: int,
    device: int,
    show_pbar: bool,
) -> np.ndarray:
    """Run ESPIRiT calibration on one [num_coils, H, W] slice."""
    eff_calib = min(calib_width, ksp.shape[-2], ksp.shape[-1])
    eff_kernel = min(kernel_width, ksp.shape[-2], ksp.shape[-1])
    calib_app = app.EspiritCalib(
        np.ascontiguousarray(ksp),
        calib_width=eff_calib,
        thresh=thresh,
        kernel_width=eff_kernel,
        crop=crop,
        max_iter=max_iter,
        device=sp.Device(device),
        show_pbar=show_pbar,
    )
    return np.asarray(calib_app.run()).astype(np.complex64)


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
    output_dir: str | Path | None = None,
) -> bool:
    """Compute and write sensitivity maps for one HDF5 file.

    Slices are calibrated and written one at a time, so peak memory is one slice
    of maps rather than a whole volume.

    Args:
        file_path: Path to the source .h5 file holding 'kspace'.
        calib_width: Calibration box size.
        thresh: Eigenvalue threshold.
        kernel_width: Kernel width.
        crop: Cropping factor.
        max_iter: Maximum iterations.
        overwrite: Overwrite existing maps if present.
        device: Device index (-1 for CPU, >= 0 for CUDA).
        show_pbar: Whether to show progress bar.
        output_dir: Write a sidecar file here instead of modifying the source.

    Returns:
        True if processed and written, False if skipped.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    in_place = output_dir is None
    if in_place:
        dest_path = file_path
    else:
        dest_dir = Path(output_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / file_path.name

        if dest_path.is_file() and not overwrite:
            with h5py.File(dest_path, "r") as existing:
                if SENS_KEY in existing:
                    logger.info("Skipping %s: sidecar already exists.", file_path.name)
                    return False

    params = (calib_width, thresh, kernel_width, crop, max_iter, device, show_pbar)

    # Opened read-write when writing back into the archive: HDF5 will not hand out a
    # second handle to a file this process already holds open read-only.
    with h5py.File(file_path, "r+" if in_place else "r") as src:
        if "kspace" not in src:
            logger.warning("Skipping %s: no 'kspace' dataset found.", file_path.name)
            return False

        ksp_ds = src["kspace"]
        if ksp_ds.ndim not in (3, 4):
            raise ValueError(
                f"Expected kspace ndim 3 or 4 in {file_path.name}, got shape {ksp_ds.shape}"
            )
        shape = tuple(ksp_ds.shape)

        if in_place:
            if SENS_KEY in src:
                if not overwrite:
                    logger.info("Skipping %s: '%s' already exists.", file_path.name, SENS_KEY)
                    return False
                del src[SENS_KEY]
            _write_maps(src, ksp_ds, params)
        else:
            # Staged through a temporary file and renamed, so an interrupted run never
            # leaves a partially populated sidecar that a later pass reads as complete.
            tmp_path = dest_path.with_suffix(f".tmp{os.getpid()}.h5")
            try:
                with h5py.File(tmp_path, "w") as dest:
                    _write_maps(dest, ksp_ds, params)
                os.replace(tmp_path, dest_path)
            except BaseException:
                tmp_path.unlink(missing_ok=True)
                raise

    logger.info("Wrote %s shape %s to %s", SENS_KEY, shape, dest_path.name)
    return True


def _write_maps(
    dest: h5py.File,
    ksp_ds: h5py.Dataset,
    params: tuple[int, float, int, float, int, int, bool],
) -> None:
    """Calibrate and write the maps one slice at a time.

    Streaming keeps peak memory at one slice of maps rather than a whole volume,
    which matters because the maps are as large as the k-space they come from.
    """
    sens_ds = dest.create_dataset(SENS_KEY, shape=ksp_ds.shape, dtype=np.complex64, chunks=True)

    if ksp_ds.ndim == 3:
        sens_ds[:] = _calibrate_slice(_extract_complex_kspace(ksp_ds[:]), *params)
        return

    for s in range(ksp_ds.shape[0]):
        sens_ds[s] = _calibrate_slice(_extract_complex_kspace(ksp_ds[s]), *params)


def _process_one(args: tuple[str, dict[str, object]]) -> tuple[str, bool, str | None]:
    """Worker entry point: run :func:`process_h5_file` and capture any failure."""
    file_path, kwargs = args
    try:
        return file_path, process_h5_file(file_path, **kwargs), None  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001 - one bad volume must not kill the run
        return file_path, False, f"{type(exc).__name__}: {exc}"


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
            "Point dataset.sens_dir at the same directory. Strongly recommended for "
            "the full cohort: the maps are as large as the k-space."
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

    if args.output_dir is None:
        logger.warning(
            "No --output_dir given: maps will be written into the source files, "
            "roughly doubling the size of %s.",
            args.data_dir,
        )

    kwargs: dict[str, object] = {
        "calib_width": args.calib_width,
        "thresh": args.thresh,
        "kernel_width": args.kernel_width,
        "crop": args.crop,
        "max_iter": args.max_iter,
        "overwrite": args.overwrite,
        "device": args.device,
        "show_pbar": args.show_pbar,
        "output_dir": args.output_dir,
    }

    logger.info("Found %d files to process in %s", len(files), args.data_dir)
    num_processed = 0
    failures: list[str] = []

    if args.num_workers > 1:
        with ProcessPoolExecutor(max_workers=args.num_workers) as pool:
            futures = [pool.submit(_process_one, (f, kwargs)) for f in files]
            for future in as_completed(futures):
                f_path, success, error = future.result()
                if error is not None:
                    logger.error("Failed %s: %s", os.path.basename(f_path), error)
                    failures.append(f_path)
                elif success:
                    num_processed += 1
    else:
        for f in files:
            f_path, success, error = _process_one((f, kwargs))
            if error is not None:
                logger.error("Failed %s: %s", os.path.basename(f_path), error)
                failures.append(f_path)
            elif success:
                num_processed += 1

    logger.info(
        "ESPIRiT precomputation finished. Processed %d/%d files, %d failed.",
        num_processed,
        len(files),
        len(failures),
    )
    if failures:
        raise SystemExit(f"{len(failures)} file(s) failed; see the log above.")


if __name__ == "__main__":
    main()
