"""ESPIRiT coil sensitivity map computation and file preparation for fastMRI."""

from __future__ import annotations

import logging
import os
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import h5py
import numpy as np

try:
    import sigpy as sp
    import sigpy.mri.app as app

    _HAS_SIGPY = True
except ImportError:
    sp = None  # type: ignore[assignment]
    app = None  # type: ignore[assignment]
    _HAS_SIGPY = False

logger = logging.getLogger(__name__)

DEFAULT_SENS_KEY = "sensitivity_maps"
SENS_KEY = DEFAULT_SENS_KEY


def _check_sigpy() -> None:
    """Ensure sigpy is installed before running ESPIRiT calibration."""
    if not _HAS_SIGPY:
        raise ImportError(
            "sigpy is required for ESPIRiT calibration. "
            "Install it via: uv sync --extra espirit (or pip install sigpy)."
        )


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
    """Run ESPIRiT calibration on one [num_coils, H, W] slice.

    Args:
        ksp: Complex slice array of shape [C, H, W].
        calib_width: ACS autocalibration region width.
        thresh: Eigenvalue threshold.
        kernel_width: Kernel width.
        crop: Threshold cropping.
        max_iter: Maximum power iterations.
        device: Device index (-1 for CPU, >= 0 for CUDA device).
        show_pbar: Whether to display progress bar.

    Returns:
        Sensitivity maps array of shape [C, H, W] and dtype np.complex64.
    """
    _check_sigpy()
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
    _check_sigpy()
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


def _write_maps(
    dest: h5py.File,
    ksp_ds: h5py.Dataset,
    params: tuple[int, float, int, float, int, int, bool],
    sens_key: str = DEFAULT_SENS_KEY,
) -> None:
    """Calibrate and write the maps one slice at a time.

    Streaming keeps peak memory at one slice of maps rather than a whole volume,
    which matters because the maps are as large as the k-space they come from.
    """
    sens_ds = dest.create_dataset(sens_key, shape=ksp_ds.shape, dtype=np.complex64, chunks=True)

    if ksp_ds.ndim == 3:
        sens_ds[:] = _calibrate_slice(_extract_complex_kspace(ksp_ds[:]), *params)
        return

    for s in range(ksp_ds.shape[0]):
        sens_ds[s] = _calibrate_slice(_extract_complex_kspace(ksp_ds[s]), *params)


def process_h5_file(
    file_path: str | Path,
    output_dir: str | Path | None = None,
    calib_width: int = 24,
    thresh: float = 0.02,
    kernel_width: int = 6,
    crop: float = 0.95,
    max_iter: int = 100,
    overwrite: bool = False,
    device: int = -1,
    show_pbar: bool = False,
    sens_key: str = DEFAULT_SENS_KEY,
) -> bool:
    """Compute and write sensitivity maps for one HDF5 file to a sidecar file.

    Slices are calibrated and written one at a time, so peak memory is one slice
    of maps rather than a whole volume. Source files are strictly opened read-only.

    Args:
        file_path: Path to the source .h5 file holding 'kspace'.
        output_dir: Directory where sidecar sensitivity map file will be written. Required.
        calib_width: Calibration box size.
        thresh: Eigenvalue threshold.
        kernel_width: Kernel width.
        crop: Cropping factor.
        max_iter: Maximum iterations.
        overwrite: Overwrite existing maps if present.
        device: Device index (-1 for CPU, >= 0 for CUDA).
        show_pbar: Whether to show progress bar.
        sens_key: Dataset name the sensitivity maps are stored under.

    Returns:
        True if processed and written, False if skipped.

    Raises:
        ValueError: If output_dir is None.
    """
    file_path = Path(file_path)
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    if output_dir is None:
        raise ValueError(
            "output_dir is required; in-place modification of source files is forbidden."
        )

    dest_dir = Path(output_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / file_path.name

    if dest_path.is_file() and not overwrite:
        with h5py.File(dest_path, "r") as existing:
            if sens_key in existing:
                logger.info("Skipping %s: sidecar already exists.", file_path.name)
                return False

    use_gpu = False
    torch_device = "cuda"
    if isinstance(device, int) and device >= 0:
        use_gpu = True
        torch_device = f"cuda:{device}"
    elif isinstance(device, str) and (device == "cuda" or device.startswith("cuda:")):
        use_gpu = True
        torch_device = device
    elif device is None:
        try:
            import torch

            if torch.cuda.is_available():
                use_gpu = True
                torch_device = "cuda"
        except ImportError:
            pass

    if use_gpu:
        from cfm.data.torch_espirit import calibrate_fastmri_file_torch

        logger.info("Calibrating %s on GPU (%s)", file_path.name, torch_device)
        return calibrate_fastmri_file_torch(
            src_path=file_path,
            dest_path=dest_path,
            device=torch_device,
            max_iter=min(max_iter, 30),
            overwrite=overwrite,
            sens_key=sens_key,
        )

    _check_sigpy()
    params = (calib_width, thresh, kernel_width, crop, max_iter, device, show_pbar)

    # Source dataset is strictly read-only: never open with 'r+' or modify in-place.
    with h5py.File(file_path, "r") as src:
        if "kspace" not in src:
            logger.warning("Skipping %s: no 'kspace' dataset found.", file_path.name)
            return False

        ksp_ds = src["kspace"]
        if ksp_ds.ndim not in (3, 4):
            raise ValueError(
                f"Expected kspace ndim 3 or 4 in {file_path.name}, got shape {ksp_ds.shape}"
            )
        shape = tuple(ksp_ds.shape)

        # Staged through a temporary file and renamed, so an interrupted run never
        # leaves a partially populated sidecar that a later pass reads as complete.
        tmp_path = dest_path.with_suffix(f".tmp{os.getpid()}.h5")
        try:
            with h5py.File(tmp_path, "w") as dest:
                _write_maps(dest, ksp_ds, params, sens_key=sens_key)
            os.replace(tmp_path, dest_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    logger.info("Wrote %s shape %s to %s", sens_key, shape, dest_path.name)
    return True


def _process_one(args: tuple[str, dict[str, Any]]) -> tuple[str, bool, str | None]:
    """Worker entry point: run process_h5_file and capture any failure."""
    file_path, kwargs = args
    try:
        return file_path, process_h5_file(file_path, **kwargs), None
    except Exception as exc:  # noqa: BLE001 - one bad volume must not kill the run
        return file_path, False, f"{type(exc).__name__}: {exc}"


def ensure_espirit_maps(
    files: Sequence[str | Path],
    output_dir: str | Path | None = None,
    num_workers: int | None = None,
    show_pbar: bool = False,
    device: int | str | None = -1,
    calib_width: int = 24,
    thresh: float = 0.02,
    kernel_width: int = 6,
    crop: float = 0.95,
    max_iter: int = 100,
    overwrite: bool = False,
    sens_key: str = DEFAULT_SENS_KEY,
) -> list[Path]:
    """Ensure ESPIRiT sensitivity maps exist for all given files, computing them if missing.

    Sensitivity maps are always written as sidecar files into output_dir. Source files
    are strictly read-only and never modified.

    Args:
        files: Sequence of paths to source HDF5 files containing 'kspace'.
        output_dir: Directory to store sidecar map files. Required.
        num_workers: Parallel workers for calibration. If None, automatically determined.
        show_pbar: Whether to display a progress bar.
        device: Device to use (-1 for CPU, >= 0 or 'cuda' for CUDA).
        calib_width: Autocalibration region width.
        thresh: Eigenvalue threshold.
        kernel_width: Kernel width.
        crop: Threshold cropping.
        max_iter: Maximum power iterations.
        overwrite: Whether to overwrite existing sensitivity maps.
        sens_key: HDF5 dataset name for sensitivity maps.

    Returns:
        List of paths to output HDF5 sidecar files containing the sensitivity maps.

    Raises:
        ValueError: If output_dir is None when files are provided.
        RuntimeError: If any volume fails calibration.
    """
    is_gpu = (isinstance(device, int) and device >= 0) or (
        isinstance(device, str) and (device == "cuda" or device.startswith("cuda:"))
    )
    if not is_gpu:
        _check_sigpy()

    if not files:
        return []

    if output_dir is None:
        raise ValueError(
            "output_dir is required; ESPIRiT sensitivity maps must be written as sidecars."
        )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    file_strs = [str(f) for f in files]
    out_paths = [out_dir / Path(f).name for f in file_strs]

    # Volumes to calibrate in parallel: CPU uses up to 4 workers; GPU runs 1 worker.
    if num_workers is None:
        effective_workers = (
            1
            if (is_gpu or len(file_strs) <= 1)
            else min(os.cpu_count() or 1, 4, len(file_strs))
        )
    else:
        effective_workers = max(1, num_workers)

    kwargs: dict[str, Any] = {
        "output_dir": out_dir,
        "calib_width": calib_width,
        "thresh": thresh,
        "kernel_width": kernel_width,
        "crop": crop,
        "max_iter": max_iter,
        "overwrite": overwrite,
        "device": device,
        "show_pbar": show_pbar and len(file_strs) == 1,
        "sens_key": sens_key,
    }

    logger.info(
        "Ensuring ESPIRiT maps for %d file(s) (workers=%d, output_dir=%s)",
        len(file_strs),
        effective_workers,
        str(out_dir),
    )

    num_processed = 0
    failures: list[str] = []

    if effective_workers > 1:
        with ProcessPoolExecutor(max_workers=effective_workers) as pool:
            futures = [pool.submit(_process_one, (f, kwargs)) for f in file_strs]
            for future in as_completed(futures):
                f_path, success, error = future.result()
                if error is not None:
                    logger.error("Failed %s: %s", os.path.basename(f_path), error)
                    failures.append(f_path)
                elif success:
                    num_processed += 1
    else:
        for f in file_strs:
            f_path, success, error = _process_one((f, kwargs))
            if error is not None:
                logger.error("Failed %s: %s", os.path.basename(f_path), error)
                failures.append(f_path)
            elif success:
                num_processed += 1

    logger.info(
        "ESPIRiT preparation finished. Processed %d/%d files, %d failed.",
        num_processed,
        len(file_strs),
        len(failures),
    )

    if failures:
        raise RuntimeError(
            f"ESPIRiT calibration failed for {len(failures)} file(s): "
            f"{[os.path.basename(f) for f in failures]}"
        )

    return out_paths


__all__ = [
    "DEFAULT_SENS_KEY",
    "SENS_KEY",
    "compute_espirit_maps",
    "ensure_espirit_maps",
    "process_h5_file",
]
