"""Streaming HDF5 dataset implementation for fastMRI multi-coil cohort."""

from __future__ import annotations

import glob
import hashlib
import json
import logging
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from cfm.core.dataset import BaseComplexDataset
from cfm.core.registry import DATASETS, MASKS
from cfm.data.hdf5_manager import WorkerHDF5Manager
from cfm.data.masks import BaseMaskGenerator
from cfm.utils.fft import fft2c, ifft2c

logger = logging.getLogger(__name__)

DEFAULT_SENS_KEY = "sensitivity_maps"


def _to_complex_tensor(raw: np.ndarray) -> torch.Tensor:
    """Convert numpy array of various complex formats into torch.complex64 tensor.

    Args:
        raw: Numpy array with complex dtype, structured ('r', 'i'), or [..., 2] float.

    Returns:
        Complex PyTorch tensor of dtype torch.complex64.
    """
    if raw.dtype.names is not None:
        if "r" in raw.dtype.names and "i" in raw.dtype.names:
            raw = raw["r"] + 1j * raw["i"]
        elif "real" in raw.dtype.names and "imag" in raw.dtype.names:
            raw = raw["real"] + 1j * raw["imag"]

    if not np.iscomplexobj(raw) and raw.shape[-1] == 2:
        raw = raw[..., 0] + 1j * raw[..., 1]

    raw = np.nan_to_num(raw)
    return torch.from_numpy(np.asarray(raw, dtype=np.complex64))


def _compute_index_hash(files: list[str]) -> str:
    """Compute deterministic SHA256 hash from sorted file paths and metadata."""
    hasher = hashlib.sha256()
    for f_path in sorted(files):
        try:
            stat = os.stat(f_path)
            hasher.update(f"{os.path.abspath(f_path)}:{stat.st_mtime_ns}:{stat.st_size}".encode())
        except OSError:
            hasher.update(f"{os.path.abspath(f_path)}".encode())
    return hasher.hexdigest()[:16]


def _center_crop_to(x: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """Center crop the trailing two dimensions of ``x`` to ``height`` x ``width``."""
    h, w = x.shape[-2], x.shape[-1]
    if (h, w) == (height, width):
        return x
    if h < height or w < width:
        raise ValueError(
            f"Cannot center crop a {h}x{w} tensor up to {height}x{width}. "
            "pre_transform may crop the image but must not enlarge it, because the "
            "sensitivity maps are cropped to match and cannot be extrapolated."
        )
    top = (h - height) // 2
    left = (w - width) // 2
    return x[..., top : top + height, left : left + width]


@DATASETS.register("fastmri")
@DATASETS.register("fast_mri")
class FastMRIDataset(BaseComplexDataset):
    """Streaming multi-coil fastMRI dataset with ESPIRiT sensitivity maps.

    Loads multi-coil k-space and precomputed sensitivity maps from HDF5 files,
    performs SENSE coil combination, and provides data for generation and
    reconstruction workflows.

    Both k-space and the sensitivity maps are handled on the *centered* grid via
    :func:`~cfm.utils.fft.fft2c` / :func:`~cfm.utils.fft.ifft2c`, which is the
    convention ``sigpy``'s ESPIRiT calibration emits its maps in. See that module
    for why the trailing shift is not optional here.

    Args:
        data_dir: Path to directory containing HDF5 files.
        transform: Optional per-slice transform callable.
        window_transform: Optional multi-slice window transform callable.
        num_slices: Number of contiguous slices in window (must be positive odd).
        mode: Operation mode ('generation' or 'reconstruction').
        acceleration: Undersampling acceleration factor (e.g. 4 or 8).
        mask: Mask generator instance, config dictionary, or registry key.
        mask_seed: Optional fixed seed for mask generation reproducibility.
        pre_transform: Optional geometric transform applied before undersampling.
            May crop the image; the sensitivity maps are center-cropped to match.
        post_transform: Optional intensity transform applied after undersampling.
            Must not change the spatial shape.
        use_cache: Whether to use persistent disk index caching.
        cache_dir: Directory to store index cache files.
        files_pattern: Optional glob pattern relative to data_dir.
        max_volumes: Optional cap on the number of volumes, applied by striding the
            sorted file list so the subset spans the cohort rather than its first N.
        sens_dir: Optional directory of sidecar HDF5 files holding the sensitivity
            maps, keyed by the same basename as the k-space file. When omitted,
            defaults to ``Path(data_dir).parent / f"{Path(data_dir).name}_sens"``.
        sens_key: Dataset name the sensitivity maps are stored under.
        auto_calibrate: Whether to automatically compute ESPIRiT sensitivity maps
            for volumes that lack them.
        calib_workers: Number of parallel worker processes for auto-calibration.
    """

    def __init__(
        self,
        data_dir: str | Path,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        window_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        num_slices: int = 1,
        mode: str = "generation",
        acceleration: int | float = 4,
        mask: BaseMaskGenerator | Mapping[str, Any] | str | None = None,
        mask_seed: int | None = None,
        pre_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        post_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        use_cache: bool = True,
        cache_dir: str | Path = ".cache",
        files_pattern: str | None = None,
        max_volumes: int | None = None,
        sens_dir: str | Path | None = None,
        sens_key: str = DEFAULT_SENS_KEY,
        auto_calibrate: bool = True,
        calib_workers: int | None = None,
    ) -> None:
        if "fastmri_local" in str(data_dir):
            from cfm.data.download import ensure_dataset_exists

            ensure_dataset_exists("fastmri", str(data_dir), mode="local")
        elif "fastmri_full" in str(data_dir):
            from cfm.data.download import ensure_dataset_exists

            ensure_dataset_exists("fastmri", str(data_dir), mode="full")

        if num_slices <= 0 or num_slices % 2 == 0:
            raise ValueError(f"num_slices must be a positive odd integer, got {num_slices}")

        if mode not in ("generation", "reconstruction"):
            raise ValueError(f"mode must be 'generation' or 'reconstruction', got {mode!r}")

        if mode == "reconstruction" and num_slices != 1:
            raise ValueError(
                f"mode='reconstruction' currently supports num_slices=1 only, got {num_slices}."
            )

        if acceleration < 1:
            raise ValueError(f"acceleration must be >= 1, got {acceleration}")

        if max_volumes is not None and max_volumes < 1:
            raise ValueError(f"max_volumes must be >= 1, got {max_volumes}")

        self.data_dir = str(data_dir)
        self.transform = transform
        self.window_transform = window_transform
        self.num_slices = num_slices
        self.mode = mode
        self.mask_seed = mask_seed
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.use_cache = use_cache
        self.cache_dir = str(cache_dir)
        if sens_dir is not None:
            self.sens_dir = str(sens_dir)
        else:
            data_path = Path(self.data_dir)
            self.sens_dir = str(data_path.parent / f"{data_path.name}_sens")
        self.sens_key = sens_key
        self.auto_calibrate = auto_calibrate
        self.calib_workers = calib_workers

        # Resolve mask generator from MASKS registry
        if isinstance(mask, BaseMaskGenerator):
            self.mask_generator = mask
        elif isinstance(mask, str):
            self.mask_generator = MASKS.build(mask, acceleration=acceleration)
        elif isinstance(mask, Mapping) or (mask is not None and hasattr(mask, "get")):
            m_dict = dict(mask)
            if "name" in m_dict:
                m_name = str(m_dict.pop("name"))
            elif "type" in m_dict:
                m_name = str(m_dict.pop("type"))
            else:
                m_name = "cartesian"
            m_accel = m_dict.pop("acceleration", acceleration)
            self.mask_generator = MASKS.build(m_name, acceleration=m_accel, **m_dict)
        elif mask is None:
            self.mask_generator = MASKS.build("cartesian", acceleration=acceleration)
        else:
            raise TypeError(f"Unsupported mask specification type: {type(mask)}")

        acc = getattr(self.mask_generator, "acceleration", acceleration)
        self.acceleration: int | float = (
            int(acc) if isinstance(acc, int | float) and float(acc).is_integer() else float(acc)
        )

        # File discovery
        if files_pattern is not None:
            self.files = sorted(glob.glob(os.path.join(self.data_dir, files_pattern)))
        else:
            self.files = sorted(glob.glob(f"{self.data_dir}/*.h5"))
            if not self.files:
                self.files = sorted(glob.glob(f"{self.data_dir}/**/*.h5", recursive=True))

        if not self.files:
            raise FileNotFoundError(f"No .h5 files found in: {self.data_dir}")

        # Stride rather than truncate: fastMRI is ordered by acquisition, so the
        # first N files are one scanner/anatomy and would misrepresent the cohort
        # a local debug subset is meant to stand in for.
        if max_volumes is not None and max_volumes < len(self.files):
            stride = len(self.files) // max_volumes
            self.files = self.files[::stride][:max_volumes]

        self.slice_map: list[tuple[str, int]] = []
        self._volume_depths: dict[str, int] = {}

        self._load_or_build_index()

        # Automatic on-demand ESPIRiT calibration for volumes missing sensitivity maps
        if self.auto_calibrate:
            missing = self._find_missing_sensitivity_maps()
            if missing:
                self._calibrate_missing(missing)

    def _has_sensitivity_map(self, f_path: str) -> bool:
        """Check if sensitivity maps exist for a volume either in sens_dir or in-file."""
        if self.sens_dir is not None:
            sidecar = os.path.join(self.sens_dir, os.path.basename(f_path))
            if os.path.isfile(sidecar):
                try:
                    with h5py.File(sidecar, "r") as hf:
                        if self.sens_key in hf:
                            return True
                except Exception:
                    pass
        if os.path.isfile(f_path):
            try:
                with h5py.File(f_path, "r") as hf:
                    if self.sens_key in hf:
                        return True
            except Exception:
                pass
        return False

    def _find_missing_sensitivity_maps(self) -> list[str]:
        """Return list of files in self.files that lack sensitivity maps."""
        return [f for f in self.files if not self._has_sensitivity_map(f)]

    def _calibrate_missing(self, missing: list[str]) -> None:
        """Run ESPIRiT calibration on missing volumes, coordinating across DDP ranks."""
        import torch.distributed as dist

        from cfm.data.espirit import ensure_espirit_maps

        msg = (
            f"[Auto-ESPIRiT] Found {len(missing)} volume(s) missing sensitivity maps. "
            f"Computing ESPIRiT calibration to {self.sens_dir}..."
        )
        calib_device: int | str = "cuda" if torch.cuda.is_available() else -1
        if dist.is_available() and dist.is_initialized():
            device = torch.device("cpu")
            try:
                if dist.get_backend() == "nccl" and torch.cuda.is_available():
                    device = torch.device("cuda", torch.cuda.current_device())
            except Exception:
                pass

            success = torch.tensor([1], dtype=torch.uint8, device=device)
            if dist.get_rank() == 0:
                print(msg, flush=True)
                try:
                    ensure_espirit_maps(
                        missing,
                        output_dir=self.sens_dir,
                        num_workers=self.calib_workers,
                        sens_key=self.sens_key,
                        device=calib_device,
                    )
                except Exception as exc:
                    logger.error("ESPIRiT calibration failed on rank 0: %s", exc)
                    success.fill_(0)
            dist.broadcast(success, src=0)
            if success.item() == 0:
                raise RuntimeError("ESPIRiT calibration failed on rank 0")
            dist.barrier()
        else:
            print(msg, flush=True)
            ensure_espirit_maps(
                missing,
                output_dir=self.sens_dir,
                num_workers=self.calib_workers,
                sens_key=self.sens_key,
                device=calib_device,
            )

    def _sens_path(self, f_path: str) -> str:
        """Resolve the file holding the sensitivity maps for a k-space file."""
        if self.sens_dir is None:
            return f_path
        return os.path.join(self.sens_dir, os.path.basename(f_path))

    def _load_or_build_index(self) -> None:
        """Load scan index from persistent cache or build from HDF5 headers."""
        cache_hash = _compute_index_hash(self.files)
        cache_file = os.path.join(self.cache_dir, f"fastmri_index_{cache_hash}.json")

        loaded = False
        if self.use_cache and os.path.isfile(cache_file):
            try:
                with open(cache_file, encoding="utf-8") as fh:
                    payload = json.load(fh)
                volume_depths: dict[str, int] = payload["volume_depths"]
                slice_map_raw: list[list[str | int]] = payload["slice_map"]

                if set(volume_depths.keys()) == set(self.files):
                    self._volume_depths = volume_depths
                    self.slice_map = [(str(p), int(idx)) for p, idx in slice_map_raw]
                    loaded = True
            except Exception:
                loaded = False

        if not loaded:
            self.slice_map = []
            self._volume_depths = {}
            # Opened and closed one at a time rather than through WorkerHDF5Manager:
            # the manager caches handles for the lifetime of the process with no
            # eviction, and the full fastMRI cohort is thousands of volumes, which
            # would exhaust the file descriptor limit before the index is built.
            for f_path in self.files:
                with h5py.File(f_path, "r") as handle:
                    kspace_shape = handle["kspace"].shape
                depth = self._volume_depth(kspace_shape, f_path)
                self._volume_depths[f_path] = depth
                for i in range(depth):
                    self.slice_map.append((f_path, i))

            if self.use_cache:
                try:
                    os.makedirs(self.cache_dir, exist_ok=True)
                    tmp_cache = f"{cache_file}.tmp.{os.getpid()}"
                    payload = {
                        "files": self.files,
                        "volume_depths": self._volume_depths,
                        "slice_map": self.slice_map,
                    }
                    with open(tmp_cache, "w", encoding="utf-8") as fh:
                        json.dump(payload, fh)
                    os.replace(tmp_cache, cache_file)
                except Exception:
                    pass

    @staticmethod
    def _volume_depth(kspace_shape: tuple[int, ...], f_path: str) -> int:
        """Number of slices in a volume, rejecting layouts this loader cannot read.

        Raises:
            ValueError: If the k-space is not 4D ``[slices, coils, H, W]``.
        """
        if len(kspace_shape) != 4:
            raise ValueError(
                f"File '{f_path}' has kspace of shape {tuple(kspace_shape)}; "
                "FastMRIDataset requires multi-coil 4D [slices, coils, H, W] data. "
                "A 3D array is ambiguous (fastMRI singlecoil stores [slices, H, W] "
                "while a single-slice multi-coil scan stores [coils, H, W]), so it is "
                "rejected rather than guessed at."
            )
        return int(kspace_shape[0])

    def __len__(self) -> int:
        """Total number of slices across all volumes in dataset."""
        return len(self.slice_map)

    @staticmethod
    def _reflect_index(i: int, depth: int) -> int:
        """Reflect out-of-range slice index without duplicating boundary."""
        if depth <= 1:
            return 0
        while i < 0 or i > depth - 1:
            i = -i if i < 0 else 2 * (depth - 1) - i
        return i

    def _mask_seed(self, f_path: str, slice_idx: int) -> int:
        """Compute deterministic seed from file path, slice index, and acceleration."""
        key = f"{os.path.basename(f_path)}:{slice_idx}:{self.acceleration}"
        if self.mask_seed is not None:
            key = f"{key}:{self.mask_seed}"
        return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")

    def _undersampling_mask(self, f_path: str, slice_idx: int, h: int, w: int) -> torch.Tensor:
        """Construct deterministic undersampling mask [1, H, W] via configured mask generator."""
        seed = self._mask_seed(f_path, slice_idx)
        return self.mask_generator.generate(shape=(h, w), seed=seed)

    @staticmethod
    def sense_combine(kspace: torch.Tensor, sensitivity_maps: torch.Tensor) -> torch.Tensor:
        """Perform adjoint SENSE coil combination from k-space and sensitivity maps.

        Args:
            kspace: Multi-coil complex tensor [num_coils, H, W], centered.
            sensitivity_maps: Multi-coil complex tensor [num_coils, H, W], centered.

        Returns:
            Single-channel complex image tensor [1, H, W].
        """
        x_coils = ifft2c(kspace)
        # SENSE combine: sum_c (S_c^* * x_c)
        return (sensitivity_maps.conj() * x_coils).sum(dim=0, keepdim=True)

    def _read_slice_data(
        self,
        handle: h5py.File,
        slice_idx: int,
        f_path: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Extract multi-coil k-space and sensitivity maps for a slice.

        Args:
            handle: Open HDF5 file handle for the k-space file.
            slice_idx: Slice index within the volume.
            f_path: File path for diagnostic error messages.

        Returns:
            Tuple (kspace [C, H, W], sensitivity_maps [C, H, W]) as complex64 tensors.
        """
        ksp_ds = handle["kspace"]
        self._volume_depth(ksp_ds.shape, f_path)

        sens_path = self._sens_path(f_path)
        if sens_path == f_path:
            sens_handle = handle
        else:
            if not os.path.isfile(sens_path):
                if self.sens_key in handle:
                    sens_handle = handle
                    sens_path = f_path
                else:
                    raise FileNotFoundError(
                        f"No sidecar sensitivity map file for '{os.path.basename(f_path)}' at "
                        f"{sens_path}. Run scripts/prep_fastmri_espirit.py with "
                        f"--output_dir {self.sens_dir}."
                    )
            else:
                sens_handle = WorkerHDF5Manager.get_instance().get_handle(sens_path)

        if self.sens_key not in sens_handle:
            raise KeyError(
                f"File '{sens_path}' is missing '{self.sens_key}'. "
                "Run scripts/prep_fastmri_espirit.py to precompute sensitivity maps."
            )

        sens_ds = sens_handle[self.sens_key]

        ksp = ksp_ds[slice_idx]
        sens = sens_ds[slice_idx]

        ksp_tensor = _to_complex_tensor(ksp)
        sens_tensor = _to_complex_tensor(sens)

        if ksp_tensor.shape != sens_tensor.shape:
            raise ValueError(
                f"Shape mismatch in '{os.path.basename(f_path)}' slice {slice_idx}: "
                f"kspace {tuple(ksp_tensor.shape)} vs {self.sens_key} "
                f"{tuple(sens_tensor.shape)}. The maps must be computed from this "
                "k-space at full resolution."
            )

        return ksp_tensor, sens_tensor

    def _reconstruction_item(
        self,
        handle: h5py.File,
        slice_idx: int,
        f_path: str,
    ) -> dict[str, torch.Tensor]:
        """Build the reconstruction-mode sample for one slice."""
        ksp_slice, sens_slice = self._read_slice_data(handle, slice_idx, f_path)
        img_complex = self.sense_combine(ksp_slice, sens_slice)

        if self.pre_transform is not None:
            img_complex = self.pre_transform(img_complex)
            # Geometry only. Running the image pipeline over the maps would
            # renormalise them and break sum_c |S_c|^2 = 1, which is what makes the
            # adjoint combination above a coil combination in the first place.
            sens_slice = _center_crop_to(sens_slice, img_complex.shape[-2], img_complex.shape[-1])

        _, h, w = img_complex.shape
        mask = self._undersampling_mask(f_path, slice_idx, h, w)

        if self.pre_transform is None:
            # Nothing has touched the image, so the measured k-space still describes
            # it exactly and is preferable to a re-synthesised copy: it carries the
            # true acquisition noise and any signal outside the span of the maps.
            y_full = ksp_slice
        else:
            y_full = fft2c(sens_slice * img_complex)

        y_under = y_full * mask

        x_alias = (sens_slice.conj() * ifft2c(y_under)).sum(dim=0, keepdim=True)

        scale = img_complex.abs().max().clamp(min=1e-8)
        x_gt = img_complex / scale
        x_alias = x_alias / scale
        masked_kspace = y_under / scale

        if self.post_transform is not None:
            before = x_gt.shape
            x_gt = self.post_transform(x_gt)
            x_alias = self.post_transform(x_alias)
            if x_gt.shape[-2:] != before[-2:]:
                raise ValueError(
                    "post_transform changed the spatial shape from "
                    f"{tuple(before[-2:])} to {tuple(x_gt.shape[-2:])}. It runs after "
                    "the mask, k-space and sensitivity maps are fixed, so a resize "
                    "here would silently desynchronise them. Use pre_transform for "
                    "anything geometric."
                )

        return {
            "input": x_alias,
            "mask": mask,
            "target": x_gt,
            "masked_kspace": masked_kspace,
            "sensitivity_maps": sens_slice,
        }

    def __getitem__(self, idx: int) -> torch.Tensor | dict[str, torch.Tensor]:
        """Retrieve slice or multi-slice window by index.

        Args:
            idx: Slice index in flat dataset map.

        Returns:
            In generation mode: complex tensor [1, H, W] or window [S, 1, H, W].
            In reconstruction mode: dictionary containing 'input', 'mask', 'target',
                'masked_kspace', and 'sensitivity_maps'.
        """
        f_path, slice_idx = self.slice_map[idx]
        manager = WorkerHDF5Manager.get_instance()
        handle = manager.get_handle(f_path)

        if self.mode == "reconstruction":
            return self._reconstruction_item(handle, slice_idx, f_path)

        if self.num_slices == 1:
            ksp_slice, sens_slice = self._read_slice_data(handle, slice_idx, f_path)
            img_complex = self.sense_combine(ksp_slice, sens_slice)

            if self.transform is not None:
                img_complex = self.transform(img_complex)

            return img_complex

        # 2.5D Multi-slice window
        depth = self._volume_depths[f_path]
        half = self.num_slices // 2
        indices = [
            self._reflect_index(slice_idx + offset, depth) for offset in range(-half, half + 1)
        ]

        slices = []
        for i in indices:
            ksp_slice, sens_slice = self._read_slice_data(handle, i, f_path)
            img_complex = self.sense_combine(ksp_slice, sens_slice)
            if self.transform is not None:
                img_complex = self.transform(img_complex)
            slices.append(img_complex)

        window = torch.stack(slices, dim=0)

        if self.window_transform is not None:
            window = self.window_transform(window)

        return window
