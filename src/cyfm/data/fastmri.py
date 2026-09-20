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

from cyfm.core.dataset import BaseComplexDataset
from cyfm.core.registry import DATASETS
from cyfm.data.hdf5_manager import WorkerHDF5Manager
from cyfm.utils.fft import fft2c, ifft2c

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
    :func:`~cyfm.utils.fft.fft2c` / :func:`~cyfm.utils.fft.ifft2c`, which is the
    convention ``sigpy``'s ESPIRiT calibration emits its maps in. See that module
    for why the trailing shift is not optional here.

    Args:
        data_dir: Path to directory containing HDF5 files.
        transform: Optional per-slice transform callable.
        window_transform: Optional multi-slice window transform callable.
        num_slices: Number of contiguous slices in window (must be positive odd).
            May crop the image; the sensitivity maps are center-cropped to match.
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
            from cyfm.data.download import ensure_dataset_exists

            ensure_dataset_exists("fastmri", str(data_dir), mode="local")
        elif "fastmri_full" in str(data_dir):
            from cyfm.data.download import ensure_dataset_exists

            ensure_dataset_exists("fastmri", str(data_dir), mode="full")

        if num_slices <= 0 or num_slices % 2 == 0:
            raise ValueError(f"num_slices must be a positive odd integer, got {num_slices}")

        if max_volumes is not None and max_volumes < 1:
            raise ValueError(f"max_volumes must be >= 1, got {max_volumes}")

        self.data_dir = str(data_dir)
        self.transform = transform
        self.window_transform = window_transform
        self.num_slices = num_slices
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
        """Run ESPIRiT calibration on missing volumes, coordinating across DDP ranks.

        Every rank calibrates its own stride of the list rather than parking on a
        collective while rank 0 does all of it: NCCL aborts a rank that waits longer
        than ``COLLECTIVE_TIMEOUT`` (45 min), and a cold cohort takes far longer than
        that. Sharding both divides the work and leaves each rank waiting only for the
        slack between shards.
        """
        import torch.distributed as dist

        from cyfm.data.espirit import ensure_espirit_maps

        distributed = dist.is_available() and dist.is_initialized()
        rank = dist.get_rank() if distributed else 0
        world_size = dist.get_world_size() if distributed else 1
        shard = missing[rank::world_size]

        if rank == 0:
            print(
                f"[Auto-ESPIRiT] Found {len(missing)} volume(s) missing sensitivity maps. "
                f"Computing ESPIRiT calibration to {self.sens_dir} "
                f"across {world_size} rank(s)...",
                flush=True,
            )

        # Each rank calibrates on the GPU it already owns; without the index every rank
        # would pile onto cuda:0.
        calib_device: int | str = -1
        if torch.cuda.is_available():
            calib_device = torch.cuda.current_device() if distributed else "cuda"

        failed = False
        if shard:
            try:
                ensure_espirit_maps(
                    shard,
                    output_dir=self.sens_dir,
                    num_workers=1 if distributed else self.calib_workers,
                    sens_key=self.sens_key,
                    device=calib_device,
                )
            except Exception as exc:
                logger.error("ESPIRiT calibration failed on rank %d: %s", rank, exc)
                failed = True

        if not distributed:
            if failed:
                raise RuntimeError("ESPIRiT calibration failed")
            return

        device = torch.device("cpu")
        if dist.get_backend() == "nccl" and torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())

        # MAX rather than a broadcast from rank 0: any rank's shard can fail, and all of
        # them must agree to stop so none is left alone in the next collective.
        status = torch.tensor([1 if failed else 0], dtype=torch.uint8, device=device)
        dist.all_reduce(status, op=dist.ReduceOp.MAX)
        if int(status.item()) != 0:
            raise RuntimeError("ESPIRiT calibration failed on at least one rank")

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

    def __getitem__(self, idx: int) -> torch.Tensor | dict[str, torch.Tensor]:
        """Retrieve slice or multi-slice window by index.

        Args:
            idx: Slice index in flat dataset map.

        Returns:
            Complex tensor [1, H, W], or a window [S, 1, H, W].
        """
        f_path, slice_idx = self.slice_map[idx]
        manager = WorkerHDF5Manager.get_instance()
        handle = manager.get_handle(f_path)

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
