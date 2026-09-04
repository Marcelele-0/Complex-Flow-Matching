"""Streaming HDF5 dataset implementation for SKM-TEA MRI knee cohort."""

from __future__ import annotations

import glob
import hashlib
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cfm.core.dataset import BaseComplexDataset
from cfm.core.registry import DATASETS, MASKS
from cfm.data.hdf5_manager import WorkerHDF5Manager
from cfm.data.masks import BaseMaskGenerator
from cfm.utils.fft import fft2c, ifft2c


def _process_slice_array(
    raw: np.ndarray,
    echo_idx: int | str = 0,
    coil_idx: int | str = 0,
) -> torch.Tensor:
    """Process raw numpy slice array into a complex PyTorch tensor.

    Args:
        raw: Array of shape [H, W], [H, W, E], or [H, W, E, C].
        echo_idx: Echo selector (int index, 'both', 'all', 'average', or 'mean').
        coil_idx: Coil selector (int index, 'rss', 'root_sum_squares', or 'all').

    Returns:
        Complex tensor of shape [C_total, H, W] with dtype torch.complex64.
    """
    # Handle compound complex dtypes if present
    if raw.dtype.names is not None:
        if "r" in raw.dtype.names and "i" in raw.dtype.names:
            raw = raw["r"] + 1j * raw["i"]
        elif "real" in raw.dtype.names and "imag" in raw.dtype.names:
            raw = raw["real"] + 1j * raw["imag"]

    raw = np.nan_to_num(raw)

    # Standardize dimensions to [H, W, E, C]
    if raw.ndim == 2:
        raw = raw[:, :, np.newaxis, np.newaxis]
    elif raw.ndim == 3:
        raw = raw[:, :, :, np.newaxis]
    elif raw.ndim == 4:
        pass
    else:
        raise ValueError(f"Unexpected slice dimensionality: {raw.ndim}, shape: {raw.shape}")

    h, w, num_echoes, num_coils = raw.shape

    # 1. Echo Selection / Combination
    if isinstance(echo_idx, int):
        if echo_idx < 0 or echo_idx >= num_echoes:
            raise IndexError(f"echo_idx={echo_idx} is out of bounds for {num_echoes} echoes.")
        raw = raw[:, :, echo_idx : echo_idx + 1, :]
    elif str(echo_idx).lower() in ("both", "all"):
        pass
    elif str(echo_idx).lower() in ("average", "mean"):
        raw = np.mean(raw, axis=2, keepdims=True)
    else:
        raise ValueError(
            f"Unsupported echo_idx={echo_idx!r}. Expected int, 'both', 'all', 'average', or 'mean'."
        )

    # 2. Coil Selection / Combination
    if isinstance(coil_idx, int):
        if coil_idx < 0 or coil_idx >= num_coils:
            raise IndexError(f"coil_idx={coil_idx} is out of bounds for {num_coils} coils.")
        raw = raw[:, :, :, coil_idx : coil_idx + 1]
    elif str(coil_idx).lower() in ("rss", "root_sum_squares"):
        # Root-sum-of-squares across coils
        rss = np.sqrt(np.sum(np.abs(raw) ** 2, axis=3, keepdims=True))
        raw = rss.astype(raw.dtype)
    elif str(coil_idx).lower() == "all":
        pass
    else:
        raise ValueError(f"Unsupported coil_idx={coil_idx!r}. Expected int, 'rss', or 'all'.")

    # raw is now [H, W, E_out, C_out] -> transpose to [E_out, C_out, H, W]
    raw = np.transpose(raw, (2, 3, 0, 1))
    # Flatten channels: [E_out * C_out, H, W]
    raw = raw.reshape(-1, h, w)

    return torch.from_numpy(raw).to(torch.complex64)


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


@DATASETS.register("skm_tea")
@DATASETS.register("skmtea")
@DATASETS.register("skmtea_streaming")
@DATASETS.register("skmtea_full")
class SKMTEADataset(BaseComplexDataset):
    """High-throughput streaming dataset for SKM-TEA knee MRI cohort.

    Supports persistent index caching, worker-safe HDF5 handles, multi-echo,
    multi-coil, and contiguous 2.5D multi-slice window streaming.

    Args:
        data_dir: Path to directory containing HDF5 files.
        transform: Optional per-slice transform callable.
        window_transform: Optional multi-slice window transform callable.
        num_slices: Number of contiguous slices in window (must be positive odd).
        mode: Operation mode ('generation' or 'reconstruction').
        acceleration: Undersampling acceleration factor (e.g. 4 or 8).
        mask_seed: Optional fixed seed for mask generation reproducibility.
        pre_transform: Optional transform before undersampling.
        post_transform: Optional transform after undersampling.
        echo_idx: Echo selector (int index, 'both', 'all', 'average', or 'mean').
        coil_idx: Coil selector (int index, 'rss', or 'all').
        use_cache: Whether to use persistent disk index caching.
        cache_dir: Directory to store index cache files.
        files_pattern: Optional glob pattern relative to data_dir.
    """

    def __init__(
        self,
        data_dir: str,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        window_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        num_slices: int = 1,
        mode: str = "generation",
        acceleration: int | float = 4,
        mask: BaseMaskGenerator | Mapping[str, Any] | str | None = None,
        mask_seed: int | None = None,
        pre_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        post_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        echo_idx: int | str = 0,
        coil_idx: int | str = 0,
        use_cache: bool = True,
        cache_dir: str | Path = ".cache",
        files_pattern: str | None = None,
    ) -> None:
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

        self.data_dir = str(data_dir)
        self.transform = transform
        self.window_transform = window_transform
        self.num_slices = num_slices
        self.mode = mode
        self.mask_seed = mask_seed
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.echo_idx = echo_idx
        self.coil_idx = coil_idx
        self.use_cache = use_cache
        self.cache_dir = str(cache_dir)

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
            self.files = sorted(glob.glob(f"{self.data_dir}/files_recon_calib-24/*.h5"))
            if not self.files:
                self.files = sorted(glob.glob(f"{self.data_dir}/*.h5"))

        if not self.files:
            raise FileNotFoundError(f"No .h5 files found in: {self.data_dir}")

        self.slice_map: list[tuple[str, int]] = []
        self._volume_depths: dict[str, int] = {}

        self._load_or_build_index()

    def _load_or_build_index(self) -> None:
        """Load scan index from persistent cache or build from HDF5 volume headers."""
        cache_hash = _compute_index_hash(self.files)
        cache_file = os.path.join(self.cache_dir, f"skmtea_index_{cache_hash}.json")

        loaded = False
        if self.use_cache and os.path.isfile(cache_file):
            try:
                with open(cache_file, encoding="utf-8") as fh:
                    payload = json.load(fh)
                volume_depths: dict[str, int] = payload["volume_depths"]
                slice_map_raw: list[list[str | int]] = payload["slice_map"]

                # Verify all files match
                if set(volume_depths.keys()) == set(self.files):
                    self._volume_depths = volume_depths
                    self.slice_map = [(str(p), int(idx)) for p, idx in slice_map_raw]
                    loaded = True
            except Exception:
                loaded = False

        if not loaded:
            self.slice_map = []
            self._volume_depths = {}
            manager = WorkerHDF5Manager.get_instance()
            for f_path in self.files:
                handle = manager.get_handle(f_path)
                depth = int(handle["target"].shape[0])
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

    def __getitem__(self, idx: int) -> torch.Tensor | dict[str, torch.Tensor]:
        """Retrieve slice or multi-slice window by index.

        Args:
            idx: Slice index in flat dataset map.

        Returns:
            In generation mode: state tensor [C, H, W] or [S, C, H, W].
            In reconstruction mode: dict with 'input', 'mask', and 'target' tensors.
        """
        f_path, slice_idx = self.slice_map[idx]
        manager = WorkerHDF5Manager.get_instance()
        handle = manager.get_handle(f_path)

        if self.mode == "reconstruction":
            raw_slice = handle["target"][slice_idx]
            img_complex = _process_slice_array(raw_slice, self.echo_idx, self.coil_idx)

            if self.pre_transform is not None:
                img_complex = self.pre_transform(img_complex)

            _, h, w = img_complex.shape
            mask = self._undersampling_mask(f_path, slice_idx, h, w)

            y = fft2c(img_complex)
            y_under = y * mask
            x_alias = ifft2c(y_under)

            # Shared scalar so input/target stay on the same amplitude scale
            scale = img_complex.abs().max().clamp(min=1e-8)
            x_gt = img_complex / scale
            x_alias = x_alias / scale

            if self.post_transform is not None:
                x_gt = self.post_transform(x_gt)
                x_alias = self.post_transform(x_alias)

            return {"input": x_alias, "mask": mask, "target": x_gt}

        if self.num_slices == 1:
            raw_slice = handle["target"][slice_idx]
            img_complex = _process_slice_array(raw_slice, self.echo_idx, self.coil_idx)

            if self.transform is not None:
                img_complex = self.transform(img_complex)

            return img_complex

        # 2.5D Multi-slice window: read contiguous memory block [lo:hi] once
        depth = self._volume_depths[f_path]
        half = self.num_slices // 2
        indices = [
            self._reflect_index(slice_idx + offset, depth) for offset in range(-half, half + 1)
        ]

        lo, hi = min(indices), max(indices) + 1
        raw_block = handle["target"][lo:hi]

        slices = []
        for i in indices:
            raw_slice = raw_block[i - lo]
            img_complex = _process_slice_array(raw_slice, self.echo_idx, self.coil_idx)

            if self.transform is not None:
                img_complex = self.transform(img_complex)

            slices.append(img_complex)

        window = torch.stack(slices, dim=0)

        if self.window_transform is not None:
            window = self.window_transform(window)

        return window
