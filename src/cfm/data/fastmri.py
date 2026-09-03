"""Streaming HDF5 dataset implementation for fastMRI multi-coil cohort."""

from __future__ import annotations

import glob
import hashlib
import json
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


@DATASETS.register("fastmri")
@DATASETS.register("fast_mri")
class FastMRIDataset(BaseComplexDataset):
    """Streaming multi-coil fastMRI dataset with ESPIRiT sensitivity maps.

    Loads multi-coil k-space and precomputed sensitivity maps from HDF5 files,
    performs SENSE coil combination, and provides data for generation and
    reconstruction workflows.

    Args:
        data_dir: Path to directory containing HDF5 files.
        transform: Optional per-slice transform callable.
        window_transform: Optional multi-slice window transform callable.
        num_slices: Number of contiguous slices in window (must be positive odd).
        mode: Operation mode ('generation' or 'reconstruction').
        acceleration: Undersampling acceleration factor (e.g. 4 or 8).
        mask: Mask generator instance, config dictionary, or registry key.
        mask_seed: Optional fixed seed for mask generation reproducibility.
        pre_transform: Optional transform before undersampling.
        post_transform: Optional transform after undersampling.
        use_cache: Whether to use persistent disk index caching.
        cache_dir: Directory to store index cache files.
        files_pattern: Optional glob pattern relative to data_dir.
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
            self.files = sorted(glob.glob(f"{self.data_dir}/*.h5"))
            if not self.files:
                self.files = sorted(glob.glob(f"{self.data_dir}/**/*.h5", recursive=True))

        if not self.files:
            raise FileNotFoundError(f"No .h5 files found in: {self.data_dir}")

        self.slice_map: list[tuple[str, int]] = []
        self._volume_depths: dict[str, int] = {}

        self._load_or_build_index()

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
            manager = WorkerHDF5Manager.get_instance()
            for f_path in self.files:
                handle = manager.get_handle(f_path)
                kspace_shape = handle["kspace"].shape
                depth = int(kspace_shape[0]) if len(kspace_shape) == 4 else 1
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

    @staticmethod
    def sense_combine(kspace: torch.Tensor, sensitivity_maps: torch.Tensor) -> torch.Tensor:
        """Perform adjoint SENSE coil combination from k-space and sensitivity maps.

        Args:
            kspace: Multi-coil complex tensor [num_coils, H, W].
            sensitivity_maps: Multi-coil complex tensor [num_coils, H, W].

        Returns:
            Single-channel complex image tensor [1, H, W].
        """
        # IFFT centered k-space to image space
        x_coils = torch.fft.ifft2(torch.fft.ifftshift(kspace, dim=(-2, -1)), norm="ortho")
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
            handle: Open HDF5 file handle.
            slice_idx: Slice index within the volume.
            f_path: File path for diagnostic error messages.

        Returns:
            Tuple (kspace [C, H, W], sensitivity_maps [C, H, W]) as complex64 tensors.
        """
        if "sensitivity_maps" not in handle:
            raise KeyError(
                f"File '{f_path}' is missing 'sensitivity_maps'. "
                "Run scripts/prep_fastmri_espirit.py to precompute sensitivity maps."
            )

        ksp_ds = handle["kspace"]
        sens_ds = handle["sensitivity_maps"]

        if ksp_ds.ndim == 4:
            ksp = ksp_ds[slice_idx]
            sens = sens_ds[slice_idx]
        elif ksp_ds.ndim == 3:
            ksp = ksp_ds[:]
            sens = sens_ds[:] if sens_ds.ndim == 3 else sens_ds[slice_idx]
        else:
            raise ValueError(f"Unexpected kspace ndim: {ksp_ds.ndim} in {f_path}")

        ksp_tensor = _to_complex_tensor(ksp)
        sens_tensor = _to_complex_tensor(sens)
        return ksp_tensor, sens_tensor

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
            ksp_slice, sens_slice = self._read_slice_data(handle, slice_idx, f_path)
            img_complex = self.sense_combine(ksp_slice, sens_slice)

            if self.pre_transform is not None:
                img_complex = self.pre_transform(img_complex)
                if img_complex.shape[-2:] != sens_slice.shape[-2:]:
                    sens_slice = self.pre_transform(sens_slice)

            _, h, w = img_complex.shape
            mask = self._undersampling_mask(f_path, slice_idx, h, w)

            # Check if spatial dimension matches raw k-space
            if ksp_slice.shape[-2:] == (h, w):
                y_under = ksp_slice * mask
            else:
                # Re-synthesize k-space after pre-transform
                y = torch.fft.fftshift(
                    torch.fft.fft2(sens_slice * img_complex, norm="ortho"), dim=(-2, -1)
                )
                y_under = y * mask

            x_alias_coils = torch.fft.ifft2(
                torch.fft.ifftshift(y_under, dim=(-2, -1)), norm="ortho"
            )
            x_alias = (sens_slice.conj() * x_alias_coils).sum(dim=0, keepdim=True)

            scale = img_complex.abs().max().clamp(min=1e-8)
            x_gt = img_complex / scale
            x_alias = x_alias / scale
            masked_kspace = y_under / scale

            if self.post_transform is not None:
                x_gt = self.post_transform(x_gt)
                x_alias = self.post_transform(x_alias)

            return {
                "input": x_alias,
                "mask": mask,
                "target": x_gt,
                "masked_kspace": masked_kspace,
                "sensitivity_maps": sens_slice,
            }

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
