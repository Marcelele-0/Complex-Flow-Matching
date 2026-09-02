"""Lazy-loading dataset implementation for SKM-TEA MRI knee cohort."""

from __future__ import annotations

import glob
import hashlib
import os
from collections.abc import Callable

import h5py
import numpy as np
import torch

from cfm.core.dataset import BaseComplexDataset
from cfm.core.registry import DATASETS


@DATASETS.register("skm_tea")
@DATASETS.register("skmtea")
class SKMTEADataset(BaseComplexDataset):
    """Lazy-loading dataset for SKM-TEA knee MRI cohort.

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
    """

    def __init__(
        self,
        data_dir: str,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        window_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        num_slices: int = 1,
        mode: str = "generation",
        acceleration: int = 4,
        mask_seed: int | None = None,
        pre_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        post_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
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

        self.files = glob.glob(f"{data_dir}/files_recon_calib-24/*.h5")
        if not self.files:
            raise FileNotFoundError(f"No .h5 files found in: {data_dir}")

        self.transform = transform
        self.window_transform = window_transform
        self.num_slices = num_slices
        self.mode = mode
        self.acceleration = acceleration
        self.mask_seed = mask_seed
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.slice_map: list[tuple[str, int]] = []
        self._volume_depths: dict[str, int] = {}

        for f_path in self.files:
            with h5py.File(f_path, "r") as f:
                depth = f["target"].shape[0]
                self._volume_depths[f_path] = depth
                for i in range(depth):
                    self.slice_map.append((f_path, i))

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
        """Construct deterministic 1D Cartesian phase-encoding mask [1, H, W]."""
        rng = np.random.default_rng(self._mask_seed(f_path, slice_idx))

        num_center_lines = min(24, w)
        center_start = w // 2 - num_center_lines // 2
        center_end = center_start + num_center_lines

        line_mask = np.zeros(w, dtype=np.float32)
        line_mask[center_start:center_end] = 1.0

        remaining_lines = int(w / self.acceleration) - num_center_lines
        if remaining_lines > 0:
            leftover_indices = np.concatenate(
                [np.arange(0, center_start), np.arange(center_end, w)]
            )
            sampled = rng.choice(leftover_indices, remaining_lines, replace=False)
            line_mask[sampled] = 1.0

        mask = np.tile(line_mask, (h, 1))
        return torch.from_numpy(mask).unsqueeze(0)

    def __getitem__(self, idx: int) -> torch.Tensor | dict[str, torch.Tensor]:
        """Retrieve slice or multi-slice window by index.

        Args:
            idx: Slice index in flat dataset map.

        Returns:
            In generation mode: state tensor [1, H, W] or [S, C, H, W].
            In reconstruction mode: dict with 'input', 'mask', and 'target' tensors.
        """
        f_path, slice_idx = self.slice_map[idx]

        if self.mode == "reconstruction":
            with h5py.File(f_path, "r") as f:
                img_np = f["target"][slice_idx, :, :, 0, 0]

            img_np = np.nan_to_num(img_np)
            img_complex = torch.from_numpy(img_np).to(torch.complex64).unsqueeze(0)

            if self.pre_transform is not None:
                img_complex = self.pre_transform(img_complex)

            _, h, w = img_complex.shape
            mask = self._undersampling_mask(f_path, slice_idx, h, w)

            y = torch.fft.fftshift(torch.fft.fft2(img_complex, norm="ortho"), dim=(-2, -1))
            y_under = y * mask
            x_alias = torch.fft.ifft2(torch.fft.ifftshift(y_under, dim=(-2, -1)), norm="ortho")

            # Shared scalar so input/target stay on the same amplitude scale;
            # AmplitudeNormalize would normalize each independently and break
            # the y_under = mask * fft(target) relationship between the pair.
            scale = img_complex.abs().max().clamp(min=1e-8)
            x_gt = img_complex / scale
            x_alias = x_alias / scale

            if self.post_transform is not None:
                x_gt = self.post_transform(x_gt)
                x_alias = self.post_transform(x_alias)

            return {"input": x_alias, "mask": mask, "target": x_gt}

        if self.num_slices == 1:
            with h5py.File(f_path, "r") as f:
                # target shape: (Nx, Ny, Nz, echoes, coils)
                # Select specific slice, first echo, first coil
                img_np = f["target"][slice_idx, :, :, 0, 0]

            # Protect against NaNs from MRI scans
            img_np = np.nan_to_num(img_np)

            # Convert to PyTorch complex tensor
            img_complex = torch.from_numpy(img_np).to(torch.complex64)

            # Force shape [1, H, W] for transformations
            img_complex = img_complex.unsqueeze(0)

            # Apply the pipeline
            if self.transform is not None:
                img_complex = self.transform(img_complex)

            return img_complex

        depth = self._volume_depths[f_path]
        half = self.num_slices // 2
        indices = [
            self._reflect_index(slice_idx + offset, depth) for offset in range(-half, half + 1)
        ]

        # Reflected indices aren't monotonic at edges (e.g. [1, 0, 1]), so read
        # the covering contiguous block once and index into it by offset.
        lo, hi = min(indices), max(indices) + 1
        with h5py.File(f_path, "r") as f:
            block = f["target"][lo:hi, :, :, 0, 0]

        slices = []
        for i in indices:
            img_np = np.nan_to_num(block[i - lo])
            img_complex = torch.from_numpy(img_np).to(torch.complex64).unsqueeze(0)

            if self.transform is not None:
                img_complex = self.transform(img_complex)

            slices.append(img_complex)

        window = torch.stack(slices, dim=0)

        if self.window_transform is not None:
            window = self.window_transform(window)

        return window
