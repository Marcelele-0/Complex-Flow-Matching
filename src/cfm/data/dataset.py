import glob
import hashlib
import os
from typing import Callable, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class SKMTEADataset(Dataset):
    """
    Lazy-loading dataset for the SKM-TEA dataset.
    Opens HDF5 files and applies the transformation pipeline on the fly.

    With num_slices > 1, each sample is a [num_slices, C, H, W] window of
    neighboring slices from the same volume (slice x channel x H x W),
    transformed per-slice, stacked, then optionally passed through
    window_transform (e.g. WindowAmplitudeNormalize, which needs to see the
    whole window at once and so cannot run per-slice). Volume edges are
    reflect-padded without duplicating the boundary slice
    (np.pad(mode="reflect")), so __len__ is unaffected by num_slices.

    Note: the window's center slice is NOT identical to what num_slices=1
    returns for the same idx whenever window_transform rescales relative to
    the whole window (e.g. WindowAmplitudeNormalize divides every slice by
    the window's amplitude maximum, not its own) - only the raw read before
    any such window-level rescaling is shared with the num_slices=1 path.
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
        window_transform: Optional[Callable] = None,
        num_slices: int = 1,
        mode: str = "generation",
        acceleration: int = 4,
        mask_seed: Optional[int] = None,
        pre_transform: Optional[Callable] = None,
        post_transform: Optional[Callable] = None,
    ) -> None:
        if num_slices <= 0 or num_slices % 2 == 0:
            raise ValueError(f"num_slices must be a positive odd integer, got {num_slices}")

        if mode not in ("generation", "reconstruction"):
            raise ValueError(f"mode must be 'generation' or 'reconstruction', got {mode!r}")

        if mode == "reconstruction" and num_slices != 1:
            raise ValueError(
                f"mode='reconstruction' currently supports num_slices=1 only, got {num_slices}. "
                "Multi-slice reconstruction needs a decision on whether the whole window "
                "shares one mask (one acquisition) or gets independent masks; out of scope here."
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
        self.slice_map = []
        self._volume_depths: dict[str, int] = {}

        # Create a map of pointers to individual image slices
        for f_path in self.files:
            with h5py.File(f_path, "r") as f:
                depth = f["target"].shape[0]
                self._volume_depths[f_path] = depth
                for i in range(depth):
                    self.slice_map.append((f_path, i))

    def __len__(self) -> int:
        return len(self.slice_map)

    @staticmethod
    def _reflect_index(i: int, depth: int) -> int:
        """Reflects an out-of-range slice index without duplicating the boundary."""
        if depth <= 1:
            return 0
        while i < 0 or i > depth - 1:
            i = -i if i < 0 else 2 * (depth - 1) - i
        return i

    def _mask_seed(self, f_path: str, slice_idx: int) -> int:
        """Deterministic seed from (file, slice, acceleration[, mask_seed]), stable
        across processes/workers (unlike Python's randomized hash())."""
        key = f"{os.path.basename(f_path)}:{slice_idx}:{self.acceleration}"
        if self.mask_seed is not None:
            key = f"{key}:{self.mask_seed}"
        return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big")

    def _undersampling_mask(self, f_path: str, slice_idx: int, h: int, w: int) -> torch.Tensor:
        """Builds a deterministic 1D Cartesian phase-encoding mask along the last
        (Nz) axis: a fixed center ACS region is always kept, and the remaining
        lines needed to reach `self.acceleration` are chosen uniformly at random
        with a per-(file, slice) seed, then tiled to [1, H, W]."""
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

    def __getitem__(self, idx: int) -> torch.Tensor:
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
