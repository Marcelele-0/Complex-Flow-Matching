import glob
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
    transformed per-slice then stacked. Volume edges are reflect-padded
    without duplicating the boundary slice (np.pad(mode="reflect")), so
    __len__ and the window's center slice are unaffected by num_slices.
    """

    def __init__(
        self,
        data_dir: str,
        transform: Optional[Callable] = None,
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

        self.files = glob.glob(f"{data_dir}/files_recon_calib-24/*.h5")
        if not self.files:
            raise FileNotFoundError(f"No .h5 files found in: {data_dir}")

        self.transform = transform
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

    def __getitem__(self, idx: int) -> torch.Tensor:
        f_path, slice_idx = self.slice_map[idx]

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

        return torch.stack(slices, dim=0)
