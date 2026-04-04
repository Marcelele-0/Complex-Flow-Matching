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
    """
    def __init__(self, data_dir: str, transform: Optional[Callable] = None) -> None:
        self.files = glob.glob(f"{data_dir}/files_recon_calib-24/*.h5")
        if not self.files:
            raise FileNotFoundError(f"No .h5 files found in: {data_dir}")
            
        self.transform = transform
        self.slice_map = []
        
        # Create a map of pointers to individual image slices
        for f_path in self.files:
            with h5py.File(f_path, 'r') as f:
                num_slices = f["target"].shape[0]
                for i in range(num_slices):
                    self.slice_map.append((f_path, i))

    def __len__(self) -> int:
        return len(self.slice_map)

    def __getitem__(self, idx: int) -> torch.Tensor:
        f_path, slice_idx = self.slice_map[idx]
        
        with h5py.File(f_path, 'r') as f:
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