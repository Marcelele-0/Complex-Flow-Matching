"""Base dataset interface for complex-valued MRI data."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset


class BaseComplexDataset(ABC, Dataset):
    """Abstract base dataset for complex-valued MRI reconstruction and synthesis."""

    @abstractmethod
    def __len__(self) -> int:
        """Total number of samples/slices in the dataset."""

    @abstractmethod
    def __getitem__(self, idx: int) -> torch.Tensor | dict[str, torch.Tensor]:
        """Retrieve slice or volume item by index.

        Returns:
            Either a complex tensor [1, H, W] / [S, 1, H, W] or a dictionary
            containing 'input', 'mask', 'target' for reconstruction tasks.
        """
