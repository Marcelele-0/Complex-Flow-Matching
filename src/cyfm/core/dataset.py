"""Base dataset interface for complex-valued fields."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch.utils.data import Dataset


class BaseComplexDataset(ABC, Dataset):
    """Abstract base dataset for complex-valued fields.

    Attributes:
        slice_map: One ``(file_path, slice_idx)`` per sample, in dataset order.
            Part of the interface rather than an implementation detail:
            :func:`~cyfm.data.splits.select_indices` reads it to gate a dataset down
            to a split, which is how ``train.py`` and ``evaluate.py`` keep their
            volumes disjoint, and the entry points use it to name samples in
            reports.
    """

    slice_map: list[tuple[str, int]]

    @abstractmethod
    def __len__(self) -> int:
        """Total number of samples/slices in the dataset."""

    @abstractmethod
    def __getitem__(self, idx: int) -> torch.Tensor | dict[str, torch.Tensor]:
        """Retrieve slice or volume item by index.

        Returns:
            A complex tensor, ``[1, H, W]`` for a single field.
        """
