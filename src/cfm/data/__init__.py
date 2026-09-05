"""Data loading, caching, transforms, and splits for complex MRI datasets."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any, cast

import torch

from cfm.core.dataset import BaseComplexDataset
from cfm.core.registry import DATASETS
from cfm.data.dataset import SKMTEADataset
from cfm.data.fastmri import FastMRIDataset
from cfm.data.hdf5_manager import WorkerHDF5Manager
from cfm.data.masks import (
    BaseMaskGenerator,
    CartesianMaskGenerator,
    PoissonDiscMaskGenerator,
)
from cfm.data.splits import load_split_file_names, select_indices
from cfm.data.transforms import (
    AmplitudeNormalize,
    CenterCropModulo,
    CenterCropOrPad,
    ComplexToCylinderTransform,
    ComplexToEuclideanTransform,
    Compose,
    EuclideanNormalize,
    WindowAmplitudeNormalize,
    WindowEuclideanNormalize,
)
from cfm.utils.config import as_plain_dict

# Config keys that belong to a dataset group but are consumed by the entry point
# rather than passed to the dataset constructor.
_ENTRY_POINT_KEYS = frozenset({"name", "split", "crop_size"})

DEFAULT_DATASET = "skm_tea"


def build_geometry_transform(
    crop_size: int | tuple[int, int] | list[int] | None,
    crop_base: int = 16,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build the shape-only transform every entry point applies to complex slices.

    Kept separate from the manifold's representation pipeline because it has to run
    on the complex tensor, and because in reconstruction mode it is handed to the
    dataset as ``pre_transform``, where the sensitivity maps are cropped to match it.

    Args:
        crop_size: Fixed ``(H, W)`` output, or ``None`` to only enforce divisibility.
        crop_base: Divisibility factor required by the model's downsampling depth.

    Returns:
        A callable mapping ``[..., H, W]`` to a model-compatible spatial shape.
    """
    stages: list[Callable[[torch.Tensor], torch.Tensor]] = []
    if crop_size is not None:
        stages.append(CenterCropOrPad(crop_size))
    stages.append(CenterCropModulo(base=crop_base))
    return Compose(stages)


def build_dataset(
    dataset_cfg: Mapping[str, Any] | None,
    **overrides: Any,
) -> BaseComplexDataset:
    """Instantiate the configured dataset through the :data:`DATASETS` registry.

    Every entry point resolves its dataset here so that ``dataset=<group>`` selects
    the same class in ``train.py``, ``evaluate.py`` and ``reconstruct.py``, and so
    that registering a class is what makes it reachable rather than a hardcoded
    branch that has to be updated in three places.

    Args:
        dataset_cfg: The ``dataset`` config group. ``name`` picks the registry key.
        **overrides: Constructor arguments that win over the config, typically the
            transforms and ``mode``, which are built by the caller. ``None`` values
            are ignored so a caller can pass an unset optional through.

    Returns:
        The constructed dataset.

    Raises:
        KeyError: If ``name`` is not a registered dataset.
        TypeError: If the config carries a key the selected dataset cannot accept.
    """
    # Resolved to plain containers up front so the dataset constructors never see a
    # DictConfig and stay usable from tests and scripts with no Hydra in the picture.
    options: dict[str, Any] = as_plain_dict(dataset_cfg)
    options.update({key: value for key, value in overrides.items() if value is not None})

    name = str(options.pop("name", DEFAULT_DATASET))
    dataset_cls = DATASETS.get(name)

    accepted = set(inspect.signature(dataset_cls).parameters) - {"self"}
    unknown = sorted(set(options) - accepted - _ENTRY_POINT_KEYS)
    if unknown:
        raise TypeError(
            f"dataset={name!r} ({dataset_cls.__name__}) does not accept "
            f"{', '.join(unknown)}. Accepted keys: {', '.join(sorted(accepted))}."
        )

    kwargs = {key: value for key, value in options.items() if key in accepted}
    return cast(BaseComplexDataset, dataset_cls(**kwargs))


__all__ = [
    "DEFAULT_DATASET",
    "BaseMaskGenerator",
    "CartesianMaskGenerator",
    "PoissonDiscMaskGenerator",
    "AmplitudeNormalize",
    "CenterCropModulo",
    "CenterCropOrPad",
    "ComplexToCylinderTransform",
    "ComplexToEuclideanTransform",
    "Compose",
    "EuclideanNormalize",
    "FastMRIDataset",
    "SKMTEADataset",
    "WindowAmplitudeNormalize",
    "WindowEuclideanNormalize",
    "WorkerHDF5Manager",
    "build_dataset",
    "build_geometry_transform",
    "load_split_file_names",
    "select_indices",
]
