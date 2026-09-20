"""Data loading, caching, transforms and splits for complex-valued fields."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from typing import Any

import torch

from cyfm.config.resolve import as_plain_dict
from cyfm.core.dataset import BaseComplexDataset
from cyfm.core.registry import DATASETS
from cyfm.data.espirit import compute_espirit_maps, ensure_espirit_maps, process_h5_file
from cyfm.data.fastmri import FastMRIDataset
from cyfm.data.splits import load_split_file_names, select_indices
from cyfm.data.stores.audio import StftStoreDataset
from cyfm.data.stores.hdf5 import WorkerHDF5Manager
from cyfm.data.stores.knee import KneeStoreDataset
from cyfm.data.stores.stft import DEFAULT_STFT, StftProtocol, forward_stft, inverse_stft
from cyfm.data.torch_espirit import calibrate_fastmri_file_torch, compute_espirit_torch
from cyfm.data.toy import CylinderToyFieldDataset, CylinderToyIIDDataset
from cyfm.data.transforms import (
    AmplitudeNormalize,
    CenterCropModulo,
    CenterCropOrPad,
    ComplexToCylinderTransform,
    ComplexToEuclideanTransform,
    Compose,
    EuclideanNormalize,
    KSpaceCenterCrop,
    WindowAmplitudeNormalize,
    WindowEuclideanNormalize,
    slice_transform,
    window_transforms,
)

# Config keys that belong to a dataset group but are consumed by the entry point
# rather than passed to the dataset constructor.
_ENTRY_POINT_KEYS = frozenset({"name", "split", "crop_size"})

DEFAULT_DATASET = "cylinder_toy_field"


def build_geometry_transform(
    crop_size: int | tuple[int, int] | list[int] | None,
    crop_base: int = 16,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build the shape-only transform every entry point applies to complex slices.

    Kept separate from the manifold's representation pipeline because it has to run
    on the complex tensor, before the geometry decides how to represent it.

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
    the same class in ``train.py`` and ``evaluate.py``, and so
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

    name = str(options.pop("name", DEFAULT_DATASET))
    dataset_cls = DATASETS.get(name)
    accepted = set(inspect.signature(dataset_cls).parameters) - {"self"}

    # Only the *config* is validated. An unknown key there is a user typo and
    # should fail loudly. The overrides are the entry point's own plumbing --
    # transforms, a data directory, a slice count -- assembled the same way for
    # every dataset, so a dataset that has no use for one is not an error; a
    # synthetic sampler has no data_dir and needs none. Validating those too would
    # make adding a dataset mean editing every caller.
    unknown = sorted(set(options) - accepted - _ENTRY_POINT_KEYS)
    if unknown:
        raise TypeError(
            f"dataset={name!r} ({dataset_cls.__name__}) does not accept "
            f"{', '.join(unknown)}. Accepted keys: {', '.join(sorted(accepted))}."
        )

    options.update({key: value for key, value in overrides.items() if value is not None})
    kwargs = {key: value for key, value in options.items() if key in accepted}
    return dataset_cls(**kwargs)


__all__ = [
    "DEFAULT_DATASET",
    "AmplitudeNormalize",
    "CenterCropModulo",
    "CenterCropOrPad",
    "ComplexToCylinderTransform",
    "ComplexToEuclideanTransform",
    "Compose",
    "CylinderToyFieldDataset",
    "CylinderToyIIDDataset",
    "EuclideanNormalize",
    "FastMRIDataset",
    "KSpaceCenterCrop",
    "KneeStoreDataset",
    "StftStoreDataset",
    "StftProtocol",
    "DEFAULT_STFT",
    "forward_stft",
    "inverse_stft",
    "WindowAmplitudeNormalize",
    "WindowEuclideanNormalize",
    "WorkerHDF5Manager",
    "build_dataset",
    "build_geometry_transform",
    "calibrate_fastmri_file_torch",
    "compute_espirit_maps",
    "compute_espirit_torch",
    "ensure_espirit_maps",
    "load_split_file_names",
    "process_h5_file",
    "select_indices",
    "slice_transform",
    "window_transforms",
]
