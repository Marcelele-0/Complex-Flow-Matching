"""Synthetic complex fields as first-class datasets.

Two variants, differing in exactly one thing: whether neighbouring coefficients
are related. Both draw their coefficients through :class:`~cyfm.data.synthetic.CylinderToy`,
so at every pixel the amplitude marginal, the phase marginal and the dependence
between them are identical across the pair, and identical across every value of
``coupling``. The iid variant is the zero-correlation-length limit of the field
variant, which is what makes the two comparable.

Why both exist
--------------
The iid variant is the control. A convolutional trunk on a field whose
coefficients are independent has nothing spatial to exploit, so a U-Net there
should do no better than :class:`~cyfm.models.mlp.PointwiseVelocityMLP`.
If it does, the difference is capacity or optimisation, not structure. Run it
before believing any architecture result on the structured variant.

The field variant is the one that can answer a question about architecture. Its
spatial correlation comes from smoothing the *latents* before the copula is
applied, never the samples: smoothing a phase field directly would drag the
phase marginal toward its circular mean and wrap badly across the branch cut,
and smoothing amplitudes would narrow their distribution. Smoothing standard
normals and renormalising leaves each latent standard normal marginally, so the
pointwise law is untouched and only the spatial dependence changes.

Neither variant touches disk. ``slice_map`` is synthesised to satisfy the
:class:`~cyfm.core.dataset.BaseComplexDataset` interface that the entry points use
to name samples; there are no files, no manifests and no splits, so a held-out
claim here means "a different seed", which is all it can mean for an analytic
distribution.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from cyfm.core.dataset import BaseComplexDataset
from cyfm.core.registry import DATASETS
from cyfm.data.synthetic import CylinderToy
from cyfm.utils.random_fields import smooth_standard_normals

__all__ = ["CylinderToyFieldDataset", "CylinderToyIIDDataset"]


class _CylinderToyDataset(BaseComplexDataset):
    """Shared machinery for the two synthetic field datasets.

    Args:
        coupling: Amplitude-phase dependence, passed to :class:`CylinderToy`.
        structure: Dependence structure, passed to :class:`CylinderToy`.
        size: Number of samples the dataset reports.
        crop_size: ``(H, W)`` of each field.
        seed: Base seed; sample ``i`` is drawn from ``seed + i`` and is therefore
            stable across epochs, workers and processes.
        transform: Applied to the complex field before it is returned. The entry
            points pass the manifold's representation pipeline here, so the
            dataset yields the state the model consumes rather than a complex
            tensor, exactly as the file-backed cohorts do.

    Raises:
        ValueError: If ``size`` is not positive or ``crop_size`` is not a pair of
            positive integers.
    """

    def __init__(
        self,
        coupling: float = 0.0,
        structure: str = "spiral",
        size: int = 4096,
        crop_size: tuple[int, int] = (64, 64),
        seed: int = 0,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        if size <= 0:
            raise ValueError(f"size must be positive, got {size}")
        if len(crop_size) != 2 or any(int(v) <= 0 for v in crop_size):
            raise ValueError(f"crop_size must be two positive integers, got {crop_size}")

        self.toy = CylinderToy(coupling=coupling, structure=structure)  # type: ignore[arg-type]
        self.size = int(size)
        self.height, self.width = int(crop_size[0]), int(crop_size[1])
        self.seed = int(seed)
        self.transform = transform
        self.slice_map = [("synthetic", index) for index in range(self.size)]

    def __len__(self) -> int:
        """Number of samples."""
        return self.size

    def _latents(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw the pair of standard normal latent fields for one sample.

        Args:
            index: Sample index.

        Returns:
            Two fields of shape ``[1, 1, H, W]``.

        Raises:
            NotImplementedError: Always; subclasses decide what structure the
                latents carry.
        """
        raise NotImplementedError

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Draw one complex field.

        Args:
            idx: Sample index.

        Returns:
            Complex tensor of shape ``[1, H, W]``, or whatever ``transform``
            turns it into.

        Raises:
            IndexError: If ``idx`` is out of range.
        """
        if not 0 <= idx < self.size:
            raise IndexError(f"index {idx} out of range for {self.size} samples")
        angle_normal, residual_normal = self._latents(idx)
        amplitude, phase = self.toy.polar_from_latents(angle_normal, residual_normal)
        field = torch.polar(amplitude, phase).reshape(1, self.height, self.width)
        return field if self.transform is None else self.transform(field)


@DATASETS.register("cylinder_toy_iid")
class CylinderToyIIDDataset(_CylinderToyDataset):
    """Every coefficient drawn independently: the control variant.

    There is no spatial information here at all, so a convolutional trunk cannot
    beat a pointwise one on it except by accident. That is what makes it useful.
    """

    def _latents(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw white latent fields.

        Args:
            index: Sample index.

        Returns:
            Two independent standard normal fields of shape ``[1, 1, H, W]``.
        """
        generator = torch.Generator().manual_seed(self.seed + index)
        shape = (1, 1, self.height, self.width)
        return (
            torch.randn(shape, generator=generator),
            torch.randn(shape, generator=generator),
        )


@DATASETS.register("cylinder_toy_field")
class CylinderToyFieldDataset(_CylinderToyDataset):
    """Spatially correlated coefficients: the variant with structure to learn.

    The constructor spells every argument out rather than forwarding ``**kwargs``:
    :func:`~cyfm.data.build_dataset` resolves config keys by inspecting the
    signature, so an inherited argument hidden behind ``**kwargs`` would be
    rejected as an unknown key.

    Args:
        coupling: Amplitude-phase dependence, passed to :class:`CylinderToy`.
        structure: Dependence structure, passed to :class:`CylinderToy`.
        size: Number of samples the dataset reports.
        crop_size: ``(H, W)`` of each field.
        seed: Base seed; sample ``i`` is drawn from ``seed + i``.
        correlation_length: Gaussian smoothing sigma in pixels applied to the
            latents. Larger means smoother fields; ``0`` reproduces the iid
            variant exactly.
        transform: Applied to the complex field before it is returned.

    Raises:
        ValueError: If ``correlation_length`` is negative, or an argument of the
            base class is out of range.
    """

    def __init__(
        self,
        coupling: float = 0.0,
        structure: str = "spiral",
        size: int = 4096,
        crop_size: tuple[int, int] = (64, 64),
        seed: int = 0,
        correlation_length: float = 4.0,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        super().__init__(
            coupling=coupling,
            structure=structure,
            size=size,
            crop_size=crop_size,
            seed=seed,
            transform=transform,
        )
        if correlation_length < 0.0:
            raise ValueError(f"correlation_length must be non-negative, got {correlation_length}")
        self.correlation_length = float(correlation_length)

    def _latents(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw smoothed latent fields.

        Args:
            index: Sample index.

        Returns:
            Two standard normal fields of shape ``[1, 1, H, W]``, each with the
            configured spatial correlation length and independent of the other.
        """
        generator = torch.Generator().manual_seed(self.seed + index)
        shape = (1, 1, self.height, self.width)
        return (
            smooth_standard_normals(
                torch.randn(shape, generator=generator), self.correlation_length
            ),
            smooth_standard_normals(
                torch.randn(shape, generator=generator), self.correlation_length
            ),
        )
