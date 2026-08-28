"""Transformation pipeline, split into per-slice and per-window stages.

Per-slice transforms (ComplexToCylinderTransform, AmplitudeNormalize) take a
single slice, [C, H, W] or [1, H, W] before the cylindrical projection.

Window transforms (WindowAmplitudeNormalize, WindowEuclideanNormalize) take a
stacked [S, C, H, W] window assembled by SKMTEADataset after the per-slice stage.

The two are NOT interchangeable. At num_slices=3 the slice axis and the
channel axis are both of size 3, so a transform applied at the wrong stage
would index the wrong axis and corrupt data silently instead of raising -
AmplitudeNormalize on a stacked window would rescale slice 0's cos/sin phase
channels. Window transforms therefore assert rank 4; CenterCropModulo is the
only rank-agnostic one, because it only ever touches the trailing two axes.
"""

from typing import Callable

import torch

from cfm.utils.complex_ops import complex_to_cylinder, complex_to_euclidean


class ComplexToCylinderTransform:
    """Maps a raw complex tensor to a 3-channel cylindrical topology."""
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Protect against dim=1 error: [C, H, W] -> [1, C, H, W]
        if x.dim() == 3:
            x = x.unsqueeze(0)

        cyl_data = complex_to_cylinder(x)

        # Squeeze back to 3D: [1, 3, H, W] -> [3, H, W]
        if cyl_data.dim() == 4:
            cyl_data = cyl_data.squeeze(0)

        return cyl_data


class AmplitudeNormalize:
    """
    Normalizes only the amplitude channel (index 0) to the [0, 1] range.
    Phase channels (index 1 and 2) remain untouched (R=1).
    """
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [3, H, W] on the cylinder
        amp = x[0:1, :, :]
        amp_max = amp.max()

        if amp_max > 0:
            amp = amp / amp_max

        x[0:1, :, :] = amp
        return x


class WindowAmplitudeNormalize:
    """Normalizes the amplitude channel across an entire multi-slice window.

    Input/output: [S, C, H, W] (slice x channel x H x W). All slices are
    divided by a single scalar - the maximum amplitude over the whole
    window - so relative brightness between neighbouring slices is
    preserved. Exactly one slice attains 1.0; the rest stay below it.

    Unlike AmplitudeNormalize this is NOT a per-slice transform and must run
    after stacking.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"WindowAmplitudeNormalize expects a stacked [S, C, H, W] window, "
                f"got a rank-{x.dim()} tensor {tuple(x.shape)}"
            )

        amp = x[:, 0:1, :, :]
        amp_max = amp.max()
        if amp_max > 0:
            x[:, 0:1, :, :] = amp / amp_max
        return x


class WindowEuclideanNormalize:
    """Divides an entire window by its peak modulus, putting |z| in the [0, 1] range.

    The window-scoped counterpart of EuclideanNormalize, and the Euclidean twin
    of WindowAmplitudeNormalize: one scalar - the largest modulus anywhere in the
    window - divides every slice, so relative brightness between neighbouring
    slices is preserved. Exactly one pixel attains 1.0.

    Scaling both channels by one factor divides the modulus by its max and leaves
    the argument untouched, so this and WindowAmplitudeNormalize put the *same*
    complex signal in front of the model. That is what keeps a 2.5D
    cylindrical-vs-Euclidean comparison a comparison of geometries.

    Like its cylindrical twin this is NOT a per-slice transform and must run
    after stacking.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(
                f"WindowEuclideanNormalize expects a stacked [S, C, H, W] window, "
                f"got a rank-{x.dim()} tensor {tuple(x.shape)}"
            )

        modulus = torch.sqrt(x[:, 0:1, :, :] ** 2 + x[:, 1:2, :, :] ** 2)
        peak = modulus.max()

        if peak > 0:
            x = x / peak

        return x


class ComplexToEuclideanTransform:
    """Maps a raw complex tensor to the flat 2-channel Euclidean representation."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Protect against dim=1 error: [C, H, W] -> [1, C, H, W]
        if x.dim() == 3:
            x = x.unsqueeze(0)

        euc_data = complex_to_euclidean(x)

        # Squeeze back to 3D: [1, 2, H, W] -> [2, H, W]
        if euc_data.dim() == 4:
            euc_data = euc_data.squeeze(0)

        return euc_data


class EuclideanNormalize:
    """
    Divides both channels by the peak modulus, putting |z| in the [0, 1] range.

    This is the exact counterpart of AmplitudeNormalize: scaling the real and
    imaginary parts by one common factor divides the modulus by its max and
    leaves the argument untouched. Both pipelines therefore carry the *identical*
    complex signal, differing only in how it is coordinatised - which is what
    makes PSNR's data_range=1.0 valid on both sides and the cylindrical-vs-
    Euclidean comparison meaningful.

    Applied before the crop, as AmplitudeNormalize is, so both geometries divide
    by a peak taken over the same (uncropped) slice.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [2, H, W]
        modulus = torch.sqrt(x[0:1, :, :] ** 2 + x[1:2, :, :] ** 2)
        peak = modulus.max()

        if peak > 0:
            x = x / peak

        return x


class CenterCropModulo:
    """
    Crops the image from the center to the nearest multiple of the given base (e.g., 16).
    Protects the U-Net architecture from Up-sampling errors.
    """
    def __init__(self, base: int = 16) -> None:
        self.base = base

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Rank-agnostic: works on [C, H, W] and stacked [S, C, H, W] alike,
        # since it only ever touches the two trailing spatial axes.
        h, w = x.shape[-2], x.shape[-1]
        new_h = (h // self.base) * self.base
        new_w = (w // self.base) * self.base

        if new_h == h and new_w == w:
            return x

        top = (h - new_h) // 2
        left = (w - new_w) // 2

        return x[..., top : top + new_h, left : left + new_w]


class Compose:
    """Connects a list of transformations into a single sequential pipeline."""
    def __init__(self, transforms: list[Callable]) -> None:
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        for t in self.transforms:
            x = t(x)
        return x
