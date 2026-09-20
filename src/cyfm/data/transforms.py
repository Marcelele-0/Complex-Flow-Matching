"""Preprocessing and transformation pipelines for complex MRI tensors.

The two functions at the foot of this module are the *only* place a geometry's
representation pipeline is composed. Each geometry declares a
:class:`~cyfm.core.manifold.Representation` and nothing more, which is what keeps
the two arms of the paper's comparison sharing a preprocessing chain by
construction rather than by two implementations agreeing to.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from cyfm.core.manifold import BaseManifold, Representation
from cyfm.utils.complex_ops import complex_to_cylinder, complex_to_euclidean
from cyfm.utils.fft import fft2c, ifft2c


class ComplexToCylinderTransform:
    """Maps complex tensor to 3-channel cylindrical manifold representation."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Convert complex tensor to cylindrical state.

        Args:
            x: Complex tensor [1, H, W] or [H, W] complex64.

        Returns:
            Cylindrical tensor [3, H, W] (amplitude, cos(phi), sin(phi)).
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)

        cyl_data = complex_to_cylinder(x)

        if cyl_data.dim() == 4:
            cyl_data = cyl_data.squeeze(0)

        return cyl_data


class AmplitudeNormalize:
    """Normalizes amplitude channel (index 0) to [0, 1] range."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize amplitude of cylindrical slice.

        Args:
            x: Cylindrical tensor [3, H, W].

        Returns:
            Normalized cylindrical tensor [3, H, W] with amplitude in [0, 1].
        """
        amp = x[0:1, :, :]
        amp_max = amp.max()

        if amp_max > 0:
            amp = amp / amp_max

        x[0:1, :, :] = amp
        return x


class WindowAmplitudeNormalize:
    """Normalizes amplitude channel across multi-slice window by shared peak."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize entire multi-slice window by global peak amplitude.

        Args:
            x: Window tensor [S, 3, H, W].

        Returns:
            Normalized window [S, 3, H, W].

        Raises:
            ValueError: If input tensor is not rank 4.
        """
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
    """Normalizes 2-channel Euclidean window by shared peak modulus."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize entire multi-slice window by global peak modulus.

        Args:
            x: Window tensor [S, 2, H, W].

        Returns:
            Normalized window [S, 2, H, W].

        Raises:
            ValueError: If input tensor is not rank 4.
        """
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
    """Maps complex tensor to flat 2-channel Euclidean representation (Re, Im)."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Convert complex tensor to Euclidean 2-channel state.

        Args:
            x: Complex tensor [1, H, W] or [H, W] complex64.

        Returns:
            Euclidean tensor [2, H, W] (real, imag).
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)

        euc_data = complex_to_euclidean(x)

        if euc_data.dim() == 4:
            euc_data = euc_data.squeeze(0)

        return euc_data


class EuclideanNormalize:
    """Normalizes real and imaginary channels by peak complex modulus to [0, 1]."""

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize Euclidean slice by peak modulus.

        Args:
            x: Euclidean tensor [2, H, W].

        Returns:
            Normalized Euclidean tensor [2, H, W].
        """
        modulus = torch.sqrt(x[0:1, :, :] ** 2 + x[1:2, :, :] ** 2)
        peak = modulus.max()

        if peak > 0:
            x = x / peak

        return x


class CenterCropModulo:
    """Center crops image spatial dimensions to nearest multiple of base.

    Args:
        base: Divisibility factor (e.g. 16 for standard 4-level U-Nets).
    """

    def __init__(self, base: int = 16) -> None:
        self.base = base

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Center crop trailing spatial dimensions.

        Args:
            x: Tensor [..., H, W].

        Returns:
            Cropped tensor [..., H_new, W_new] where H_new, W_new are multiples of base.
        """
        h, w = x.shape[-2], x.shape[-1]
        new_h = (h // self.base) * self.base
        new_w = (w // self.base) * self.base

        if new_h == h and new_w == w:
            return x

        top = (h - new_h) // 2
        left = (w - new_w) // 2

        return x[..., top : top + new_h, left : left + new_w]


class CenterCropOrPad:
    """Resizes the trailing spatial dimensions to a fixed shape by cropping or zero-padding.

    :class:`CenterCropModulo` rounds each image down to a multiple of ``base``, which
    is enough for a U-Net but does *not* give every sample the same shape. fastMRI is
    not a uniform cohort - knee volumes are 640x368 and 640x372, brain volumes include
    640x320 and 768x396 - so a batch drawn across volumes cannot be collated without a
    fixed target size. Cropping also removes the 2x readout oversampling that fastMRI
    k-space carries, which is why the reference pipeline centre-crops to 320x320.

    Padding is supported so a volume smaller than the target in some dimension still
    produces the agreed shape instead of aborting the epoch.

    Args:
        size: Target ``(H, W)``, or a single int for a square output.
    """

    def __init__(self, size: int | tuple[int, int] | list[int]) -> None:
        if isinstance(size, int):
            height, width = size, size
        else:
            height, width = int(size[0]), int(size[1])
        if height < 1 or width < 1:
            raise ValueError(f"crop size must be positive, got {(height, width)}")
        self.height = height
        self.width = width

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Center crop or zero-pad the trailing spatial dimensions.

        Args:
            x: Tensor [..., H, W].

        Returns:
            Tensor [..., self.height, self.width].
        """
        h, w = x.shape[-2], x.shape[-1]

        crop_h = min(h, self.height)
        crop_w = min(w, self.width)
        top = (h - crop_h) // 2
        left = (w - crop_w) // 2
        x = x[..., top : top + crop_h, left : left + crop_w]

        if crop_h == self.height and crop_w == self.width:
            return x

        pad_top = (self.height - crop_h) // 2
        pad_bottom = self.height - crop_h - pad_top
        pad_left = (self.width - crop_w) // 2
        pad_right = self.width - crop_w - pad_left
        # F.pad does not accept complex tensors, so pad the parts separately.
        if x.is_complex():
            pads = (pad_left, pad_right, pad_top, pad_bottom)
            real = torch.nn.functional.pad(x.real, pads)
            imag = torch.nn.functional.pad(x.imag, pads)
            return torch.complex(real, imag)
        return torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom))


class KSpaceCenterCrop:
    """Reduces resolution by keeping only the centre of k-space.

    This is a lower-resolution acquisition, not a resampled image. The stores this runs on
    hold images *after* ESPIRiT coil combination, so the transform has to return to k-space
    itself: :func:`~cyfm.utils.fft.fft2c`, keep the centre block about DC, then
    :func:`~cyfm.utils.fft.ifft2c`. Truncating the spectrum is what a scanner does when it
    acquires a smaller matrix. Cropping the image instead would shrink the field of view,
    and interpolating it would invent detail the measurement never contained.

    The block is anchored to the DC bin at ``H // 2`` rather than to the array centre,
    which is the convention k-space centre cropping uses. The two agree
    whenever both sizes are even, and the DC-anchored form stays symmetric about DC for odd
    ones as well.

    An orthonormal transform cropped from ``N`` to ``M`` leaves the image scaled by
    ``N / M``, since the forward transform normalises by ``N`` and the inverse by ``M``.
    Nothing here corrects that, because every pipeline using this divides by the field's own
    peak modulus afterwards (:class:`AmplitudeNormalize`, :class:`EuclideanNormalize`) and a
    global factor is removed there. A correction would be counted twice.

    Args:
        size: Target ``(H, W)``, or a single int for a square output.

    Raises:
        ValueError: If ``size`` is not positive.
    """

    def __init__(self, size: int | tuple[int, int] | list[int]) -> None:
        if isinstance(size, int):
            height, width = size, size
        else:
            height, width = int(size[0]), int(size[1])
        if height < 1 or width < 1:
            raise ValueError(f"crop size must be positive, got {(height, width)}")
        self.height = height
        self.width = width

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Keep the centre of the spectrum and transform back to the image domain.

        Args:
            x: Complex image-domain tensor ``[..., H, W]``, centered.

        Returns:
            Complex tensor ``[..., self.height, self.width]``.

        Raises:
            ValueError: If ``x`` is not complex, or the target exceeds its spatial shape.
        """
        if not x.is_complex():
            raise ValueError(f"k-space crop needs a complex field, got {x.dtype}")

        h, w = x.shape[-2], x.shape[-1]
        if self.height > h or self.width > w:
            raise ValueError(
                f"cannot enlarge {(h, w)} to {(self.height, self.width)}: zero-filling "
                "k-space interpolates rather than measures"
            )
        if self.height == h and self.width == w:
            return x

        top = h // 2 - self.height // 2
        left = w // 2 - self.width // 2
        kept = fft2c(x)[..., top : top + self.height, left : left + self.width]
        return ifft2c(kept)


class Compose:
    """Sequentially chains a list of data transforms.

    Args:
        transforms: Sequence of transform callables.
    """

    def __init__(self, transforms: list[Callable[[torch.Tensor], torch.Tensor]]) -> None:
        self.transforms = transforms

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply transforms sequentially.

        Args:
            x: Input tensor.

        Returns:
            Transformed output tensor.
        """
        for t in self.transforms:
            x = t(x)
        return x


# What each representation swaps in. Only three parts differ between the arms:
# the channel layout, the per-slice normaliser and its per-window counterpart.
# The ordering around them -- normalise before cropping, one peak per window,
# the same crop base -- is shared, and is stated once below.
_Transform = Callable[[torch.Tensor], torch.Tensor]
_TransformFactory = Callable[[], _Transform]

_PIPELINES: dict[Representation, tuple[_TransformFactory, _TransformFactory, _TransformFactory]] = {
    Representation.CYLINDER: (
        ComplexToCylinderTransform,
        AmplitudeNormalize,
        WindowAmplitudeNormalize,
    ),
    Representation.PLANE: (
        ComplexToEuclideanTransform,
        EuclideanNormalize,
        WindowEuclideanNormalize,
    ),
}


def slice_transform(
    manifold: BaseManifold, crop_base: int = 16
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Build the preprocessing pipeline for single complex slices.

    Normalisation runs before the crop, so every arm divides by a peak modulus
    taken over the same uncropped slice. Cropping first would give each geometry
    a different peak and move every absolute number in the tables.

    Args:
        manifold: The geometry, read only for its
            :class:`~cyfm.core.manifold.Representation`.
        crop_base: Divisibility the model's downsampling depth requires.

    Returns:
        A callable mapping a complex ``[1, H, W]`` slice to a manifold state.

    Raises:
        KeyError: If the geometry declares a representation with no pipeline.
    """
    to_state, normalize, _ = _PIPELINES[manifold.representation]
    return Compose([to_state(), normalize(), CenterCropModulo(base=crop_base)])


def window_transforms(
    manifold: BaseManifold, crop_base: int = 16
) -> tuple[Callable[[torch.Tensor], torch.Tensor], Callable[[torch.Tensor], torch.Tensor]]:
    """Build the 2.5D pair: a per-slice stage and a per-window stage.

    Split where it has to be. The per-slice normaliser would divide each slice by
    its own peak and flatten the inter-slice brightness the window exists to
    carry; the window stage takes one peak over the whole stack instead.

    Args:
        manifold: The geometry, read only for its representation.
        crop_base: Divisibility the model's downsampling depth requires.

    Returns:
        ``(slice_stage, window_stage)``, applied in that order.

    Raises:
        KeyError: If the geometry declares a representation with no pipeline.
    """
    to_state, _, window_normalize = _PIPELINES[manifold.representation]
    return (
        Compose([to_state()]),
        Compose([window_normalize(), CenterCropModulo(base=crop_base)]),
    )
