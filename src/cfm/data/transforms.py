"""Preprocessing and transformation pipelines for complex MRI tensors."""

from __future__ import annotations

from collections.abc import Callable

import torch

from cfm.utils.complex_ops import complex_to_cylinder, complex_to_euclidean


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
