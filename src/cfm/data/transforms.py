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
