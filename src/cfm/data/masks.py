"""K-space undersampling mask generators for MRI reconstruction.

Provides abstract base class and modular generators for 1D Cartesian and 2D
Poisson-Disc undersampling trajectories across arbitrary acceleration factors.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch

from cfm.core.registry import MASKS


class BaseMaskGenerator(ABC):
    """Abstract base class for k-space undersampling mask generators.

    Args:
        acceleration: Undersampling acceleration factor R >= 1.0.
    """

    def __init__(self, acceleration: float = 4.0, **kwargs: Any) -> None:
        if acceleration < 1.0:
            raise ValueError(f"acceleration must be >= 1.0, got {acceleration}")
        self.acceleration = float(acceleration)

    @abstractmethod
    def generate(
        self,
        shape: tuple[int, ...] | int,
        *args: int,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Generate a binary undersampling mask.

        Args:
            shape: Tensor spatial shape (..., H, W) or H when passed with W as args.
            *args: Additional spatial dimensions if shape is an int.
            seed: Optional random seed for deterministic mask generation.

        Returns:
            torch.Tensor: Binary mask of shape [1, H, W] with dtype torch.float32.
        """
        ...

    def __call__(
        self,
        shape: tuple[int, ...] | int,
        *args: int,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Alias for generate()."""
        return self.generate(shape, *args, seed=seed)


@MASKS.register("cartesian")
@MASKS.register("cartesian1d")
class CartesianMaskGenerator(BaseMaskGenerator):
    """1D Cartesian phase-encoding undersampling mask generator.

    Generates 1D undersampling masks along a specified axis (typically phase encoding)
    with a fully sampled low-frequency autocalibration signal (ACS) center and
    random uniform sampling in the outer k-space region.

    Args:
        acceleration: Undersampling acceleration factor (e.g. 4.0, 8.0, 12.0).
        num_center_lines: Number of low-frequency center lines to fully sample.
        center_fraction: Optional fraction of center lines to fully sample.
            If specified, overrides num_center_lines.
        axis: Dimension along which to undersample (-1 for width, -2 for height).
    """

    def __init__(
        self,
        acceleration: float = 4.0,
        num_center_lines: int | None = None,
        center_fraction: float | None = None,
        axis: int = -1,
        **kwargs: Any,
    ) -> None:
        super().__init__(acceleration=acceleration, **kwargs)
        if center_fraction is not None and not (0.0 < center_fraction <= 1.0):
            raise ValueError(f"center_fraction must be in (0, 1], got {center_fraction}")
        if axis not in (-1, -2, 0, 1):
            raise ValueError(f"axis must be -1, -2, 0, or 1, got {axis}")

        self.num_center_lines = num_center_lines
        self.center_fraction = center_fraction
        self.axis = axis

    def generate(
        self,
        shape: tuple[int, ...] | int,
        *args: int,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Generate 1D Cartesian mask of shape [1, H, W].

        Args:
            shape: Shape (H, W) or (..., H, W), or int H.
            *args: W if shape was int H.
            seed: Deterministic random seed.

        Returns:
            torch.Tensor: Binary mask of shape [1, H, W] and dtype torch.float32.
        """
        if isinstance(shape, int):
            shape = (shape, *args)
        if len(shape) < 2:
            raise ValueError(f"shape must have at least 2 dimensions, got {shape}")

        h, w = int(shape[-2]), int(shape[-1])
        dim = w if self.axis in (-1, 1) else h

        if self.acceleration <= 1.0:
            return torch.ones((1, h, w), dtype=torch.float32)

        target_lines = int(round(dim / self.acceleration))
        target_lines = max(1, min(dim, target_lines))

        # Determine center lines budget
        if self.center_fraction is not None:
            center_lines = max(1, int(round(dim * self.center_fraction)))
            center_lines = min(center_lines, target_lines, dim)
        elif self.num_center_lines is not None:
            center_lines = min(self.num_center_lines, target_lines, dim)
        else:
            center_lines = min(24, dim)
            if center_lines > target_lines:
                center_lines = max(1, target_lines // 2)

        center_start = dim // 2 - center_lines // 2
        center_end = center_start + center_lines

        line_mask = np.zeros(dim, dtype=np.float32)
        line_mask[center_start:center_end] = 1.0

        remaining_lines = target_lines - center_lines
        if remaining_lines > 0:
            rng = np.random.default_rng(seed)
            leftover_indices = np.concatenate(
                [np.arange(0, center_start), np.arange(center_end, dim)]
            )
            sampled = rng.choice(leftover_indices, remaining_lines, replace=False)
            line_mask[sampled] = 1.0

        if self.axis in (-1, 1):
            mask = np.tile(line_mask, (h, 1))
        else:
            mask = np.tile(line_mask[:, None], (1, w))

        return torch.from_numpy(mask).unsqueeze(0).to(torch.float32)


@MASKS.register("poisson_disc")
@MASKS.register("poisson")
class PoissonDiscMaskGenerator(BaseMaskGenerator):
    """2D Variable-Density Poisson-Disc undersampling mask generator.

    Generates pseudo-random 2D Poisson-Disc masks with a minimum distance exclusion
    zone between samples and a fully sampled central calibration region. Variable
    density sampling ensures higher sampling frequency near k-space center with
    calibrated acceleration factor matching.

    Args:
        acceleration: Undersampling acceleration factor (e.g. 4.0, 8.0, 12.0).
        calib_size: Explicit calibration box size (calib_h, calib_w) or int.
        calib_fraction: Fraction of low frequencies to fully sample in each dimension.
        crop_corners: Whether to crop outer k-space corners (|k| > 1).
        power: Power law for variable density radius expansion r(k) = r0 * (1 + 2 * |k|^power).
        max_iter: Maximum binary search iterations to calibrate radius scale.
        tol: Tolerance for acceleration matching.
    """

    def __init__(
        self,
        acceleration: float = 4.0,
        calib_size: int | tuple[int, int] | None = None,
        calib_fraction: float | tuple[float, float] | None = None,
        crop_corners: bool = False,
        power: float = 1.5,
        max_iter: int = 15,
        tol: float = 0.03,
        **kwargs: Any,
    ) -> None:
        super().__init__(acceleration=acceleration, **kwargs)
        self.calib_size = calib_size
        self.calib_fraction = calib_fraction
        self.crop_corners = crop_corners
        self.power = power
        self.max_iter = max_iter
        self.tol = tol

    def generate(
        self,
        shape: tuple[int, ...] | int,
        *args: int,
        seed: int | None = None,
    ) -> torch.Tensor:
        """Generate 2D Poisson-Disc mask of shape [1, H, W].

        Args:
            shape: Shape (H, W) or (..., H, W), or int H.
            *args: W if shape was int H.
            seed: Deterministic random seed.

        Returns:
            torch.Tensor: Binary mask of shape [1, H, W] and dtype torch.float32.
        """
        if isinstance(shape, int):
            shape = (shape, *args)
        if len(shape) < 2:
            raise ValueError(f"shape must have at least 2 dimensions, got {shape}")

        h, w = int(shape[-2]), int(shape[-1])

        if self.acceleration <= 1.0:
            return torch.ones((1, h, w), dtype=torch.float32)

        target_samples = int(round((h * w) / self.acceleration))
        target_samples = max(1, min(h * w, target_samples))

        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0

        # Calibration region dimensions
        if self.calib_size is not None:
            if isinstance(self.calib_size, int):
                ch, cw = self.calib_size, self.calib_size
            else:
                ch, cw = int(self.calib_size[0]), int(self.calib_size[1])
        elif self.calib_fraction is not None:
            if isinstance(self.calib_fraction, (int, float)):
                fh, fw = float(self.calib_fraction), float(self.calib_fraction)
            else:
                fh, fw = float(self.calib_fraction[0]), float(self.calib_fraction[1])
            ch, cw = int(round(h * fh)), int(round(w * fw))
        else:
            ch = min(24, max(1, int(round(h * 0.08))))
            cw = min(24, max(1, int(round(w * 0.08))))

        ch = min(ch, h)
        cw = min(cw, w)

        # Ensure calibration region does not exceed target sample budget
        if ch * cw > target_samples * 0.6:
            scale = np.sqrt((target_samples * 0.6) / (ch * cw))
            ch = max(1, int(ch * scale))
            cw = max(1, int(cw * scale))

        c_y0 = max(0, int(np.floor(cy - ch / 2.0)))
        c_y1 = min(h, int(np.ceil(cy + ch / 2.0)))
        c_x0 = max(0, int(np.floor(cx - cw / 2.0)))
        c_x1 = min(w, int(np.ceil(cx + cw / 2.0)))

        y, x = np.ogrid[:h, :w]
        ny = (y - cy) / (h / 2.0)
        nx = (x - cx) / (w / 2.0)
        r_unclipped = np.sqrt(ny**2 + nx**2)
        r_norm = np.clip(r_unclipped, 0.0, 1.0)

        r_low = 0.5
        r_high = max(h, w) / 2.0
        best_mask: np.ndarray | None = None
        best_diff = float("inf")

        for it in range(self.max_iter):
            r_mid = (r_low + r_high) / 2.0
            r_map = r_mid * (1.0 + 2.0 * (r_norm**self.power))
            r_map[c_y0:c_y1, c_x0:c_x1] = 0.0

            rng = np.random.default_rng(seed + it * 1000 if seed is not None else None)
            weight = 1.0 / np.maximum(r_map, 0.5) ** 2
            keys = rng.uniform(0.0, 1.0, size=(h, w)) / weight
            keys[c_y0:c_y1, c_x0:c_x1] = 0.0

            flat_idx = np.argsort(keys.ravel())
            y_coords, x_coords = np.unravel_index(flat_idx, (h, w))

            mask = np.zeros((h, w), dtype=np.float32)
            excluded = np.zeros((h, w), dtype=bool)

            mask[c_y0:c_y1, c_x0:c_x1] = 1.0
            excluded[c_y0:c_y1, c_x0:c_x1] = True

            num_sampled = (c_y1 - c_y0) * (c_x1 - c_x0)

            for yi, xi in zip(y_coords, x_coords):
                if num_sampled >= target_samples:
                    break
                if self.crop_corners and r_unclipped[yi, xi] > 1.0:
                    continue
                if excluded[yi, xi]:
                    continue

                mask[yi, xi] = 1.0
                num_sampled += 1

                r_val = r_map[yi, xi]
                if r_val <= 0:
                    continue
                ir = int(np.ceil(r_val))
                y_min, y_max = max(0, yi - ir), min(h, yi + ir + 1)
                x_min, x_max = max(0, xi - ir), min(w, xi + ir + 1)
                dy, dx = np.ogrid[y_min - yi : y_max - yi, x_min - xi : x_max - xi]
                excluded[y_min:y_max, x_min:x_max] |= (dy**2 + dx**2 < r_val**2)

            cur_accel = (h * w) / max(1, num_sampled)
            diff = abs(cur_accel - self.acceleration)
            if diff < best_diff:
                best_diff = diff
                best_mask = mask

            if abs(num_sampled - target_samples) / target_samples <= self.tol:
                break

            if num_sampled > target_samples:
                r_low = r_mid
            else:
                r_high = r_mid

        if best_mask is None:
            best_mask = mask

        if self.crop_corners:
            best_mask[r_unclipped > 1.0] = 0.0

        return torch.from_numpy(best_mask).unsqueeze(0).to(torch.float32)
