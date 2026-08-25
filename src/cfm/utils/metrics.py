"""Numerical metrics for scoring generated complex-valued MRI.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = [
    "circular_phase_error",
    "peak_signal_noise_ratio",
    "shortest_angular_difference",
    "structural_similarity",
]

_REDUCTIONS = ("mean", "sum", "none")


def _as_amplitude(x: torch.Tensor) -> torch.Tensor:
    """Return the amplitude of ``x``, accepting complex or already-real input."""
    return torch.abs(x) if torch.is_complex(x) else x


def _as_phase(x: torch.Tensor) -> torch.Tensor:
    """Return the phase of ``x`` in radians, accepting complex or real input."""
    return torch.angle(x) if torch.is_complex(x) else x


def _check_shapes(pred: torch.Tensor, target: torch.Tensor) -> None:
    if pred.shape != target.shape:
        raise ValueError(
            f"pred and target must have the same shape, "
            f"got {tuple(pred.shape)} vs {tuple(target.shape)}"
        )


def _reduce(values: torch.Tensor, reduction: str) -> torch.Tensor:
    """Collapse per-sample values into the requested output form.

    ``mean``/``sum`` propagate NaN by design: a NaN here means a sample could not
    be scored (see :func:`circular_phase_error`), and silently dropping it would
    bias the reported number.
    """
    if reduction == "mean":
        return values.mean()
    if reduction == "sum":
        return values.sum()
    if reduction == "none":
        return values
    raise ValueError(f"reduction must be one of {_REDUCTIONS}, got {reduction!r}")


def _sample_dims(x: torch.Tensor) -> tuple[int, ...]:
    """All dims except the leading batch dim, so metrics are computed per sample."""
    return tuple(range(1, x.dim()))


def _gaussian_window(
    window_size: int, sigma: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Build a normalised 2D Gaussian kernel of shape ``[window_size, window_size]``."""
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    return torch.outer(g, g)



def shortest_angular_difference(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Signed shortest angular distance ``pred - target``, wrapped to ``[-pi, pi]``.

    Implemented as ``atan2(sin(d), cos(d))`` rather than modular arithmetic: the
    two agree numerically, but ``atan2`` is smooth everywhere including at the
    wrap point, so this helper stays safe if it is ever used inside a loss.

    Args:
        pred: Predicted angles in radians (any real shape). Need not be pre-wrapped.
        target: Target angles in radians, same shape as ``pred``.

    Returns:
        Signed differences in ``[-pi, pi]``, same shape as the inputs.
    """
    diff = pred - target
    return torch.atan2(torch.sin(diff), torch.cos(diff))


def peak_signal_noise_ratio(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """Peak signal-to-noise ratio (dB), computed on the amplitude.

    MSE is taken per sample and converted to dB per sample before reduction,
    which is the usual convention: averaging in the dB domain stops one very bad
    slice from dominating the batch the way a pooled MSE would.

    Args:
        pred: Predicted image, complex ``[B, C, H, W]`` or real amplitude.
        target: Reference image, same shape and kind as ``pred``.
        data_range: Peak value of the signal. Use ``1.0`` for amplitude-normalised
            data (what :class:`cfm.data.transforms.AmplitudeNormalize` produces);
            pass the true dynamic range otherwise, or the number is meaningless.
        reduction: ``"mean"``, ``"sum"`` or ``"none"`` (per-sample values).

    Returns:
        Scalar tensor in dB, or ``[B]`` when ``reduction="none"``.
        A perfect match yields ``+inf``.
    """
    pred_amp = _as_amplitude(pred)
    target_amp = _as_amplitude(target)
    _check_shapes(pred_amp, target_amp)

    if data_range <= 0:
        raise ValueError(f"data_range must be positive, got {data_range}")

    mse = ((pred_amp - target_amp) ** 2).mean(dim=_sample_dims(pred_amp))
    # mse == 0 divides to +inf, which is the correct PSNR for an exact match.
    psnr = 10.0 * torch.log10(data_range**2 / mse)
    return _reduce(psnr, reduction)


def structural_similarity(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    reduction: str = "mean",
) -> torch.Tensor:
    """Structural similarity index, computed on the amplitude.

    Uses the standard Gaussian-weighted formulation (11x11 window, sigma 1.5) with
    a *valid* convolution: no zero padding, so border pixels are not compared
    against fabricated black neighbours. The output map is therefore
    ``window_size - 1`` pixels smaller in each spatial dimension, which is only
    used internally for the average.

    Args:
        pred: Predicted image, complex ``[B, C, H, W]`` or real amplitude.
        target: Reference image, same shape and kind as ``pred``.
        data_range: Peak value of the signal; sets the stabilising constants
            ``C1 = (0.01 * data_range)^2`` and ``C2 = (0.03 * data_range)^2``.
        window_size: Side length of the Gaussian window. Must be odd and no larger
            than the smaller spatial dimension.
        sigma: Standard deviation of the Gaussian window.
        reduction: ``"mean"``, ``"sum"`` or ``"none"`` (per-sample values).

    Returns:
        Scalar tensor in ``[-1, 1]``, or ``[B]`` when ``reduction="none"``.
        A perfect match yields ``1.0``.
    """
    pred_amp = _as_amplitude(pred).float()
    target_amp = _as_amplitude(target).float()
    _check_shapes(pred_amp, target_amp)

    if pred_amp.dim() != 4:
        raise ValueError(f"structural_similarity expects [B, C, H, W], got {tuple(pred_amp.shape)}")
    if window_size % 2 == 0:
        raise ValueError(f"window_size must be odd, got {window_size}")
    if data_range <= 0:
        raise ValueError(f"data_range must be positive, got {data_range}")

    _, channels, height, width = pred_amp.shape
    if min(height, width) < window_size:
        raise ValueError(
            f"window_size={window_size} does not fit an image of size {height}x{width}"
        )

    window = _gaussian_window(window_size, sigma, pred_amp.device, pred_amp.dtype)
    window = window.expand(channels, 1, window_size, window_size).contiguous()

    def filt(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, window, groups=channels)

    mu_p = filt(pred_amp)
    mu_t = filt(target_amp)

    mu_p_sq = mu_p * mu_p
    mu_t_sq = mu_t * mu_t
    mu_pt = mu_p * mu_t

    # E[x^2] - E[x]^2, with the same Gaussian weighting as the means.
    sigma_p = filt(pred_amp * pred_amp) - mu_p_sq
    sigma_t = filt(target_amp * target_amp) - mu_t_sq
    sigma_pt = filt(pred_amp * target_amp) - mu_pt

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    ssim_map = ((2 * mu_pt + c1) * (2 * sigma_pt + c2)) / (
        (mu_p_sq + mu_t_sq + c1) * (sigma_p + sigma_t + c2)
    )

    ssim_per_sample = ssim_map.mean(dim=_sample_dims(ssim_map))
    return _reduce(ssim_per_sample, reduction)


def circular_phase_error(
    pred: torch.Tensor,
    target: torch.Tensor,
    amplitude_mask: torch.Tensor | None = None,
    mask_threshold: float | None = None,
    reduction: str = "mean",
) -> torch.Tensor:
    """Mean shortest angular distance between predicted and target phase, in radians.

    Background pixels carry essentially random phase (there is no signal there to
    predict), so an unmasked score over a knee MRI is dominated by air. Supply
    either an explicit ``amplitude_mask`` or a ``mask_threshold`` to restrict the
    average to pixels that actually contain tissue.

    Args:
        pred: Predicted image, complex ``[B, C, H, W]`` or real phase in radians.
        target: Reference image, same shape and kind as ``pred``.
        amplitude_mask: Optional boolean tensor broadcastable to the error map.
            Only ``True`` pixels contribute to the average.
        mask_threshold: Optional fraction in ``[0, 1)``. When given (and
            ``amplitude_mask`` is ``None``), builds a mask from the *target*
            amplitude as ``amp > mask_threshold * amp.max()``, with the max taken
            per sample. Requires a complex ``target``.
        reduction: ``"mean"``, ``"sum"`` or ``"none"`` (per-sample values).

    Returns:
        Scalar tensor in ``[0, pi]``, or ``[B]`` when ``reduction="none"``.
        A perfect match yields ``0.0``. A sample whose mask selects no pixels
        yields ``NaN``: it was not scored, which is not the same as scoring
        perfectly.
    """
    pred_phase = _as_phase(pred)
    target_phase = _as_phase(target)
    _check_shapes(pred_phase, target_phase)

    if amplitude_mask is not None and mask_threshold is not None:
        raise ValueError("pass either amplitude_mask or mask_threshold, not both")

    abs_error = shortest_angular_difference(pred_phase, target_phase).abs()
    dims = _sample_dims(abs_error)

    if mask_threshold is not None:
        if not 0.0 <= mask_threshold < 1.0:
            raise ValueError(f"mask_threshold must be in [0, 1), got {mask_threshold}")
        if not torch.is_complex(target):
            raise ValueError(
                "mask_threshold needs a complex target to derive the amplitude from; "
                "pass amplitude_mask instead"
            )
        target_amp = torch.abs(target)
        per_sample_max = target_amp.amax(dim=dims, keepdim=True)
        amplitude_mask = target_amp > (mask_threshold * per_sample_max)

    if amplitude_mask is None:
        error_per_sample = abs_error.mean(dim=dims)
    else:
        mask = amplitude_mask.expand_as(abs_error).to(abs_error.dtype)
        valid = mask.sum(dim=dims)
        total = (abs_error * mask).sum(dim=dims)
        # 0 / 0 -> NaN, flagging samples with an empty mask rather than faking a 0.
        error_per_sample = total / valid

    return _reduce(error_per_sample, reduction)