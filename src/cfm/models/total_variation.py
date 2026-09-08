"""Total variation regularised reconstruction baseline via sigpy.

Zero-filling is the smoke alarm of this project: an untrained adjoint that any
working method must clear, reported in `docs/GATE_63_PLAN.md` rather than in a
paper table. Total variation is the floor that actually appears in the
literature - it is the classical row in Chung & Ye's tables, and the one a
reviewer checks. Losing to a supervised network is ordinary for a generative
reconstructor; losing to TV is not defensible in any framing, which is why this
baseline is measured early rather than at the end.

Solves the standard formulation

    min_x  0.5 * || P F S x - y ||_2^2  +  lamda * || G x ||_1

through :class:`sigpy.mri.app.TotalVariationRecon`, where P is the sampling
operator, F the Fourier transform, S the SENSE operator and G the spatial
gradient. sigpy's transform defaults (``center=True``, ``norm='ortho'``) match
:func:`~cfm.utils.fft.fft2c`, so k-space handed to this reconstructor needs no
reinterpretation; :func:`tests.test_models.test_total_variation` pins that
agreement rather than trusting it.

The solver runs on CPU: it is a primal-dual iteration in NumPy, and the GPU
path would require cupy, which is not a dependency. Cost is therefore linear in
``max_iter`` and in the number of slices scored.

sigpy estimates the largest eigenvalue of the normal operator by power iteration
from a **random** start, which makes an unseeded run irreproducible - measured at
2.8e-4 spread on a unit-scale image, which is small but not nothing for a number
that goes in a table. Each slice is therefore reconstructed under a seeded NumPy
RNG, with the global state saved and restored so nothing else in the process sees
the change.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from cfm.core.reconstructor import BaseReconstructor
from cfm.core.registry import MODELS, RECONSTRUCTORS

try:
    from sigpy.mri.app import TotalVariationRecon

    _SIGPY_AVAILABLE = True
except ImportError:  # pragma: no cover - sigpy is a hard dependency via ESPIRiT
    TotalVariationRecon = None  # type: ignore[assignment,misc]
    _SIGPY_AVAILABLE = False


@MODELS.register("total_variation")
@RECONSTRUCTORS.register("total_variation")
class TotalVariationReconstructor(BaseReconstructor):
    """Compressed-sensing baseline: TV-regularised reconstruction.

    Registered in ``MODELS`` as well as ``RECONSTRUCTORS`` so ``evaluate.py``
    scores it through :func:`~cfm.utils.inference.build_model` like every other
    arm, under the same masks, metrics and seed. It carries no learned
    parameters, so it is evaluated with ``evaluate.run_name=SKIP``.

    Args:
        lamda: Regularisation weight on the total variation term. Tune it on the
            training split only - fitting it to the scored volume would make the
            floor artificially weak, which defeats the purpose of measuring it.
        max_iter: Primal-dual iterations. Runtime is linear in this.
        seed: Base seed for sigpy's power iteration. Slice ``i`` of a batch is
            solved under ``seed + i``, so a result depends on the slice's position
            in the dataset rather than on the batch size it happened to be scored
            with.
        **kwargs: Ignored. Absorbs ``in_channels`` / ``out_channels`` from the
            generic registry build path, which every architecture receives.

    Raises:
        ImportError: If sigpy is unavailable.
    """

    def __init__(
        self, lamda: float = 0.005, max_iter: int = 100, seed: int = 0, **kwargs: Any
    ) -> None:
        del kwargs
        super().__init__()
        if not _SIGPY_AVAILABLE:
            raise ImportError(
                "TotalVariationReconstructor needs sigpy. It ships with the ESPIRiT "
                "calibration path, so a missing sigpy means a broken environment "
                "rather than an optional extra."
            )
        if lamda < 0.0:
            raise ValueError(f"lamda must be >= 0, got {lamda}")
        if max_iter < 1:
            raise ValueError(f"max_iter must be >= 1, got {max_iter}")
        self.lamda = float(lamda)
        self.max_iter = int(max_iter)
        self.seed = int(seed)

    def reconstruct(
        self,
        masked_kspace: torch.Tensor,
        mask: torch.Tensor,
        sensitivity_maps: torch.Tensor | None = None,
        num_steps: int | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Reconstruct a batch, one slice at a time.

        sigpy solves per slice on CPU, so the batch is a loop rather than a
        vectorised call. Results are stacked back onto the input device and
        dtype so the caller cannot tell the difference.

        Args:
            masked_kspace: Undersampled k-space ``[B, C, H, W]`` complex, zero at
                the unsampled positions.
            mask: Sampling mask broadcastable to ``[B, 1, H, W]``, used as the
                data-consistency weighting.
            sensitivity_maps: Coil sensitivities ``[B, C, H, W]`` complex. ``None``
                is treated as single-coil with a unit map, which is what the
                SKM-TEA path supplies.
            num_steps: Ignored; iteration count is fixed by ``max_iter``.
            **kwargs: Ignored.

        Returns:
            Reconstructed image ``[B, 1, H, W]`` complex, on the input device.
        """
        del num_steps, kwargs

        device, dtype = masked_kspace.device, masked_kspace.dtype
        y_all = masked_kspace.detach().cpu().numpy().astype(np.complex64)
        b, c, h, w = y_all.shape

        # The mask arrives as [B, 1, H, W] or [1, 1, H, W]; sigpy wants one
        # real weight per k-space location, broadcast over coils.
        weights_all = mask.detach().cpu().numpy().real.astype(np.float32)
        weights_all = np.broadcast_to(
            weights_all.reshape(-1, h, w)[:, None],
            (weights_all.reshape(-1, h, w).shape[0], 1, h, w),
        )

        if sensitivity_maps is not None:
            mps_all = sensitivity_maps.detach().cpu().numpy().astype(np.complex64)
        else:
            mps_all = np.ones((b, c, h, w), dtype=np.complex64)

        out = np.empty((b, 1, h, w), dtype=np.complex64)
        rng_state = np.random.get_state()
        try:
            for i in range(b):
                weights = weights_all[i % weights_all.shape[0], 0]
                # Seeded per slice, not per batch: the power iteration's random
                # start would otherwise make the result depend on how the loader
                # happened to group slices.
                np.random.seed(self.seed + i)
                recon = TotalVariationRecon(
                    y_all[i],
                    mps_all[i],
                    self.lamda,
                    weights=weights,
                    max_iter=self.max_iter,
                    show_pbar=False,
                ).run()
                out[i, 0] = np.asarray(recon, dtype=np.complex64)
        finally:
            np.random.set_state(rng_state)

        return torch.from_numpy(out).to(device=device, dtype=dtype)
