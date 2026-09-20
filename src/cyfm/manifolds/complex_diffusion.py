"""Complex Diffusion Manifold using Variance-Exploding SDE (VE-SDE).

Models complex MRI in Cartesian R^2 with a continuous VE-SDE forward process,
denoising score matching with likelihood weighting, and Predictor-Corrector sampling.

The framework is Song et al. (2021), score-based generative modelling through
SDEs; Chung & Ye (2022) is the MRI application that motivated it. This arm is
*not* their prior: Score-MRI trains on magnitude images and the complex part
enters at inference through data consistency against the measurement. Here the
process runs on complex pixels as ``(Re z, Im z)`` and there is no measurement at
any point, so it is a VE-SDE prior over complex fields implemented in this
repository, and should be described as that.

The representation is inherited from
:class:`~cyfm.manifolds.flat.FlatComplexRepresentation`, the same one the Cartesian
flow arm uses. That is deliberate and load-bearing: Table 5 compares this arm
against that one, so a transform of its own would let the two preprocess
differently and move every absolute number without failing a test.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import torch

from cyfm.core.manifold import BaseManifold
from cyfm.core.registry import MANIFOLDS
from cyfm.flow.solvers import (
    PredictorCorrectorSolver,
    heun_evaluations,
    plan_within_budget,
)
from cyfm.manifolds.flat import FlatComplexRepresentation

# How the solver spends the step count the evaluation sweep hands it.
#   "heun"   match the Heun budget of the flow arms, 2n-1 model calls, which is
#            the like-for-like column of the table
#   "native" take the step count literally, the many-step regime this sampler is
#            actually used in
NFE_MODES = ("heun", "native")

# Measured with calibrate_sigma_max on peak-normalised 320x320 knee fields, the
# resolution Table 5 runs at. It is a distance between whole fields, so it scales
# roughly with sqrt(H * W): the same data at 64x64 measures about 33, and a value
# carried across resolutions is wrong in one direction or the other. The recovered
# default was 378.0, which belongs to a dataset with a different scale and would
# leave the prior far wider than the data it has to forget.
DEFAULT_SIGMA_MAX_320 = 140.0


def _match_shape(val: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Broadcast scalar or 1D tensor to match target tensor dimensions."""
    val = val.to(target.device)
    if val.dim() == target.dim():
        return val
    if val.dim() == 0:
        return val
    if val.dim() == 1 and val.shape[0] == target.shape[0]:
        return val.view(val.shape[0], *([1] * (target.dim() - 1)))
    return val


@MANIFOLDS.register("complex_diffusion")
class ComplexDiffusionManifold(FlatComplexRepresentation, BaseManifold):
    """Complex Diffusion on R^2 via Variance-Exploding SDE (VE-SDE).

    Forward process: x(t) = x(0) + sigma(t) * z, where z ~ N(0, I).
    Noise schedule: sigma(t) = sigma_min * (sigma_max / sigma_min)^t.
    Loss: Denoising score matching with likelihood weighting g(t)^2.

    Args:
        sigma_min: Smallest noise scale at t=0. Default 0.01.
        sigma_max: Largest noise scale at t=1. This is a distance between whole
            fields, not between two pixels, so it grows with the field size and
            the default belongs to one resolution only. Measure it with
            :func:`calibrate_sigma_max`.
        eps: Smallest positive time threshold. Default 1e-5.
        likelihood_weighting: Whether to use likelihood weighting g(t)^2. Default True.
        snr: Signal-to-noise ratio parameter r for Langevin Dynamics. Default 0.16.
        corrector_steps: Number of Langevin corrector steps M. Default 1.
        num_steps: Default number of integration steps. Default 50.
        nfe_mode: One of :data:`NFE_MODES`. How the evaluation sweep's step count
            is read -- as the flow arms' cost, or literally.
        **kwargs: Unrecognized arguments; triggers fail-fast error.

    Raises:
        ValueError: If parameters are invalid or unexpected kwargs passed.
    """

    name = "complex_diffusion"
    state_channels = 2
    velocity_channels = 2
    # The network regresses a score. Straightness and the induced angular velocity
    # are properties of a velocity field and are not reported for it.
    predicts_velocity = False

    def __init__(
        self,
        sigma_min: float = 0.01,
        sigma_max: float = DEFAULT_SIGMA_MAX_320,
        eps: float = 1e-5,
        likelihood_weighting: bool = True,
        snr: float = 0.16,
        corrector_steps: int = 1,
        num_steps: int = 50,
        nfe_mode: str = "heun",
        **kwargs: Any,
    ) -> None:
        if kwargs:
            raise ValueError(f"Unexpected keyword arguments for ComplexDiffusionManifold: {kwargs}")
        if nfe_mode not in NFE_MODES:
            raise ValueError(f"nfe_mode must be one of {sorted(NFE_MODES)}, got {nfe_mode!r}")
        if sigma_min <= 0:
            raise ValueError(f"sigma_min must be positive, got {sigma_min}")
        if sigma_max <= sigma_min:
            raise ValueError(
                f"sigma_max must be greater than sigma_min, got {sigma_max} <= {sigma_min}"
            )
        if eps <= 0 or eps >= 1.0:
            raise ValueError(f"eps must be in (0, 1), got {eps}")
        if snr < 0:
            raise ValueError(f"snr must be non-negative, got {snr}")
        if corrector_steps < 0:
            raise ValueError(f"corrector_steps must be non-negative, got {corrector_steps}")
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")

        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.eps = float(eps)
        self.likelihood_weighting = bool(likelihood_weighting)
        self.snr = float(snr)
        self.corrector_steps = int(corrector_steps)
        self.num_steps = int(num_steps)
        self.nfe_mode = nfe_mode
        self.log_ratio = math.log(self.sigma_max / self.sigma_min)

        self._last_t: torch.Tensor | None = None
        self._last_z: torch.Tensor | None = None

    def to(self, device: torch.device) -> ComplexDiffusionManifold:
        """Move internal buffers to device (chainable)."""
        del device
        return self

    def sigma(self, t: float | torch.Tensor) -> torch.Tensor:
        """Compute noise standard deviation sigma(t) = sigma_min * (sigma_max / sigma_min)^t.

        Args:
            t: Time scalar or tensor in [0, 1].

        Returns:
            Noise scale sigma(t).
        """
        if isinstance(t, int | float):
            val = self.sigma_min * (self.sigma_max / self.sigma_min) ** float(t)
            return torch.tensor(val, dtype=torch.float32)
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    def diffusion(self, t: float | torch.Tensor) -> torch.Tensor:
        """Compute diffusion coefficient g(t) = sigma(t) * sqrt(2 * log(sigma_max / sigma_min)).

        Args:
            t: Time scalar or tensor.

        Returns:
            Diffusion coefficient g(t).
        """
        sig = self.sigma(t)
        return sig * math.sqrt(2.0 * self.log_ratio)

    def marginal_prob(
        self, x_0: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute mean and standard deviation of transition kernel p_t(x_t | x_0).

        Args:
            x_0: Clean state [B, C, H, W].
            t: Time tensor [B] or [B, 1, 1, 1].

        Returns:
            Tuple of (mean, std).
        """
        mean = x_0
        std = _match_shape(self.sigma(t), x_0)
        return mean, std

    def forward_process(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        z: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward VE-SDE perturbation: x(t) = x(0) + sigma(t) * z.

        Args:
            x_0: Clean state [B, 2, H, W].
            t: Time tensor [B] or [B, 1, 1, 1] in [0, 1].
            z: Optional standard Gaussian noise [B, 2, H, W].
            generator: Optional RNG generator.

        Returns:
            Tuple of (x_t, z).
        """
        if z is None:
            z = torch.randn(x_0.shape, device=x_0.device, dtype=x_0.dtype, generator=generator)
        std = _match_shape(self.sigma(t), x_0)
        x_t = x_0 + std * z
        return x_t, z

    # Aliases for convenience
    forward_sde = forward_process
    perturb = forward_process

    def target_score(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Compute target conditional score grad_{x_t} log p_t(x_t | x_0) = -z / sigma(t).

        Args:
            z: Standard normal noise tensor [B, 2, H, W].
            t: Time tensor [B] or [B, 1, 1, 1].

        Returns:
            Target score tensor [B, 2, H, W].
        """
        std = _match_shape(self.sigma(t), z)
        return -z / std

    def geodesic_path(self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Interpolate state along VE-SDE forward path."""
        x_t, _ = self.bridge(x_0, x_1, t)
        return x_t

    def target_velocity(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Compute target velocity / score field."""
        if t is None:
            t = torch.ones(x_0.shape[0], 1, 1, 1, device=x_0.device, dtype=x_0.dtype)
        _, u_t = self.bridge(x_0, x_1, t)
        return u_t

    def sample_noise(
        self,
        batch: int,
        height: int,
        width: int,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw sample from prior distribution at t=1: N(0, sigma_max^2 * I).

        Args:
            batch: Batch size.
            height: Image height.
            width: Image width.
            device: Allocation device.
            generator: Optional RNG generator.

        Returns:
            Noise tensor [B, 2, H, W].
        """
        return (
            torch.randn(
                batch,
                self.state_channels,
                height,
                width,
                device=device,
                generator=generator,
            )
            * self.sigma_max
        )

    def bridge(
        self, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute perturbed state x_t and target score from prior noise x_0 and data x_1.

        Args:
            x_0: Prior noise state [B, 2, H, W].
            x_1: Clean data state [B, 2, H, W].
            t: Time tensor [B, 1, 1, 1] in [0, 1].

        Returns:
            Tuple of (x_t, target_score).
        """
        # CFM convention: x_0 is noise scaled by sigma_max.
        z = x_0 / self.sigma_max

        # CFM convention: t=0 is noise, t=1 is data.
        # VE-SDE convention: t=0 is data, t=1 is noise.
        t_internal = 1.0 - t

        std = _match_shape(self.sigma(t_internal), x_1)
        x_t = x_1 + std * z
        u_t = -z / std

        self._last_t = t_internal
        self._last_z = z
        return x_t, u_t

    def loss(
        self,
        pred_v: torch.Tensor,
        target_v: torch.Tensor,
        target_x1: torch.Tensor | None = None,
        t: torch.Tensor | None = None,
        likelihood_weighting: bool | None = None,
        **kwargs: Any,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute Denoising Score Matching loss with likelihood weighting.

        Loss = E [ g(t)^2 * || pred_score - target_score ||^2 ].

        Args:
            pred_v: Predicted score field [B, 2, H, W].
            target_v: Target score field [B, 2, H, W].
            target_x1: Optional clean data state [B, 2, H, W].
            t: Optional time tensor [B] or [B, 1, 1, 1].
            likelihood_weighting: Override for likelihood weighting flag.

        Returns:
            Tuple of (total_loss, {"dsm": total_loss}).
        """
        del target_x1
        time_t = (1.0 - t) if t is not None else self._last_t
        use_lw = (
            self.likelihood_weighting
            if likelihood_weighting is None
            else bool(likelihood_weighting)
        )

        diff = pred_v - target_v

        if use_lw and time_t is not None:
            time_tensor = _match_shape(time_t, pred_v)
            std = _match_shape(self.sigma(time_tensor), pred_v)
            g2 = 2.0 * self.log_ratio * (std**2)
            err = g2 * (diff**2)
        else:
            # Unweighted: the raw residual, which is what the flag name says. The
            # recovered version applied sigma^2 here and still called it
            # unweighted; this branch is an ablation and should be the thing it
            # is named after. The large-t end then dominates -- sigma is smallest
            # there, so the target score -z/sigma is largest -- which is the
            # behaviour the ablation exists to show.
            err = diff**2

        total_loss = torch.mean(err)
        return total_loss, {"dsm": total_loss}

    def score_matching_loss(
        self,
        model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        x_0: torch.Tensor,
        t: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        eps: float | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Convenience end-to-end forward perturbation and DSM loss computation.

        Args:
            model: Neural score network (x, t) -> score.
            x_0: Clean data state [B, 2, H, W].
            t: Optional time tensor [B].
            z: Optional noise tensor [B, 2, H, W].
            eps: Smallest time step. Defaults to self.eps.
            generator: Optional RNG generator.

        Returns:
            Tuple of (total_loss, {"dsm": total_loss}).
        """
        b = x_0.shape[0]
        min_eps = self.eps if eps is None else float(eps)
        if t is None:
            t = (
                torch.rand(b, device=x_0.device, dtype=x_0.dtype, generator=generator)
                * (1.0 - min_eps)
                + min_eps
            )
        t_model = t if t.dim() == 1 else t.view(b)
        x_t, noise_z = self.forward_process(x_0, t, z=z, generator=generator)
        pred_score = model(x_t, t_model)
        target = self.target_score(noise_z, t)
        return self.loss(pred_score, target, target_x1=x_0, t=t)

    def wrap_model(
        self, model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Read the network's output as a unit-scale residual and divide by sigma here.

        The score of the perturbation kernel is ``-z / sigma``, so its magnitude runs
        over the whole schedule: with ``sigma_min = 0.01`` and a calibrated
        ``sigma_max`` of order ten, a network asked to emit it directly has to cover
        four orders of magnitude from one time to another. It does not. Measured on a
        ``16x16`` synthetic target with everything else held fixed, moving the
        ``1 / sigma`` outside the network takes the sliced $W_2$ from ``22.0`` to
        ``0.54`` at 20 steps and from ``11.1`` to ``0.23`` at 100, against a data
        scale of ``0.74``; the raw-output arm diverges at every budget. This is the
        standard preconditioning of score-based models
        \citep{song2021score, karras2022elucidating}, not a correction specific to
        this repository.

        The factor is ``sigma(1 - t)`` because that is the scale the bridge gives the
        state at time ``t``: :meth:`bridge` evaluates the schedule at ``1 - t`` so that
        ``t = 0`` is noise, as the flow arms' convention requires. The solver passes
        the same argument, so one wrapper serves training and sampling.

        Args:
            model: Callable mapping ``(state, time)`` to a unit-scale residual.

        Returns:
            A callable emitting the score.
        """

        def score(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            raw = model(x, t)
            std = _match_shape(self.sigma(1.0 - t), raw)
            return raw / std

        return score

    def solver_plan(self, num_steps: int) -> tuple[int, int]:
        """Steps and corrections this arm runs when the sweep asks ``num_steps``.

        In ``"heun"`` mode the request is read as the flow arms' *cost* and
        converted, so both sides of the table spend the same number of model
        calls; in ``"native"`` mode it is taken literally. Corrections are part of
        the answer because at the tightest budget they are what has to give.

        Args:
            num_steps: Step count from the evaluation sweep.

        Returns:
            Tuple of steps and corrections per step.
        """
        if self.nfe_mode == "native":
            return num_steps, self.corrector_steps
        return plan_within_budget(heun_evaluations(num_steps), self.corrector_steps)

    def make_solver(self, num_steps: int) -> PredictorCorrectorSolver:
        """Construct the Predictor-Corrector solver for this manifold."""
        steps, correctors = self.solver_plan(num_steps)
        return PredictorCorrectorSolver(
            num_steps=steps,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            snr=self.snr,
            m_steps=correctors,
            eps=self.eps,
            manifold=self,
        )


def calibrate_sigma_max(fields: torch.Tensor) -> float:
    """Largest pairwise distance in a batch, the principled ``sigma_max``.

    The variance-exploding prior has to be wide enough that it forgets the data,
    and the standard rule is the maximum Euclidean distance between two training
    fields. It is reported here rather than inherited because the usual published
    value belongs to a dataset with a different scale, and this pipeline
    peak-normalises every field.

    Args:
        fields: Real states ``[B, 2, H, W]`` from the training transform.

    Returns:
        The maximum pairwise distance.

    Raises:
        ValueError: If fewer than two fields are given.
    """
    if fields.ndim != 4 or fields.shape[0] < 2:
        raise ValueError(
            f"need at least two [B, 2, H, W] fields to measure a distance, "
            f"got shape {tuple(fields.shape)}"
        )
    flat = fields.reshape(fields.shape[0], -1)
    return float(torch.cdist(flat, flat).max())
