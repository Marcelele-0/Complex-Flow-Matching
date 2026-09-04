"""Reconstruction-style evaluation of a trained flow-matching model, any geometry.

``generate.py`` samples from pure noise, so it has no ground truth and PSNR/SSIM
have nothing to compare against. This script scores a reconstruction instead: a
real slice becomes ``x_1``, is pushed part-way back toward noise through the
selected manifold's bridge at a configurable ``t_start``, then integrated forward
to ``t=1`` by the model and compared against the slice it came from.

The geometry enters only through :class:`~cfm.manifolds.base.Manifold`. The
cylindrical model and the Euclidean baseline are scored by this same code, with
the same slices, the same integration schedule, the same metrics and the same
accumulator, which is what makes the two columns of a comparison table
comparable. Scoring always happens in the complex domain, where both
representations agree on what a pixel means.

``t_start`` sets how much work the model is asked to do:

* ``0.0``: the bridge state is pure noise, so this is plain generation. Metrics
  are poor by construction; an unconditional sample has no reason to match the
  particular slice it was paired with.
* ``0.5``: half-noised, the model restores the rest.
* ``1.0``: the bridge returns the target, so metrics come out near-perfect. Tests
  the eval plumbing, not the model.

``split`` selects the volumes to score through
:func:`~cfm.data.splits.load_split_file_names`, the same function ``train.py``
gates its dataset with. Setting ``dataset.split=train`` and ``evaluate.split=test``
therefore holds the scored volumes out for real. Both default that way; set either
to ``null`` only for a directory with no ``annotations/``, and then no held-out
claim can be made about the numbers.

Everything above :func:`main` is pure (no Hydra, no HDF5, no filesystem except
the split helpers), so the whole path is unit-testable with a dummy model and
synthetic tensors.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, cast

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from cfm.core.solver import BaseODESolver, BaseSDESolver
from cfm.data import build_dataset, build_geometry_transform
from cfm.data.splits import load_split_file_names, select_indices
from cfm.manifolds import Manifold, build_manifold
from cfm.utils.config import as_plain_dict as _as_plain_dict
from cfm.utils.fft import fft2c, ifft2c
from cfm.utils.inference import (
    build_model,
    load_weights,
    reject_unsupported_sampling_model,
    resolve_checkpoint,
)
from cfm.utils.metrics import (
    circular_phase_error,
    data_consistency_error,
    peak_signal_noise_ratio,
    structural_similarity,
)

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# --------------------------------------------------------------------------- #
# Integration
# --------------------------------------------------------------------------- #
def dc_project(
    x_state: torch.Tensor,
    manifold: Manifold,
    y_measured_kspace: torch.Tensor,
    mask: torch.Tensor,
    sensitivity_maps: torch.Tensor | None = None,
) -> torch.Tensor:
    """Replace measured k-space lines in the current state.

    Args:
        x_state: Current state in the manifold's representation, ``[B, C, H, W]``.
        manifold: Supplies ``to_complex``/``from_complex`` for the geometry.
        y_measured_kspace: Measured k-space, centered. ``[B, 1, H, W]`` single-coil,
            or ``[B, num_coils, H, W]`` when ``sensitivity_maps`` is given.
        mask: Binary sampling mask, centered, broadcastable to the k-space shape.
        sensitivity_maps: Optional coil sensitivities ``[B, num_coils, H, W]``. When
            given, the state is projected onto the coils before the k-space
            replacement and SENSE-combined back, so the measured multi-coil data is
            enforced rather than a single-coil stand-in derived from the target.

    Returns:
        The projected state in the manifold's representation.
    """
    z = manifold.to_complex(x_state)
    if sensitivity_maps is not None:
        z_coils = fft2c(sensitivity_maps * z)
        z_dc_coils = z_coils * (1 - mask) + y_measured_kspace * mask
        z_dc = (sensitivity_maps.conj() * ifft2c(z_dc_coils)).sum(dim=1, keepdim=True)
    else:
        z_k = fft2c(z)
        z_dc_k = z_k * (1 - mask) + y_measured_kspace * mask
        z_dc = ifft2c(z_dc_k)
    return manifold.from_complex(z_dc)


@torch.no_grad()
def integrate_from_t(
    model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    solver: BaseODESolver | BaseSDESolver | Any,
    x_start: torch.Tensor,
    t_start: float,
    manifold: Manifold | None = None,
    y_measured_kspace: torch.Tensor | None = None,
    sampling_mask: torch.Tensor | None = None,
    use_dc_projection: bool = False,
    sensitivity_maps: torch.Tensor | None = None,
) -> torch.Tensor:
    """Integrate the flow from ``t_start`` to ``t=1``, in whatever geometry ``solver`` carries.

    Mirrors :meth:`~cfm.flow.solver.HeunODESolver.sample` (Heun
    predictor/corrector, Euler on the final step) but starts at an arbitrary
    ``t_start``, which ``sample`` cannot do because it hardcodes ``t=0``. State
    updates go through the public ``solver.step``, so a geometry's constraint -
    or its deliberate absence, in the Euclidean case - is never reimplemented here.

    Args:
        model: Callable ``(state [B, C, H, W], time [B]) -> velocity [B, 2, H, W]``.
        solver: Supplies ``num_steps`` and the geometry's ``step``.
        x_start: State at ``t_start``, shape ``[B, C, H, W]``.
        t_start: Absolute start time in ``[0, 1]``. At ``0`` this reproduces
            ``sample`` bit-for-bit; at ``1`` the interval is empty and the state
            returns unchanged up to whatever re-projection ``step`` applies.
        manifold: Optional geometry manifold for data consistency projections.
        y_measured_kspace: Optional measured k-space for data consistency.
        sampling_mask: Optional binary sampling mask.
        use_dc_projection: Whether to apply data consistency projections.
        sensitivity_maps: Optional coil sensitivities, enabling multi-coil DC.

    Returns:
        The state at ``t=1``, shape ``[B, C, H, W]``.

    Raises:
        ValueError: If ``t_start`` is outside ``[0, 1]`` or ``num_steps < 1``.
    """
    if not 0.0 <= t_start <= 1.0:
        raise ValueError(f"t_start must be in [0, 1], got {t_start}")
    if solver.num_steps < 1:
        raise ValueError(f"solver.num_steps must be >= 1, got {solver.num_steps}")

    num_steps = solver.num_steps
    device = x_start.device
    b = x_start.shape[0]

    span = 1.0 - t_start
    dt = span / num_steps
    x_t = x_start

    for i in range(num_steps):
        t_val = t_start + span * (i / num_steps)
        t_next_val = t_start + span * ((i + 1) / num_steps)

        t_tensor = torch.full((b,), t_val, device=device, dtype=torch.float32)
        t_next_tensor = torch.full((b,), t_next_val, device=device, dtype=torch.float32)

        # Predictor: current vector field.
        v_t = model(x_t, t_tensor)

        # Final step is Euler only, so the field is never evaluated past t=1.
        if i == num_steps - 1:
            x_t = solver.step(x_t, v_t, dt)
            if (
                use_dc_projection
                and manifold is not None
                and y_measured_kspace is not None
                and sampling_mask is not None
            ):
                x_t = dc_project(x_t, manifold, y_measured_kspace, sampling_mask, sensitivity_maps)
            return x_t

        # Euler probe, field at the probe, averaged velocity.
        x_pred = solver.step(x_t, v_t, dt)
        v_next = model(x_pred, t_next_tensor)
        v_avg = 0.5 * (v_t + v_next)

        # Corrected step, taken from x_t.
        x_t = solver.step(x_t, v_avg, dt)

        if (
            use_dc_projection
            and manifold is not None
            and y_measured_kspace is not None
            and sampling_mask is not None
        ):
            x_t = dc_project(x_t, manifold, y_measured_kspace, sampling_mask, sensitivity_maps)

    return x_t


def reconstruct_batch(
    model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    manifold: Manifold,
    solver: BaseODESolver | BaseSDESolver | Any,
    x_1: torch.Tensor,
    t_start: float,
    generator: torch.Generator | None = None,
    x_alias: torch.Tensor | None = None,
    sampling_mask: torch.Tensor | None = None,
    y_measured_kspace: torch.Tensor | None = None,
    use_dc_projection: bool = False,
    sensitivity_maps: torch.Tensor | None = None,
) -> torch.Tensor:
    """Noise the target to ``t_start`` via the bridge, then integrate back to ``t=1``.

    Args:
        model: The velocity field, ``(state, time) -> velocity``.
        manifold: Supplies the noise prior and the bridge.
        solver: ODE solver supplying ``num_steps`` and ``step``.
        x_1: Clean target in the manifold's representation, ``[B, C, H, W]``.
        t_start: Where on the noise-to-data path to start from.
        generator: Optional RNG for the noise draw. One seed makes a run
            repeatable, but it does not by itself pair the two geometries: the
            Euclidean prior defaults to ``uniform``, which draws the same *law* as
            the cylindrical prior but not the same sample. Under
            ``manifold.noise_prior=matched`` one seed does give both arms the same
            complex noise field, so they are scored on the same perturbation
            rather than merely the same slice.

    Returns:
        The reconstructed state at ``t=1``, shape ``[B, C, H, W]``.
    """
    b, _, h, w = x_1.shape
    x_0 = manifold.sample_noise(b, h, w, x_1.device, generator)

    t = torch.full((b, 1, 1, 1), t_start, device=x_1.device, dtype=torch.float32)
    # target_v is the training signal; only the state matters at eval time.
    if use_dc_projection and x_alias is not None:
        x_t, _ = manifold.bridge(x_0, x_alias, t)
    else:
        x_t, _ = manifold.bridge(x_0, x_1, t)

    return integrate_from_t(
        model,
        solver,
        x_t,
        t_start,
        manifold=manifold if use_dc_projection else None,
        y_measured_kspace=y_measured_kspace if use_dc_projection else None,
        sampling_mask=sampling_mask if use_dc_projection else None,
        use_dc_projection=use_dc_projection,
        sensitivity_maps=sensitivity_maps if use_dc_projection else None,
    )


def compute_batch_metrics(
    manifold: Manifold,
    pred_state: torch.Tensor,
    target_state: torch.Tensor,
    mask_threshold: float | None,
    sampling_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Score one batch in the complex domain, per sample.

    Both sides go through ``manifold.to_complex`` first. That is what makes the
    numbers comparable across geometries - a magnitude and a phase mean the same
    thing whatever coordinates produced them - and it is required anyway, since
    ``circular_phase_error``'s ``mask_threshold`` derives its mask from a complex
    target's amplitude.

    The manifold is a required argument rather than a defaulted one on purpose:
    scoring a Euclidean reconstruction through the cylindrical back-transform
    would silently produce plausible-looking, wrong numbers.

    The unmasked phase error is reported alongside the masked one: the masked
    value is the meaningful score (phase is noise in air), the unmasked value
    shows how much of the image the mask discards.

    Args:
        manifold: Supplies ``to_complex`` for the geometry both tensors are in.
        pred_state: Reconstruction, ``[B, C, H, W]`` in the manifold's representation.
        target_state: Ground truth, same shape and representation.
        mask_threshold: Fraction of the per-slice max amplitude below which pixels
            are excluded from the phase error. ``None`` scores every pixel, in
            which case masked and unmasked coincide.
        sampling_mask: Optional k-space undersampling mask in centered convention,
            broadcastable to ``[B, 1, H, W]``. When given, adds the data
            consistency error: relative k-space residual on the sampled lines.

    Returns:
        Mapping from metric name to a per-sample tensor of shape ``[B]``.
    """
    pred = manifold.to_complex(pred_state)
    target = manifold.to_complex(target_state)

    metrics = {
        # data_range=1.0 holds because every manifold's transform normalises the
        # modulus to [0, 1] (AmplitudeNormalize / EuclideanNormalize); a different
        # pipeline invalidates it.
        "psnr_db": peak_signal_noise_ratio(pred, target, data_range=1.0, reduction="none"),
        "ssim": structural_similarity(pred, target, data_range=1.0, reduction="none"),
        "phase_error_rad": circular_phase_error(
            pred, target, mask_threshold=mask_threshold, reduction="none"
        ),
    }
    if mask_threshold is not None:
        metrics["phase_error_rad_unmasked"] = circular_phase_error(pred, target, reduction="none")

    if sampling_mask is not None:
        metrics["data_consistency_error"] = data_consistency_error(
            pred, target, mask=sampling_mask, reduction="none"
        )
    return metrics


# --------------------------------------------------------------------------- #
# Config resolution
# --------------------------------------------------------------------------- #
def resolve_dc_mask(
    dataset_cfg: Mapping[str, Any] | Any,
    eval_cfg: Mapping[str, Any] | Any,
) -> dict[str, Any]:
    """Merge ``evaluate.mask`` over ``dataset.mask``, key by key.

    ``dataset.mask`` describes the trajectory the model was trained under, and
    evaluation should score it under the same one unless explicitly told otherwise.
    Merging per key rather than picking whichever block is non-empty is the whole
    point: choosing one wholesale meant a non-empty ``evaluate.mask`` discarded
    ``dataset.mask.type``, so a Poisson-disc model was silently scored under the
    Cartesian default that the resolver falls back to when no type is given.

    Args:
        dataset_cfg: The ``dataset`` config group.
        eval_cfg: The ``evaluate`` config group.

    Returns:
        A mask specification with ``acceleration`` always populated, defaulting to
        ``dataset.acceleration`` when neither mask block sets one.
    """
    dataset_plain = _as_plain_dict(dataset_cfg)
    merged: dict[str, Any] = {
        **_as_plain_dict(dataset_plain.get("mask")),
        **_as_plain_dict(_as_plain_dict(eval_cfg).get("mask")),
    }
    merged["acceleration"] = int(merged.get("acceleration", dataset_plain.get("acceleration", 4)))
    return merged


def resolve_split(
    dataset_cfg: Mapping[str, Any] | Any,
    eval_cfg: Mapping[str, Any] | Any,
) -> str | None:
    """Decide which split manifest to score against.

    A dataset group that sets ``split: null`` is declaring that it ships no
    manifests at all - fastMRI has no ``annotations/`` directory - so there is
    nothing for ``evaluate.split`` to select and asking for ``'test'`` could only
    raise. Any other value and ``evaluate.split`` governs, keeping the held-out
    gate under its own key.

    Args:
        dataset_cfg: The ``dataset`` config group.
        eval_cfg: The ``evaluate`` config group.

    Returns:
        The split name, or ``None`` to score every file in ``data_dir``.
    """
    dataset_plain = _as_plain_dict(dataset_cfg)
    if "split" in dataset_plain and dataset_plain["split"] is None:
        return None
    split = _as_plain_dict(eval_cfg).get("split", "test")
    return None if split is None else str(split)


# --------------------------------------------------------------------------- #
# Accumulation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MetricSummary:
    """Outcome of one metric over a whole evaluation run."""

    name: str
    total: int
    scored: int
    nan: int
    pos_inf: int
    neg_inf: int
    mean: float
    std: float
    minimum: float
    maximum: float
    # Kept apart because they mean opposite things: perfect_ids scored as well as
    # possible, unscored_ids could not be scored at all. Both sit outside the mean.
    perfect_ids: tuple[str, ...]
    unscored_ids: tuple[str, ...]


class MetricAccumulator:
    """Collects per-sample metric values across batches.

    :mod:`cfm.utils.metrics` returns ``NaN`` for a sample it could not score
    (empty amplitude mask, i.e. an all-air slice) and ``+inf`` for an exact match.
    Both are information, not noise, and a naive ``cat(...).mean()`` would let one
    such sample turn the entire run into ``NaN``/``inf``.

    Values are therefore partitioned, never silently filtered: the mean covers the
    finite samples, and the non-finite ones are counted and named. Dropping them
    quietly would bias the number; propagating them would destroy it.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._chunks: list[torch.Tensor] = []
        self._ids: list[str] = []

    def update(self, values: torch.Tensor, sample_ids: Sequence[str] | None = None) -> None:
        """Add one batch of per-sample values.

        Args:
            values: Per-sample metric values, shape ``[B]``.
            sample_ids: Optional identifiers, one per value, used to name unscored
                samples in the summary.

        Raises:
            ValueError: If ``values`` is not 1-D, or ``sample_ids`` length differs.
        """
        if values.dim() != 1:
            raise ValueError(f"{self.name}: expected per-sample [B], got {tuple(values.shape)}")
        if sample_ids is not None and len(sample_ids) != values.numel():
            raise ValueError(
                f"{self.name}: got {values.numel()} values but {len(sample_ids)} sample_ids"
            )

        # float64 keeps a long run's mean from drifting; inf/nan survive the cast.
        self._chunks.append(values.detach().to(torch.float64).cpu())
        if sample_ids is None:
            start = len(self._ids)
            self._ids.extend(f"#{start + i}" for i in range(values.numel()))
        else:
            self._ids.extend(sample_ids)

    def values(self) -> torch.Tensor:
        """All accumulated per-sample values as a single 1-D tensor."""
        if not self._chunks:
            return torch.empty(0, dtype=torch.float64)
        return torch.cat(self._chunks)

    def summary(self, max_reported_ids: int = 5) -> MetricSummary:
        """Partition the accumulated values and reduce the finite ones.

        Args:
            max_reported_ids: How many unscored sample ids to keep for display.

        Returns:
            A :class:`MetricSummary`. With nothing scored the statistics are
            ``NaN``: an unscored run is not a perfect one.
        """
        v = self.values()
        finite = torch.isfinite(v)
        scored = v[finite]

        # +inf is an exact match, so it is a perfect score rather than a failure;
        # it stays out of the mean but is reported separately from NaN. -inf is
        # pathological and groups with the failures.
        perfect_mask = torch.isposinf(v)
        unscored_mask = torch.isnan(v) | torch.isneginf(v)
        perfect = [self._ids[i] for i in perfect_mask.nonzero().flatten().tolist()]
        bad = [self._ids[i] for i in unscored_mask.nonzero().flatten().tolist()]

        if scored.numel() == 0:
            mean = std = lo = hi = float("nan")
        else:
            mean = float(scored.mean())
            std = float(scored.std(correction=0))
            lo = float(scored.min())
            hi = float(scored.max())

        return MetricSummary(
            name=self.name,
            total=int(v.numel()),
            scored=int(scored.numel()),
            nan=int(torch.isnan(v).sum()),
            pos_inf=int(torch.isposinf(v).sum()),
            neg_inf=int(torch.isneginf(v).sum()),
            mean=mean,
            std=std,
            minimum=lo,
            maximum=hi,
            perfect_ids=tuple(perfect[:max_reported_ids]),
            unscored_ids=tuple(bad[:max_reported_ids]),
        )


def format_summary_table(summaries: Sequence[MetricSummary]) -> str:
    """Render metric summaries as a fixed-width table.

    Args:
        summaries: One summary per metric, in display order.

    Returns:
        The table as a multi-line string, with a legend whenever a sample went
        unscored.
    """
    header = (
        f"{'metric':<26}{'mean':>10}{'std':>10}{'min':>10}{'max':>10}"
        f"{'scored':>12}{'nan':>7}{'+inf':>7}"
    )
    rule = "-" * len(header)
    lines = [header, rule]

    any_unscored = False
    for s in summaries:
        if s.scored == 0:
            cells = f"{'n/a':>10}{'n/a':>10}{'n/a':>10}{'n/a':>10}"
        else:
            cells = f"{s.mean:>10.4f}{s.std:>10.4f}{s.minimum:>10.4f}{s.maximum:>10.4f}"
        scored = f"{s.scored}/{s.total}"
        lines.append(f"{s.name:<26}{cells}{scored:>12}{s.nan:>7}{s.pos_inf:>7}")
        if s.nan or s.pos_inf or s.neg_inf:
            any_unscored = True

    lines.append(rule)

    if any_unscored:
        lines.append("mean/std/min/max are over the 'scored' (finite) samples only.")
        lines.append("  nan  = could not be scored (empty amplitude mask -> all-air slice)")
        lines.append("  +inf = exact match; a perfect score excluded from the mean, not a failure")
        for s in summaries:
            if s.perfect_ids:
                lines.append(f"perfect (excluded from mean) {s.name}: {', '.join(s.perfect_ids)}")
            if s.unscored_ids:
                lines.append(f"unscored {s.name}: {', '.join(s.unscored_ids)}")

    for s in summaries:
        if s.scored == 0:
            lines.append(f"WARNING: {s.name} could not be scored for any sample.")

    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Split selection
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Hydra entry point
# --------------------------------------------------------------------------- #
@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Score a checkpoint by reconstructing held-out slices from a noised state."""
    eval_cfg = cfg.get("evaluate", {})

    # 1. Validate before any slow work: this script otherwise dies 20 minutes in.
    t_start = float(eval_cfg.get("t_start", 0.5))
    num_steps = int(eval_cfg.get("num_steps", 100))
    mask_threshold = eval_cfg.get("mask_threshold", 0.05)
    batch_size = int(eval_cfg.get("batch_size", 4))
    max_samples = eval_cfg.get("max_samples")
    num_workers = int(eval_cfg.get("num_workers", 4))
    seed = int(eval_cfg.get("seed", 0))
    dataset_cfg = cfg.get("dataset", {})

    split = resolve_split(dataset_cfg, eval_cfg)
    dc_mask_dict = resolve_dc_mask(dataset_cfg, eval_cfg)
    dc_acceleration = int(dc_mask_dict["acceleration"])
    dc_center_fraction = dc_mask_dict.get("center_fraction")
    use_dc_projection = eval_cfg.get("use_dc_projection", False)

    if dc_acceleration < 1:
        raise ValueError(f"evaluate.mask.acceleration must be >= 1, got {dc_acceleration}")
    if dc_center_fraction is not None:
        dc_center_fraction = float(dc_center_fraction)
        if not 0.0 < dc_center_fraction <= 1.0:
            raise ValueError(
                f"evaluate.mask.center_fraction must be in (0, 1], got {dc_center_fraction}"
            )
        dc_mask_dict["center_fraction"] = dc_center_fraction

    if not 0.0 <= t_start <= 1.0:
        raise ValueError(f"evaluate.t_start must be in [0, 1], got {t_start}")
    if num_steps < 1:
        raise ValueError(f"evaluate.num_steps must be >= 1, got {num_steps}")
    if mask_threshold is not None:
        mask_threshold = float(mask_threshold)
        if not 0.0 <= mask_threshold < 1.0:
            raise ValueError(f"evaluate.mask_threshold must be in [0, 1), got {mask_threshold}")
    if batch_size < 1:
        raise ValueError(f"evaluate.batch_size must be >= 1, got {batch_size}")
    if max_samples is not None:
        max_samples = int(max_samples)
        if max_samples < 1:
            raise ValueError(f"evaluate.max_samples must be >= 1, got {max_samples}")

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    precision = cfg.get("training", {}).get("matmul_precision", "highest")
    torch.set_float32_matmul_precision(precision)
    print(f"Starting evaluation on: {device}")

    # 2. Geometry. Must match the one the checkpoint was trained under; a mismatch
    # surfaces immediately as a state-dict shape error on init_conv rather than as
    # quietly wrong metrics.
    manifold = build_manifold(cfg).to(device)

    # 3. Model and checkpoint. Not compiled: one-shot eval pays the compile cost
    # for nothing. load_weights also puts the model in eval mode.
    reject_unsupported_sampling_model(cfg)

    orig_cwd = hydra.utils.get_original_cwd()
    run_name = eval_cfg.get("run_name") or cfg.get("logging", {}).get("experiment_name")
    model = build_model(
        cfg,
        device,
        in_channels=manifold.state_channels,
        out_channels=manifold.velocity_channels,
    )
    checkpoint_path = resolve_checkpoint(cfg, "evaluate", orig_cwd)
    load_weights(model, checkpoint_path, device)

    # 4. Data. Reconstruction mode normalises the target to a peak modulus of 1
    # itself, which is what keeps data_range=1.0 valid; the pipeline handed to the
    # dataset is therefore shape-only, and doubles as the crop the sensitivity maps
    # are matched to for multi-coil cohorts.
    data_dir = eval_cfg.get("data_dir") or dataset_cfg.get(
        "data_dir", "data/skm-tea-mini/v1-release"
    )
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(orig_cwd, data_dir)

    geometry = build_geometry_transform(dataset_cfg.get("crop_size"), crop_base=16)

    # The dataset opens every .h5 under data_dir to count slices before split
    # filtering is possible, so a corrupt file aborts the run even when it is not
    # in the chosen split. Intentional: swallowing a corrupt-data error here is how
    # you end up reporting metrics on half a dataset.
    dataset = build_dataset(
        dataset_cfg,
        data_dir=data_dir,
        mode="reconstruction",
        acceleration=dc_acceleration,
        mask=dc_mask_dict,
        pre_transform=geometry,
    )
    file_names = load_split_file_names(data_dir, split)
    indices = select_indices(dataset.slice_map, file_names, max_samples)
    sample_ids = [
        f"{os.path.basename(path)}[{slice_idx}]"
        for path, slice_idx in (dataset.slice_map[i] for i in indices)
    ]

    num_files = len({os.path.basename(dataset.slice_map[i][0]) for i in indices})
    print(
        f"Split {split!r}: {num_files} file(s), "
        f"{len(indices)} of {len(dataset.slice_map)} slices selected"
    )

    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    # 5. Optional W&B, same import guard as train.py.
    use_wandb = cfg.get("logging", {}).get("use_wandb", False)
    output_dir = cfg.get("paths", {}).get("output_dir", ".")
    os.makedirs(output_dir, exist_ok=True)

    if use_wandb and HAS_WANDB:
        print("Weights & Biases logging enabled.")
        config_dict = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
        wandb.init(
            project=cfg.get("logging", {}).get("project_name", "Cylindrical-Flow-Matching"),
            name=f"eval_{manifold.name}_{run_name}_t{t_start}",
            dir=output_dir,
            config=config_dict,
        )
    else:
        use_wandb = False
        print("Local logging only.")

    # 6. Evaluation loop.
    solver = manifold.make_solver(num_steps)
    accumulators: dict[str, MetricAccumulator] = {}

    span = 1.0 - t_start
    print(
        f"Reconstructing from t_start={t_start} (span {span:.3f} over {num_steps} steps "
        f"-> dt {span / num_steps:.6f})"
    )

    # shuffle=False and drop_last=False keep this position aligned with sample_ids.
    position = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            target_complex = batch["target"].to(device)
            x_1 = manifold.from_complex(target_complex)
            sampling_mask = batch["mask"].to(device)
            batch_ids = sample_ids[position : position + x_1.shape[0]]
            position += x_1.shape[0]

            if use_dc_projection and t_start < 1.0:
                x_alias = batch["input"].to(device)

                # Multi-coil cohorts carry the actual undersampled measurement and
                # the maps that relate it to the image, so DC enforces the real
                # k-space. Single-coil cohorts have no acquisition mask to load, so
                # the measurement is simulated from the target, as documented in
                # conf/evaluate/default.yaml.
                if "masked_kspace" in batch and "sensitivity_maps" in batch:
                    y_kspace = batch["masked_kspace"].to(device)
                    sens_maps = batch["sensitivity_maps"].to(device)
                else:
                    y_kspace = fft2c(target_complex) * sampling_mask
                    sens_maps = None

                # The aliased input needs to be in the manifold representation
                x_alias_manifold = manifold.from_complex(x_alias)

                pred = reconstruct_batch(
                    model,
                    manifold,
                    solver,
                    x_1,
                    t_start,
                    generator,
                    x_alias=x_alias_manifold,
                    sampling_mask=sampling_mask,
                    y_measured_kspace=y_kspace,
                    use_dc_projection=True,
                    sensitivity_maps=sens_maps,
                )
            else:
                pred = reconstruct_batch(model, manifold, solver, x_1, t_start, generator)

            batch_metrics = compute_batch_metrics(
                manifold, pred, x_1, mask_threshold, sampling_mask
            )

            for name, values in batch_metrics.items():
                if name not in accumulators:
                    accumulators[name] = MetricAccumulator(name)
                accumulators[name].update(values, batch_ids)

    # 7. Report.
    summaries = [acc.summary() for acc in accumulators.values()]

    print()
    print("=" * 88)
    print("Reconstruction evaluation")
    print(f"  manifold   : {manifold.name}")
    print(f"  checkpoint : {checkpoint_path}")
    print(f"  split      : {split}   ({num_files} file(s), {len(indices)} slices)")
    print(f"  t_start    : {t_start:.3f}   num_steps: {num_steps}")
    print(f"  mask thr.  : {mask_threshold}")
    if dc_center_fraction is not None:
        print(
            f"  dc mask    : R={dc_acceleration}, center_fraction={dc_center_fraction} (simulated)"
        )
    else:
        print(f"  dc mask    : R={dc_acceleration} (simulated)")
    print(f"  batch/dev  : {batch_size} on {device}   seed: {seed}")
    print("=" * 88)
    print(format_summary_table(summaries))
    print("=" * 88)

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                # Recorded so a metrics.json can never be mistaken for the other
                # arm of the comparison once it is out of its output directory.
                "manifold": manifold.name,
                "checkpoint": checkpoint_path,
                "split": split,
                "num_slices": len(indices),
                "t_start": t_start,
                "num_steps": num_steps,
                "mask_threshold": mask_threshold,
                "seed": seed,
                "metrics": [asdict(s) for s in summaries],
            },
            fh,
            indent=2,
        )
    print(f"Wrote {metrics_path}")

    if use_wandb:
        log_dict: dict[str, float | int | str] = {
            "eval/manifold": manifold.name,
            "eval/t_start": t_start,
            "eval/num_steps": num_steps,
            "eval/num_samples": len(indices),
            "eval/split": str(split),
        }
        for s in summaries:
            # Partitioned statistics only, never the raw mean: one NaN/inf sample
            # would have poisoned it.
            log_dict[f"eval/{s.name}_mean"] = s.mean
            log_dict[f"eval/{s.name}_std"] = s.std
            log_dict[f"eval/{s.name}_min"] = s.minimum
            log_dict[f"eval/{s.name}_max"] = s.maximum
            log_dict[f"eval/{s.name}_scored"] = s.scored
            log_dict[f"eval/{s.name}_nan"] = s.nan
            log_dict[f"eval/{s.name}_inf"] = s.pos_inf
        wandb.log(log_dict)
        wandb.finish()


if __name__ == "__main__":
    main()
