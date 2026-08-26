"""Reconstruction-style evaluation of a trained cylindrical flow-matching model.

``generate.py`` samples from pure noise, so it has no ground truth and PSNR/SSIM
have nothing to compare against. This script scores a reconstruction instead: a
real slice becomes ``x_1``, is pushed part-way back toward noise through
:class:`~cfm.flow.bridge.GeodesicFlowBridge` at a configurable ``t_start``, then
integrated forward to ``t=1`` by the model and compared against the slice it came
from.

``t_start`` sets how much work the model is asked to do:

* ``0.0``: the bridge state is pure noise, so this is plain generation. Metrics
  are poor by construction; an unconditional sample has no reason to match the
  particular slice it was paired with.
* ``0.5``: half-noised, the model restores the rest.
* ``1.0``: the bridge returns the target, so metrics come out near-perfect. Tests
  the eval plumbing, not the model.

.. warning::
   ``split`` chooses which files are *scored*; it does not hold them out.
   ``train.py`` hands ``data_dir`` straight to
   :class:`~cfm.data.dataset.SKMTEADataset`, which globs every ``.h5`` with no
   split filter, so the volumes scored here were almost certainly in the training
   set. Read these numbers as reconstruction fidelity, not generalization, until
   the same manifests gate training.

Everything above :func:`main` is pure (no Hydra, no HDF5, no filesystem except
:func:`load_split_file_names`), so the whole path is unit-testable with a dummy
model and synthetic tensors.
"""

from __future__ import annotations

import glob
import json
import math
import os
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from cfm.data.dataset import SKMTEADataset
from cfm.data.transforms import (
    AmplitudeNormalize,
    CenterCropModulo,
    ComplexToCylinderTransform,
    Compose,
)
from cfm.flow.bridge import GeodesicFlowBridge
from cfm.flow.solver import CylindricalODESolver
from cfm.utils.complex_ops import cylinder_to_complex
from cfm.utils.inference import build_model, load_weights, resolve_checkpoint
from cfm.utils.metrics import (
    circular_phase_error,
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
@torch.no_grad()
def integrate_from_t(
    model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    solver: CylindricalODESolver,
    x_start: torch.Tensor,
    t_start: float,
) -> torch.Tensor:
    """Integrate the cylindrical flow from ``t_start`` to ``t=1``.

    Mirrors :meth:`~cfm.flow.solver.CylindricalODESolver.sample` (Heun
    predictor/corrector, Euler on the final step) but starts at an arbitrary
    ``t_start``, which ``sample`` cannot do because it hardcodes ``t=0``. State
    updates go through the public ``solver.step``, so the manifold projection is
    never reimplemented here.

    Args:
        model: Callable ``(state [B, 3, H, W], time [B]) -> velocity [B, 2, H, W]``.
        solver: Supplies ``num_steps`` and the manifold-safe ``step``.
        x_start: State on the cylinder at ``t_start``, shape ``[B, 3, H, W]``.
        t_start: Absolute start time in ``[0, 1]``. At ``0`` this reproduces
            ``sample`` bit-for-bit; at ``1`` the interval is empty and the state
            returns unchanged up to the phase reprojection inside ``step``.

    Returns:
        The state at ``t=1``, shape ``[B, 3, H, W]``.

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
        # Absolute times in [t_start, 1), not step indices. Written as
        # t_start + span * (i / num_steps) rather than t_start + i * dt so
        # t_start=0 collapses to exactly i / num_steps, matching the float
        # solver.sample() feeds the model, with no drift accumulation.
        t_val = t_start + span * (i / num_steps)
        t_next_val = t_start + span * ((i + 1) / num_steps)

        t_tensor = torch.full((b,), t_val, device=device, dtype=torch.float32)
        t_next_tensor = torch.full((b,), t_next_val, device=device, dtype=torch.float32)

        # Predictor: current vector field.
        v_t = model(x_t, t_tensor)

        # Final step is Euler only, so the field is never evaluated past t=1.
        if i == num_steps - 1:
            return solver.step(x_t, v_t, dt)

        # Euler probe, field at the probe, averaged velocity.
        x_pred = solver.step(x_t, v_t, dt)
        v_next = model(x_pred, t_next_tensor)
        v_avg = 0.5 * (v_t + v_next)

        # Corrected step, taken from x_t.
        x_t = solver.step(x_t, v_avg, dt)

    return x_t


def sample_cylindrical_noise(
    batch: int,
    height: int,
    width: int,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Draw ``x_0`` on the cylinder exactly as ``train.py`` does.

    Amplitude is ``U[0, 1]``, phase is ``U[0, 2*pi)`` mapped onto the unit circle.
    Any other distribution puts the model off its training manifold at ``t=0``.

    Args:
        batch: Number of samples.
        height: Spatial height.
        width: Spatial width.
        device: Device to allocate on.
        generator: Optional RNG for reproducibility.

    Returns:
        Cylindrical noise of shape ``[B, 3, H, W]``.
    """
    amp = torch.rand(batch, 1, height, width, device=device, generator=generator)
    phi = torch.rand(batch, 1, height, width, device=device, generator=generator) * 2 * math.pi
    return torch.cat([amp, torch.cos(phi), torch.sin(phi)], dim=1)


def reconstruct_batch(
    model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    solver: CylindricalODESolver,
    bridge: GeodesicFlowBridge,
    x_1: torch.Tensor,
    t_start: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Noise the target to ``t_start`` via the bridge, then integrate back to ``t=1``.

    Args:
        model: The velocity field, ``(state, time) -> velocity``.
        solver: ODE solver supplying ``num_steps`` and ``step``.
        bridge: Bridge used to build the partially-noised state.
        x_1: Clean cylindrical target, shape ``[B, 3, H, W]``.
        t_start: Where on the noise-to-data path to start from.
        generator: Optional RNG for the noise draw.

    Returns:
        The reconstructed cylindrical state at ``t=1``, shape ``[B, 3, H, W]``.
    """
    b, _, h, w = x_1.shape
    x_0 = sample_cylindrical_noise(b, h, w, x_1.device, generator)

    t = torch.full((b, 1, 1, 1), t_start, device=x_1.device, dtype=torch.float32)
    # target_v is the training signal; only the state matters at eval time.
    x_t, _ = bridge.forward(cyl_noise=x_0, cyl_data=x_1, t=t)

    return integrate_from_t(model, solver, x_t, t_start)


def compute_batch_metrics(
    pred_cyl: torch.Tensor,
    target_cyl: torch.Tensor,
    mask_threshold: float | None,
) -> dict[str, torch.Tensor]:
    """Score one batch in the complex domain, per sample.

    Both sides go through :func:`~cfm.utils.complex_ops.cylinder_to_complex`
    first, since ``circular_phase_error``'s ``mask_threshold`` derives its mask
    from a complex target's amplitude.

    The unmasked phase error is reported alongside the masked one: the masked
    value is the meaningful score (phase is noise in air), the unmasked value
    shows how much of the image the mask discards.

    Args:
        pred_cyl: Reconstruction, cylindrical ``[B, 3, H, W]``.
        target_cyl: Ground truth, cylindrical ``[B, 3, H, W]``.
        mask_threshold: Fraction of the per-slice max amplitude below which pixels
            are excluded from the phase error. ``None`` scores every pixel, in
            which case masked and unmasked coincide.

    Returns:
        Mapping from metric name to a per-sample tensor of shape ``[B]``.
    """
    pred = cylinder_to_complex(pred_cyl)
    target = cylinder_to_complex(target_cyl)

    metrics = {
        # data_range=1.0 holds only because AmplitudeNormalize maps the amplitude
        # channel to [0, 1]; a different pipeline invalidates it.
        "psnr_db": peak_signal_noise_ratio(pred, target, data_range=1.0, reduction="none"),
        "ssim": structural_similarity(pred, target, data_range=1.0, reduction="none"),
        "phase_error_rad": circular_phase_error(
            pred, target, mask_threshold=mask_threshold, reduction="none"
        ),
    }
    if mask_threshold is not None:
        metrics["phase_error_rad_unmasked"] = circular_phase_error(pred, target, reduction="none")
    return metrics


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
def load_split_file_names(
    data_dir: str,
    split: str | None,
    annotations_subdir: str = "annotations/v1.0.0",
) -> set[str] | None:
    """Read the COCO-style manifest for ``split`` and return its ``.h5`` basenames.

    Args:
        data_dir: Dataset root, the same one handed to :class:`SKMTEADataset`.
        split: Manifest name (``"train"``, ``"val"``, ``"test"``), or ``None`` for
            every file in ``data_dir``, the escape hatch for a directory with no
            ``annotations/``.
        annotations_subdir: Where the manifests live under ``data_dir``.

    Returns:
        The set of ``.h5`` basenames in the split, or ``None`` when ``split`` is
        ``None``.

    Raises:
        FileNotFoundError: If the manifest for ``split`` does not exist.
        ValueError: If the manifest lists no images.
    """
    if split is None:
        return None

    manifest = os.path.join(data_dir, annotations_subdir, f"{split}.json")
    if not os.path.isfile(manifest):
        available = sorted(glob.glob(os.path.join(data_dir, annotations_subdir, "*.json")))
        names = [os.path.basename(p) for p in available] or "none"
        raise FileNotFoundError(
            f"No manifest for split={split!r} at {manifest}. Available: {names}. "
            f"Use evaluate.split=null to evaluate every file in data_dir."
        )

    with open(manifest, encoding="utf-8") as fh:
        payload = json.load(fh)

    file_names = {os.path.basename(img["file_name"]) for img in payload.get("images", [])}
    if not file_names:
        raise ValueError(f"Manifest {manifest} lists no images under 'images'.")
    return file_names


def select_indices(
    slice_map: Sequence[tuple[str, int]],
    file_names: set[str] | None,
    max_samples: int | None = None,
) -> list[int]:
    """Pick the dataset indices to evaluate.

    Args:
        slice_map: :attr:`SKMTEADataset.slice_map`, one ``(file_path, slice_idx)``
            per slice in dataset order.
        file_names: ``.h5`` basenames to keep, or ``None`` to keep everything.
        max_samples: Optional cap, applied by striding rather than truncation.
            Slices from one volume are highly correlated, so the first N all come
            from one end of one knee and misrepresent the split.

    Returns:
        Ascending dataset indices.

    Raises:
        ValueError: If no slice matches the requested split.
    """
    if file_names is None:
        indices = list(range(len(slice_map)))
    else:
        indices = [
            i for i, (path, _) in enumerate(slice_map) if os.path.basename(path) in file_names
        ]

    if not indices:
        present = sorted({os.path.basename(p) for p, _ in slice_map})
        raise ValueError(
            f"No slices matched the split. Manifest lists {sorted(file_names or [])}, "
            f"but the files present on disk are {present}. Download the missing volumes, "
            f"pick another split, or set evaluate.split=null to evaluate whatever is there."
        )

    if max_samples is not None and 0 < max_samples < len(indices):
        stride = len(indices) // max_samples
        indices = indices[::stride][:max_samples]

    return indices


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
    split = eval_cfg.get("split", "test")

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

    # 2. Model and checkpoint. Not compiled: one-shot eval pays the compile cost
    # for nothing. load_weights also puts the model in eval mode.
    orig_cwd = hydra.utils.get_original_cwd()
    run_name = eval_cfg.get("run_name") or cfg.get("logging", {}).get("experiment_name")
    model = build_model(cfg, device)
    checkpoint_path = resolve_checkpoint(cfg, "evaluate", orig_cwd)
    load_weights(model, checkpoint_path, device)

    # 4. Data. Same pipeline as train.py; a mismatch invalidates data_range=1.0.
    pipeline = Compose(
        [ComplexToCylinderTransform(), AmplitudeNormalize(), CenterCropModulo(base=16)]
    )
    data_dir = eval_cfg.get("data_dir") or cfg.get("dataset", {}).get(
        "data_dir", "data/skm-tea-mini/v1-release"
    )
    if not os.path.isabs(data_dir):
        data_dir = os.path.join(orig_cwd, data_dir)

    # SKMTEADataset opens every .h5 under data_dir to count slices before split
    # filtering is possible, so a corrupt file aborts the run even when it is not
    # in the chosen split. Intentional: swallowing a corrupt-data error here is how
    # you end up reporting metrics on half a dataset.
    dataset = SKMTEADataset(data_dir=data_dir, transform=pipeline)
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
        wandb.init(
            project=cfg.get("logging", {}).get("project_name", "Cylindrical-Flow-Matching"),
            name=f"eval_{run_name}_t{t_start}",
            dir=output_dir,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    else:
        use_wandb = False
        print("Local logging only.")

    # 6. Evaluation loop.
    solver = CylindricalODESolver(num_steps=num_steps)
    bridge = GeodesicFlowBridge()
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
            x_1 = batch.to(device)
            batch_ids = sample_ids[position : position + x_1.shape[0]]
            position += x_1.shape[0]

            pred = reconstruct_batch(model, solver, bridge, x_1, t_start, generator)
            batch_metrics = compute_batch_metrics(pred, x_1, mask_threshold)

            for name, values in batch_metrics.items():
                if name not in accumulators:
                    accumulators[name] = MetricAccumulator(name)
                accumulators[name].update(values, batch_ids)

    # 7. Report.
    summaries = [acc.summary() for acc in accumulators.values()]

    print()
    print("=" * 88)
    print("Reconstruction evaluation")
    print(f"  checkpoint : {checkpoint_path}")
    print(f"  split      : {split}   ({num_files} file(s), {len(indices)} slices)")
    print(f"  t_start    : {t_start:.3f}   num_steps: {num_steps}")
    print(f"  mask thr.  : {mask_threshold}")
    print(f"  batch/dev  : {batch_size} on {device}   seed: {seed}")
    print("=" * 88)
    print(format_summary_table(summaries))
    print("=" * 88)

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
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
