"""Generative evaluation, as a pipeline that returns its result instead of printing it.

The measurement and the report were one function. Everything below ``execute``
produces data; :func:`render_report` turns it into the text a human reads, and
the caller writes the JSON. That split is what lets the reproduction layer
re-render an archived result without re-running a sweep, and it is the same split
the network-free probes still lack.

**The absent-key convention.** ``metrics.json`` *omits* the families it could not
compute rather than writing ``null`` or ``0.0``, and the table scripts rely on
the absence. A reader must not be able to mistake "not applicable" for "measured,
and small". Two things are gated, and on different questions: straightness and
the angular probe both need the network to emit a velocity
(``manifold.predicts_velocity``), and the angular probe additionally needs the
geometry to define the quantity
(``manifold.reports_induced_angular_velocity``).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Self

import torch
from torch.utils.data import DataLoader

from cyfm.config.adapters import manifold_from_config
from cyfm.config.resolve import as_plain_dict
from cyfm.config.schema import RunConfig
from cyfm.core.manifold import BaseManifold
from cyfm.core.pipeline import BasePipeline
from cyfm.data import build_dataset, build_geometry_transform
from cyfm.data.transforms import Compose, slice_transform
from cyfm.flow.couplings import build_coupling
from cyfm.metrics import AngularVelocityProbe, generative_metrics, straightness
from cyfm.utils.inference import build_model, load_weights, resolve_checkpoint

__all__ = [
    "CROP_BASE",
    "EvaluationPipeline",
    "assert_training_domain",
    "format_table",
    "render_report",
    "training_pipeline",
]

# Divisibility the U-Net's downsampling depth requires of the field.
CROP_BASE = 16

# Columns the headline table shows, in order.
HEADLINE = (
    "sliced_w2_complex",
    "w2_amplitude",
    "w2_phase_circular",
    "dependence_gap",
    "spatial_lag1_gap",
    "radial_spectrum_gap",
)


def training_pipeline(dataset_cfg: Mapping[str, Any], manifold: BaseManifold) -> Compose:
    """The transform training applies to every field, reproduced for evaluation.

    The reference distribution has to live where the model's samples live.
    Training hands the model fields that went through the geometry crop and the
    manifold's representation pipeline, which divides every field by its own peak
    modulus, so a model that learned the training distribution perfectly
    generates normalised fields. Scoring those against raw fields penalises the
    normalisation rather than anything the model got wrong -- on the synthetic
    target the raw amplitude mean is 0.671 and the normalised one 0.618, a bias
    that earlier versions of this module carried into every absolute number.

    Args:
        dataset_cfg: The ``dataset`` config group; ``crop_size`` is read.
        manifold: The geometry whose representation pipeline training used.

    Returns:
        A callable mapping a complex field to the manifold state training saw.
    """
    return Compose(
        [
            build_geometry_transform(dataset_cfg.get("crop_size"), crop_base=CROP_BASE),
            slice_transform(manifold, crop_base=CROP_BASE),
        ]
    )


def assert_training_domain(reference: torch.Tensor, tolerance: float = 1e-4) -> float:
    """Check the reference batch carries the normalisation training applied.

    Both geometries divide a field by its own peak modulus before the crop, so
    every coefficient training ever saw has modulus at most one, and so does
    every sample a converged model draws. A reference batch that breaks the bound
    did not come through :func:`training_pipeline`, and scoring against it
    measures the missing normalisation instead of the model -- the defect that
    once biased every absolute W2 in this module.

    The bound is one-sided on purpose. Normalisation happens before the crop, so
    a field whose peak modulus was cropped away is legitimately below one and
    must not fail; only exceeding one proves the division never happened.

    Args:
        reference: Complex reference fields of shape ``[B, 1, H, W]``.
        tolerance: Slack over one, for the rounding in a float32 divide.

    Returns:
        The largest peak modulus observed, for the run's record.

    Raises:
        ValueError: If any field's peak modulus exceeds ``1 + tolerance``.
    """
    peaks = reference.abs().amax(dim=(-3, -2, -1))
    largest = float(peaks.max())
    if largest > 1.0 + tolerance:
        offenders = int((peaks > 1.0 + tolerance).sum())
        raise ValueError(
            f"reference fields are not in the training domain: {offenders} of "
            f"{peaks.numel()} have a peak modulus above 1 (largest {largest:.6g}). "
            "Peak normalisation is applied by the manifold's transform, so this "
            "batch bypassed training_pipeline(); scoring against it would measure "
            "the normalisation rather than the model."
        )
    return largest


def format_table(rows: Sequence[tuple[int, dict[str, float]]], metrics: Sequence[str]) -> str:
    """Render the sweep as a fixed-width table.

    Args:
        rows: ``(num_steps, metrics)`` pairs, in sweep order.
        metrics: Metric names to show, in column order.

    Returns:
        The rendered table.
    """
    header = f"{'steps':>7}" + "".join(f"{name:>24}" for name in metrics)
    lines = [header, "-" * len(header)]
    for steps, values in rows:
        lines.append(f"{steps:>7}" + "".join(f"{values[name]:>24.5f}" for name in metrics))
    return "\n".join(lines)


class EvaluationPipeline(BasePipeline[dict[str, Any]]):
    """Score a checkpoint's samples against the data distribution.

    Args:
        cfg: The whole config tree.
        orig_cwd: ``hydra.utils.get_original_cwd()``, since Hydra moves the
            working directory and the checkpoint lookup must not follow it.
    """

    def __init__(self, cfg: Mapping[str, Any], orig_cwd: str = ".") -> None:
        self.cfg = cfg
        self.orig_cwd = orig_cwd
        self.config = RunConfig.from_config(cfg)
        self.dataset_cfg = as_plain_dict(cfg.get("dataset"))
        self.data_states: torch.Tensor | None = None
        self.reference: torch.Tensor | None = None

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any], orig_cwd: str = ".") -> Self:
        """Build from a composed config tree.

        Args:
            cfg: The whole config.
            orig_cwd: Directory the run was launched from.

        Returns:
            The pipeline, not yet set up.
        """
        return cls(cfg, orig_cwd)

    def setup(self) -> None:
        """Load the checkpoint and the reference cohort it will be scored against."""
        settings = self.config.evaluate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Two generators from one seed: torch requires a generator on the same
        # device as the tensor it fills, and the metrics run on CPU while the
        # sampling runs wherever the model does. Splitting them keeps both halves
        # reproducible.
        self.device_generator = torch.Generator(device=self.device).manual_seed(settings.seed)
        self.metric_generator = torch.Generator().manual_seed(settings.seed)
        print(f"Evaluating generation on: {self.device}")

        self.manifold = manifold_from_config(self.cfg).to(self.device)
        self.model = build_model(
            self.cfg,
            self.device,
            in_channels=self.manifold.state_channels,
            out_channels=self.manifold.velocity_channels,
            velocity_bound=(
                self.manifold.velocity_bound
                if bool(as_plain_dict(self.cfg.get("model")).get("bounded_velocity", False))
                else None
            ),
        )
        load_weights(
            self.model,
            resolve_checkpoint(self.cfg, "evaluate", self.orig_cwd),
            self.device,
        )
        self.model.eval()

        self._load_reference()

    def _load_reference(self) -> None:
        """Draw the cohort the samples are scored against, in the training domain."""
        settings = self.config.evaluate
        dataset = build_dataset(
            self.dataset_cfg, transform=training_pipeline(self.dataset_cfg, self.manifold)
        )
        loader = DataLoader(
            dataset,
            batch_size=settings.batch_size,
            shuffle=False,
            num_workers=settings.num_workers,
        )

        batches: list[torch.Tensor] = []
        collected = 0
        for batch in loader:
            batches.append(batch)
            collected += batch.shape[0]
            if collected >= settings.num_fields:
                break
        if hasattr(dataset, "close") and callable(dataset.close):
            dataset.close()
        if not batches:
            raise ValueError("dataset yielded no samples to evaluate against")

        # Straightness needs the data in the manifold's representation and the
        # distributional metrics need it complex; both come from the same fields,
        # in the domain training used, so the two numbers describe one batch.
        self.data_states = torch.cat(batches, dim=0)[: settings.num_fields].to(self.device)
        self.reference = self.manifold.to_complex(self.data_states).cpu()
        _, _, self.height, self.width = self.data_states.shape
        # Checked, not assumed: the domain is what makes the absolute numbers mean
        # anything, and it is cheap enough to verify on every run that reports one.
        self.peak_modulus = assert_training_domain(self.reference)
        print(
            f"Reference: {self.data_states.shape[0]} fields of {self.height}x{self.width}, "
            f"in the training domain (peak modulus {self.peak_modulus:.6g})"
        )

    def execute(self) -> dict[str, Any]:
        """Measure straightness and sweep the solver step counts.

        Returns:
            The payload written to ``metrics.json``, with the families this arm
            cannot report absent rather than null.
        """
        if self.reference is None or self.data_states is None:
            raise RuntimeError("EvaluationPipeline.setup() must be run before execute().")
        straightness_value = self._straightness()
        rows, probes = self._sweep()
        settings = self.config.evaluate
        nfe = list(settings.nfe)

        # Described from what ran, not from a constant: this module serves a Heun
        # integrator and a stochastic sampler, and a note naming the wrong one is
        # worse than no note at all.
        solver_name = type(self.manifold.make_solver(nfe[0])).__name__
        note_on_nfe = (
            "Function evaluations are counted per solver and reported as `nfe` on "
            "each sweep row, beside the `num_steps` the sweep requested and the "
            "`executed_steps` actually run. Heun evaluates twice per step less the "
            "corrector its final step skips (1, 3, 7, 15, 199 for 1, 2, 4, 8, 100); "
            "a predictor-corrector sampler evaluates 1 + corrector_steps per step."
        )
        return {
            "manifold": self.manifold.name,
            "model": type(self.model).__name__,
            "dataset": self.dataset_cfg.get("name"),
            "num_fields": self.reference.shape[0],
            "field_shape": [self.height, self.width],
            "seed": settings.seed,
            "num_projections": settings.num_projections,
            "reference_domain": "training transform",
            # Evidence for the line above rather than a restatement of it: the
            # largest peak modulus in the reference batch, which
            # assert_training_domain required to be at most one before any metric
            # was computed.
            "reference_peak_modulus": self.peak_modulus,
            # Absent, not null, for an arm where a path's straightness is undefined.
            **(
                {}
                if straightness_value is None
                else {
                    "straightness": straightness_value,
                    "straightness_pairing": self.config.training.coupling,
                }
            ),
            "solver": solver_name,
            "note_on_nfe": note_on_nfe,
            # `nfe` is the cost actually paid, read off the solver, and
            # `num_steps` the count the sweep asked for; in matched mode they are
            # not the same and `executed_steps` in each row says what ran.
            "sweep": [
                {
                    "num_steps": steps,
                    "nfe": int(probes[steps]["model_evaluations"]),
                    **values,
                    **probes[steps],
                }
                for steps, values in rows
            ],
        }

    def _straightness(self) -> float | None:
        """The regression residual of the conditional velocity, or ``None``.

        An arm regressing a score has no such comparison to make: the second
        return of its bridge is ``-z/sigma``, whose scale runs away as sigma
        falls, so the ratio would be dominated by the ``t`` near 1 end and would
        sit in the table looking comparable to the flow arms' path straightness.
        """
        if self.data_states is None:
            raise RuntimeError(
                "EvaluationPipeline.setup() must be run before measuring straightness."
            )
        if not self.manifold.predicts_velocity:
            print(
                f"straightness: not reported for {self.manifold.name} "
                "(its output is not a velocity)"
            )
            return None
        chunk = self.config.evaluate.straightness_batch_size
        return straightness(
            self.model,
            self.manifold,
            self.data_states,
            self.device,
            self.device_generator,
            build_coupling(self.config.training.coupling),
            chunk=chunk,
        )

    def _sweep(self) -> tuple[list[tuple[int, dict[str, float]]], dict[int, dict[str, float]]]:
        """Generate and score at every requested step count.

        Returns:
            ``(rows, probes)``: the distributional metrics per step count, and
            the path probe plus cost for each.
        """
        if self.reference is None:
            raise RuntimeError("EvaluationPipeline.setup() must be run before sweeping solvers.")
        settings = self.config.evaluate
        rows: list[tuple[int, dict[str, float]]] = []
        probes: dict[int, dict[str, float]] = {}

        for steps in settings.nfe:
            solver = self.manifold.make_solver(steps)
            generated_batches = []
            remaining = self.reference.shape[0]
            probe = AngularVelocityProbe(
                self.manifold.wrap_model(self.model),
                self.manifold,
                enabled=self.manifold.predicts_velocity,
            )
            while remaining > 0:
                size = min(settings.batch_size, remaining)
                prior = self.manifold.sample_noise(
                    size, self.height, self.width, self.device, generator=self.device_generator
                )
                probe.start_batch()
                with torch.no_grad():
                    # The generator reaches the sampler too: a stochastic one
                    # draws inside sample(), and without it those draws would
                    # come from the global RNG while the record still claimed a
                    # seed.
                    generated_batches.append(
                        self.manifold.to_complex(
                            solver.sample(probe, prior, generator=self.device_generator)
                        ).cpu()
                    )
                remaining -= size

            row_probe: dict[str, float] = {}
            if probe.enabled:
                assert probe.peak_angular is not None and probe.min_amplitude is not None
                row_probe = {
                    "peak_angular_velocity_median": float(probe.peak_angular.median()),
                    "peak_angular_velocity_max": float(probe.peak_angular.max()),
                    "peak_angular_velocity_near_t_half": probe.peak_angular_mid,
                    "min_amplitude_mean": float(probe.min_amplitude.mean()),
                    "min_amplitude_min": float(probe.min_amplitude.min()),
                }
            # What the sweep's step count actually cost, and what it actually ran.
            # Heun spends 2n-1 calls and the diffusion sampler (1+M) per step, so
            # the column header alone does not say whether two rows had the same
            # budget; and in matched mode the executed step count is not the
            # requested one.
            row_probe["model_evaluations"] = float(solver.evaluations)
            row_probe["executed_steps"] = float(solver.num_steps)
            probes[steps] = row_probe

            generated = torch.cat(generated_batches, dim=0)
            rows.append(
                (
                    steps,
                    generative_metrics(
                        generated,
                        self.reference,
                        settings.num_projections,
                        self.metric_generator,
                    ),
                )
            )
            print(f"  steps={steps:<4} sliced_w2={rows[-1][1]['sliced_w2_complex']:.5f}")

        return rows, probes

    def teardown(self) -> None:
        """Release reference tensors and clear cached CUDA memory."""
        self.data_states = None
        self.reference = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def render_report(payload: Mapping[str, Any]) -> str:
    """Turn a result into the text a human reads.

    Separate from the measurement on purpose: the same payload can be re-rendered
    from an archive without regenerating a single sample.

    Args:
        payload: What :meth:`EvaluationPipeline.execute` returned.

    Returns:
        The report, ready to print.
    """
    sweep = list(payload["sweep"])
    rows = [(int(row["num_steps"]), dict(row)) for row in sweep]
    first = rows[0][1]

    lines = [
        "",
        "=" * 120,
        "GENERATIVE EVALUATION",
        "=" * 120,
        format_table(rows, HEADLINE),
        "-" * 120,
    ]
    if "straightness" in payload:
        lines.append(
            f"straightness ({payload['straightness_pairing']} pairing): "
            f"{payload['straightness']:.5f}"
        )
    lines.append(
        f"dependence: reference {first['dependence_reference']:.4f}, "
        f"generated {first['dependence_generated']:.4f} at {rows[0][0]} step(s)"
    )
    lines.append(
        f"spatial lag-1: reference {first['spatial_lag1_reference']:.4f}, "
        f"generated {first['spatial_lag1_generated']:.4f} at {rows[0][0]} step(s)"
    )

    # Only for arms whose output is a velocity; for the others the quantity is
    # undefined and the section is omitted rather than filled with zeros.
    if "peak_angular_velocity_median" in first:
        lines += [
            "",
            "-" * 120,
            "ANGULAR VELOCITY ALONG THE PATH   "
            "(cylinder is bounded by pi = 3.1416; the plane is not)",
            f"{'steps':>7}{'NFE':>7}{'median':>12}{'max':>12}{'near t=0.5':>14}{'min |z|':>10}",
        ]
        for steps, row in rows:
            lines.append(
                f"{steps:>7}{int(row['model_evaluations']):>7}"
                f"{row['peak_angular_velocity_median']:>14.4f}"
                f"{row['peak_angular_velocity_max']:>12.4f}"
                f"{row['peak_angular_velocity_near_t_half']:>18.4f}"
                f"{row['min_amplitude_min']:>10.5f}"
            )
    return "\n".join(lines)
