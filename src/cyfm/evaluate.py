"""Generative evaluation: does the model reproduce the data distribution?

This entry point replaces a paired one. The reconstruction path scored a
prediction against the specific slice it was derived from -- PSNR, SSIM, a phase
error and a data-consistency term, each meaningful only because the target was
known. Pure synthesis has no such target: a sample drawn from the prior has no
reason to match any particular slice, and every paired metric is therefore poor
*by construction* at ``t = 0``, which is precisely the setting the project now
cares about. Scoring synthesis needs distributional metrics, and those are what
this module computes.

What is measured, and why each one is here
------------------------------------------
**Sliced 2-Wasserstein on the complex plane.** The headline number. Coefficients
from generated and reference fields are pooled and compared as points in ``R^2``.
Sliced rather than exact because the exact assignment's finite-sample floor is
large enough to swallow the differences worth seeing: measured on this project's
synthetic target, the exact estimator's floor at 2048 samples is 0.078 while the
sliced estimator reaches 0.004 at 32768, against a separation of 0.188 between
genuinely different distributions.

**Exact transport on each marginal.** Amplitude on the line and phase on the
circle, both solved exactly. A model can match a pooled two-dimensional cloud
while getting one marginal wrong in a way slicing averages away, and the phase
marginal is the one this project makes claims about.

**The dependence gap.** The circular-linear correlation of the generated
coefficients against the reference's. Amplitude-phase dependence is the axis the
synthetic datasets sweep and the thing a factorised coupling destroys, so a model
that reproduces both marginals and none of the dependence has to be visible as a
number rather than as an argument.

**The spatial gap.** Lag-one autocorrelation of the amplitude field, generated
against reference. Every metric above pools coefficients and is therefore blind
to spatial structure entirely; without this one a model could match the pointwise
law perfectly and produce noise where the data has fields.

**Straightness.** The regression residual of the conditional velocity,
normalised by the displacement it had to explain. Both bridges carry a velocity
constant along the path, so this is the rectified-flow straightness statistic
directly rather than the training loss under another name. Computed under the
coupling the checkpoint was trained with, read from ``training.coupling``.

All of the above are swept over solver step counts, because the number of
function evaluations a geometry needs is a claim this project makes and a table
it has to be able to produce. Every metric lives in :mod:`cyfm.metrics` -- the
path probes included, which used to stay here on the grounds that they need the
model and the manifold. They do, and taking both as arguments costs nothing;
what it buys is that they can be tested without standing up a Hydra ``main``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from cyfm.config.adapters import manifold_from_config
from cyfm.data import build_dataset, build_geometry_transform
from cyfm.data.transforms import Compose, slice_transform
from cyfm.flow.couplings import build_coupling
from cyfm.metrics import AngularVelocityProbe, generative_metrics, straightness
from cyfm.utils.inference import build_model, load_weights, resolve_checkpoint


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


def training_pipeline(dataset_cfg: Any, manifold: Any) -> Compose:
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
            build_geometry_transform(dataset_cfg.get("crop_size"), crop_base=16),
            slice_transform(manifold, crop_base=16),
        ]
    )


def assert_training_domain(reference: torch.Tensor, tolerance: float = 1e-4) -> float:
    """Check the reference batch carries the normalisation training applied.

    Both geometries divide a field by its own peak modulus before the crop, so
    every coefficient training ever saw has modulus at most one, and so does
    every sample a converged model draws. A reference batch that breaks the
    bound did not come through :func:`training_pipeline`, and scoring against it
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


@hydra.main(version_base="1.3", config_path="../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Score a checkpoint's samples against the data distribution."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    settings = cfg.get("evaluate", {})
    seed = int(settings.get("seed", 0))
    # Two generators from one seed: torch requires a generator on the same device
    # as the tensor it fills, and the metrics run on CPU while the sampling runs
    # wherever the model does. Splitting them keeps both halves reproducible.
    device_generator = torch.Generator(device=device).manual_seed(seed)
    metric_generator = torch.Generator().manual_seed(seed)

    print(OmegaConf.to_yaml(cfg))
    print(f"Evaluating generation on: {device}")

    manifold = manifold_from_config(cfg).to(device)
    model = build_model(
        cfg,
        device,
        in_channels=manifold.state_channels,
        out_channels=manifold.velocity_channels,
        velocity_bound=(
            manifold.velocity_bound
            if bool(cfg.get("model", {}).get("bounded_velocity", False))
            else None
        ),
    )
    load_weights(model, resolve_checkpoint(cfg, "evaluate", hydra.utils.get_original_cwd()), device)
    model.eval()

    dataset_cfg = cfg.get("dataset", {})
    dataset = build_dataset(dataset_cfg, transform=training_pipeline(dataset_cfg, manifold))

    num_fields = int(settings.get("num_fields", 64))
    batch_size = int(settings.get("batch_size", 16))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(settings.get("num_workers", 0)),
    )

    state_batches: list[torch.Tensor] = []
    collected = 0
    for batch in loader:
        state_batches.append(batch)
        collected += batch.shape[0]
        if collected >= num_fields:
            break
    if not state_batches:
        raise ValueError("dataset yielded no samples to evaluate against")

    # Straightness needs the data in the manifold's representation and the
    # distributional metrics need it complex; both come from the same fields, in
    # the domain training used, so the two numbers describe one batch.
    data_states = torch.cat(state_batches, dim=0)[:num_fields].to(device)
    reference = manifold.to_complex(data_states).cpu()
    _, _, height, width = data_states.shape
    # Checked, not assumed: the domain is what makes the absolute numbers mean
    # anything, and it is cheap enough to verify on every run that reports one.
    peak_modulus = assert_training_domain(reference)
    print(
        f"Reference: {data_states.shape[0]} fields of {height}x{width}, "
        f"in the training domain (peak modulus {peak_modulus:.6g})"
    )

    coupling_name = str(cfg.get("training", {}).get("coupling", "independent"))
    straightness_chunk = settings.get("straightness_batch_size")
    # Straightness measures how close a predicted velocity is to the displacement
    # it should equal. An arm regressing a score has no such comparison to make:
    # the second return of its bridge is -z/sigma, whose scale runs away as sigma
    # falls, so the ratio would be dominated by the t near 1 end and would sit in
    # the table looking comparable to the flow arms' path straightness.
    straightness_value: float | None = None
    if manifold.predicts_velocity:
        straightness_value = straightness(
            model,
            manifold,
            data_states,
            device,
            device_generator,
            build_coupling(coupling_name),
            chunk=None if straightness_chunk is None else int(straightness_chunk),
        )
    else:
        print(f"straightness: not reported for {manifold.name} (its output is not a velocity)")

    nfe: Sequence[int] = [int(n) for n in settings.get("nfe", [1, 2, 4, 8, 16, 32, 64, 100])]
    projections = int(settings.get("num_projections", 256))

    rows: list[tuple[int, dict[str, float]]] = []
    probes: dict[int, dict[str, float]] = {}
    for steps in nfe:
        solver = manifold.make_solver(steps)
        generated_batches = []
        remaining = reference.shape[0]
        probe = AngularVelocityProbe(
            manifold.wrap_model(model), manifold, enabled=manifold.predicts_velocity
        )
        while remaining > 0:
            size = min(batch_size, remaining)
            prior = manifold.sample_noise(size, height, width, device, generator=device_generator)
            probe.start_batch()
            with torch.no_grad():
                # The generator reaches the sampler too: a stochastic one draws
                # inside sample(), and without it those draws would come from the
                # global RNG while the record still claimed a seed.
                generated_batches.append(
                    manifold.to_complex(
                        solver.sample(probe, prior, generator=device_generator)
                    ).cpu()
                )
            remaining -= size
        # Recorded only where an angular velocity is defined; the keys are absent
        # rather than zero for the other arms, so a reader cannot mistake "not
        # applicable" for "measured, and small".
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
        # Heun spends 2n-1 calls and the diffusion sampler (1+M) per step, so the
        # column header alone does not say whether two rows had the same budget;
        # and in matched mode the executed step count is not the requested one.
        row_probe["model_evaluations"] = float(solver.evaluations)
        row_probe["executed_steps"] = float(solver.num_steps)
        probes[steps] = row_probe
        generated = torch.cat(generated_batches, dim=0)
        rows.append(
            (steps, generative_metrics(generated, reference, projections, metric_generator))
        )
        print(f"  steps={steps:<4} sliced_w2={rows[-1][1]['sliced_w2_complex']:.5f}")

    headline = [
        "sliced_w2_complex",
        "w2_amplitude",
        "w2_phase_circular",
        "dependence_gap",
        "spatial_lag1_gap",
        "radial_spectrum_gap",
    ]
    print("\n" + "=" * 120)
    print("GENERATIVE EVALUATION")
    print("=" * 120)
    print(format_table(rows, headline))
    print("-" * 120)
    if straightness_value is not None:
        print(f"straightness ({coupling_name} pairing): {straightness_value:.5f}")
    print(
        f"dependence: reference {rows[0][1]['dependence_reference']:.4f}, "
        f"generated {rows[0][1]['dependence_generated']:.4f} at {nfe[0]} step(s)"
    )
    print(
        f"spatial lag-1: reference {rows[0][1]['spatial_lag1_reference']:.4f}, "
        f"generated {rows[0][1]['spatial_lag1_generated']:.4f} at {nfe[0]} step(s)"
    )
    # Only for arms whose output is a velocity; for the others the quantity is
    # undefined and the section is omitted rather than filled with zeros.
    if manifold.predicts_velocity:
        print("\n" + "-" * 120)
        print(
            "ANGULAR VELOCITY ALONG THE PATH   "
            "(cylinder is bounded by pi = 3.1416; the plane is not)"
        )
        print(f"{'steps':>7}{'NFE':>7}{'median':>12}{'max':>12}{'near t=0.5':>14}{'min |z|':>10}")
        for steps in nfe:
            p = probes[steps]
            print(
                f"{steps:>7}{int(p['model_evaluations']):>7}"
                f"{p['peak_angular_velocity_median']:>14.4f}{p['peak_angular_velocity_max']:>12.4f}"
                f"{p['peak_angular_velocity_near_t_half']:>18.4f}{p['min_amplitude_min']:>10.5f}"
            )

    # Described from what ran, not from a constant: this module serves a Heun
    # integrator and a stochastic sampler, and a note naming the wrong one is
    # worse than no note at all.
    solver_name = type(manifold.make_solver(nfe[0])).__name__
    note_on_nfe = (
        "Function evaluations are counted per solver and reported as `nfe` on "
        "each sweep row, beside the `num_steps` the sweep requested and the "
        "`executed_steps` actually run. Heun evaluates twice per step less the "
        "corrector its final step skips (1, 3, 7, 15, 199 for 1, 2, 4, 8, 100); "
        "a predictor-corrector sampler evaluates 1 + corrector_steps per step."
    )

    output_dir = Path(cfg.get("paths", {}).get("output_dir", "."))
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "manifold": manifold.name,
        "model": type(model).__name__,
        "dataset": dataset_cfg.get("name"),
        "num_fields": reference.shape[0],
        "field_shape": [height, width],
        "seed": seed,
        "num_projections": projections,
        "reference_domain": "training transform",
        # Evidence for the line above, rather than a restatement of it: the
        # largest peak modulus in the reference batch, which assert_training_domain
        # required to be at most one before any metric was computed.
        "reference_peak_modulus": peak_modulus,
        # Absent, not null, for an arm where a path's straightness is undefined.
        **({} if straightness_value is None else {"straightness": straightness_value}),
        **({} if straightness_value is None else {"straightness_pairing": coupling_name}),
        "solver": solver_name,
        "note_on_nfe": note_on_nfe,
        # `nfe` is the cost actually paid, read off the solver, and `num_steps`
        # the count the sweep asked for; in matched mode they are not the same
        # and `executed_steps` in each row says what ran.
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
    destination = output_dir / "metrics.json"
    destination.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {destination}")


if __name__ == "__main__":
    main()
