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
it has to be able to produce. The distributional metrics live in
:mod:`cfm.utils.metrics`; straightness and the angular-velocity probe stay here
because they need the model and the manifold.
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

from cfm.data import build_dataset, build_geometry_transform
from cfm.data.transforms import Compose
from cfm.flow.coupling import BaseCoupling, build_coupling
from cfm.manifolds import build_manifold
from cfm.utils.inference import build_model, load_weights, resolve_checkpoint
from cfm.utils.metrics import distributional_metrics

# Guards the induced angular velocity against a literal division by zero; the
# divergence it is meant to expose happens well above this.
_ANGULAR_FLOOR = 1e-12


class _AngularProbe:
    """Wraps a velocity field and records what the solver actually meets.

    Theorem 3 is a statement about the path, not the endpoint: under a Cartesian
    field the induced angular velocity

        theta_dot = (x v_y - y v_x) / A^2

    diverges as ``A -> 0``, while on the cylinder the angular velocity *is* a
    coordinate of the prediction and is bounded by ``pi`` from the range of
    ``atan2``. Recording both along the same trajectories is what turns that from
    an argument into a number.

    Args:
        network: The velocity field to wrap.
        geometry: ``"euclidean"`` to induce the angular velocity, ``"cylindrical"``
            to read it off the prediction.
        mid_window: Half-width of the band around ``t = 0.5`` reported separately,
            which is where a chordal path passes closest to the origin.
    """

    def __init__(self, network: torch.nn.Module, geometry: str, mid_window: float = 0.1) -> None:
        self.network = network
        self.geometry = geometry
        self.mid_window = mid_window
        self.min_amplitude: torch.Tensor | None = None
        self.peak_angular: torch.Tensor | None = None
        self.peak_angular_mid = 0.0

    def __call__(self, state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Record, then delegate.

        Args:
            state: State tensor ``[B, C, H, W]``.
            t: Times ``[B]``.

        Returns:
            The wrapped field's velocity.
        """
        velocity = self.network(state, t)
        if self.geometry == "euclidean":
            real, imag = state[:, 0], state[:, 1]
            squared = (real * real + imag * imag).clamp_min(_ANGULAR_FLOOR)
            amplitude = squared.sqrt()
            angular = (real * velocity[:, 1] - imag * velocity[:, 0]) / squared
        else:
            amplitude = state[:, 0].clamp_min(0.0)
            angular = velocity[:, 1]

        magnitude = angular.abs()
        self.min_amplitude = (
            amplitude
            if self.min_amplitude is None
            else torch.minimum(self.min_amplitude, amplitude)
        )
        self.peak_angular = (
            magnitude if self.peak_angular is None else torch.maximum(self.peak_angular, magnitude)
        )
        if abs(float(t.reshape(-1)[0]) - 0.5) <= self.mid_window:
            self.peak_angular_mid = max(self.peak_angular_mid, float(magnitude.max()))
        return velocity


@torch.no_grad()
def straightness(
    model: torch.nn.Module,
    manifold: Any,
    data_states: torch.Tensor,
    device: torch.device,
    generator: torch.Generator,
    coupling: BaseCoupling | None = None,
    chunk: int | None = None,
) -> float:
    """Regression residual of the conditional velocity, normalised.

    Computed under the coupling the model was trained with. A model regressed
    toward optimal-transport pairs has to be scored against optimal-transport
    pairs: scoring it against independent ones measures a different regression,
    and once made an OT arm look exactly as unlearnable as its baseline. A
    coupling that removes crossings should lower this, and lowering it is the
    mechanism behind any reduction in solver steps.

    Args:
        model: The trained velocity field.
        manifold: The geometry, supplying the prior and the bridge.
        data_states: Data in the manifold's state representation,
            ``[B, state_channels, H, W]``.
        device: Device to compute on.
        generator: RNG for the prior draw and the time sample. Must live on
            ``device``: torch refuses a CPU generator for a CUDA draw.
        coupling: The pairing training used. ``None`` pairs independently.
        chunk: Fields per forward pass. ``None`` sends the whole batch at once,
            which is what every synthetic run did and what keeps their numbers
            reproducible; a 320x320 cohort does not fit that way, so the fastMRI
            experiments set it. Chunking changes how the generator is consumed, so
            two chunk sizes give slightly different draws and must not be mixed
            within one table.

    Returns:
        ``0`` for a perfectly straight field, ``1`` for one that explains none of
        the displacement.
    """
    batch, _, height, width = data_states.shape
    step = batch if chunk is None else min(chunk, batch)

    residuals, displacements = [], []
    for start in range(0, batch, step):
        block = data_states[start : start + step]
        size = block.shape[0]
        prior = manifold.sample_noise(size, height, width, device, generator=generator)
        times = torch.rand(size, 1, 1, 1, device=device, generator=generator)
        if coupling is not None:
            block = coupling(prior, block, manifold)
        state, target = manifold.bridge(prior, block, times)
        residuals.append((model(state, times.reshape(-1)) - target).square().flatten(1).sum(1))
        displacements.append(target.square().flatten(1).sum(1))

    displacement = torch.cat(displacements)
    total = displacement.mean()
    if float(total) <= 0.0:
        return 0.0
    return float(torch.cat(residuals).mean() / total)


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
            manifold.build_transform(crop_base=16),
        ]
    )


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

    manifold = build_manifold(cfg).to(device)
    model = build_model(
        cfg,
        device,
        in_channels=manifold.state_channels,
        out_channels=manifold.velocity_channels,
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
    print(f"Reference: {data_states.shape[0]} fields of {height}x{width}, in the training domain")

    coupling_name = str(cfg.get("training", {}).get("coupling", "independent"))
    straightness_chunk = settings.get("straightness_batch_size")
    straightness_value = straightness(
        model,
        manifold,
        data_states,
        device,
        device_generator,
        build_coupling(coupling_name),
        chunk=None if straightness_chunk is None else int(straightness_chunk),
    )

    nfe: Sequence[int] = [int(n) for n in settings.get("nfe", [1, 2, 4, 8, 16, 32, 64, 100])]
    projections = int(settings.get("num_projections", 256))

    rows: list[tuple[int, dict[str, float]]] = []
    probes: dict[int, dict[str, float]] = {}
    for steps in nfe:
        solver = manifold.make_solver(steps)
        generated_batches = []
        remaining = reference.shape[0]
        probe = _AngularProbe(model, manifold.name)
        while remaining > 0:
            size = min(batch_size, remaining)
            prior = manifold.sample_noise(size, height, width, device, generator=device_generator)
            with torch.no_grad():
                generated_batches.append(manifold.to_complex(solver.sample(probe, prior)).cpu())
            remaining -= size
        assert probe.peak_angular is not None and probe.min_amplitude is not None
        probes[steps] = {
            "peak_angular_velocity_median": float(probe.peak_angular.median()),
            "peak_angular_velocity_max": float(probe.peak_angular.max()),
            "peak_angular_velocity_near_t_half": probe.peak_angular_mid,
            "min_amplitude_mean": float(probe.min_amplitude.mean()),
            "min_amplitude_min": float(probe.min_amplitude.min()),
        }
        generated = torch.cat(generated_batches, dim=0)
        rows.append(
            (steps, distributional_metrics(generated, reference, projections, metric_generator))
        )
        print(f"  steps={steps:<4} sliced_w2={rows[-1][1]['sliced_w2_complex']:.5f}")

    headline = [
        "sliced_w2_complex",
        "w2_amplitude",
        "w2_phase_circular",
        "dependence_gap",
        "spatial_lag1_gap",
    ]
    print("\n" + "=" * 120)
    print("GENERATIVE EVALUATION")
    print("=" * 120)
    print(format_table(rows, headline))
    print("-" * 120)
    print(f"straightness ({coupling_name} pairing): {straightness_value:.5f}")
    print(
        f"dependence: reference {rows[0][1]['dependence_reference']:.4f}, "
        f"generated {rows[0][1]['dependence_generated']:.4f} at {nfe[0]} step(s)"
    )
    print(
        f"spatial lag-1: reference {rows[0][1]['spatial_lag1_reference']:.4f}, "
        f"generated {rows[0][1]['spatial_lag1_generated']:.4f} at {nfe[0]} step(s)"
    )
    print("\n" + "-" * 120)
    print(
        "ANGULAR VELOCITY ALONG THE PATH   (cylinder is bounded by pi = 3.1416; the plane is not)"
    )
    print(f"{'steps':>7}{'NFE':>7}{'median':>12}{'max':>12}{'near t=0.5':>14}{'min |z|':>10}")
    for steps in nfe:
        p = probes[steps]
        print(
            f"{steps:>7}{2 * steps - 1:>7}"
            f"{p['peak_angular_velocity_median']:>14.4f}{p['peak_angular_velocity_max']:>12.4f}"
            f"{p['peak_angular_velocity_near_t_half']:>18.4f}{p['min_amplitude_min']:>10.5f}"
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
        "straightness": straightness_value,
        "straightness_pairing": coupling_name,
        "solver": "heun",
        "note_on_nfe": (
            "The production solver is Heun, a two-evaluation predictor-corrector, "
            "and the last step skips the corrector. Function evaluations are "
            "therefore 2 * num_steps - 1, not num_steps: 1, 3, 7, 15, 199 for "
            "num_steps 1, 2, 4, 8, 100."
        ),
        "sweep": [
            {"num_steps": steps, "nfe": 2 * steps - 1, **values, **probes[steps]}
            for steps, values in rows
        ],
    }
    destination = output_dir / "metrics.json"
    destination.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {destination}")


if __name__ == "__main__":
    main()
