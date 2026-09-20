"""Sliced W2 of the raw prior against the data: no model, no integration.

This answers the reviewer question "what does the noise alone give", which is the
denominator every few-step claim is implicitly measured against. A single Euler
step is only impressive if it beats not running the model at all.

Written because the prior control quoted in the September 2026 notes (synthetic
0.1502, speech 0.3781, knee 0.2239) had no source anywhere in the repository. No
zero-step evaluation path existed, no ``prior`` arm appears in
``docs/reproduce/paper_results/``, and the only occurrences of those values in the
git history are *different metrics on trained arms* -- 0.1502 as
``w2_phase_circular`` at ``num_steps: 8``, 0.2239 as an amplitude error at
``k = 1``. So the figures had to be measured rather than recovered.

**Measured, the synthetic row reproduces:** 0.1498 against the notes' 0.1502, and a phase
prior of 0.3173 against their 0.3238. Those collisions were coincidence, and the missing
provenance was not evidence of error. The speech and knee rows are still unchecked, because
their stores are not local.

The protocol mirrors :mod:`cyfm.evaluate` rather than reimplementing it: the same
dataset registry, the same ``training_pipeline`` (which divides every field by its
own peak modulus -- scoring against raw fields biases every absolute number, as
that function's own docstring records), the same ``manifold.sample_noise``, the
same ``to_complex``, the same ``distributional_metrics``, and the same
``num_fields`` / ``batch_size`` / ``num_projections`` / ``seed`` defaults as
``conf/evaluate/default.yaml``. That is what makes the output comparable with the
``k = 1`` column of Table 2 instead of merely similar to it.

The prior is drawn from the cylindrical manifold, but the metric lives on complex
fields and both geometries share one prior, so the number is geometry-independent.

**The reference population is the caller's problem, deliberately.** ``cyfm.evaluate``
never touches ``dataset.role``; ``scripts/paper/run_arm.sh`` supplies it per arm through
its ``EVAL_ONLY`` overrides, and the two file-backed cohorts do not agree on what
held-out means. Speech hashes one store by speaker, so its evaluation reads
``role=holdout``. Knee MRI does not hash at all: training reads ``train.h5`` and the
evaluation scores against ``val.h5`` with ``role=all``, which is fastMRI's own
patient-disjoint split and the protocol the score-based-prior literature reports. So this
script hardcodes neither, and forcing ``role=holdout`` on the knee would measure the prior
against a different population than its table's reference fields.

Usage::

    uv run python scripts/prior_control.py
    uv run python scripts/prior_control.py --side 16
    uv run python scripts/prior_control.py --dataset librispeech_stft --role holdout
    uv run python scripts/prior_control.py --dataset fastmri_knee_pd --role all \
        --store val.h5 --data-dir "$PDDIR/CyFM/data/knee_pd"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from cyfm.data import build_dataset
from cyfm.evaluate import training_pipeline
from cyfm.manifolds import build_manifold
from cyfm.utils.metrics import distributional_metrics


def parse_args() -> argparse.Namespace:
    """Command-line arguments; every default matches the paper's protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=str,
        default="cylinder_toy_field",
        help="A conf/dataset/ group name, loaded as that file rather than reconstructed here.",
    )
    parser.add_argument(
        "--role",
        type=str,
        default=None,
        help="Override dataset.role. The configs ship role=fit, so measuring against a "
        "table's reference fields needs the role that table's evaluation used: holdout for "
        "speech, all together with --store val.h5 for knee MRI.",
    )
    parser.add_argument(
        "--store", type=str, default=None, help="Override dataset.store, e.g. val.h5."
    )
    parser.add_argument("--data-dir", type=str, default=None, help="Override dataset.data_dir.")
    parser.add_argument(
        "--side",
        type=int,
        default=None,
        help="Override crop_size with a square side. Left unset the config decides, which is "
        "what the file-backed cohorts need: their fields are cut at build time.",
    )
    parser.add_argument(
        "--coupling",
        type=float,
        default=0.5,
        help="Copula correlation of the synthetic target. conf/dataset/cylinder_toy_field.yaml "
        "ships 0.0 and conf/experiment/paper_unet.yaml overrides it to 0.5, so the protocol's "
        "value has to be restated here. Ignored by configs that have no such key.",
    )
    parser.add_argument(
        "--correlation-length",
        type=float,
        default=4.0,
        help="Latent smoothing sigma in pixels. Ignored by configs that have no such key.",
    )
    parser.add_argument("--num-fields", type=int, default=64, help="Fields compared per side.")
    parser.add_argument("--batch-size", type=int, default=16, help="Fields per prior draw.")
    parser.add_argument(
        "--num-projections", type=int, default=256, help="Directions for the sliced estimator."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seeds the prior, projections and subsampling."
    )
    parser.add_argument("--device", type=str, default=None, help="Override the device.")
    return parser.parse_args()


def cylinder() -> Any:
    """The cylindrical manifold, built the way the entry points build it."""
    return build_manifold(
        OmegaConf.create(
            {
                "manifold": {"name": "cylindrical", "spatial_correlation": None},
                "training": {"loss": {}},
            }
        )
    )


def dataset_config(args: argparse.Namespace) -> DictConfig:
    """Load a ``conf/dataset/`` group and apply the caller's evaluation overrides.

    The group is read from the file rather than rebuilt here, so a change to the
    shipped protocol reaches this script without an edit.

    Args:
        args: Parsed arguments.

    Returns:
        The dataset config group, with only the keys the caller asked to change.

    Raises:
        ValueError: If the named group does not exist or is not a mapping.
    """
    path = Path(__file__).resolve().parent.parent / "conf" / "dataset" / f"{args.dataset}.yaml"
    if not path.is_file():
        raise ValueError(f"no such dataset group: {path}")
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise ValueError(f"dataset group must be a mapping, got {type(cfg).__name__}")

    overrides = {
        "role": args.role,
        "store": args.store,
        "data_dir": args.data_dir,
        "coupling": args.coupling if "coupling" in cfg else None,
        "correlation_length": args.correlation_length if "correlation_length" in cfg else None,
    }
    for key, value in overrides.items():
        if value is not None:
            cfg[key] = value
    if args.side is not None:
        cfg["crop_size"] = [args.side, args.side]
    return cfg


def reference_fields(
    args: argparse.Namespace, dataset_cfg: DictConfig, manifold: Any, device: torch.device
) -> torch.Tensor:
    """Draw the data cohort as complex fields, in the domain training used.

    Args:
        args: Parsed arguments.
        dataset_cfg: The dataset config group to instantiate.
        manifold: Geometry whose representation pipeline training used.
        device: Device to stage the states on.

    Returns:
        Complex reference fields of shape ``[num_fields, 1, H, W]`` on the CPU.

    Raises:
        ValueError: If the dataset yields no samples.
    """
    dataset = build_dataset(dataset_cfg, transform=training_pipeline(dataset_cfg, manifold))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    batches: list[torch.Tensor] = []
    collected = 0
    for batch in loader:
        batches.append(batch)
        collected += batch.shape[0]
        if collected >= args.num_fields:
            break
    if not batches:
        raise ValueError("dataset yielded no samples to compare against")

    states = torch.cat(batches, dim=0)[: args.num_fields].to(device)
    return manifold.to_complex(states).cpu()


def prior_fields(
    args: argparse.Namespace,
    manifold: Any,
    device: torch.device,
    height: int,
    width: int,
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    """Draw the untouched prior as complex fields, in the same batches evaluation uses.

    Args:
        args: Parsed arguments.
        manifold: Geometry supplying the prior.
        device: Device to draw on.
        height: Field height.
        width: Field width.
        count: Number of fields to draw.
        generator: RNG, so the draw is reproducible.

    Returns:
        Complex prior fields of shape ``[count, 1, H, W]`` on the CPU.
    """
    batches: list[torch.Tensor] = []
    remaining = count
    while remaining > 0:
        size = min(args.batch_size, remaining)
        noise = manifold.sample_noise(size, height, width, device, generator=generator)
        batches.append(manifold.to_complex(noise).cpu())
        remaining -= size
    return torch.cat(batches, dim=0)


def main() -> None:
    """Measure and report the prior baseline."""
    args = parse_args()
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    manifold = cylinder()
    dataset_cfg = dataset_config(args)
    reference = reference_fields(args, dataset_cfg, manifold, device)
    _, _, height, width = reference.shape

    device_generator = torch.Generator(device=device).manual_seed(args.seed)
    metric_generator = torch.Generator().manual_seed(args.seed)
    prior = prior_fields(
        args, manifold, device, height, width, reference.shape[0], device_generator
    )

    metrics = distributional_metrics(
        generated=prior,
        reference=reference,
        num_projections=args.num_projections,
        generator=metric_generator,
    )

    provenance = ", ".join(
        f"{key} {dataset_cfg[key]}"
        for key in ("role", "store", "coupling", "correlation_length")
        if key in dataset_cfg
    )
    print(
        f"prior vs data: {args.dataset}, {reference.shape[0]} fields of {height}x{width}, "
        f"{provenance}, seed {args.seed}"
    )
    print(f"  sliced_w2_complex  = {metrics['sliced_w2_complex']:.4f}")
    print(f"  w2_amplitude       = {metrics['w2_amplitude']:.4f}")
    print(f"  w2_phase_circular  = {metrics['w2_phase_circular']:.4f}")


if __name__ == "__main__":
    main()
