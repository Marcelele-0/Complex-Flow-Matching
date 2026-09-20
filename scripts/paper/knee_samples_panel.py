"""Generated knee wavefields beside real ones: amplitude and phase, one row per arm.

The manuscript argues about the spatial structure of synthesised complex MRI fields and
shows none of them. Every entry of Table 5 is a summary statistic, and a reader who wants
to know what "reproduces the reference's amplitude texture twice as closely" looks like has
to take it on trust. This draws the panel that lets them check.

Real fields come from the held-out store through :func:`cyfm.evaluate.training_pipeline`,
which is the transform training applied, so the comparison is between fields living in one
domain -- the same reason that function exists rather than a hand-built ``Compose`` here.
Generated fields come from a checkpoint resolved by run name and are drawn with the
manifold's own solver at the requested step counts, with one shared prior seed across arms
so a difference in the picture is a difference in the model and not in the noise.

Amplitude and phase are drawn as two blocks side by side, one row per condition, because
phase is the part the eye is worst at: an amplitude panel can look like anatomy while its
phase is noise, which is the failure the pooled $W_2$ cannot see and ``phase_lag_one`` was
added to catch. Phase gets a cyclic colormap because it is an angle, and every panel shares
one amplitude scale (printed with the figure) so no arm is flattered by its own normalisation.

Usage::

    uv run python scripts/paper/knee_samples_panel.py
    uv run python scripts/paper/knee_samples_panel.py --columns 4 --steps 1 8 100
"""

from __future__ import annotations

import argparse
import pathlib
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from cyfm.data import build_dataset  # noqa: E402
from cyfm.evaluate import training_pipeline  # noqa: E402
from cyfm.manifolds import build_manifold  # noqa: E402
from cyfm.utils.inference import build_model, load_weights  # noqa: E402

# The two arms of the knee 64x64 block whose checkpoints are in the repository, with the
# geometry each was trained on. Both come from scripts/wcss/table5_fastmri.sbatch with
# dataset.kspace_crop=64; the geometry is not stored in the weights, so naming it here is
# what keeps a 3-channel cylindrical checkpoint from being loaded as a 2-channel plane.
ARMS: tuple[tuple[str, str, str], ...] = (
    ("t5c64_cylindrical_ot_s0", "cylindrical", "CyFM + OT"),
    ("t5c64_euclidean_independent_s0", "euclidean", "Cartesian"),
)


def parse_args() -> argparse.Namespace:
    """Command-line arguments; the data defaults match the Table 5 evaluation protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=str, default="data/knee_pd", help="Store directory.")
    parser.add_argument(
        "--store",
        type=str,
        default="val.h5",
        help="Store to read. val.h5 with --role all is fastMRI's own patient-disjoint "
        "split, which is what the knee arms were scored against.",
    )
    parser.add_argument("--role", type=str, default="all", help="Volume-split role.")
    parser.add_argument(
        "--kspace-crop",
        type=int,
        default=64,
        help="Acquisition matrix. Must match the arms: these checkpoints were trained at 64.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        default=[1, 100],
        help="Solver step counts to show, one row per arm and count.",
    )
    parser.add_argument("--columns", type=int, default=3, help="Fields shown per row.")
    parser.add_argument("--seed", type=int, default=0, help="Seeds the prior draw.")
    parser.add_argument("--device", type=str, default=None, help="Override the device.")
    parser.add_argument(
        "--out",
        type=str,
        default="docs/notes/figures/knee_samples_64.png",
        help="Where to write the panel.",
    )
    return parser.parse_args()


def manifold_for(name: str) -> Any:
    """Build one geometry the way the entry points build it.

    Args:
        name: Manifold group name, ``cylindrical`` or ``euclidean``.

    Returns:
        The manifold.
    """
    return build_manifold(
        OmegaConf.create(
            {
                "manifold": {"name": name, "spatial_correlation": None},
                "training": {"loss": {}},
            }
        )
    )


def dataset_config(args: argparse.Namespace) -> DictConfig:
    """Load ``conf/dataset/fastmri_knee_pd.yaml`` and apply the evaluation overrides.

    Read from the file rather than rebuilt here, so a change to the shipped protocol
    reaches this script without an edit -- the same contract as
    ``scripts/prior_control.py``.

    Args:
        args: Parsed arguments.

    Returns:
        The dataset config group with the caller's overrides applied.

    Raises:
        ValueError: If the group is missing or is not a mapping.
    """
    root = pathlib.Path(__file__).resolve().parents[2]
    path = root / "conf" / "dataset" / "fastmri_knee_pd.yaml"
    if not path.is_file():
        raise ValueError(f"no such dataset group: {path}")
    cfg = OmegaConf.load(path)
    if not isinstance(cfg, DictConfig):
        raise ValueError(f"dataset group must be a mapping, got {type(cfg).__name__}")
    cfg["data_dir"] = args.data_dir
    cfg["store"] = args.store
    cfg["role"] = args.role
    cfg["kspace_crop"] = args.kspace_crop
    return cfg


def resolve_checkpoint(run_name: str) -> pathlib.Path:
    """The checkpoint of a run's newest attempt, highest epoch.

    ``cyfm.utils.inference.find_latest_checkpoint`` picks by modification time, which is
    right on the cluster and wrong here: rsync preserves mtimes, so a run copied back in
    two stamped directories -- ``t5c64_cylindrical_ot_s0`` has two, from a superseded job
    array -- would be resolved by whichever transfer happened to touch last. Newest stamp,
    then highest epoch, is the same choice made deliberately.

    Args:
        run_name: Training run directory under ``outputs/train``.

    Returns:
        Path to the checkpoint.

    Raises:
        FileNotFoundError: If no checkpoint is present, which usually means the run is
            still only on WCSS.
    """
    root = pathlib.Path("outputs/train") / run_name
    stamps = sorted(p for p in root.glob("*") if (p / "checkpoints").is_dir())
    for stamp in reversed(stamps):
        found = sorted(
            (stamp / "checkpoints").glob("checkpoint_epoch_*.pt"),
            key=lambda p: int(p.stem.rsplit("_", 1)[1]),
        )
        if found:
            return found[-1]
    raise FileNotFoundError(
        f"no checkpoint under {root}. The knee arms train on WCSS; copy one back with "
        "rsync --no-perms --no-owner --no-group before drawing the panel."
    )


def real_fields(args: argparse.Namespace, manifold: Any, count: int) -> torch.Tensor:
    """Held-out slices, in the domain training used.

    The slices are taken evenly spaced through the store rather than as the first batch.
    The store holds the central 11 slices of each volume in order, so consecutive indices
    are neighbouring cuts of one knee: a first-batch panel shows the same joint three
    times and invites the reader to conclude the reference distribution is narrow.

    Args:
        args: Parsed arguments.
        manifold: Geometry supplying the representation pipeline. The metric domain is
            complex and both geometries normalise identically, so which one is passed
            does not change the fields that come back.
        count: Fields to take.

    Returns:
        Complex fields of shape ``[count, 1, H, W]`` on the CPU.

    Raises:
        ValueError: If the store yields nothing.
    """
    cfg = dataset_config(args)
    dataset = build_dataset(cfg, transform=training_pipeline(cfg, manifold))
    if len(dataset) == 0:
        raise ValueError(f"store {cfg['store']} yielded no slices with role={cfg['role']}")
    stride = max(1, len(dataset) // count)
    indices = [i * stride for i in range(count)][:count]
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=count, shuffle=False, num_workers=0)
    print(f"reference: {len(dataset)} slices in {cfg['store']}, showing indices {indices}")
    return manifold.to_complex(next(iter(loader)))


def generated_fields(
    run_name: str,
    geometry: str,
    steps: int,
    count: int,
    side: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    """Sample one arm at one step count.

    Args:
        run_name: Training run whose checkpoint to load.
        geometry: Manifold the run was trained on.
        steps: Solver steps.
        count: Fields to draw.
        side: Field height and width.
        seed: Prior seed, shared across arms so the rows are comparable.
        device: Device to sample on.

    Returns:
        Complex fields of shape ``[count, 1, side, side]`` on the CPU.
    """
    manifold = manifold_for(geometry)
    model = build_model(
        OmegaConf.create({"model": {"name": "c_unet", "base_channels": 64}}),
        device,
        in_channels=manifold.state_channels,
        out_channels=manifold.velocity_channels,
    )
    load_weights(model, str(resolve_checkpoint(run_name)), device)
    solver = manifold.make_solver(steps)
    generator = torch.Generator(device=device).manual_seed(seed)
    prior = manifold.sample_noise(count, side, side, device, generator=generator)
    with torch.no_grad():
        state = solver.sample(manifold.wrap_model(model), prior, generator=generator)
    return manifold.to_complex(state).cpu()


def draw(rows: list[tuple[str, torch.Tensor]], out: pathlib.Path, side: int, vmax: float) -> None:
    """Write the panel: amplitude block left, phase block right, one row per condition.

    Args:
        rows: Labelled batches, drawn top to bottom.
        out: File to write.
        side: Acquisition matrix, for the title.
        vmax: Shared amplitude ceiling.
    """
    columns = min(batch.shape[0] for _, batch in rows)
    width_ratios = [1.0] * columns + [0.35] + [1.0] * columns
    fig, axes = plt.subplots(
        len(rows),
        2 * columns + 1,
        figsize=(1.05 * (2 * columns + 1), 1.15 * len(rows) + 0.7),
        gridspec_kw={"width_ratios": width_ratios, "hspace": 0.06, "wspace": 0.06},
        squeeze=False,
    )
    for row, (label, batch) in enumerate(rows):
        axes[row][columns].axis("off")
        for index in range(columns):
            field = batch[index, 0]
            left, right = axes[row][index], axes[row][columns + 1 + index]
            left.imshow(field.abs(), cmap="gray", vmin=0.0, vmax=vmax)
            right.imshow(field.angle(), cmap="twilight", vmin=-torch.pi, vmax=torch.pi)
            for axis in (left, right):
                axis.set_xticks([])
                axis.set_yticks([])
        axes[row][0].set_ylabel(label, fontsize=7, rotation=0, ha="right", va="center")
    axes[0][0].set_title("amplitude", fontsize=8, loc="left")
    axes[0][columns + 1].set_title("phase", fontsize=8, loc="left")
    fig.suptitle(
        f"fastMRI knee CORPD, {side}x{side} acquisition matrix, unconditional samples "
        f"(amplitude on a shared scale, 0 to {vmax:.2f})",
        fontsize=8,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"wrote {out}")


def main() -> None:
    """Draw real and generated knee fields in one panel."""
    args = parse_args()
    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    reference = real_fields(args, manifold_for("cylindrical"), args.columns)
    side = reference.shape[-1]
    if side != args.kspace_crop:
        raise ValueError(
            f"store fields are {side}x{side} but --kspace-crop is {args.kspace_crop}; the "
            "arms were trained at the crop and their samples would not be comparable"
        )
    rows: list[tuple[str, torch.Tensor]] = [("real\n(held out)", reference)]
    for run_name, geometry, label in ARMS:
        for steps in args.steps:
            rows.append(
                (
                    f"{label}\nk = {steps}",
                    generated_fields(
                        run_name, geometry, steps, args.columns, side, args.seed, device
                    ),
                )
            )
    # One ceiling for every panel, taken as a high quantile rather than the maximum so a
    # single hot coefficient in one arm cannot darken every other picture in the figure.
    vmax = max(float(batch.abs().flatten().quantile(0.995)) for _, batch in rows)
    for label, batch in rows:
        print(
            f"  {label.replace(chr(10), ' '):<22} amplitude mean {float(batch.abs().mean()):.4f}"
            f"  peak {float(batch.abs().max()):.4f}"
        )
    draw(rows, pathlib.Path(args.out), side, vmax)


if __name__ == "__main__":
    main()
