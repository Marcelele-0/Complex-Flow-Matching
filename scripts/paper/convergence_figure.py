r"""Two-panel convergence figure: error against solver steps, on the two domains of Table 2.

The table gives the numbers; this gives the shape. What a column of numbers does not make
obvious is how far apart the geometries are in the few-step regime and how completely that
gap closes by convergence -- the two facts the section is about. It covers the same two
domains as Table~\ref{tab:unet_baseline} and is read beside it, not instead of it: the
table keeps the exact values, the separation marks and $k^\star$.

Read from the frozen archives rather than from the manuscript, so the figure cannot drift
away from the tables it sits beside.

**Per-seed bands, not error bars.** The shaded region is the min-to-max envelope over the
five seeds. Two arms whose bands do not touch are separated in the sense the paper uses --
every seed of one below every seed of the other -- so the figure carries the same
statement as an asterisk in the table, and carries it continuously in the step count
rather than at the sampled points alone.

Usage::

    uv run python scripts/paper/convergence_figure.py
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS = ROOT / "docs" / "reproduce" / "paper_results"
STEPS = (1, 2, 4, 8, 100)

# Categorical slots 1-4 of the validated palette, assigned in fixed order and never
# cycled. Marker and dash differ per arm as well: the figure has to survive greyscale
# printing and colour-vision deficiency, where hue alone carries nothing.
ARMS = (
    ("cylindrical", "ot", "CyFM + OT (ours)", "#2a78d6", "o", "-"),
    ("cylindrical", "independent", "CyFM, independent", "#1baf7a", "s", "--"),
    ("euclidean", "ot", "Cartesian + OT", "#eb6834", "^", "-."),
    ("euclidean", "independent", "Cartesian, independent", "#eda100", "D", ":"),
)


def synthetic_key(geometry: str, coupling: str, seed: int) -> str:
    """Archive key for one synthetic arm at 64x64."""
    return f"un_{geometry}_{coupling}_scnull_s{seed}_eval"


def speech_key(geometry: str, coupling: str, seed: int) -> str:
    """Archive key for one speech arm; the published cohort trained for 40 epochs."""
    return f"t6e40_{geometry}_{coupling}_s{seed}_eval"


def series(archive: dict, key_of, geometry: str, coupling: str) -> tuple[list[float], ...]:
    """Mean, minimum and maximum over seeds at each step count.

    Args:
        archive: A loaded archive.
        key_of: Builds the archive key from geometry, coupling and seed.
        geometry: Manifold name.
        coupling: Coupling name.

    Returns:
        Three lists parallel to ``STEPS``: the seed mean, minimum and maximum.
    """
    mean, low, high = [], [], []
    for step in STEPS:
        values = [
            next(
                row["sliced_w2_complex"]
                for row in archive[key_of(geometry, coupling, seed)]["sweep"]
                if row["num_steps"] == step
            )
            for seed in range(5)
        ]
        mean.append(statistics.fmean(values))
        low.append(min(values))
        high.append(max(values))
    return mean, low, high


def panel(axis, archive: dict, key_of, title: str) -> None:
    """Draw one domain: four arms, each a mean line over its per-seed envelope.

    Args:
        axis: Matplotlib axes to draw on.
        archive: The archive holding this domain.
        key_of: Key builder for this domain.
        title: Panel title.
    """
    for geometry, coupling, label, colour, marker, dash in ARMS:
        mean, low, high = series(archive, key_of, geometry, coupling)
        axis.fill_between(STEPS, low, high, color=colour, alpha=0.15, linewidth=0)
        axis.plot(
            STEPS,
            mean,
            color=colour,
            marker=marker,
            linestyle=dash,
            linewidth=1.6,
            markersize=4.5,
            label=label,
        )
    # E* is the worst converged error in the block: the level every arm is asked to
    # reach, so that arms converging to different asymptotes are still comparable. The
    # step count at which each curve crosses it is k*, read off a denser grid than the
    # five points plotted here and tabulated rather than marked, since for the synthetic
    # block it falls between them.
    worst = max(
        series(archive, key_of, geometry, coupling)[0][-1] for geometry, coupling, *_ in ARMS
    )
    axis.axhline(worst, color="#52514e", linewidth=0.8, linestyle=(0, (4, 3)), zorder=0)
    axis.annotate(
        r"$E^\star$",
        xy=(1.05, worst),
        xytext=(1.05, worst),
        fontsize=7,
        color="#52514e",
        va="bottom",
    )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xticks(STEPS)
    axis.set_xticklabels([str(s) for s in STEPS])
    axis.set_xlabel("solver steps $k$")
    axis.set_title(title, fontsize=9)
    # Recessive grid: it locates a point without competing with the data.
    axis.grid(True, which="major", linewidth=0.4, alpha=0.3)
    axis.grid(False, which="minor")
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def main() -> None:
    """Write the figure."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=ROOT / "paper" / "ICLR Main" / "figures" / "convergence.pdf",
        help="Where to write the figure; a .pdf keeps the text vector in the manuscript.",
    )
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
        }
    )
    synthetic = json.loads((RESULTS / "unet_eval_metrics_l2u.json").read_text())
    speech = json.loads((RESULTS / "table6_audio_metrics.json").read_text())

    figure, axes = plt.subplots(1, 2, figsize=(5.5, 2.15))
    panel(axes[0], synthetic, synthetic_key, r"Synthetic fields, $64\times64$")
    panel(axes[1], speech, speech_key, r"Speech STFT, $64\times64$")
    axes[0].set_ylabel(r"sliced $W_2$")

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.10),
        handlelength=2.4,
        columnspacing=1.4,
    )
    figure.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
