"""Amplitude and phase panels from the knee store, and a number for whether the phase is noise.

Issue #76, step 4. A wrong coil combination can leave the magnitude looking like a
knee while scrambling the phase, and the phase is what this paper models, so the phase
is what has to be checked before anything is trained on the store.

Inside anatomy the phase of a correct adjoint SENSE combination changes slowly from one
pixel to the next. Independent, uniformly random phase gives a mean absolute wrapped
difference between neighbours of exactly pi/2. The script reports that statistic
inside the anatomy -- pixels above a share of the slice's peak amplitude, because the
background is noise in any correct image -- for the central slice of every volume, so
the verdict rests on the whole store rather than on the few panels drawn.

Usage::

    uv run python scripts/data/knee_store_panels.py \
        --store data/knee_pd/val.h5 --out outputs/gate/knee_panels.png
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Any

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 - the backend has to be chosen first
import numpy as np

NOISE_STEP = math.pi / 2
# Above this share of NOISE_STEP the phase is called noise-like. A heuristic, not a
# derived bound: smooth phase sits far below it and scrambled phase sits at it.
NOISE_RATIO = 0.8


def adjacent_phase_step(field: np.ndarray, threshold: float) -> float:
    """Mean absolute phase change between neighbouring pixels inside the anatomy.

    Args:
        field: Complex slice ``[H, W]``.
        threshold: Share of the slice's peak amplitude a pixel needs to count.

    Returns:
        Mean ``|angle(z_i * conj(z_j))|`` over horizontally and vertically adjacent
        pairs that are both above the threshold; ``nan`` if no pair is.
    """
    amplitude = np.abs(field)
    mask = amplitude > threshold * amplitude.max()
    across = np.abs(np.angle(field[:, 1:] * np.conj(field[:, :-1])))[mask[:, 1:] & mask[:, :-1]]
    down = np.abs(np.angle(field[1:, :] * np.conj(field[:-1, :])))[mask[1:, :] & mask[:-1, :]]
    steps = np.concatenate([across, down])
    return float(steps.mean()) if steps.size else float("nan")


def central_slices(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The middle stored slice of every volume, in volume order."""
    by_volume: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_volume.setdefault(str(record["volume"]), []).append(record)
    return [
        sorted(rows, key=lambda record: int(record["slice"]))[len(rows) // 2]
        for _, rows in sorted(by_volume.items())
    ]


def draw(
    fields: list[tuple[dict[str, Any], np.ndarray, float]], threshold: float, out: pathlib.Path
) -> None:
    """Amplitude, phase, and phase inside the anatomy, one column per slice."""
    columns = len(fields)
    figure, axes = plt.subplots(3, columns, figsize=(2.6 * columns, 8.4), squeeze=False)
    labels = ("amplitude", "phase", f"phase where |z| > {threshold:g} peak")
    for column, (record, field, step) in enumerate(fields):
        amplitude = np.abs(field)
        phase = np.angle(field)
        inside = np.where(amplitude > threshold * amplitude.max(), phase, np.nan)
        axes[0, column].imshow(amplitude, cmap="gray", vmin=0, vmax=np.percentile(amplitude, 99.5))
        axes[1, column].imshow(phase, cmap="twilight", vmin=-math.pi, vmax=math.pi)
        axes[2, column].imshow(inside, cmap="twilight", vmin=-math.pi, vmax=math.pi)
        axes[0, column].set_title(
            f"{record['volume']}\nslice {record['slice']}  step {step:.2f}", fontsize=8
        )
        for row in range(3):
            axes[row, column].axis("off")
    for row, label in enumerate(labels):
        axes[row, 0].text(
            -0.06,
            0.5,
            label,
            transform=axes[row, 0].transAxes,
            rotation=90,
            va="center",
            ha="right",
            fontsize=9,
        )
    figure.suptitle(
        f"knee store, central slices -- noise would give step {NOISE_STEP:.2f}", fontsize=10
    )
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=120)
    plt.close(figure)


def main() -> int:
    """Draw the panel of sample fields from a built knee store."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--store", required=True, help="The store, e.g. data/knee_pd/val.h5.")
    parser.add_argument("--out", required=True, help="PNG to write.")
    parser.add_argument("--count", type=int, default=6, help="Slices to draw.")
    parser.add_argument(
        "--threshold", type=float, default=0.1, help="Share of peak amplitude counted as anatomy."
    )
    args = parser.parse_args()

    store = pathlib.Path(args.store)
    manifest = json.loads(store.with_suffix(".json").read_text())
    chosen = central_slices(manifest["slices"])
    picks = sorted(
        {int(i) for i in np.linspace(0, len(chosen) - 1, min(args.count, len(chosen))).round()}
    )

    steps: list[float] = []
    fields: list[tuple[dict[str, Any], np.ndarray, float]] = []
    with h5py.File(store, "r") as handle:
        images = handle["images"]
        for index, record in enumerate(chosen):
            field = np.asarray(images[int(record["row"])])
            steps.append(adjacent_phase_step(field, args.threshold))
            if index in picks:
                fields.append((record, field, steps[-1]))

    draw(fields, args.threshold, pathlib.Path(args.out))

    finite = np.array([step for step in steps if math.isfinite(step)])
    if finite.size == 0:
        print("no slice had pixels above the threshold; lower --threshold")
        return 1
    median = float(np.median(finite))
    ratio = median / NOISE_STEP
    print(
        f"phase_step_median={median:.3f} phase_step_min={finite.min():.3f} "
        f"phase_step_max={finite.max():.3f} noise_reference={NOISE_STEP:.3f} "
        f"ratio={ratio:.2f} volumes={finite.size}"
    )
    print(f"panels {args.out}")
    if ratio > NOISE_RATIO:
        print(
            "VERDICT: the phase is noise-like inside the anatomy -- check the coil "
            "combination before training anything on this store."
        )
        return 1
    print("VERDICT: the phase is spatially coherent inside the anatomy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
