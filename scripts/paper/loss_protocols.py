"""Cylinder against Cartesian under two loss protocols, with per-cell statistics.

Protocols, both on the Tables 2-3 grid (16x16, 32x32, 64x64; k = 1, 2, 4, 8, 100):

* **matched loss** -- both geometries trained with L2 and an unweighted phase
  term (``l2u``); cylinder + joint OT against the better Cartesian coupling.
* **best loss per geometry** -- each geometry takes, per cell, the lowest mean
  over the same four variants {L1, L2} x {independent, joint OT}. For the
  cylinder L1 means ``l1u`` (unweighted phase), the counterpart of Cartesian L1,
  which has no phase weighting. The selection is made on the evaluation itself,
  which flatters both geometries equally.

For each cell it prints the means, the difference with a 95% bootstrap interval,
the exact two-sided Mann-Whitney p and whether the seeds separate. With five
seeds per arm the smallest attainable p is 2/252 = 0.008, reached exactly when
the seeds separate.

Sources: ``outputs/evaluate`` by run-name prefix (default), or with
``--archive`` the three archives in ``docs/paper_results/``.

Usage::

    uv run python scripts/paper/loss_protocols.py
    uv run python scripts/paper/loss_protocols.py --archive
    uv run python scripts/paper/loss_protocols.py --metric w2_phase_circular
"""

from __future__ import annotations

import argparse
import json
import pathlib
from typing import Any

import numpy as np
from scipy.stats import mannwhitneyu

ROOT = pathlib.Path(__file__).resolve().parents[2]
EVALUATIONS = ROOT / "outputs" / "evaluate"
RESULTS = ROOT / "docs" / "paper_results"

SIDES = (16, 32, 64)
STEPS = (1, 2, 4, 8, 100)
COUPLINGS = ("independent", "ot")
SEEDS = (0, 1, 2, 3, 4)
METRICS = ("sliced_w2_complex", "w2_amplitude", "w2_phase_circular", "dependence_gap")

# loss label -> (run-name prefix, archive file, geometries it was run for)
SOURCES: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "l2u": ("l2u_", "unet_eval_metrics_l2u.json", ("cylindrical", "euclidean")),
    "l1u": ("l1u_", "unet_eval_metrics_l1u_cylindrical.json", ("cylindrical",)),
    "l1w": ("l1w_", "unet_eval_metrics_l1w_cartesian.json", ("euclidean",)),
}
# Which loss plays "L1" for each geometry in the best-loss protocol.
L1_LOSS = {"cylindrical": "l1u", "euclidean": "l1w"}

Runs = dict[str, dict[str, Any]]


def run_name(side: int, geometry: str, coupling: str, seed: int) -> str:
    """Unprefixed evaluation name, as written by the table scripts."""
    if side == 64:
        return f"un_{geometry}_{coupling}_scnull_s{seed}_eval"
    return f"unsz_{geometry}_{coupling}_{side}_s{seed}_eval"


def load(archive: bool) -> dict[str, Runs]:
    """loss label -> evaluation name -> metrics."""
    sources: dict[str, Runs] = {}
    for label, (prefix, filename, _) in SOURCES.items():
        if archive:
            sources[label] = json.loads((RESULTS / filename).read_text())
            continue
        runs: Runs = {}
        for path in sorted(EVALUATIONS.glob(f"{prefix}*_eval")):
            found = sorted(path.glob("*/metrics.json"))
            if found:
                runs[path.name.removeprefix(prefix)] = json.loads(found[-1].read_text())
        sources[label] = runs
    return sources


def values(
    sources: dict[str, Runs],
    loss: str,
    side: int,
    geometry: str,
    coupling: str,
    step: int,
    metric: str,
) -> np.ndarray:
    """Per-seed values of one arm; raises if any seed is missing."""
    out = []
    for seed in SEEDS:
        run = sources[loss][run_name(side, geometry, coupling, seed)]
        out.append(next(row[metric] for row in run["sweep"] if row["num_steps"] == step))
    return np.array(out, dtype=float)


def best(
    sources: dict[str, Runs],
    candidates: list[tuple[str, str]],
    side: int,
    geometry: str,
    step: int,
    metric: str,
) -> tuple[str, np.ndarray]:
    """The (loss, coupling) candidate with the lowest mean, and its values."""
    arms = {
        f"{loss} {coupling[:3]}": values(sources, loss, side, geometry, coupling, step, metric)
        for loss, coupling in candidates
    }
    label = min(arms, key=lambda name: float(arms[name].mean()))
    return label, arms[label]


def compare(cyl: np.ndarray, cart: np.ndarray, rng: np.random.Generator) -> str:
    """Difference, bootstrap interval, exact Mann-Whitney p and separation."""
    p = mannwhitneyu(cyl, cart, alternative="two-sided", method="exact").pvalue
    cyl_means = rng.choice(cyl, (20000, len(cyl))).mean(axis=1)
    cart_means = rng.choice(cart, (20000, len(cart))).mean(axis=1)
    diffs = cyl_means - cart_means
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    if cyl.max() < cart.min():
        verdict = "cylinder, separated"
    elif cart.max() < cyl.min():
        verdict = "Cartesian, separated"
    else:
        verdict = "overlap"
    diff = cyl.mean() - cart.mean()
    return f"diff {diff:+.3f} [{lo:+.3f}, {hi:+.3f}]  p={p:.3f}  {verdict}"


def print_protocol(
    title: str,
    sources: dict[str, Runs],
    cylinder: list[tuple[str, str]],
    cartesian: list[tuple[str, str]],
    metric: str,
    rng: np.random.Generator,
) -> None:
    print(f"\n{title}  ({metric}, {len(SEEDS)} seeds per arm)")
    for side in SIDES:
        print(f" {side}x{side}")
        for step in STEPS:
            cyl_label, cyl = best(sources, cylinder, side, "cylindrical", step, metric)
            cart_label, cart = best(sources, cartesian, side, "euclidean", step, metric)
            print(
                f"  k={step:<4} cylinder {cyl.mean():.3f} ({cyl_label:<7})"
                f"  Cartesian {cart.mean():.3f} ({cart_label:<7})  {compare(cyl, cart, rng)}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--archive", action="store_true", help="Read docs/paper_results/.")
    parser.add_argument("--metric", choices=METRICS, default=METRICS[0])
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    args = parser.parse_args()

    sources = load(args.archive)
    rng = np.random.default_rng(args.bootstrap_seed)
    print_protocol(
        "MATCHED LOSS (L2, unweighted phase, both geometries)",
        sources,
        [("l2u", "ot")],
        [("l2u", coupling) for coupling in COUPLINGS],
        args.metric,
        rng,
    )
    print_protocol(
        "BEST LOSS PER GEOMETRY (each picks from {L1, L2} x {independent, OT})",
        sources,
        [(loss, c) for loss in ("l2u", L1_LOSS["cylindrical"]) for c in COUPLINGS],
        [(loss, c) for loss in ("l2u", L1_LOSS["euclidean"]) for c in COUPLINGS],
        args.metric,
        rng,
    )


if __name__ == "__main__":
    main()
