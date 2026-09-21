"""Gate A: does a factorised coupling still transport to the target distribution?

The cylindrical coupling described in the project's mathematics note solves two
one-dimensional transport problems -- amplitude on the line, phase on the circle
-- and pairs their solutions. That is only a transport plan between the joint
distributions when both are products of their marginals,

    min_pi Int (c_A + c_theta) dpi
        = min_{pi_A} Int c_A dpi_A  +  min_{pi_theta} Int c_theta dpi_theta,

which requires ``p(A, theta) = p(A) p(theta)`` on both sides. Real k-space and
real spectrograms do not satisfy it. This script measures what that costs, with
no network anywhere: the prior, the target and both couplings are analytic, so
nothing measured here can be attributed to training, to a loss weighting, or to
a reconstruction pipeline.

Three things are reported.

**One: the marginals survive while the joint does not.** The factorised pairing
takes amplitudes from one permutation of the target and phases from another, so
each marginal of the resulting cloud is *exactly* the target's marginal -- the
same multiset, reordered. Every marginal diagnostic therefore passes while the
joint distribution has been replaced. The circular-linear correlation and the
2-Wasserstein distance on the cylinder both see the joint, and both are reported
beside the marginal statistics that miss it.

**Two: whether the damage is bias or sampling noise.** Two finite clouds drawn
from one distribution sit some distance apart, and that distance shrinks as the
batch grows. A bias does not. The sweep over batch size separates the two, with
an independent draw of the target as the noise floor.

**Three: the cost gap.** The factorised solution reports a transport cost below
the true optimum, because it is minimising over a strictly larger set than the
feasible plans. A number that beats the optimum is not a better solution; it is
the signature of an infeasible one.

Usage::

    uv run python scripts/coupling_gate.py
    uv run python scripts/coupling_gate.py --samples 2048 --seeds 16
    uv run python scripts/coupling_gate.py --plot /tmp/gate_a.png
"""

from __future__ import annotations

import argparse

import torch

from cyfm.data.synthetic import (
    COUPLING_PRESETS,
    CylinderToy,
)
from cyfm.experiments.factorised_trap import (
    FactorisedTrapExperiment,
    draw,
    render as render_trap,
)
from cyfm.flow.transport import (
    wasserstein2_cylinder,
)


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        type=int,
        default=1024,
        help="Batch size for the per-coupling diagnostics.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=8,
        help="Independent repeats averaged into every reported number.",
    )
    parser.add_argument(
        "--batch-sweep",
        type=int,
        nargs="+",
        default=(64, 256, 1024, 4096),
        help="Batch sizes for the bias-versus-noise sweep.",
    )
    parser.add_argument(
        "--phase-weight",
        type=float,
        default=1.0,
        help="Weight on the angular term of the product metric; 1.0 is flat.",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Optional path for a scatter of target versus factorised cloud.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base seed.")
    return parser.parse_args()


def report_bias_versus_noise(args: argparse.Namespace) -> None:
    """Whether the damage shrinks with batch size."""
    print("\n" + "=" * 96)
    print(f"2. BIAS OR SAMPLING NOISE   (spiral, {args.seeds} seeds, W2 to the target cloud)")
    print("=" * 96)
    header = "".join(f"{f'n={n}':>12}" for n in args.batch_sweep)
    print(f"{'rho':<8}{header}")
    print("-" * 96)

    for coupling in COUPLING_PRESETS.values():
        toy = CylinderToy(coupling=coupling, structure="spiral")
        row = []
        for n in args.batch_sweep:
            total = 0.0
            for repeat in range(args.seeds):
                clouds = draw(toy, n, args.seed + 1000 * repeat, args.phase_weight)
                total += float(
                    wasserstein2_cylinder(
                        clouds.chimera_amplitude,
                        clouds.chimera_phase,
                        clouds.target_amplitude,
                        clouds.target_phase,
                        args.phase_weight,
                    )
                )
            row.append(total / args.seeds)
        print(f"{coupling:<8.2f}" + "".join(f"{value:>12.4f}" for value in row))

    print("-" * 96)
    print("  A number that falls with n is sampling noise. One that plateaus is a bias:")
    print("  the coupling converges, and it converges to a distribution that is not the target.")


def write_plot(args: argparse.Namespace, path: str) -> None:
    """Scatter the target cloud against the factorised cloud for each coupling.

    Args:
        args: Parsed arguments.
        path: Destination for the PNG.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    couplings = list(COUPLING_PRESETS.values())
    figure, axes = plt.subplots(
        2, len(couplings), figsize=(4 * len(couplings), 7), sharex=True, sharey=True
    )

    for column, coupling in enumerate(couplings):
        toy = CylinderToy(coupling=coupling, structure="spiral")
        clouds = draw(toy, min(args.samples, 2048), args.seed, args.phase_weight)
        panels = (
            ("target", clouds.target_phase, clouds.target_amplitude),
            ("factorised coupling", clouds.chimera_phase, clouds.chimera_amplitude),
        )
        for row, (label, phase, amplitude) in enumerate(panels):
            axis = axes[row][column]
            axis.scatter(phase.numpy(), amplitude.numpy(), s=2, alpha=0.35, linewidths=0)
            axis.set_title(f"{label}, rho = {coupling}", fontsize=10)
            if row == 1:
                axis.set_xlabel("phase (rad)")
            if column == 0:
                axis.set_ylabel("amplitude")

    figure.suptitle(
        "Gate A: the factorised coupling preserves both marginals and destroys the joint",
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    print(f"\nplot written to {path}")


def main() -> None:
    """Run the gate."""
    args = parse_args()
    torch.manual_seed(args.seed)

    print("=" * 96)
    print("GATE A -- is the factorised cylindrical coupling a transport plan at all?")
    print("=" * 96)
    print("prior: circularly symmetric complex Gaussian (Rayleigh amplitude, uniform phase)")
    print("target: CylinderToy, marginals held fixed across every value of rho")
    print(f"phase weight on the product metric: {args.phase_weight}")

    # Sections 1 and 3 are the two Table 4 prints, so they are the two that had
    # to become re-renderable: the measurement lives in FactorisedTrapExperiment
    # and returns its numbers. Section 2 is a diagnostic the paper does not
    # publish and stays here.
    experiment = FactorisedTrapExperiment(
        samples=args.samples,
        seeds=args.seeds,
        seed=args.seed,
        phase_weight=args.phase_weight,
    )
    damage, cost_gap = render_trap(experiment.run())
    print(damage)
    report_bias_versus_noise(args)
    print(cost_gap)

    if args.plot is not None:
        write_plot(args, args.plot)


if __name__ == "__main__":
    main()
