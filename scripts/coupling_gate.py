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
from dataclasses import dataclass

import torch

from cyfm.data.synthetic import (
    COUPLING_PRESETS,
    CylinderToy,
    Structure,
    cylinder_prior,
)
from cyfm.flow.transport import (
    circular_transport_permutation,
    cylinder_transport_permutation,
    sorted_transport_permutation,
    transport_cost,
    wasserstein2_cylinder,
)
from cyfm.metrics import circular_linear_correlation

# Typed as the Literal the toy accepts, not as bare str: iterating an untyped
# tuple hands CylinderToy a widened `str` and loses the only check that the
# two names here still match the ones it implements.
STRUCTURES: tuple[Structure, ...] = ("spiral", "cardioid")


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


@dataclass(frozen=True)
class Clouds:
    """One draw of the prior, the target, and both coupled endpoint clouds.

    Attributes:
        target_amplitude: Target amplitudes, shape ``[n]``.
        target_phase: Target phases, shape ``[n]``.
        chimera_amplitude: Endpoint amplitudes under the factorised coupling.
        chimera_phase: Endpoint phases under the factorised coupling.
        factorised_cost: Mean transport cost the factorised solution reports.
        joint_cost: Mean transport cost of the exact joint plan.
    """

    target_amplitude: torch.Tensor
    target_phase: torch.Tensor
    chimera_amplitude: torch.Tensor
    chimera_phase: torch.Tensor
    factorised_cost: float
    joint_cost: float


def draw(toy: CylinderToy, n: int, seed: int, phase_weight: float) -> Clouds:
    """Draw one prior/target pair and couple it both ways.

    Args:
        toy: The target sampler.
        n: Batch size.
        seed: Seed for this repeat.
        phase_weight: Weight on the angular term of the product metric.

    Returns:
        The clouds and the two transport costs.
    """
    prior = cylinder_prior(n, generator=torch.Generator().manual_seed(seed))
    amplitude, phase = toy.sample_polar(n, generator=torch.Generator().manual_seed(seed + 7919))
    prior_amplitude, prior_phase = prior.abs(), prior.angle()

    amplitude_perm = sorted_transport_permutation(prior_amplitude, amplitude)
    phase_perm = circular_transport_permutation(prior_phase, phase)
    chimera_amplitude, chimera_phase = amplitude[amplitude_perm], phase[phase_perm]

    joint_perm = cylinder_transport_permutation(
        prior_amplitude, prior_phase, amplitude, phase, phase_weight
    )

    return Clouds(
        target_amplitude=amplitude,
        target_phase=phase,
        chimera_amplitude=chimera_amplitude,
        chimera_phase=chimera_phase,
        factorised_cost=float(
            transport_cost(
                prior_amplitude, prior_phase, chimera_amplitude, chimera_phase, phase_weight
            )
        ),
        joint_cost=float(
            transport_cost(
                prior_amplitude,
                prior_phase,
                amplitude[joint_perm],
                phase[joint_perm],
                phase_weight,
            )
        ),
    )


def marginal_gap(first: torch.Tensor, second: torch.Tensor) -> float:
    """Two-sample Kolmogorov-Smirnov statistic between 1D clouds."""
    first_sorted, second_sorted = first.sort().values, second.sort().values
    grid = torch.cat([first_sorted, second_sorted]).sort().values
    cdf_first = torch.searchsorted(first_sorted, grid, right=True).float() / first.numel()
    cdf_second = torch.searchsorted(second_sorted, grid, right=True).float() / second.numel()
    return float((cdf_first - cdf_second).abs().max())


def report_joint_damage(args: argparse.Namespace) -> None:
    """What every marginal diagnostic misses."""
    print("\n" + "=" * 96)
    print(f"1. MARGINALS SURVIVE, JOINT DOES NOT   (n = {args.samples}, {args.seeds} seeds)")
    print("=" * 96)
    print(
        f"{'structure':<10} {'rho':>5} {'r_cl target':>12} {'r_cl chimera':>13} "
        f"{'KS(A)':>8} {'KS(theta)':>10} {'W2 chimera':>12} {'W2 floor':>10}"
    )
    print("-" * 96)

    for structure in STRUCTURES:
        for coupling in COUPLING_PRESETS.values():
            toy = CylinderToy(coupling=coupling, structure=structure)
            stats = {key: 0.0 for key in ("rt", "rc", "ka", "kp", "w2", "floor")}

            for repeat in range(args.seeds):
                seed = args.seed + 1000 * repeat
                clouds = draw(toy, args.samples, seed, args.phase_weight)
                other_amplitude, other_phase = toy.sample_polar(
                    args.samples, generator=torch.Generator().manual_seed(seed + 104729)
                )

                stats["rt"] += float(
                    circular_linear_correlation(clouds.target_amplitude, clouds.target_phase)
                )
                stats["rc"] += float(
                    circular_linear_correlation(clouds.chimera_amplitude, clouds.chimera_phase)
                )
                stats["ka"] += marginal_gap(clouds.chimera_amplitude, clouds.target_amplitude)
                stats["kp"] += marginal_gap(clouds.chimera_phase, clouds.target_phase)
                stats["w2"] += float(
                    wasserstein2_cylinder(
                        clouds.chimera_amplitude,
                        clouds.chimera_phase,
                        clouds.target_amplitude,
                        clouds.target_phase,
                        args.phase_weight,
                    )
                )
                stats["floor"] += float(
                    wasserstein2_cylinder(
                        other_amplitude,
                        other_phase,
                        clouds.target_amplitude,
                        clouds.target_phase,
                        args.phase_weight,
                    )
                )

            mean = {key: value / args.seeds for key, value in stats.items()}
            print(
                f"{structure:<10} {coupling:>5.2f} {mean['rt']:>12.4f} {mean['rc']:>13.4f} "
                f"{mean['ka']:>8.5f} {mean['kp']:>10.5f} {mean['w2']:>12.4f} "
                f"{mean['floor']:>10.4f}"
            )
        print("-" * 96)
    print(f"{'':<10} {'':>5}  r_cl = circular-linear correlation; KS compares the chimera")
    print(
        f"{'':<10} {'':>5}  cloud's marginals with the target's; W2 floor is an independent draw."
    )


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


def report_cost_gap(args: argparse.Namespace) -> None:
    """The cost the factorised solution reports against the cost that is attainable."""
    print("\n" + "=" * 96)
    print(f"3. THE FACTORISED COST IS NOT ATTAINABLE   (n = {args.samples}, {args.seeds} seeds)")
    print("=" * 96)
    print(f"{'structure':<10} {'rho':>5} {'factorised':>12} {'joint (exact)':>14} {'deficit':>10}")
    print("-" * 96)

    for structure in STRUCTURES:
        for coupling in COUPLING_PRESETS.values():
            toy = CylinderToy(coupling=coupling, structure=structure)
            factorised = joint = 0.0
            for repeat in range(args.seeds):
                clouds = draw(toy, args.samples, args.seed + 1000 * repeat, args.phase_weight)
                factorised += clouds.factorised_cost
                joint += clouds.joint_cost
            factorised /= args.seeds
            joint /= args.seeds
            print(
                f"{structure:<10} {coupling:>5.2f} {factorised:>12.4f} {joint:>14.4f} "
                f"{joint - factorised:>10.4f}"
            )
        print("-" * 96)
    print("  The joint plan is exact, so its cost is the true optimum over feasible plans.")
    print("  Any deficit is the factorised relaxation scoring a plan that does not exist.")


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

    report_joint_damage(args)
    report_bias_versus_noise(args)
    report_cost_gap(args)

    if args.plot is not None:
        write_plot(args, args.plot)


if __name__ == "__main__":
    main()
