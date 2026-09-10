"""Gate B: does the cylinder plus a correct coupling actually buy anything?

Gate A settled what a coupling must *not* be. This one asks what the geometry and
the coupling are worth once both are correct, in the only setting where the
question can be answered without confounds: single complex numbers, where the
prior, the network, its input encoding, its parameter count and the loss are
identical across arms, and the analytic target lets a distributional metric be
computed exactly.

The design is a 2 x 2 crossed with the dependence knob:

    {euclidean, cylindrical}  x  {independent pairing, batch-level OT}   at rho in {0, 0.5, 1}

Varying the coupling *within* each geometry is what separates "the cylinder
helps" from "optimal transport helps", which no measurement in this project has
done yet. Varying rho asks the question the abandoned coupling-gap narrative was
supposed to answer, but from the other side: how much does a correct coupling buy
as amplitude and phase become dependent?

Two numbers are reported.

**Straightness.** Both bridges carry a conditional velocity that is constant
along the path, so the converged regression residual is the straightness
statistic ``E ||(x_1 - x_0) - v(x_t, t)||^2`` directly. Normalised by the mean
squared displacement, it is comparable across geometries. This is the *mechanism*
claim: a coupling that removes crossings should lower it.

**Distributional error against solver steps.** 2-Wasserstein between generated
and held-out target samples, measured for both arms in the *same* fixed cylinder
metric so the comparison is apples to apples, swept over the number of Heun
steps from ``t = 0``. This is the consequence claim, and unlike every paired
metric on MRI it is meaningful at pure generation.

Usage::

    uv run python scripts/coupling_gate_b.py
    uv run python scripts/coupling_gate_b.py --train-steps 6000 --eval-samples 2048
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Literal

import torch

from cfm.data.synthetic import CylinderToy, cylinder_prior
from cfm.flow.optimal_transport import (
    cylinder_transport_permutation,
    euclidean_transport_permutation,
    wasserstein2_cylinder,
)
from cfm.flow.toy_flow import (
    Geometry,
    ToyVelocityField,
    build_bridge,
    build_solver,
    from_state,
    straightness,
    to_state,
)

# Guards the induced angular velocity against a literal division by zero; the
# divergence it is meant to expose happens well above this.
_ANGULAR_FLOOR = 1e-12

GEOMETRIES: tuple[Geometry, ...] = ("euclidean", "cylindrical")
Coupling = Literal["independent", "ot"]
COUPLINGS: tuple[Coupling, ...] = ("independent", "ot")


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-steps", type=int, default=4000, help="Optimiser steps per arm.")
    parser.add_argument("--batch-size", type=int, default=256, help="Samples per step.")
    parser.add_argument("--width", type=int, default=128, help="Hidden width, identical per arm.")
    parser.add_argument("--depth", type=int, default=3, help="Hidden layers, identical per arm.")
    parser.add_argument("--lr", type=float, default=2e-3, help="Adam learning rate.")
    parser.add_argument(
        "--eval-samples", type=int, default=2048, help="Samples per distributional measurement."
    )
    parser.add_argument(
        "--nfe",
        type=int,
        nargs="+",
        default=(1, 2, 4, 8, 16, 32, 64, 100),
        help="Solver step counts to sweep.",
    )
    parser.add_argument(
        "--phase-weight",
        type=float,
        default=1.0,
        help="Angular weight of the scoring metric and of the cylindrical coupling.",
    )
    parser.add_argument(
        "--rho",
        type=float,
        nargs="+",
        default=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        help="Dependence values to sweep; monotonicity is measured, not assumed.",
    )
    parser.add_argument("--seeds", type=int, default=3, help="Independent repeats per arm.")
    parser.add_argument(
        "--singularity-radius",
        type=float,
        default=0.05,
        help="Amplitude below which a trajectory counts as visiting the chart singularity.",
    )
    parser.add_argument("--structure", type=str, default="spiral", help="Toy dependence structure.")
    parser.add_argument("--seed", type=int, default=0, help="Base seed.")
    parser.add_argument("--device", type=str, default=None, help="Override the device.")
    return parser.parse_args()


@dataclass(frozen=True)
class ArmResult:
    """Everything one trained arm produced.

    Attributes:
        straightness: Normalised regression residual after training.
        distributional: 2-Wasserstein to the target per solver step count.
        min_amplitude: Mean over samples of the smallest amplitude the solver
            visited while integrating, at the finest step count.
        peak_angular: Median over samples of the largest induced angular
            velocity met along the trajectory. Bounded by pi on the cylinder by
            construction; unbounded in the plane.
        singularity_fraction: Fraction of trajectories that came within
            ``--singularity-radius`` of the origin, where the cylinder chart is
            singular and the Cartesian angular velocity diverges.
        seconds: Wall-clock training time.
    """

    straightness: float
    distributional: dict[int, float]
    min_amplitude: float
    peak_angular: float
    singularity_fraction: float
    seconds: float


class _AmplitudeProbe:
    """Wraps a velocity field and records the amplitudes the solver visits.

    Theorem 3 predicts that a Cartesian field's angular velocity diverges as
    ``O(A^-1)``, so the damage it does under coarse integration should be
    governed by how close its trajectories come to the origin. That is a claim
    about the path, not the endpoint, and this is what turns it from a story
    into a measurement.

    Args:
        network: The velocity field to wrap.
        geometry: How to read an amplitude out of the state.
    """

    def __init__(self, network: ToyVelocityField, geometry: Geometry) -> None:
        self.network = network
        self.geometry = geometry
        self.minimum: torch.Tensor | None = None
        self.peak_angular: torch.Tensor | None = None

    def __call__(self, state: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Record the amplitude and the induced angular velocity, then delegate.

        The angular velocity is the quantity Theorem 3 is about. On the cylinder
        it *is* a coordinate of the predicted velocity and is bounded by pi from
        the range of ``atan2``. In the plane it has to be induced,

            theta_dot = (x v_y - y v_x) / A^2,

        which is the ``O(A^-1)`` term. Recording both along the same trajectories
        is what makes the theorem's prediction falsifiable here.

        Args:
            state: State tensor of shape ``[n, C, 1, 1]``.
            t: Times of shape ``[n]``.

        Returns:
            The wrapped field's velocity.
        """
        velocity = self.network(state, t)
        flat = state.reshape(state.shape[0], state.shape[1])
        tangent = velocity.reshape(velocity.shape[0], 2)

        if self.geometry == "euclidean":
            real, imag = flat[:, 0], flat[:, 1]
            squared = (real * real + imag * imag).clamp_min(_ANGULAR_FLOOR)
            amplitude = squared.sqrt()
            angular = (real * tangent[:, 1] - imag * tangent[:, 0]) / squared
        else:
            amplitude = flat[:, 0]
            angular = tangent[:, 1]

        magnitude = angular.abs()
        self.minimum = amplitude if self.minimum is None else torch.minimum(self.minimum, amplitude)
        self.peak_angular = (
            magnitude if self.peak_angular is None else torch.maximum(self.peak_angular, magnitude)
        )
        return velocity


def couple(
    prior: torch.Tensor, target: torch.Tensor, geometry: Geometry, coupling: Coupling, weight: float
) -> torch.Tensor:
    """Reorder the target batch according to the arm's coupling.

    Each geometry receives the optimal coupling *in its own metric*, so the
    comparison varies the geometry rather than handing one arm a better solver.

    Args:
        prior: Complex prior samples, shape ``[n]``.
        target: Complex target samples, shape ``[n]``.
        geometry: Which metric the coupling is solved in.
        coupling: ``"independent"`` leaves the batch as drawn; ``"ot"`` solves
            the exact assignment.
        weight: Angular weight for the cylindrical cost.

    Returns:
        The reordered target, shape ``[n]``.
    """
    if coupling == "independent":
        return target
    if geometry == "euclidean":
        return target[euclidean_transport_permutation(prior, target)]
    permutation = cylinder_transport_permutation(
        prior.abs(), prior.angle(), target.abs(), target.angle(), weight
    )
    return target[permutation]


def train_arm(
    toy: CylinderToy,
    geometry: Geometry,
    coupling: Coupling,
    args: argparse.Namespace,
    device: torch.device,
    seed: int,
) -> ArmResult:
    """Train one arm and measure it.

    Args:
        toy: The target sampler.
        geometry: Which geometry this arm uses.
        coupling: Which coupling this arm uses.
        args: Parsed arguments.
        device: Device to train on.
        seed: Seed for this arm.

    Returns:
        The arm's measurements.
    """
    torch.manual_seed(seed)
    network = ToyVelocityField(geometry, width=args.width, depth=args.depth).to(device)
    optimiser = torch.optim.Adam(network.parameters(), lr=args.lr)
    bridge = build_bridge(geometry)

    started = time.perf_counter()
    for step in range(args.train_steps):
        generator = torch.Generator().manual_seed(seed + step)
        prior = cylinder_prior(args.batch_size, generator=generator)
        target = couple(
            prior,
            toy.sample(args.batch_size, generator=generator),
            geometry,
            coupling,
            args.phase_weight,
        )

        start_state = to_state(prior, geometry).to(device)
        end_state = to_state(target, geometry).to(device)
        times = torch.rand(args.batch_size, 1, 1, 1, device=device)
        interpolated, velocity_target = bridge.forward(start_state, end_state, times)

        loss = (network(interpolated, times.reshape(-1)) - velocity_target).square().mean()
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    seconds = time.perf_counter() - started

    network.eval()
    with torch.no_grad():
        generator = torch.Generator().manual_seed(seed + 999_983)
        prior = cylinder_prior(args.eval_samples, generator=generator)
        target = couple(
            prior,
            toy.sample(args.eval_samples, generator=generator),
            geometry,
            coupling,
            args.phase_weight,
        )
        start_state = to_state(prior, geometry).to(device)
        end_state = to_state(target, geometry).to(device)
        times = torch.rand(args.eval_samples, 1, 1, 1, device=device)
        interpolated, velocity_target = bridge.forward(start_state, end_state, times)
        residual = (
            (network(interpolated, times.reshape(-1)) - velocity_target).square().flatten(1).sum(1)
        )
        displacement = velocity_target.square().flatten(1).sum(1)
        measured_straightness = float(straightness(residual, displacement))

        # Distributional error is scored against a fresh, uncoupled target draw:
        # the model must reproduce the distribution, not the batch it was paired
        # with.
        reference = toy.sample(
            args.eval_samples, generator=torch.Generator().manual_seed(seed + 15_485_863)
        )
        fresh_prior = cylinder_prior(
            args.eval_samples, generator=torch.Generator().manual_seed(seed + 32_452_843)
        )
        initial = to_state(fresh_prior, geometry).to(device)

        distributional: dict[int, float] = {}
        probe_minimum: torch.Tensor | None = None
        probe_angular: torch.Tensor | None = None
        for steps in args.nfe:
            probe = _AmplitudeProbe(network, geometry)
            generated = from_state(build_solver(geometry, steps).sample(probe, initial), geometry)
            if steps == max(args.nfe):
                probe_minimum = probe.minimum
                probe_angular = probe.peak_angular
            generated = generated.cpu()
            distributional[steps] = float(
                wasserstein2_cylinder(
                    generated.abs(),
                    generated.angle(),
                    reference.abs(),
                    reference.angle(),
                    args.phase_weight,
                )
            )

        assert probe_minimum is not None
        assert probe_angular is not None
        visited = probe_minimum.cpu()
        angular = probe_angular.cpu()

    return ArmResult(
        measured_straightness,
        distributional,
        float(visited.mean()),
        float(angular.median()),
        float((visited < args.singularity_radius).float().mean()),
        seconds,
    )


def main() -> None:
    """Run the gate."""
    args = parse_args()
    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    finest = max(args.nfe)

    print("=" * 112)
    print("GATE B -- what the cylinder and a correct coupling are worth, with no confounds")
    print("=" * 112)
    print(f"device: {device} | steps: {args.train_steps} | batch: {args.batch_size} | ")
    print(f"identical prior, encoding, width ({args.width}), depth ({args.depth}) and loss per arm")
    print(f"rho grid: {list(args.rho)} | {args.seeds} seeds per arm")

    results: dict[tuple[float, Geometry, Coupling], list[ArmResult]] = {}
    arm_index = 0
    for rho in args.rho:
        toy = CylinderToy(coupling=rho, structure=args.structure)
        for geometry in GEOMETRIES:
            for coupling in COUPLINGS:
                runs = []
                for _repeat in range(args.seeds):
                    # Arm index rather than hash(): string hashing is salted per
                    # process, which would make the table irreproducible.
                    runs.append(
                        train_arm(
                            toy, geometry, coupling, args, device, args.seed + 7919 * arm_index
                        )
                    )
                    arm_index += 1
                    print(".", end="", flush=True)
                results[(rho, geometry, coupling)] = runs
    print()

    def summarise(values: list[float]) -> str:
        """Mean and full spread of a small sample."""
        return f"{sum(values) / len(values):>8.4f} +-{max(values) - min(values):>6.4f}"

    print("\n" + "=" * 112)
    print(
        f"1. PRICE OF ONE SOLVER STEP   W2(n=1) - W2(n={finest}), paired inside each trained model"
    )
    print("=" * 112)
    print("   The pairing is what makes this readable: both terms carry the same")
    print("   finite-sample bias of the estimator, so the difference does not.")
    print()
    print(f"{'rho':<6}" + "".join(f"{g + '/' + c:>26}" for g in GEOMETRIES for c in COUPLINGS))
    print("-" * 112)
    for rho in args.rho:
        row = ""
        for geometry in GEOMETRIES:
            for coupling in COUPLINGS:
                gaps = [
                    arm.distributional[1] - arm.distributional[finest]
                    for arm in results[(rho, geometry, coupling)]
                ]
                row += f"{summarise(gaps):>26}"
        print(f"{rho:<6.2f}{row}")

    print("\n" + "=" * 112)
    print("2. DOES THE TRAJECTORY VISIT THE SINGULARITY?   Theorem 3's mechanism, measured")
    print("=" * 112)
    print("   left: smallest amplitude reached while integrating.")
    print("   right: peak induced angular velocity |theta_dot| along the path.")
    print(f"   The cylinder is bounded by pi = {math.pi:.2f} by construction; the plane is not.")
    print()
    print(f"{'rho':<6}" + "".join(f"{g + '/' + c:>26}" for g in GEOMETRIES for c in COUPLINGS))
    print("-" * 112)
    for rho in args.rho:
        row = ""
        for geometry in GEOMETRIES:
            for coupling in COUPLINGS:
                runs = results[(rho, geometry, coupling)]
                mean_min = sum(a.min_amplitude for a in runs) / len(runs)
                angular = sum(a.peak_angular for a in runs) / len(runs)
                row += f"{mean_min:>12.3f} |{angular:>11.2f}"
        print(f"{rho:<6.2f}{row}")

    print("\n" + "=" * 112)
    print("3. STRAIGHTNESS   normalised regression residual; lower is straighter")
    print("=" * 112)
    print(f"{'rho':<6}" + "".join(f"{g + '/' + c:>26}" for g in GEOMETRIES for c in COUPLINGS))
    print("-" * 112)
    for rho in args.rho:
        row = ""
        for geometry in GEOMETRIES:
            for coupling in COUPLINGS:
                row += (
                    f"{summarise([a.straightness for a in results[(rho, geometry, coupling)]]):>26}"
                )
        print(f"{rho:<6.2f}{row}")

    print("\n" + "=" * 112)
    print(f"4. FULL W2 vs SOLVER STEPS   mean over {args.seeds} seeds")
    print("=" * 112)
    for rho in args.rho:
        print(f"\n  rho = {rho}")
        print(f"  {'arm':<24}" + "".join(f"{f'n={n}':>9}" for n in args.nfe))
        print("  " + "-" * (24 + 9 * len(args.nfe)))
        for geometry in GEOMETRIES:
            for coupling in COUPLINGS:
                runs = results[(rho, geometry, coupling)]
                row = "".join(
                    f"{sum(a.distributional[n] for a in runs) / len(runs):>9.4f}" for n in args.nfe
                )
                print(f"  {geometry + '/' + coupling:<24}{row}")

    total = sum(a.seconds for runs in results.values() for a in runs)
    arms = sum(len(runs) for runs in results.values())
    print(f"\ntotal training time: {total:.1f} s across {arms} arms")


if __name__ == "__main__":
    main()
