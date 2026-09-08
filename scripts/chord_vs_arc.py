"""The chord is not the arc: two consequences of one geometric identity.

Two unit phasors separated by an angle ``dtheta`` admit two natural notions of
distance, and the difference between them is not cosmetic. Writing both out:

    midpoint modulus of the average   =  |cos(dtheta/2)|
    chordal distance                  =  2 |sin(dtheta/2)|
    geodesic distance                 =  |dtheta|

Each line has a consequence in a different part of the literature, and this
script measures both in isolation -- no network, no loss weighting, no learned
representation, so nothing here can be attributed to a confound.

**Consequence one, on probability paths.** Linear interpolation between two
complex numbers passes inside the unit circle whenever their phases differ, so
the modulus is attenuated at intermediate times. Under an i.i.d. uniform phase
prior the expected midpoint modulus is ``E|cos(dtheta/2)| = 2/pi``, i.e. an
attenuation of ``1 - 2/pi ~ 36.33%``. A geodesic on the cylinder interpolates
amplitude and angle separately and therefore has none. This matters for
generative models, where the path *is* the model: sampling integrates through
the intermediate states, and no penalty acts at ``t = 0.5``.

**Consequence two, on penalties.** A squared chordal penalty is
``2(1 - cos dtheta)``, whose derivative ``2 sin dtheta`` vanishes as
``dtheta -> pi``. The gradient is weakest exactly where the phase error is
largest. A squared geodesic penalty is ``dtheta^2``, whose derivative ``2 dtheta``
is largest there. This matters for the regularisation literature, which
penalises phase through Cartesian or sign-vector residuals.

The third experiment turns that gradient statement into an observable: gradient
descent on each penalty, from a controlled initial error, for a fixed budget.
The chordal objective is expected to leave a large residual for initial errors
near ``pi`` while the geodesic one does not.

Usage::

    uv run python scripts/chord_vs_arc.py
    uv run python scripts/chord_vs_arc.py --steps 200 --lr 0.05 --samples 200000
"""

from __future__ import annotations

import argparse
import math

import torch


def parse_args() -> argparse.Namespace:
    """Command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples",
        type=int,
        default=200_000,
        help="Monte Carlo draws for the uniform-prior expectation.",
    )
    parser.add_argument(
        "--grid",
        type=int,
        default=9,
        help="Number of dtheta values tabulated over [0, pi].",
    )
    parser.add_argument(
        "--steps", type=int, default=200, help="Gradient-descent budget in the recovery test."
    )
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate for that descent.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def shortest_angle(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Signed shortest angular difference ``a - b``, wrapped to ``[-pi, pi]``.

    Args:
        a: Angles in radians.
        b: Angles in radians, broadcastable against ``a``.

    Returns:
        The wrapped difference, same shape as the broadcast of the inputs.
    """
    return torch.atan2(torch.sin(a - b), torch.cos(a - b))


def midpoint_modulus(dtheta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Modulus at ``t = 0.5`` of a Euclidean and of a geodesic path.

    Both paths run between two unit phasors separated by ``dtheta``. The
    Euclidean midpoint is the average of the two complex numbers; the geodesic
    midpoint interpolates amplitude and angle separately, so its modulus is
    exactly the interpolated amplitude.

    Args:
        dtheta: Angular separations in radians.

    Returns:
        A tuple ``(euclidean, geodesic)`` of moduli, both shaped like ``dtheta``.
    """
    z0 = torch.ones_like(dtheta, dtype=torch.complex64)
    z1 = torch.polar(torch.ones_like(dtheta), dtheta)
    euclidean = (0.5 * (z0 + z1)).abs()
    geodesic = torch.ones_like(dtheta)
    return euclidean, geodesic


def penalty_gradients(dtheta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Gradient magnitude of a squared chordal and a squared geodesic penalty.

    Args:
        dtheta: Phase errors in radians.

    Returns:
        A tuple ``(chordal, geodesic)`` of ``|d/d dtheta|``, shaped like the input.
    """
    return (2.0 * torch.sin(dtheta)).abs(), (2.0 * dtheta).abs()


def recover_phase(
    initial_error: torch.Tensor, steps: int, lr: float, chordal: bool
) -> torch.Tensor:
    """Descend one of the two penalties and report the residual phase error.

    The target phase is zero without loss of generality, so ``initial_error`` is
    the starting angle. Optimising the chordal objective is done on the embedded
    unit vector, which is how the reconstruction literature penalises phase; the
    geodesic objective is optimised on the angle directly.

    Args:
        initial_error: Starting phase errors in radians, shape ``[N]``.
        steps: Number of gradient-descent iterations.
        lr: Step size.
        chordal: Descend the chordal penalty when ``True``, geodesic otherwise.

    Returns:
        Residual ``|dtheta|`` after the budget, shape ``[N]``.
    """
    theta = initial_error.clone().requires_grad_(True)
    optimiser = torch.optim.SGD([theta], lr=lr)

    for _ in range(steps):
        optimiser.zero_grad()
        if chordal:
            # ||e^{i theta} - 1||^2, the residual a Cartesian formulation sees.
            loss = ((torch.cos(theta) - 1.0) ** 2 + torch.sin(theta) ** 2).sum()
        else:
            loss = (shortest_angle(theta, torch.zeros_like(theta)) ** 2).sum()
        loss.backward()
        optimiser.step()

    return shortest_angle(theta.detach(), torch.zeros_like(theta)).abs()


def main() -> None:
    """Run the three measurements and print them."""
    args = parse_args()
    torch.manual_seed(args.seed)

    grid = torch.linspace(0.0, math.pi, args.grid)

    print("=" * 76)
    print("1. Modulus at t = 0.5, Euclidean path vs geodesic path")
    print("=" * 76)
    euclidean, geodesic = midpoint_modulus(grid)
    print(f"{'dtheta':>10}{'euclidean':>14}{'geodesic':>12}{'attenuation':>14}")
    for i in range(args.grid):
        d = grid[i].item()
        att = 1.0 - euclidean[i].item()
        print(f"{d:>10.4f}{euclidean[i].item():>14.4f}{geodesic[i].item():>12.4f}{att:>13.2%}")

    uniform = (torch.rand(args.samples) * 2.0 - 1.0) * math.pi
    measured = 1.0 - (0.5 * (1.0 + torch.polar(torch.ones_like(uniform), uniform))).abs().mean()
    predicted = 1.0 - 2.0 / math.pi
    print(
        f"\nUnder an i.i.d. uniform phase prior "
        f"({args.samples} draws): measured {measured.item():.4%}, "
        f"predicted 1 - 2/pi = {predicted:.4%}"
    )

    print()
    print("=" * 76)
    print("2. Gradient magnitude of the two penalties")
    print("=" * 76)
    chordal_grad, geodesic_grad = penalty_gradients(grid)
    print(f"{'dtheta':>10}{'chordal':>14}{'geodesic':>12}{'ratio':>10}")
    for i in range(args.grid):
        ratio = chordal_grad[i].item() / max(geodesic_grad[i].item(), 1e-12)
        print(
            f"{grid[i].item():>10.4f}{chordal_grad[i].item():>14.4f}"
            f"{geodesic_grad[i].item():>12.4f}{ratio:>10.3f}"
        )

    print()
    print("=" * 76)
    print(f"3. Residual after {args.steps} descent steps at lr={args.lr}")
    print("=" * 76)
    starts = grid[1:]  # zero error is a fixed point of both, and uninformative
    residual_chordal = recover_phase(starts, args.steps, args.lr, chordal=True)
    residual_geodesic = recover_phase(starts, args.steps, args.lr, chordal=False)
    print(f"{'start':>10}{'chordal':>14}{'geodesic':>12}{'recovered':>12}")
    for i in range(starts.numel()):
        frac = 1.0 - residual_chordal[i].item() / max(starts[i].item(), 1e-12)
        print(
            f"{starts[i].item():>10.4f}{residual_chordal[i].item():>14.4f}"
            f"{residual_geodesic[i].item():>12.4f}{frac:>11.1%}"
        )

    print()
    print("Reading: attenuation is a property of the path and no penalty acts on it;")
    print("the chordal gradient collapses toward pi, so large phase errors are")
    print("corrected slowly or not at all within a fixed budget.")


if __name__ == "__main__":
    main()
