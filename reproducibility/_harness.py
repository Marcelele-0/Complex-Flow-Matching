"""Shared machinery: three outcomes, one report format, one exit code.

Why three and not two. ``FAIL`` means a number was computed and disagrees with
the paper. ``MISSING INPUT`` means there was nothing to compute it from -- no
archive, no store, no checkpoint. A committee has to be able to tell those apart:
the first is a result that does not hold, the second is a result this repository
cannot show you. Collapsing them into "not OK" hides which one you are looking
at, and collapsing ``MISSING INPUT`` into ``PASS`` would be worse.

Both are failures for the exit code. A script that cannot find its input exits
non-zero and says what to run to produce it; it never prints a quiet "OK".
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from enum import Enum

from reproducibility.expected import SEED_SENSITIVE, Expectation

__all__ = ["Check", "MissingInput", "Outcome", "compare", "report"]


class Outcome(Enum):
    """What happened to one check."""

    PASS = "PASS"
    FAIL = "FAIL"
    MISSING_INPUT = "MISSING INPUT"


class MissingInput(Exception):
    """Raised when the input a check needs is not in this checkout.

    Args:
        what: The missing thing, named as a path where possible.
        how: The command that would produce it.
    """

    def __init__(self, what: str, how: str) -> None:
        super().__init__(what)
        self.what = what
        self.how = how


@dataclass(frozen=True)
class Check:
    """One comparison of a measured number against a published one.

    Attributes:
        label: What was measured.
        outcome: Pass, fail or missing input.
        measured: The number this run produced, or ``None``.
        expectation: The published number and its tolerance.
        detail: A line explaining the outcome.
    """

    label: str
    outcome: Outcome
    measured: float | None
    expectation: Expectation | None
    detail: str = ""


def compare(label: str, measured: float, expectation: Expectation) -> Check:
    """Check one measured number against its published counterpart.

    A ``SEED_SENSITIVE`` expectation is never failed on the digit. It records a
    number the paper prints as fixed that is in fact one draw of a statistic
    whose seed-to-seed spread is wider than its printed precision; failing it
    would be honest and useless, so the check reports where the measurement fell
    relative to the observed spread and passes if it is inside it.

    Args:
        label: What was measured.
        measured: This run's number.
        expectation: The published value and its tolerance.

    Returns:
        The check.
    """
    if expectation.tolerance is SEED_SENSITIVE:
        low, high = expectation.spread or (measured, measured)
        # The recorded spread is rounded to the precision it is written at, so a
        # measurement that *is* one of its endpoints can land a rounding step
        # outside it. Compared at that precision rather than exactly.
        edge = 5e-4
        inside = low - edge <= measured <= high + edge
        return Check(
            label=label,
            outcome=Outcome.PASS if inside else Outcome.FAIL,
            measured=measured,
            expectation=expectation,
            detail=(
                f"paper prints {expectation.value:+.3f}; this run {measured:+.3f}; "
                f"seed spread [{low:+.3f}, {high:+.3f}] -- "
                + ("inside" if inside else "OUTSIDE the observed spread")
            ),
        )

    gap = abs(measured - expectation.value)
    tolerance = expectation.tolerance or 0.0
    # An EXACT expectation is compared at the precision the paper prints, which
    # for a percentage rounded to one decimal is half of 0.001.
    slack = tolerance if tolerance else 5e-4
    ok = gap <= slack
    return Check(
        label=label,
        outcome=Outcome.PASS if ok else Outcome.FAIL,
        measured=measured,
        expectation=expectation,
        detail=f"paper {expectation.value:.4f}  measured {measured:.4f}  gap {gap:.4f} "
        f"(tolerance {slack:.4f})",
    )


def missing(label: str, error: MissingInput) -> Check:
    """Turn a missing input into a check that says how to produce it.

    Args:
        label: What could not be measured.
        error: The missing input and the command that makes it.

    Returns:
        The check.
    """
    return Check(
        label=label,
        outcome=Outcome.MISSING_INPUT,
        measured=None,
        expectation=None,
        detail=f"{error.what}\n      produce it with: {error.how}",
    )


def report(title: str, checks: list[Check], notes: list[str] | None = None) -> int:
    """Print one result's checks and return the process exit code.

    Args:
        title: What was reproduced.
        checks: The comparisons made.
        notes: Lines printed after the table, for anything a reader of a
            surprising outcome needs to know.

    Returns:
        ``0`` if every check passed, ``1`` otherwise.
    """
    width = max((len(c.label) for c in checks), default=20)
    print("=" * 100)
    print(title)
    print("=" * 100)
    for check in checks:
        print(f"  {check.outcome.value:<14}{check.label:<{width}}  {check.detail}")
        if check.expectation is not None and check.expectation.note:
            print(f"  {'':<14}{'':<{width}}  note: {check.expectation.note}")
        if check.expectation is not None:
            print(f"  {'':<14}{'':<{width}}  source: {check.expectation.source}")

    for note in notes or []:
        print(f"\n{note}")

    failed = [c for c in checks if c.outcome is not Outcome.PASS]
    print("-" * 100)
    if not failed:
        print(f"PASS  {len(checks)} of {len(checks)} checks reproduce the published numbers.")
        return 0
    counts = {o: sum(1 for c in failed if c.outcome is o) for o in Outcome}
    print(
        f"NOT REPRODUCED  {counts[Outcome.FAIL]} failed, "
        f"{counts[Outcome.MISSING_INPUT]} missing input, "
        f"{len(checks) - len(failed)} passed."
    )
    return 1


def exit_with(code: int) -> None:
    """Leave the process with ``code``; never exit 0 on an unreproduced result."""
    sys.exit(code)
