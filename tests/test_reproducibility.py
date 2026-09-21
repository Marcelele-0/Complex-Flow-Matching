"""The reproduction harness: three outcomes, and an exit code that means them.

The value of this layer is entirely in what it refuses to call success. A check
that cannot find its input must not be reported the same way as one that found it
and agreed, and neither may exit 0.
"""

from __future__ import annotations

from reproducibility._harness import MissingInput, Outcome, compare, missing, report
from reproducibility.expected import EXACT, SEED_SENSITIVE, Expectation


def _exact(value: float) -> Expectation:
    return Expectation(value, EXACT, "test")


class TestCompare:
    def test_a_matching_number_passes(self) -> None:
        assert compare("x", 0.4600, _exact(0.460)).outcome is Outcome.PASS

    def test_a_disagreeing_number_fails(self) -> None:
        assert compare("x", 0.4700, _exact(0.460)).outcome is Outcome.FAIL

    def test_exact_compares_at_the_printed_precision(self) -> None:
        """A percentage printed to one decimal carries half of 0.001 of slack.

        Tighter would fail on the last float bit; looser would let a genuinely
        different number through.
        """
        assert compare("x", 0.46004, _exact(0.460)).outcome is Outcome.PASS
        assert compare("x", 0.46200, _exact(0.460)).outcome is Outcome.FAIL

    def test_an_explicit_tolerance_is_honoured(self) -> None:
        loose = Expectation(1.0, 0.05, "test")
        assert compare("x", 1.04, loose).outcome is Outcome.PASS
        assert compare("x", 1.06, loose).outcome is Outcome.FAIL


class TestSeedSensitive:
    """The one published number whose digit does not reproduce.

    Section 5.4 prints 0.044 for the seam correlation; the shipped seed gives
    -0.008, and both are ordinary draws from a spread of [-0.008, +0.055].
    Failing on the digit would be honest and useless, so the check asks whether
    the measurement fell inside the observed spread -- and still reports the
    published value beside it, so the discrepancy stays visible.
    """

    SPREAD = Expectation(0.044, SEED_SENSITIVE, "test", spread=(-0.008, 0.055))

    def test_a_draw_inside_the_spread_passes(self) -> None:
        assert compare("seam", -0.008, self.SPREAD).outcome is Outcome.PASS
        assert compare("seam", 0.030, self.SPREAD).outcome is Outcome.PASS

    def test_a_draw_outside_it_fails(self) -> None:
        assert compare("seam", 0.400, self.SPREAD).outcome is Outcome.FAIL

    def test_an_endpoint_is_inside_despite_rounding(self) -> None:
        """The recorded spread is rounded to where it is printed.

        A measurement that *is* an endpoint can land a rounding step outside it,
        which reported the shipped seed as OUTSIDE its own spread.
        """
        assert compare("seam", -0.00804, self.SPREAD).outcome is Outcome.PASS

    def test_the_published_digit_is_still_shown(self) -> None:
        detail = compare("seam", -0.008, self.SPREAD).detail
        assert "0.044" in detail and "-0.008" in detail


class TestReport:
    def test_all_passing_exits_zero(self, capsys) -> None:
        code = report("t", [compare("x", 1.0, _exact(1.0))])
        assert code == 0
        assert "PASS" in capsys.readouterr().out

    def test_a_failure_exits_non_zero(self) -> None:
        assert report("t", [compare("x", 2.0, _exact(1.0))]) == 1

    def test_missing_input_exits_non_zero_and_says_how(self, capsys) -> None:
        """Never a quiet OK: a missing input is a non-result, not a pass."""
        check = missing("block", MissingInput("data/x does not exist", "build it with foo"))
        assert report("t", [check]) == 1
        out = capsys.readouterr().out
        assert "MISSING INPUT" in out
        assert "build it with foo" in out

    def test_the_two_failure_kinds_are_counted_apart(self, capsys) -> None:
        """A reviewer must see which is which: a wrong number, or no number."""
        checks = [
            compare("a", 1.0, _exact(1.0)),
            compare("b", 2.0, _exact(1.0)),
            missing("c", MissingInput("gone", "make it")),
        ]
        assert report("t", checks) == 1
        summary = capsys.readouterr().out
        assert "1 failed" in summary
        assert "1 missing input" in summary
        assert "1 passed" in summary


class TestExpectedConstants:
    def test_every_expectation_names_its_source(self) -> None:
        """A number with no provenance is a number nobody can check."""
        from reproducibility import expected

        found = [
            v
            for name, value in vars(expected).items()
            if not name.startswith("_")
            for v in _walk(value)
        ]
        assert found
        for expectation in found:
            assert expectation.source, f"{expectation} has no source"

    def test_the_seed_sensitive_one_records_its_spread(self) -> None:
        from reproducibility.expected import SEAM_LAG1

        published = SEAM_LAG1["published"]
        assert published.tolerance is SEED_SENSITIVE
        assert published.spread is not None
        assert published.note


def _walk(value: object) -> list[Expectation]:
    """Every Expectation nested anywhere inside ``value``."""
    if isinstance(value, Expectation):
        return [value]
    if isinstance(value, dict):
        return [e for v in value.values() for e in _walk(v)]
    if isinstance(value, list | tuple):
        return [e for v in value for e in _walk(v)]
    return []
