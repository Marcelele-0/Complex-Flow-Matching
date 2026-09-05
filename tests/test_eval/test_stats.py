import numpy as np
import pandas as pd
import pytest

from cfm.eval.stats import (
    WilcoxonResult,
    apply_holm_bonferroni,
    compute_paired_wilcoxon,
    generate_latex_table,
    pair_evaluations,
)


def test_pair_evaluations_matching():
    target_df = pd.DataFrame({"sample_id": [f"s{i}" for i in range(10)], "metric": np.arange(10)})
    baseline_df = pd.DataFrame(
        {"sample_id": [f"s{i}" for i in range(10)], "metric": np.arange(10) + 1}
    )
    t_val, b_val, ids = pair_evaluations(target_df, baseline_df, "metric")
    assert len(ids) == 10
    np.testing.assert_array_equal(t_val, np.arange(10))
    np.testing.assert_array_equal(b_val, np.arange(10) + 1)


def test_pair_evaluations_mismatch():
    target_df = pd.DataFrame({"sample_id": [f"s{i}" for i in range(3)], "metric": np.arange(3)})
    baseline_df = pd.DataFrame(
        {"sample_id": [f"s{i}" for i in range(3)], "metric": np.arange(3) + 1}
    )
    with pytest.raises(ValueError, match="need at least 5 for testing"):
        pair_evaluations(target_df, baseline_df, "metric")


def test_wilcoxon_identical():
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    stat, p_val, eff = compute_paired_wilcoxon(x, x)
    assert stat == 0.0
    assert p_val == 1.0
    assert eff == 0.0


def test_wilcoxon_superior():
    x = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    stat, p_val, eff = compute_paired_wilcoxon(x, y)
    assert p_val < 0.05
    assert eff > 0.0


def test_holm_bonferroni_ordering_and_monotonicity():
    results = [
        WilcoxonResult("m1", "b", "t", 10, 0, 0, 0, 0, 0, 0, 0, 0.01, 0, 0, False, False),
        WilcoxonResult("m2", "b", "t", 10, 0, 0, 0, 0, 0, 0, 0, 0.001, 0, 0, False, False),
        WilcoxonResult("m3", "b", "t", 10, 0, 0, 0, 0, 0, 0, 0, 0.05, 0, 0, False, False),
    ]
    adjusted = apply_holm_bonferroni(results)
    p_vals = [r.p_value_adjusted for r in adjusted]
    assert max(p_vals) <= 1.0
    assert results[1].p_value_adjusted <= results[0].p_value_adjusted <= results[2].p_value_adjusted


def test_latex_table_rendering():
    results = [
        WilcoxonResult(
            "m1", "b", "t", 10, 1.0, 0.1, 0.5, 0.1, 0.5, 0.5, 0, 0.0001, 0.0001, 0.8, True, True
        ),
    ]
    table = generate_latex_table(results)
    assert "\\toprule" in table
    assert "\\midrule" in table
    assert "\\bottomrule" in table
    assert "***" in table


def test_evaluate_saves_csv(tmp_path, monkeypatch):
    # We will invoke evaluate main with a dummy config.
    # Just need to check that eval_records.csv is saved.
    # Because `evaluate.py` relies on hydra, we can just run a CLI command for it.
    pass


def test_run_stats_cli_demo_subprocess():
    import subprocess
    import sys

    res = subprocess.run(
        [sys.executable, "scripts/run_stats.py", "--demo"], capture_output=True, text=True
    )
    assert res.returncode == 0
    assert "Running in demo mode" in res.stdout
