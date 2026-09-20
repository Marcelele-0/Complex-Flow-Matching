import numpy as np
import pandas as pd
import pytest

from cyfm.eval.stats import (
    WilcoxonResult,
    apply_holm_bonferroni,
    cli_main,
    compute_paired_wilcoxon,
    generate_latex_table,
    load_eval_records,
    pair_evaluations,
    print_table,
    process_stats,
    run_demo,
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


def test_pair_evaluations_duplicate_sample_id_raises():
    target_df = pd.DataFrame(
        {"sample_id": ["s1", "s1", "s2", "s3", "s4", "s5"], "metric": range(6)}
    )
    baseline_df = pd.DataFrame(
        {"sample_id": ["s1", "s2", "s3", "s4", "s5", "s6"], "metric": range(6)}
    )
    with pytest.raises(pd.errors.MergeError):
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
    stat_f, p_f, eff_f = compute_paired_wilcoxon(x, y)
    stat_r, p_r, eff_r = compute_paired_wilcoxon(y, x)
    assert p_f < 0.05
    assert eff_f == pytest.approx(-eff_r)
    assert eff_f > 0.0


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
    assert [r.p_value_adjusted for r in adjusted] == pytest.approx([0.02, 0.003, 0.05])


def test_latex_table_rendering():
    results = [
        WilcoxonResult(
            "psnr_db",
            "unet_baseline_seed0",
            "cylindrical_ours",
            10,
            1.0,
            0.1,
            0.5,
            0.1,
            0.5,
            0.5,
            0,
            0.0001,
            0.0001,
            0.8,
            True,
            True,
        ),
    ]
    table = generate_latex_table(results, target_name="cylindrical_ours")
    assert "\\toprule" in table
    assert "\\midrule" in table
    assert "\\bottomrule" in table
    assert "***" in table
    assert "\\begin{tabular}{llrrrrrr}" in table
    assert "psnr\\_db" in table
    assert "unet\\_baseline\\_seed0" in table
    assert "cylindrical\\_ours" in table
    assert "Metric & Baseline & N &" in table

    # Verify 8 columns in header and data rows
    for line in table.splitlines():
        if "&" in line:
            cols = [c.strip() for c in line.rstrip("\\ ").split("&")]
            assert len(cols) == 8


def test_load_eval_records_parquet_raises(tmp_path):
    parquet_file = tmp_path / "test.parquet"
    parquet_file.touch()
    with pytest.raises(
        ValueError,
        match=(
            "Parquet format is not supported; eval_records are emitted as CSV. "
            "Please provide a CSV file."
        ),
    ):
        load_eval_records(parquet_file)


def test_run_stats_cli_demo_subprocess():
    import subprocess
    import sys

    res = subprocess.run(
        [sys.executable, "scripts/run_stats.py", "--demo"], capture_output=True, text=True
    )
    assert res.returncode == 0
    assert "Running in demo mode" in res.stdout


def test_cli_main_demo():
    code = cli_main(["--demo"])
    assert code == 0


def test_cli_main_missing_args(capsys):
    assert cli_main([]) == 1
    assert "Error: --target is required" in capsys.readouterr().out

    assert cli_main(["--target", "some_target.csv"]) == 1
    assert "Error: --baselines is required" in capsys.readouterr().out


def test_scripts_run_stats_reexports():
    import importlib.util
    from pathlib import Path

    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_stats.py"
    spec = importlib.util.spec_from_file_location("run_stats", script_path)
    assert spec is not None and spec.loader is not None
    srs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(srs)

    assert callable(srs.cli_main)
    assert callable(srs.main)
    assert callable(srs.process_stats)
    assert callable(srs.print_table)
    assert callable(srs.run_demo)


def test_process_stats_missing_metric_raises():
    target_df = pd.DataFrame({"sample_id": ["s1", "s2"], "psnr_db": [30.0, 31.0]})
    baseline_df = pd.DataFrame({"sample_id": ["s1", "s2"], "ssim": [0.8, 0.9]})
    with pytest.raises(KeyError, match="Metric 'psnr_db' not found"):
        process_stats(target_df, {"b": baseline_df}, ["psnr_db"])


def test_run_demo_direct(capsys):
    run_demo()
    assert "Running in demo mode with synthetic data..." in capsys.readouterr().out


def test_print_table_direct(capsys):
    print_table([])
    assert "Metric" in capsys.readouterr().out
