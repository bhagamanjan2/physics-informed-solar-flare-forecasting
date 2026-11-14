"""
Smoke tests for Solar Flare Forecasting CLI subcommands.
Owned by Agent A8 (CLI and Tests).
"""

import sys
import subprocess
import pytest
import pandas as pd
import numpy as np


def run_cli(args):
    """Run CLI module with args and return CompletedProcess."""
    cmd = [sys.executable, "-m", "solarflare.cli"] + args
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result


def test_cli_help():
    """Verify CLI main entry point and help display."""
    res = run_cli(["--help"])
    assert res.returncode == 0
    assert "Solar Flare Forecasting CLI" in res.stdout
    for subcmd in ["predict", "ingest", "features", "train", "evaluate", "ablate", "report"]:
        assert subcmd in res.stdout


@pytest.mark.parametrize("subcmd", ["predict", "ingest", "features", "train", "evaluate", "ablate", "report"])
def test_subcommand_help(subcmd):
    """Verify each subcommand provides helpful usage descriptions."""
    res = run_cli([subcmd, "--help"])
    assert res.returncode == 0
    assert f"usage: " in res.stdout


def test_cli_predict_smoke(tmp_path):
    """Smoke test for real-time flare prediction on custom input flux."""
    # Create a small flux input CSV
    flux_file = tmp_path / "recent_flux.csv"
    times = pd.date_range("2024-01-01 12:00:00", periods=60, freq="1min")
    flux_vals = 1.5e-7 + 0.1e-7 * np.sin(np.linspace(0, 3, 60))
    df = pd.DataFrame({
        "time": times.strftime("%Y-%m-%d %H:%M:%S"),
        "flux": flux_vals,
        "quality_ok": [True] * 60
    })
    df.to_csv(flux_file, index=False)

    res = run_cli(["predict", "--input", str(flux_file), "--horizon", "30"])
    assert res.returncode == 0
    assert "Solar Flare Early Warning Output" in res.stdout
    assert "P(Flare" in res.stdout
    assert "Risk Tier" in res.stdout
    assert "Golden Rule enforced" in res.stdout
    assert "Viva / Operations Brief" in res.stdout
    assert "Flare probability is" in res.stdout
    assert "Operational status:" in res.stdout


def test_cli_predict_nonexistent_fallback():
    """Verify predict gracefully generates fallback sample when input file is not found."""
    res = run_cli(["predict", "--input", "nonexistent_sample.csv", "--horizon", "15"])
    assert res.returncode == 0
    assert "Input file 'nonexistent_sample.csv' not found. Generating sample" in res.stdout
    assert "P(Flare" in res.stdout
    assert "Flare probability is" in res.stdout


def test_cli_ingest_smoke():
    """Smoke test for ingest command."""
    res = run_cli(["ingest"])
    assert res.returncode == 0
    assert "Data Ingestion Summary" in res.stdout
    assert "SIGNIFICANT_FLARE" in res.stdout


def test_cli_features_smoke():
    """Smoke test for features command."""
    res = run_cli(["features"])
    assert res.returncode == 0
    assert "Feature Engineering Dataset Matrix" in res.stdout
    assert "G0" in res.stdout
    assert "G3" in res.stdout


def test_cli_train_smoke():
    """Smoke test for train command."""
    res = run_cli(["train"])
    assert res.returncode == 0
    assert "Model Training Performance" in res.stdout
    assert "Random Forest" in res.stdout


def test_cli_evaluate_smoke():
    """Smoke test for evaluate command."""
    res = run_cli(["evaluate"])
    assert res.returncode == 0
    assert "Evaluation Metrics" in res.stdout
    assert "HSS" in res.stdout
    assert "TSS" in res.stdout


def test_cli_ablate_smoke():
    """Smoke test for ablate command."""
    res = run_cli(["ablate"])
    assert res.returncode == 0
    assert "Feature Ablation Study" in res.stdout
    assert "Research Question" in res.stdout


def test_cli_report_smoke():
    """Smoke test for report command."""
    res = run_cli(["report"])
    assert res.returncode == 0
    assert "CAPSTONE EXECUTIVE REPORT" in res.stdout
    assert "Viva Cheat Sheet" in res.stdout
