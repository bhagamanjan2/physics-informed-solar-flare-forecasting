"""
Unit tests for metrics and evaluation modules.
Verifies worked example from project brief:
TP=8, FN=2, FP=20, TN=970 gives TSS about 0.78 and HSS about 0.41.
Also verifies threshold tuning, day-block bootstrap CI, and selection rule.
"""

import pytest
import numpy as np
import pandas as pd
from solarflare.metrics import (
    compute_tss,
    compute_hss,
    compute_classification_metrics,
)
from solarflare.evaluate import (
    tune_threshold_validation,
    compute_threshold_curve,
    bootstrap_day_block_ci,
    compute_calibration,
    apply_selection_rule,
)


def test_worked_example_metrics():
    """Verify exact formula values on the brief's worked example."""
    tp = 8
    fn = 2
    fp = 20
    tn = 970

    tss = compute_tss(tp, fn, fp, tn)
    hss = compute_hss(tp, fn, fp, tn)

    assert pytest.approx(tss, abs=0.01) == 0.78
    assert pytest.approx(hss, abs=0.01) == 0.41

    # Verify via compute_classification_metrics
    y_true = np.array([1] * 8 + [1] * 2 + [0] * 20 + [0] * 970)
    proba = np.array([0.9] * 8 + [0.1] * 2 + [0.8] * 20 + [0.1] * 970)
    m = compute_classification_metrics(y_true, proba, threshold=0.5)

    assert m["tp"] == 8
    assert m["fn"] == 2
    assert m["fp"] == 20
    assert m["tn"] == 970
    assert pytest.approx(m["tss"], abs=0.01) == 0.78
    assert pytest.approx(m["hss"], abs=0.01) == 0.41
    assert pytest.approx(m["recall"], abs=0.01) == 0.80
    assert pytest.approx(m["far"], abs=0.01) == 0.714
    assert pytest.approx(m["pofd"], abs=0.01) == 0.020
    assert pytest.approx(m["precision"], abs=0.01) == 0.286
    assert pytest.approx(m["accuracy"], abs=0.01) == 0.978


def test_threshold_tuning():
    """Verify threshold tuning finds the threshold maximizing HSS on validation."""
    y = np.array([0] * 900 + [1] * 100)
    # Clear separation with some overlap
    p = np.array([0.15] * 850 + [0.45] * 50 + [0.40] * 20 + [0.85] * 80)
    best_th = tune_threshold_validation(y, p, metric="hss", num_steps=50)

    assert 0.15 < best_th < 0.85

    # Check threshold curve
    curve = compute_threshold_curve(y, p, num_steps=20)
    assert len(curve) == 20
    assert "hss" in curve.columns
    assert "tss" in curve.columns


def test_day_block_bootstrap_ci():
    """Verify fast day-block bootstrap confidence intervals."""
    np.random.seed(42)
    n_days = 20
    days = np.repeat([f"2024-01-{i+1:02d}" for i in range(n_days)], 100)
    y = np.random.binomial(1, 0.1, size=len(days))
    p = np.clip(y * 0.5 + np.random.uniform(0, 0.3, size=len(days)), 0, 1)

    df = pd.DataFrame({"day": days, "y": y, "p": p})
    ci_df = bootstrap_day_block_ci(df, "y", "p", threshold=0.3, n_bootstraps=500)

    assert not ci_df.empty
    assert "metric" in ci_df.columns
    assert "ci_lower" in ci_df.columns
    assert "ci_upper" in ci_df.columns

    hss_row = ci_df[ci_df["metric"] == "hss"].iloc[0]
    assert hss_row["ci_lower"] <= hss_row["mean"] <= hss_row["ci_upper"]


def test_calibration_curve():
    """Verify calibration curve computation."""
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    res = compute_calibration(y, p, n_bins=3)

    assert "prob_true" in res
    assert "prob_pred" in res
    assert "bin_counts" in res
    assert "brier_score" in res
    assert len(res["bin_counts"]) == 3


def test_model_selection_rule():
    """Verify fixed selection rule prefers highest HSS and uses simplicity as tie-breaker."""
    cv_data = pd.DataFrame([
        {"model": "logistic_regression", "fold": 1, "hss": 0.40, "tss": 0.70, "brier": 0.05},
        {"model": "logistic_regression", "fold": 2, "hss": 0.42, "tss": 0.72, "brier": 0.05},
        {"model": "random_forest", "fold": 1, "hss": 0.41, "tss": 0.71, "brier": 0.05},
        {"model": "random_forest", "fold": 2, "hss": 0.41, "tss": 0.71, "brier": 0.05},
    ])
    sel = apply_selection_rule(cv_data, prefer_simpler_within_se=True)
    assert sel["selected_model"] == "logistic_regression"
