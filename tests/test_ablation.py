"""
Unit tests for ablation and interpretability modules.
Tests filter_features_by_set, run_feature_ablation, compute_permutation_importance, error_analysis.
"""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from solarflare.ablation import filter_features_by_set, run_feature_ablation, FEATURE_SETS
from solarflare.interpret import compute_permutation_importance, error_analysis


def test_filter_features_by_set():
    all_cols = [
        "g0_log_net_flux", "g0_flux_slope_5", "g0_flux_mean_15",
        "g1_dt_last_flare", "g1_n_flares_6h",
        "g2_band_ratio_1", "g2_band_ratio_2",
        "g3_phase",
        "non_feature_col"
    ]
    
    a0 = filter_features_by_set(all_cols, "A0_brightness")
    assert all(c.startswith("g0_") for c in a0)
    assert len(a0) == 3

    a1 = filter_features_by_set(all_cols, "A1_history")
    assert all(c.startswith(("g0_", "g1_")) for c in a1)
    assert len(a1) == 5

    a2 = filter_features_by_set(all_cols, "A2_pulsations")
    assert all(c.startswith(("g0_", "g2_")) for c in a2)
    assert len(a2) == 5

    a3 = filter_features_by_set(all_cols, "A3_phase")
    assert all(c.startswith(("g0_", "g3_")) for c in a3)
    assert len(a3) == 4

    a4 = filter_features_by_set(all_cols, "A4_all")
    assert len(a4) == 8
    assert "non_feature_col" not in a4


def test_compute_permutation_importance():
    np.random.seed(42)
    n = 200
    X = pd.DataFrame({
        "g0_flux": np.random.randn(n),
        "g0_slope": np.random.randn(n),
        "g1_history": np.random.randn(n),
        "g2_stft": np.random.randn(n),
        "g3_phase": np.random.randint(0, 4, size=n),
    })
    # Signal strongly correlated with g0_flux
    prob = 1 / (1 + np.exp(-2.0 * X["g0_flux"]))
    y = (prob >= 0.5).astype(int)

    model = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    model.fit(X, y)

    df_imp = compute_permutation_importance(model, X, y, scoring="hss", n_repeats=3)
    assert isinstance(df_imp, pd.DataFrame)
    assert "feature" in df_imp.columns
    assert "importance_mean" in df_imp.columns
    assert "group" in df_imp.columns
    assert len(df_imp) == 5
    # g0_flux should be most important
    assert df_imp.iloc[0]["feature"] == "g0_flux"


def test_error_analysis():
    times = pd.date_range("2024-01-01 00:00:00", periods=100, freq="1min")
    df_test = pd.DataFrame({
        "time": times,
        "y": [0]*90 + [1]*10,
        "y_pred": [0]*85 + [1]*10 + [0]*5, # 5 FPs, 5 TPs, 5 FNs
        "g0_log_net_flux": np.linspace(-7, -4, 100),
    })

    flares_df = pd.DataFrame([
        {"flare_id": "test_1", "tstart": "2024-01-01 01:30:00", "tpeak": "2024-01-01 01:35:00", "tend": "2024-01-01 01:40:00", "class_letter": "C", "peak_flux": 2e-6},
        {"flare_id": "test_2", "tstart": "2024-01-01 01:35:00", "tpeak": "2024-01-01 01:38:00", "tend": "2024-01-01 01:42:00", "class_letter": "M", "peak_flux": 2e-5},
    ])

    results = error_analysis(df_test, flares_df=flares_df, horizon_min=30)
    assert results["n_tp"] == 5
    assert results["n_fn"] == 5
    assert results["n_fp"] == 5
    assert "false_alarm_profile" in results
    assert "flares_by_class" in results
    assert "C" in results["flares_by_class"]
    assert "M" in results["flares_by_class"]


def test_run_feature_ablation_synthetic():
    np.random.seed(42)
    n = 300
    times = pd.date_range("2024-01-01 00:00:00", periods=n, freq="1min")
    days = [str(t.date()) for t in times]
    
    df = pd.DataFrame({
        "time": times,
        "day": days,
        "split": ["train"]*180 + ["val"]*45 + ["test"]*75,
        "y": np.random.choice([0, 1], size=n, p=[0.85, 0.15]),
        "g0_flux": np.random.randn(n),
        "g1_dt": np.random.randn(n),
        "g2_ratio": np.random.randn(n),
        "g3_phase": np.random.randint(0, 4, size=n),
    })

    pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression())])
    models = {"LR": pipe}
    datasets = {30: df}

    df_ablation = run_feature_ablation(models, datasets=datasets, horizons=[30], fast_mode=True)
    assert isinstance(df_ablation, pd.DataFrame)
    assert len(df_ablation) == 5 # A0 through A4
    assert set(df_ablation["feature_set"]) == {"A0_brightness", "A1_history", "A2_pulsations", "A3_phase", "A4_all"}
    assert "delta_hss" in df_ablation.columns
