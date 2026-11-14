"""
Model interpretability and error analysis module.
Owned by Agent A5 (Ablation and Insight).
Computes permutation importance and inspects false alarms and missed flares.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix

from solarflare.metrics import compute_hss, compute_tss


def _get_hss_scorer(threshold: float = 0.5):
    """Return custom scorer for permutation_importance calculating HSS."""
    def scorer(estimator, X, y):
        if hasattr(estimator, "predict_proba"):
            probs = estimator.predict_proba(X)[:, 1]
            preds = (probs >= threshold).astype(int)
        else:
            preds = estimator.predict(X)
        tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
        return compute_hss(tp, fn, fp, tn)
    return scorer


def _get_tss_scorer(threshold: float = 0.5):
    """Return custom scorer for permutation_importance calculating TSS."""
    def scorer(estimator, X, y):
        if hasattr(estimator, "predict_proba"):
            probs = estimator.predict_proba(X)[:, 1]
            preds = (probs >= threshold).astype(int)
        else:
            preds = estimator.predict(X)
        tn, fp, fn, tp = confusion_matrix(y, preds, labels=[0, 1]).ravel()
        return compute_tss(tp, fn, fp, tn)
    return scorer


def compute_permutation_importance(
    model,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    scoring: str = "hss",
    threshold: float = 0.5,
    n_repeats: int = 10,
    max_samples: int = 10000,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Calculate permutation feature importance on test partition.
    Supports HSS, TSS, ROC-AUC scoring, with automatic fast subsampling if test size > max_samples.
    """
    X_eval = X_test
    y_eval = y_test

    # Subsample if dataset is very large to ensure fast execution on laptop
    if len(X_eval) > max_samples:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X_eval), size=max_samples, replace=False)
        X_eval = X_eval.iloc[idx].reset_index(drop=True)
        y_eval = y_eval[idx]

    # Select scoring function
    if scoring == "hss":
        scorer = _get_hss_scorer(threshold=threshold)
    elif scoring == "tss":
        scorer = _get_tss_scorer(threshold=threshold)
    else:
        scorer = scoring

    result = permutation_importance(
        model, X_eval, y_eval, scoring=scorer, n_repeats=n_repeats, random_state=random_state, n_jobs=-1
    )

    df_imp = pd.DataFrame({
        "feature": X_eval.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    # Assign clue group for color-coding in plots
    groups = []
    for f in df_imp["feature"]:
        if f.startswith("g0_"):
            groups.append("G0: Brightness")
        elif f.startswith("g1_"):
            groups.append("G1: History")
        elif f.startswith("g2_"):
            groups.append("G2: Pulsations")
        elif f.startswith("g3_"):
            groups.append("G3: Phase")
        else:
            groups.append("Other")
    df_imp["group"] = groups

    return df_imp


def error_analysis(
    df_test: pd.DataFrame,
    flares_df: pd.DataFrame | None = None,
    y_true: np.ndarray | None = None,
    y_pred: np.ndarray | None = None,
    horizon_min: int = 30,
) -> Dict[str, Any]:
    """
    Characterize missed flares (FN) by class and profile false alarm (FP) periods.
    Returns dictionary with confusion counts, breakdown by flare class, and false alarm profiles.
    """
    if y_true is None:
        y_true = df_test["y"].values
    if y_pred is None:
        if "y_pred" in df_test.columns:
            y_pred = df_test["y_pred"].values
        elif "proba" in df_test.columns:
            y_pred = (df_test["proba"].values >= 0.5).astype(int)
        else:
            raise ValueError("y_pred or proba column required for error analysis")

    fp_indices = np.where((y_true == 0) & (y_pred == 1))[0]
    fn_indices = np.where((y_true == 1) & (y_pred == 0))[0]
    tp_indices = np.where((y_true == 1) & (y_pred == 1))[0]
    tn_indices = np.where((y_true == 0) & (y_pred == 0))[0]

    out: Dict[str, Any] = {
        "n_tp": len(tp_indices),
        "n_fn": len(fn_indices),
        "n_fp": len(fp_indices),
        "n_tn": len(tn_indices),
        "fp_rows": df_test.iloc[fp_indices].copy() if not df_test.empty else pd.DataFrame(),
        "fn_rows": df_test.iloc[fn_indices].copy() if not df_test.empty else pd.DataFrame(),
    }

    # Profile false alarm characteristics
    fp_profile: Dict[str, float] = {}
    if len(fp_indices) > 0 and not df_test.empty:
        fp_df = df_test.iloc[fp_indices]
        tn_df = df_test.iloc[tn_indices]

        # Inspect brightness clues if present
        for col in ["g0_log_net_flux", "g0_flux_slope_5", "net_flux", "flux"]:
            if col in df_test.columns:
                fp_profile[f"{col}_fp_mean"] = float(fp_df[col].mean())
                fp_profile[f"{col}_tn_mean"] = float(tn_df[col].mean()) if len(tn_indices) > 0 else 0.0

    out["false_alarm_profile"] = fp_profile

    # Characterize caught vs missed flares by class if flares_df is supplied
    if flares_df is not None and not flares_df.empty and "time" in df_test.columns:
        t_test_times = pd.to_datetime(df_test["time"])
        t_min = t_test_times.min()
        t_max = t_test_times.max()

        fl_copy = flares_df.copy()
        fl_copy["tstart_dt"] = pd.to_datetime(fl_copy["tstart"])
        # Flares that started during test set
        test_flares = fl_copy[(fl_copy["tstart_dt"] >= t_min) & (fl_copy["tstart_dt"] <= t_max)]

        class_stats: Dict[str, Dict[str, Any]] = {"C": {"total": 0, "caught": 0, "missed": 0, "recall": 0.0},
                                                   "M": {"total": 0, "caught": 0, "missed": 0, "recall": 0.0},
                                                   "X": {"total": 0, "caught": 0, "missed": 0, "recall": 0.0}}

        alarm_times = set(t_test_times[y_pred == 1])
        missed_list: List[Dict[str, Any]] = []

        for _, flare_row in test_flares.iterrows():
            c = str(flare_row.get("class_letter", "C")).upper()
            if c not in class_stats:
                class_stats[c] = {"total": 0, "caught": 0, "missed": 0, "recall": 0.0}

            tstart = flare_row["tstart_dt"]
            # Look-back window (tstart - horizon, tstart]
            window_times = pd.date_range(tstart - pd.Timedelta(minutes=horizon_min), tstart, freq="1min")
            is_caught = any(wt in alarm_times for wt in window_times)

            class_stats[c]["total"] += 1
            if is_caught:
                class_stats[c]["caught"] += 1
            else:
                class_stats[c]["missed"] += 1
                missed_list.append({
                    "flare_id": flare_row.get("flare_id", ""),
                    "tstart": str(flare_row.get("tstart", "")),
                    "class_letter": c,
                    "peak_flux": float(flare_row.get("peak_flux", 0.0)),
                })

        for c in class_stats:
            tot = class_stats[c]["total"]
            cgt = class_stats[c]["caught"]
            class_stats[c]["recall"] = float(cgt / tot) if tot > 0 else 0.0

        out["flares_by_class"] = class_stats
        out["missed_flares"] = missed_list

    return out
