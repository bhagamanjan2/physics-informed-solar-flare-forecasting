"""
Ablation study module for feature importance across clue groups.
Owned by Agent A5 (Ablation and Insight).

Feature sets:
  A0 = G0 (Brightness)
  A1 = G0 + G1 (Brightness + Flare History)
  A2 = G0 + G2 (Brightness + Pulsations/STFT)
  A3 = G0 + G3 (Brightness + Phase Flags)
  A4 = All Clues (G0 + G1 + G2 + G3)
"""

from typing import Dict, List, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.base import clone

from solarflare.metrics import compute_classification_metrics
from solarflare.evaluate import tune_threshold_validation, bootstrap_day_block_ci


FEATURE_SETS = {
    "A0_brightness": ["g0_"],
    "A1_history": ["g0_", "g1_"],
    "A2_pulsations": ["g0_", "g2_"],
    "A3_phase": ["g0_", "g3_"],
    "A4_all": ["g0_", "g1_", "g2_", "g3_"],
}

FEATURE_SET_DESCRIPTIONS = {
    "A0_brightness": "G0 (Recent Brightness)",
    "A1_history": "G0 + G1 (Brightness + History)",
    "A2_pulsations": "G0 + G2 (Brightness + Pulsations)",
    "A3_phase": "G0 + G3 (Brightness + Phase)",
    "A4_all": "All Clues (G0 + G1 + G2 + G3)",
}


def filter_features_by_set(feature_cols: List[str], set_name: str) -> List[str]:
    """Return feature column subset matching specified clue set."""
    prefixes = FEATURE_SETS.get(set_name, ["g0_", "g1_", "g2_", "g3_"])
    return [c for c in feature_cols if any(c.startswith(p) for p in prefixes)]


def _subsample_split(
    df: pd.DataFrame, max_samples: int = 40000, random_state: int = 42
) -> pd.DataFrame:
    """Subsample dataset preserving positive class proportion if size exceeds max_samples."""
    if len(df) <= max_samples:
        return df
    pos = df[df["y"] == 1]
    neg = df[df["y"] == 0]
    frac = max_samples / len(df)
    n_pos = max(1, int(len(pos) * frac))
    n_neg = max_samples - n_pos
    pos_sample = pos.sample(n=min(n_pos, len(pos)), random_state=random_state)
    neg_sample = neg.sample(n=min(n_neg, len(neg)), random_state=random_state)
    return pd.concat([pos_sample, neg_sample]).sort_values("time").reset_index(drop=True)


def _evaluate_single_ablation(
    model_template: Any,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    days_test: np.ndarray | None = None,
) -> Tuple[Dict[str, Any], float, float]:
    """Train pipeline on subset and evaluate on validation and test sets."""
    pipeline = clone(model_template)
    pipeline.fit(X_train, y_train)

    p_val = pipeline.predict_proba(X_val)[:, 1] if hasattr(pipeline, "predict_proba") else pipeline.predict(X_val)
    best_th = tune_threshold_validation(y_val, p_val, metric="hss")

    p_test = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, "predict_proba") else pipeline.predict(X_test)
    metrics = compute_classification_metrics(y_test, p_test, threshold=best_th)

    ci_lower, ci_upper = metrics["hss"], metrics["hss"]
    if days_test is not None:
        try:
            df_boot = pd.DataFrame({"y_true": y_test, "proba": p_test, "day": days_test})
            ci_df = bootstrap_day_block_ci(df_boot, "y_true", "proba", day_col="day", threshold=best_th, n_bootstraps=100)
            hss_row = ci_df[ci_df["metric"] == "hss"]
            if not hss_row.empty:
                ci_lower = float(hss_row["ci_lower"].values[0])
                ci_upper = float(hss_row["ci_upper"].values[0])
        except Exception:
            pass

    return metrics, ci_lower, ci_upper


def run_feature_ablation(
    best_models: Any,
    datasets: Dict[int, pd.DataFrame] | None = None,
    horizons: List[int] = [5, 15, 30, 60],
    fast_mode: bool = True,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Evaluate A0 (G0), A1 (G0+G1), A2 (G0+G2), A3 (G0+G3), A4 (All) feature sets across horizons.
    Returns DataFrame reporting test HSS, delta HSS over A0 baseline, TSS, and confidence intervals.
    """
    records: List[Dict[str, Any]] = []

    model_dict: Dict[str, Any] = {}
    if isinstance(best_models, dict):
        model_dict = best_models
    elif isinstance(best_models, list):
        model_dict = {f"Model_{i+1}": m for i, m in enumerate(best_models)}
    else:
        model_dict = {"Best_Model": best_models}

    for horizon in horizons:
        df_h = None
        if datasets and horizon in datasets:
            df_h = datasets[horizon]
        else:
            path = f"data/processed/dataset_N{horizon}.csv"
            try:
                df_h = pd.read_csv(path)
            except Exception:
                continue

        # Split partitions
        train_df = df_h[df_h["split"] == "train"]
        val_df = df_h[df_h["split"] == "val"]
        test_df = df_h[df_h["split"] == "test"]

        if train_df.empty or test_df.empty:
            continue

        if fast_mode and len(train_df) > 40000:
            train_df = _subsample_split(train_df, max_samples=40000, random_state=random_state)

        all_clue_cols = [c for c in df_h.columns if c.startswith(("g0_", "g1_", "g2_", "g3_"))]
        y_tr, y_val, y_te = train_df["y"].values, val_df["y"].values, test_df["y"].values
        days_te = test_df["day"].values if "day" in test_df.columns else None

        for m_name, m_pipeline in model_dict.items():
            hss_a0 = 0.0
            tss_a0 = 0.0

            for set_key in ["A0_brightness", "A1_history", "A2_pulsations", "A3_phase", "A4_all"]:
                cols = filter_features_by_set(all_clue_cols, set_key)
                if not cols:
                    continue

                X_tr = train_df[cols]
                X_v = val_df[cols]
                X_te = test_df[cols]

                metrics, ci_lo, ci_up = _evaluate_single_ablation(
                    m_pipeline, X_tr, y_tr, X_v, y_val, X_te, y_te, days_test=days_te
                )

                if set_key == "A0_brightness":
                    hss_a0 = metrics["hss"]
                    tss_a0 = metrics["tss"]

                records.append({
                    "horizon": horizon,
                    "model": m_name,
                    "feature_set": set_key,
                    "description": FEATURE_SET_DESCRIPTIONS.get(set_key, set_key),
                    "n_features": len(cols),
                    "hss": metrics["hss"],
                    "delta_hss": metrics["hss"] - hss_a0,
                    "tss": metrics["tss"],
                    "delta_tss": metrics["tss"] - tss_a0,
                    "ci_lower": ci_lo,
                    "ci_upper": ci_up,
                    "brier": metrics["brier"],
                    "recall": metrics["recall"],
                    "precision": metrics["precision"],
                    "far": metrics["far"],
                    "threshold": metrics["threshold"],
                })

    return pd.DataFrame(records)
