"""
Train all classification models and baselines across specified horizons.
Owned by Agent A3 (Split and Models).

Usage:
    python scripts/03_train_all.py --horizon 30
    python scripts/03_train_all.py --all-horizons
    python scripts/03_train_all.py --models lr,rf,gb --horizon 30
"""

import os
import argparse
import yaml
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Any

from solarflare.split import blocked_time_series_cv
from solarflare.models import (
    build_pipeline,
    get_param_distributions,
    train_and_tune,
)
from solarflare.metrics import compute_classification_metrics


DEFAULT_MODELS = [
    "lr",
    "knn",
    "dt",
    "rf",
    "svm",
    "nb",
    "gb",
    "all_quiet",
    "flux_threshold",
    "recent_flare",
]

CANONICAL_ALIASES = {
    "lr": "logistic_regression",
    "dt": "decision_tree",
    "rf": "random_forest",
    "svm": "svm_rbf",
    "nb": "gaussian_nb",
    "gb": "gradient_boosting",
    "recent_flare": "persistence",
}


def load_config(config_path: str = "config.yaml") -> dict:
    """Load project YAML configuration."""
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    return {
        "fast_mode": True,
        "seed": 42,
        "horizons": [30, 5, 15, 60],
        "split": {"cv_folds": 5, "gap_hours": 25},
    }


def train_horizon(
    horizon: int,
    models: List[str],
    config: dict,
    fast_mode: bool = True,
    dataset_path: str = None,
) -> pd.DataFrame:
    """
    Train and evaluate models for a single prediction horizon.
    """
    seed = config.get("seed", 42)
    n_splits = config.get("split", {}).get("cv_folds", 5)
    gap_hours = config.get("split", {}).get("gap_hours", 25)

    if dataset_path is None:
        dataset_path = f"data/processed/dataset_N{horizon}.csv"

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. "
            f"Please run scripts/02_build_dataset.py --horizon {horizon} first."
        )

    print(f"\n========================================================")
    print(f" TRAINING PIPELINE: Horizon N = {horizon} minutes")
    print(f" Dataset: {dataset_path} | Fast Mode: {fast_mode}")
    print(f"========================================================")

    df = pd.read_csv(dataset_path)
    feature_cols = [c for c in df.columns if c.startswith(("g0_", "g1_", "g2_", "g3_"))]
    print(f"Loaded {len(df):,} total rows with {len(feature_cols)} features:")
    print(f"  Features: {', '.join(feature_cols[:6])}... ({len(feature_cols)} total)")

    # Identify partitions
    train_mask = df["split"] == "train"
    val_mask = df["split"] == "val"
    test_mask = df["split"] == "test"

    df_train = df[train_mask].reset_index(drop=True)
    df_val = df[val_mask].reset_index(drop=True)
    df_test = df[test_mask].reset_index(drop=True)

    print(f"Partitions: Train={len(df_train):,} | Val={len(df_val):,} | Test={len(df_test):,}")
    print(f"Positive rate: Train={df_train['y'].mean():.2%} | Val={df_val['y'].mean():.2%} | Test={df_test['y'].mean():.2%}")

    X_train = df_train[feature_cols]
    y_train = df_train["y"]

    X_val = df_val[feature_cols]
    y_val = df_val["y"]

    X_test = df_test[feature_cols]
    y_test = df_test["y"]

    # Blocked time-series CV on train partition
    print(f"\nGenerating {n_splits} blocked time-series CV folds with {gap_hours}h blackout gaps...")
    cv_splits = blocked_time_series_cv(df_train, n_splits=n_splits, gap_hours=gap_hours)
    print(f"Constructed {len(cv_splits)} temporal CV splits.")

    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    cv_records = []

    for model_name in models:
        print(f"\n--- Training Model: {model_name.upper()} ---")
        pipeline = build_pipeline(model_name, random_state=seed)
        param_dist = get_param_distributions(model_name)

        # 1. Out-of-fold CV evaluation
        fold_scores = []
        for fold_idx, (tr_idx, va_idx) in enumerate(cv_splits):
            X_fold_tr = X_train.iloc[tr_idx]
            y_fold_tr = y_train.iloc[tr_idx]
            X_fold_va = X_train.iloc[va_idx]
            y_fold_va = y_train.iloc[va_idx]

            pipe_fold = build_pipeline(model_name, random_state=seed)
            if fast_mode and len(X_fold_tr) > 20000:
                s_idx = np.linspace(0, len(X_fold_tr) - 1, 20000, dtype=int)
                pipe_fold.fit(X_fold_tr.iloc[s_idx], y_fold_tr.iloc[s_idx])
            else:
                pipe_fold.fit(X_fold_tr, y_fold_tr)

            proba_va = pipe_fold.predict_proba(X_fold_va)[:, 1]
            metrics_va = compute_classification_metrics(y_fold_va, proba_va, threshold=0.5)

            # Record with both names for compatibility
            canonical_name = CANONICAL_ALIASES.get(model_name, model_name)
            for m_key in set([model_name, canonical_name]):
                cv_records.append({
                    "model": m_key,
                    "horizon": horizon,
                    "fold": fold_idx + 1,
                    "train_size": len(tr_idx),
                    "val_size": len(va_idx),
                    "hss": metrics_va["hss"],
                    "tss": metrics_va["tss"],
                    "brier": metrics_va["brier"],
                    "roc_auc": metrics_va["roc_auc"],
                    "pr_auc": metrics_va["pr_auc"],
                    "precision": metrics_va["precision"],
                    "recall": metrics_va["recall"],
                })
            fold_scores.append(metrics_va)

        mean_hss = np.mean([s["hss"] for s in fold_scores])
        mean_tss = np.mean([s["tss"] for s in fold_scores])
        mean_auc = np.mean([s["roc_auc"] for s in fold_scores])
        print(f"  CV Results (mean across {len(cv_splits)} folds): HSS={mean_hss:.4f} | TSS={mean_tss:.4f} | AUC={mean_auc:.4f}")

        # 2. Fit final model on full train partition
        print(f"  Fitting full model on training set (fast_mode={fast_mode})...")
        fitted_model = train_and_tune(
            pipeline,
            param_dist=param_dist,
            X_train=X_train,
            y_train=y_train,
            cv_splits=cv_splits,
            fast_mode=fast_mode,
            random_state=seed,
        )

        # 3. Save model joblib (both short and canonical names)
        canonical_name = CANONICAL_ALIASES.get(model_name, model_name)
        for m_key in set([model_name, canonical_name]):
            model_filename = f"models/{m_key}_N{horizon}.joblib"
            joblib.dump(fitted_model, model_filename)

        print(f"  Saved model artifacts to models/{model_name}_N{horizon}.joblib")

        # 4. Generate predictions across validation and test partitions
        proba_val = fitted_model.predict_proba(X_val)[:, 1]
        proba_test = fitted_model.predict_proba(X_test)[:, 1]

        df_pred_val = pd.DataFrame({
            "time": df_val["time"].values,
            "y_true": y_val.values,
            "proba": proba_val,
            "split": "val",
        })

        df_pred_test = pd.DataFrame({
            "time": df_test["time"].values,
            "y_true": y_test.values,
            "proba": proba_test,
            "split": "test",
        })

        df_predictions = pd.concat([df_pred_val, df_pred_test], ignore_index=True)

        for m_key in set([model_name, canonical_name]):
            pred_path = f"results/predictions_N{horizon}_{m_key}.csv"
            df_predictions.to_csv(pred_path, index=False)

        print(f"  Saved predictions ({len(df_predictions):,} rows) to results/predictions_N{horizon}_{model_name}.csv")

    df_cv_all = pd.DataFrame(cv_records)
    return df_cv_all


def main():
    parser = argparse.ArgumentParser(description="Train and evaluate solar flare prediction models.")
    parser.add_argument("--horizon", type=int, default=30, help="Horizon in minutes (5, 15, 30, 60).")
    parser.add_argument("--all-horizons", action="store_true", help="Train on all horizons.")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model names.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml.")
    parser.add_argument("--fast-mode", dest="fast_mode", action="store_true", default=None, help="Enable fast mode.")
    parser.add_argument("--no-fast-mode", dest="fast_mode", action="store_false", help="Disable fast mode.")
    args = parser.parse_args()

    config = load_config(args.config)
    fast_mode = args.fast_mode if args.fast_mode is not None else config.get("fast_mode", True)

    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = DEFAULT_MODELS

    horizons = [30, 5, 15, 60] if args.all_horizons else [args.horizon]

    all_cv_dfs = []
    for h in horizons:
        try:
            df_cv_h = train_horizon(horizon=h, models=models, config=config, fast_mode=fast_mode)
            all_cv_dfs.append(df_cv_h)
        except FileNotFoundError as e:
            print(f"Skipping horizon N={h}: {e}")

    if all_cv_dfs:
        df_cv_total = pd.concat(all_cv_dfs, ignore_index=True)
        cv_csv_path = "results/cv_scores.csv"

        if os.path.exists(cv_csv_path):
            existing_cv = pd.read_csv(cv_csv_path)
            existing_filtered = existing_cv[~existing_cv["horizon"].isin(horizons)]
            df_cv_total = pd.concat([existing_filtered, df_cv_total], ignore_index=True)

        df_cv_total.to_csv(cv_csv_path, index=False)
        print(f"\nSuccessfully saved updated CV scores to {cv_csv_path}")

        print("\n=== SUMMARY OF CROSS-VALIDATION PERFORMANCE (MEAN HSS & TSS) ===")
        summary = (
            df_cv_total.groupby(["horizon", "model"])[["hss", "tss", "roc_auc", "brier"]]
            .mean()
            .reset_index()
        )
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
