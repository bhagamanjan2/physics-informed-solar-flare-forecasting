"""
Script 05: Feature Set Ablation and Interpretability.
Owned by Agent A5 (Ablation and Insight).

Runs:
1. Feature set ablations A0 (G0) through A4 (All) on the best ML models across all horizons.
2. Permutation feature importance on the test split for N=30.
3. Error analysis characterizing false alarms and missed flares by class.

Outputs:
- results/ablation.csv
- results/importance.csv
- results/error_analysis.json
"""

import sys
import os
import json
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from solarflare.models import build_pipeline
from solarflare.ablation import run_feature_ablation
from solarflare.interpret import compute_permutation_importance, error_analysis
from solarflare.evaluate import tune_threshold_validation


def get_top_ml_models(selection_file: Path) -> list[str]:
    """Extract top 2 non-baseline ML models from selection.json."""
    default_models = ["random_forest", "logistic_regression"]
    if not selection_file.exists():
        return default_models

    try:
        with open(selection_file, "r") as f:
            sel_data = json.load(f)

        candidates = []
        # Check by_horizon first (prefer N30)
        by_h = sel_data.get("by_horizon", {})
        target_h = by_h.get("N30", {}) or next(iter(by_h.values()), {})
        ranking = target_h.get("ranking", [])
        for r in ranking:
            m = r.get("model", "")
            if m and m not in ["persistence", "all_quiet", "flux_threshold", "recent_flare"]:
                if m not in candidates:
                    candidates.append(m)

        if len(candidates) >= 2:
            return candidates[:2]
        elif len(candidates) == 1:
            candidates.append("logistic_regression")
            return candidates
    except Exception as e:
        print(f"Note: Error parsing selection.json: {e}")

    return default_models


def main():
    config_path = Path("config.yaml")
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {"horizons": [5, 15, 30, 60], "fast_mode": True}

    horizons = config.get("horizons", [5, 15, 30, 60])
    fast_mode = config.get("fast_mode", True)
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    print("==================================================")
    print(" Solar Flare Forecasting: 05 Ablation & Insight   ")
    print("==================================================")

    # 1. Determine best ML models to ablate
    selection_file = results_dir / "selection.json"
    best_model_names = get_top_ml_models(selection_file)
    print(f"Top 2 ML Models selected for ablation: {best_model_names}")

    models_to_test = {
        name: build_pipeline(name)
        for name in best_model_names
    }

    # 2. Run feature ablation across horizons
    print(f"\n--- Running Feature Ablation (A0 - A4) across horizons {horizons} ---")
    df_ablation = run_feature_ablation(
        best_models=models_to_test,
        horizons=horizons,
        fast_mode=fast_mode,
        random_state=config.get("seed", 42),
    )

    if not df_ablation.empty:
        ablation_csv = results_dir / "ablation.csv"
        df_ablation.to_csv(ablation_csv, index=False)
        print(f"Saved ablation results to {ablation_csv}")
        print("\nSummary of Ablation Results (HSS & Delta HSS):")
        cols_show = ["horizon", "model", "feature_set", "hss", "delta_hss", "ci_lower", "ci_upper"]
        print(df_ablation[[c for c in cols_show if c in df_ablation.columns]].to_string(index=False))
    else:
        print("Warning: No ablation records generated (check if dataset files exist).")

    # 3. Permutation Feature Importance for N=30
    h_target = 30 if 30 in horizons else horizons[0]
    ds_target_path = Path(f"data/processed/dataset_N{h_target}.csv")

    if ds_target_path.exists():
        print(f"\n--- Computing Permutation Importance for N={h_target} ---")
        df_target = pd.read_csv(ds_target_path)
        train_df = df_target[df_target["split"] == "train"]
        val_df = df_target[df_target["split"] == "val"]
        test_df = df_target[df_target["split"] == "test"]

        clue_cols = [c for c in df_target.columns if c.startswith(("g0_", "g1_", "g2_", "g3_"))]
        
        # Fit primary model on train
        primary_name = best_model_names[0]
        model_pipe = build_pipeline(primary_name)
        
        # Subsample train if in fast_mode
        if fast_mode and len(train_df) > 40000:
            from solarflare.ablation import _subsample_split
            train_df = _subsample_split(train_df, max_samples=40000, random_state=config.get("seed", 42))

        X_tr = train_df[clue_cols]
        y_tr = train_df["y"].values
        X_val = val_df[clue_cols]
        y_val = val_df["y"].values
        X_te = test_df[clue_cols]
        y_te = test_df["y"].values

        model_pipe.fit(X_tr, y_tr)
        p_val = model_pipe.predict_proba(X_val)[:, 1] if hasattr(model_pipe, "predict_proba") else model_pipe.predict(X_val)
        th = tune_threshold_validation(y_val, p_val, metric="hss")

        df_imp = compute_permutation_importance(
            model=model_pipe,
            X_test=X_te,
            y_test=y_te,
            scoring="hss",
            threshold=th,
            n_repeats=5 if fast_mode else 10,
        )

        imp_csv = results_dir / "importance.csv"
        df_imp.to_csv(imp_csv, index=False)
        print(f"Saved feature importance to {imp_csv}")
        print("\nTop 10 Most Important Features:")
        print(df_imp.head(10).to_string(index=False))

        # 4. Error Analysis
        print(f"\n--- Running Error Analysis for N={h_target} ---")
        flares_path = Path("data/processed/flares.csv")
        flares_df = pd.read_csv(flares_path) if flares_path.exists() else None

        p_test = model_pipe.predict_proba(X_te)[:, 1] if hasattr(model_pipe, "predict_proba") else model_pipe.predict(X_te)
        y_pred_te = (p_test >= th).astype(int)

        err_res = error_analysis(
            df_test=test_df,
            flares_df=flares_df,
            y_true=y_te,
            y_pred=y_pred_te,
            horizon_min=h_target,
        )

        # Remove non-serializable DataFrame rows for json dump
        err_serializable = {
            "n_tp": err_res["n_tp"],
            "n_fn": err_res["n_fn"],
            "n_fp": err_res["n_fp"],
            "n_tn": err_res["n_tn"],
            "false_alarm_profile": err_res.get("false_alarm_profile", {}),
            "flares_by_class": err_res.get("flares_by_class", {}),
            "missed_flares": err_res.get("missed_flares", []),
        }

        err_json = results_dir / "error_analysis.json"
        with open(err_json, "w") as f:
            json.dump(err_serializable, f, indent=2)
        print(f"Saved error analysis to {err_json}")
        print(f"Caught vs Missed by Class: {err_res.get('flares_by_class')}")
    else:
        print(f"Dataset {ds_target_path} not found. Stand by for Phase 2 datasets.")

    print("\n[Done] Script 05 execution complete.")


if __name__ == "__main__":
    main()
