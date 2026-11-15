"""
Model evaluation, validation threshold tuning, day-block bootstrap confidence intervals,
and deterministic model selection rule.

Owned by Agent A4 (Metrics and Evaluation).
"""

from typing import Dict, Any, List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from solarflare.metrics import (
    compute_tss,
    compute_hss,
    compute_classification_metrics,
)


MODEL_COMPLEXITY_ORDER: Dict[str, int] = {
    # Baseline heuristics (simplest)
    "all_quiet": 0,
    "flux_threshold": 1,
    "persistence": 2,
    # Simple linear / generative parametric models
    "logistic_regression": 10,
    "logisticregression": 10,
    "lr": 10,
    "gaussian_nb": 11,
    "gaussiannb": 11,
    "gnb": 11,
    # Non-linear tree / instance models
    "decision_tree": 20,
    "decisiontree": 20,
    "dt": 20,
    "knn": 25,
    "kneighbors": 25,
    # Complex ensembles and kernel methods
    "random_forest": 30,
    "randomforest": 30,
    "rf": 30,
    "gradient_boosting": 35,
    "gradientboosting": 35,
    "gb": 35,
    "svm_rbf": 40,
    "svm": 40,
    "svc": 40,
}


def get_model_complexity(model_name: str) -> int:
    """Return numeric complexity rank for a model (lower means simpler)."""
    norm_name = str(model_name).lower().replace(" ", "_").replace("-", "_")
    for key, val in MODEL_COMPLEXITY_ORDER.items():
        if key in norm_name or norm_name in key:
            return val
    return 50  # Default complexity if unrecognized


def tune_threshold_validation(
    y_val: Union[np.ndarray, pd.Series, list],
    proba_val: Union[np.ndarray, pd.Series, list],
    metric: str = "tss",
    num_steps: int = 100,
) -> float:
    """
    Tune classification decision threshold strictly on the validation set to maximize metric.
    Never uses the test set.

    Parameters:
        y_val: Validation ground truth binary labels (0 or 1).
        proba_val: Validation predicted probabilities in [0, 1].
        metric: Optimization criterion ('hss', 'tss', 'f1', etc.). Default 'hss'.
        num_steps: Number of candidate thresholds to test between 0.01 and 0.99.

    Returns:
        best_threshold: Optimal decision threshold as float.
    """
    y_arr = np.asarray(y_val).astype(int)
    p_arr = np.asarray(proba_val).astype(float)

    total_pos = int(np.sum(y_arr == 1))
    total_neg = int(len(y_arr) - total_pos)

    thresholds = np.linspace(0.01, 0.99, num_steps)
    best_score = -999.0
    best_thresh = 0.5

    for th in thresholds:
        y_pred = (p_arr >= th)
        tp = int(np.sum(y_pred & (y_arr == 1)))
        fp = int(np.sum(y_pred & (y_arr == 0)))
        fn = total_pos - tp
        tn = total_neg - fp

        if metric == "hss":
            score = compute_hss(tp, fn, fp, tn)
        elif metric == "tss":
            score = compute_tss(tp, fn, fp, tn)
        elif metric == "f1":
            denom = 2 * tp + fp + fn
            score = (2.0 * tp / denom) if denom > 0 else 0.0
        elif metric == "recall":
            score = (tp / total_pos) if total_pos > 0 else 0.0
        elif metric == "precision":
            score = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        else:
            m = compute_classification_metrics(y_arr, p_arr, threshold=th)
            score = m.get(metric, 0.0)

        if score > best_score:
            best_score = score
            best_thresh = float(th)

    return float(best_thresh)


def compute_threshold_curve(
    y_true: Union[np.ndarray, pd.Series, list],
    proba: Union[np.ndarray, pd.Series, list],
    num_steps: int = 100,
) -> pd.DataFrame:
    """
    Compute classification metrics across candidate thresholds.
    Used for plotting the validation threshold sweep curve (Figure F15).
    """
    y_arr = np.asarray(y_true).astype(int)
    p_arr = np.asarray(proba).astype(float)

    total_pos = int(np.sum(y_arr == 1))
    total_neg = int(len(y_arr) - total_pos)

    thresholds = np.linspace(0.01, 0.99, num_steps)
    rows = []

    for th in thresholds:
        y_pred = (p_arr >= th)
        tp = int(np.sum(y_pred & (y_arr == 1)))
        fp = int(np.sum(y_pred & (y_arr == 0)))
        fn = total_pos - tp
        tn = total_neg - fp

        tss = compute_tss(tp, fn, fp, tn)
        hss = compute_hss(tp, fn, fp, tn)
        recall = (tp / total_pos) if total_pos > 0 else 0.0
        far = (fp / (tp + fp)) if (tp + fp) > 0 else 0.0
        pofd = (fp / total_neg) if total_neg > 0 else 0.0
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        f1 = (2.0 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0

        rows.append({
            "threshold": float(th),
            "hss": float(hss),
            "tss": float(tss),
            "recall": float(recall),
            "far": float(far),
            "pofd": float(pofd),
            "precision": float(precision),
            "f1": float(f1),
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
        })

    return pd.DataFrame(rows)


def precompute_day_confusion(
    df_eval: pd.DataFrame,
    y_true_col: str,
    proba_col: str,
    day_col: str = "day",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Precompute per-day confusion matrix components (TP, FN, FP, TN, Brier sum, N).
    Enables instant day-block bootstrap resampling without expensive row-level concatenations.
    """
    df = df_eval.copy()
    if day_col not in df.columns:
        if "time" in df.columns:
            df[day_col] = pd.to_datetime(df["time"]).dt.strftime("%Y-%m-%d")
        else:
            # Fallback: divide into 1440-minute blocks (1 day equivalent)
            df[day_col] = (np.arange(len(df)) // 1440).astype(str)

    y_arr = df[y_true_col].values.astype(int)
    p_arr = df[proba_col].values.astype(float)
    y_pred = (p_arr >= threshold).astype(int)

    tp_mask = (y_arr == 1) & (y_pred == 1)
    fn_mask = (y_arr == 1) & (y_pred == 0)
    fp_mask = (y_arr == 0) & (y_pred == 1)
    tn_mask = (y_arr == 0) & (y_pred == 0)
    brier_sq = (p_arr - y_arr) ** 2

    temp_df = pd.DataFrame({
        day_col: df[day_col].values,
        "tp": tp_mask.astype(int),
        "fn": fn_mask.astype(int),
        "fp": fp_mask.astype(int),
        "tn": tn_mask.astype(int),
        "brier_sum": brier_sq.astype(float),
        "n_samples": 1,
    })

    day_agg = temp_df.groupby(day_col, as_index=False).sum()
    return day_agg


def bootstrap_day_block_ci(
    df_eval: pd.DataFrame,
    y_true_col: str,
    proba_col: str,
    day_col: str = "day",
    threshold: float = 0.5,
    n_bootstraps: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute 95% confidence intervals by block-bootstrapping whole days.
    Preserves intra-day temporal autocorrelation.

    Uses PRECOMPUTED per-day confusion counts for ultra-fast, vectorized execution.
    """
    day_agg = precompute_day_confusion(
        df_eval=df_eval,
        y_true_col=y_true_col,
        proba_col=proba_col,
        day_col=day_col,
        threshold=threshold,
    )

    n_days = len(day_agg)
    if n_days == 0:
        return pd.DataFrame()

    tp_days = day_agg["tp"].values
    fn_days = day_agg["fn"].values
    fp_days = day_agg["fp"].values
    tn_days = day_agg["tn"].values
    brier_days = day_agg["brier_sum"].values
    count_days = day_agg["n_samples"].values

    # Pre-draw bootstrap indices: shape (n_bootstraps, n_days)
    rng = np.random.default_rng(random_state)
    boot_indices = rng.integers(0, n_days, size=(n_bootstraps, n_days))

    # Vectorized summation across resampled days
    tp_b = tp_days[boot_indices].sum(axis=1)
    fn_b = fn_days[boot_indices].sum(axis=1)
    fp_b = fp_days[boot_indices].sum(axis=1)
    tn_b = tn_days[boot_indices].sum(axis=1)
    brier_sum_b = brier_days[boot_indices].sum(axis=1)
    n_total_b = count_days[boot_indices].sum(axis=1)

    # Derived metrics across all bootstrap replications (safe division)
    pos_b = tp_b + fn_b
    neg_b = fp_b + tn_b
    pred_pos_b = tp_b + fp_b
    pred_neg_b = fn_b + tn_b

    tpr_b = np.divide(tp_b, pos_b, out=np.zeros_like(tp_b, dtype=float), where=pos_b > 0)
    fpr_b = np.divide(fp_b, neg_b, out=np.zeros_like(fp_b, dtype=float), where=neg_b > 0)
    tss_b = tpr_b - fpr_b

    hss_num = 2.0 * (tp_b * tn_b - fn_b * fp_b)
    hss_denom = (pos_b * pred_neg_b) + (pred_pos_b * neg_b)
    hss_b = np.divide(hss_num, hss_denom, out=np.zeros_like(hss_num, dtype=float), where=hss_denom > 0)

    recall_b = tpr_b
    far_b = np.divide(fp_b, pred_pos_b, out=np.zeros_like(fp_b, dtype=float), where=pred_pos_b > 0)
    pofd_b = fpr_b
    precision_b = np.divide(tp_b, pred_pos_b, out=np.zeros_like(tp_b, dtype=float), where=pred_pos_b > 0)
    f1_denom = (2.0 * tp_b) + fp_b + fn_b
    f1_b = np.divide(2.0 * tp_b, f1_denom, out=np.zeros_like(tp_b, dtype=float), where=f1_denom > 0)
    brier_b = np.divide(brier_sum_b, n_total_b, out=np.zeros_like(brier_sum_b, dtype=float), where=n_total_b > 0)
    accuracy_b = np.divide(tp_b + tn_b, n_total_b, out=np.zeros_like(tp_b, dtype=float), where=n_total_b > 0)

    # Point estimates on the whole sample
    point_metrics = compute_classification_metrics(
        df_eval[y_true_col].values, df_eval[proba_col].values, threshold=threshold
    )

    metric_arrays = {
        "tss": (tss_b, point_metrics["tss"]),
        "hss": (hss_b, point_metrics["hss"]),
        "recall": (recall_b, point_metrics["recall"]),
        "far": (far_b, point_metrics["far"]),
        "pofd": (pofd_b, point_metrics["pofd"]),
        "precision": (precision_b, point_metrics["precision"]),
        "f1": (f1_b, point_metrics["f1"]),
        "brier": (brier_b, point_metrics["brier"]),
        "accuracy": (accuracy_b, point_metrics["accuracy"]),
        "roc_auc": (None, point_metrics["roc_auc"]),
        "pr_auc": (None, point_metrics["pr_auc"]),
    }

    summary = []
    low_p = 100.0 * (alpha / 2.0)
    high_p = 100.0 * (1.0 - alpha / 2.0)

    for m_name, (boot_vals, pt_est) in metric_arrays.items():
        if boot_vals is not None:
            mean_val = float(np.mean(boot_vals))
            ci_low = float(np.percentile(boot_vals, low_p))
            ci_high = float(np.percentile(boot_vals, high_p))
            std_val = float(np.std(boot_vals))
        else:
            # Ranking metrics on full dataset
            mean_val = float(pt_est)
            ci_low = float(pt_est)
            ci_high = float(pt_est)
            std_val = 0.0

        summary.append({
            "metric": m_name,
            "point_estimate": float(pt_est),
            "mean": mean_val,
            "ci_lower": ci_low,
            "ci_upper": ci_high,
            "std": std_val,
        })

    return pd.DataFrame(summary)


def compute_calibration(
    y_true: Union[np.ndarray, pd.Series, list],
    proba: Union[np.ndarray, pd.Series, list],
    n_bins: int = 10,
    strategy: str = "uniform",
) -> Dict[str, Any]:
    """
    Compute calibration curve and probability bin distribution for reliability diagrams (Figure F16).

    Parameters:
        y_true: True binary target labels.
        proba: Forecasted probabilities.
        n_bins: Number of probability bins (default 10).
        strategy: 'uniform' (equal width bins) or 'quantile' (equal sample bins).

    Returns:
        Dictionary with:
            - prob_true: Empirical fraction of positives in each bin
            - prob_pred: Mean predicted probability in each bin
            - bin_counts: Number of samples falling into each bin
            - bin_edges: Array of bin boundary values
            - brier_score: Overall Brier score loss
    """
    y_arr = np.asarray(y_true).astype(int)
    p_arr = np.asarray(proba).astype(float)

    prob_true, prob_pred = calibration_curve(
        y_arr, p_arr, n_bins=n_bins, strategy=strategy
    )
    brier = float(brier_score_loss(y_arr, p_arr))

    # Histogram of predicted probabilities
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_counts, _ = np.histogram(p_arr, bins=bin_edges)

    return {
        "prob_true": prob_true.tolist(),
        "prob_pred": prob_pred.tolist(),
        "bin_counts": bin_counts.tolist(),
        "bin_edges": bin_edges.tolist(),
        "brier_score": brier,
    }


def apply_selection_rule(
    cv_scores_df: pd.DataFrame,
    thresholds: Optional[Dict[str, float]] = None,
    prefer_simpler_within_se: bool = True,
) -> Dict[str, Any]:
    """
    Fixed Model Selection Rule (locked in advance, TSS primary):
    1. Highest mean TSS across blocked CV folds at the tuned validation threshold.
    2. Ties broken by higher HSS.
    3. Ties broken by lower Brier score.
    4. Ties broken by simpler model (complexity ascending).
    5. If candidate models are within 1 standard error of top TSS, prefer the simpler model.

    Parameters:
        cv_scores_df: DataFrame of cross-validation results.
                      Expected columns: 'model', and either 'hss' or ('hss_mean', 'hss_std').
        thresholds: Optional dictionary mapping model names to tuned validation thresholds.
        prefer_simpler_within_se: Whether to apply 1-SE parsimony rule.

    Returns:
        Dictionary with:
            - selected_model: Winning model name
            - rationale: Multi-line plain-English defense for viva
            - ranking: List of models ranked by CV performance
            - summary_table: Detailed summary DataFrame
    """
    df = cv_scores_df.copy()

    # Aggregate if multiple folds per model
    if "fold" in df.columns:
        agg = df.groupby("model").agg(
            hss_mean=("hss", "mean"),
            hss_std=("hss", "std"),
            tss_mean=("tss", "mean"),
            tss_std=("tss", "std"),
            brier_mean=("brier", "mean"),
            brier_std=("brier", "std"),
            n_folds=("fold", "count"),
        ).reset_index()
    else:
        # Already aggregated
        agg = df.copy()
        if "hss_mean" not in agg.columns and "hss" in agg.columns:
            agg["hss_mean"] = agg["hss"]
            agg["hss_std"] = 0.0
        if "tss_mean" not in agg.columns and "tss" in agg.columns:
            agg["tss_mean"] = agg["tss"]
            agg["tss_std"] = 0.0
        if "brier_mean" not in agg.columns and "brier" in agg.columns:
            agg["brier_mean"] = agg["brier"]
            agg["brier_std"] = 0.0
        if "n_folds" not in agg.columns:
            agg["n_folds"] = 5

    # Standard error of TSS and HSS
    agg["tss_se"] = agg["tss_std"] / np.sqrt(np.maximum(agg["n_folds"], 1))
    agg["hss_se"] = agg["hss_std"] / np.sqrt(np.maximum(agg["n_folds"], 1))
    agg["complexity"] = agg["model"].apply(get_model_complexity)

    # Sort strictly by: 1. TSS desc, 2. HSS desc, 3. Brier asc, 4. Complexity asc
    agg_sorted = agg.sort_values(
        by=["tss_mean", "hss_mean", "brier_mean", "complexity"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)

    best_raw_model = agg_sorted.iloc[0]["model"]
    best_raw_tss = agg_sorted.iloc[0]["tss_mean"]
    best_raw_se = agg_sorted.iloc[0]["tss_se"]

    selected_model = best_raw_model
    rule_note = "Strict highest mean TSS in blocked cross-validation"

    if prefer_simpler_within_se:
        # Check models within 1 standard error of best TSS
        se_bound = best_raw_tss - best_raw_se
        within_se = agg_sorted[agg_sorted["tss_mean"] >= se_bound].copy()
        within_se_sorted = within_se.sort_values(
            by=["complexity", "tss_mean"], ascending=[True, False]
        )
        simpler_model = within_se_sorted.iloc[0]["model"]
        if simpler_model != best_raw_model:
            selected_model = simpler_model
            rule_note = f"Parsimony rule: {simpler_model} is simpler and within 1 SE ({best_raw_se:.4f}) of top TSS"

    # Generate plain-English viva rationale
    top_row = agg_sorted[agg_sorted["model"] == selected_model].iloc[0]
    rationale = (
        f"Selected '{selected_model}' based on the locked decision rule: "
        f"Mean CV TSS = {top_row['tss_mean']:.4f} (+/- {top_row['tss_se']:.4f} SE), "
        f"HSS = {top_row['hss_mean']:.4f}, Brier = {top_row['brier_mean']:.4f}. "
        f"({rule_note})."
    )

    ranking_list = []
    for rank_idx, r in agg_sorted.iterrows():
        ranking_list.append({
            "rank": rank_idx + 1,
            "model": r["model"],
            "tss_mean": float(r["tss_mean"]),
            "tss_std": float(r["tss_std"]),
            "tss_se": float(r["tss_se"]),
            "hss_mean": float(r["hss_mean"]),
            "hss_std": float(r["hss_std"]),
            "hss_se": float(r["hss_se"]),
            "brier_mean": float(r["brier_mean"]),
            "complexity": int(r["complexity"]),
        })

    return {
        "selected_model": str(selected_model),
        "best_raw_model": str(best_raw_model),
        "rule_applied": rule_note,
        "rationale": rationale,
        "ranking": ranking_list,
        "summary_df": agg_sorted,
    }
