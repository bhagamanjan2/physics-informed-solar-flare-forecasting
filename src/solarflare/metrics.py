"""
Metrics module for solar flare forecasting evaluation.
Includes True Skill Statistic (TSS), Heidke Skill Score (HSS), Brier score,
and standard meteorological verification metrics.

Owned by Agent A4 (Metrics and Evaluation).
"""

from typing import Dict, Any, Union
import numpy as np
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    brier_score_loss,
    roc_auc_score,
    precision_recall_curve,
    auc,
)


def compute_tss(tp: int, fn: int, fp: int, tn: int) -> float:
    """
    Compute True Skill Statistic (TSS), also known as the Hanssen-Kuipers discriminant or Peirce skill score.

    Formula:
        TSS = TPR - FPR = TP / (TP + FN) - FP / (FP + TN)

    Properties:
        - Range: [-1, +1]
        - TSS = +1: Perfect forecast (100% hits, 0% false alarms)
        - TSS = 0: No skill (forecast is no better than random or constant)
        - TSS < 0: Forecast is worse than random (systematically inverted)
        - Crucial property: Independent of the base rate / class prevalence.
          TSS is unaffected by changing the ratio of flares to quiet periods.

    Worked Example:
        TP=8, FN=2, FP=20, TN=970 -> TPR = 8/10 = 0.80, FPR = 20/990 = 0.0202 -> TSS = 0.7798 (~0.78)
    """
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return float(tpr - fpr)


def compute_hss(tp: int, fn: int, fp: int, tn: int) -> float:
    """
    Compute Heidke Skill Score (HSS).

    Measures the fractional improvement of the forecast over random chance:
        HSS = (Accuracy - Expected_Random_Accuracy) / (1 - Expected_Random_Accuracy)

    Expanded 2x2 formula:
        HSS = 2 * (TP * TN - FN * FP) / [(TP + FN)*(FN + TN) + (TP + FP)*(FP + TN)]

    Properties:
        - Range: [-inf, +1], typically [-1, +1]
        - HSS = +1: Perfect forecast
        - HSS = 0: No skill over a random forecast that matches the marginal frequencies
        - HSS < 0: Skill worse than random chance
        - Unlike TSS, HSS IS sensitive to class imbalance, heavily penalizing false alarms.
          It is the primary operational metric in space weather forecasting (NOAA/SWPC).

    Worked Example:
        TP=8, FN=2, FP=20, TN=970 -> Numerator = 15,440, Denominator = 37,440 -> HSS = 0.4124 (~0.41)
    """
    numerator = 2.0 * (tp * tn - fn * fp)
    denominator = float((tp + fn) * (fn + tn) + (tp + fp) * (fp + tn))
    if denominator == 0.0:
        return 0.0
    return float(numerator / denominator)


def compute_classification_metrics(
    y_true: Union[np.ndarray, list],
    proba: Union[np.ndarray, list],
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute full dictionary of classification metrics at a given probability threshold.

    Parameters:
        y_true: Array-like of ground truth binary labels (0 or 1).
        proba: Array-like of predicted probabilities in [0, 1].
        threshold: Decision threshold to convert probabilities to binary predictions (default 0.5).

    Returns:
        Dictionary containing:
            - tp, fn, fp, tn: Confusion counts
            - tss: True Skill Statistic (Hanssen-Kuipers discriminant)
            - hss: Heidke Skill Score
            - recall: Probability of Detection (POD) / True Positive Rate = TP / (TP + FN)
            - far: False Alarm Ratio = FP / (TP + FP) [fraction of alarms that were false]
            - pofd: Probability of False Detection = FP / (FP + TN) [False Positive Rate]
            - precision: TP / (TP + FP) = 1 - FAR
            - f1: Harmonic mean of precision and recall
            - accuracy: Overall fraction correct = (TP + TN) / Total
            - brier: Mean squared probability error (Brier score loss, lower is better)
            - roc_auc: Area under ROC curve (threshold-independent)
            - pr_auc: Area under Precision-Recall curve (threshold-independent)
            - threshold: Evaluated decision threshold
    """
    y_true_arr = np.asarray(y_true).astype(int)
    proba_arr = np.asarray(proba).astype(float)

    y_pred = (proba_arr >= threshold).astype(int)
    cm = confusion_matrix(y_true_arr, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    tss = compute_tss(tp, fn, fp, tn)
    hss = compute_hss(tp, fn, fp, tn)

    rec = float(recall_score(y_true_arr, y_pred, zero_division=0))
    far = float(fp / (tp + fp)) if (tp + fp) > 0 else 0.0
    pofd = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    prec = float(precision_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))
    acc = float(accuracy_score(y_true_arr, y_pred))
    brier = float(brier_score_loss(y_true_arr, proba_arr))

    # ROC AUC calculation (handle single class safely)
    try:
        if len(np.unique(y_true_arr)) > 1:
            roc_auc = float(roc_auc_score(y_true_arr, proba_arr))
        else:
            roc_auc = 0.5
    except (ValueError, TypeError):
        roc_auc = 0.5

    # PR AUC calculation
    try:
        if len(np.unique(y_true_arr)) > 1:
            p_curve, r_curve, _ = precision_recall_curve(y_true_arr, proba_arr)
            pr_auc_val = float(auc(r_curve, p_curve))
            pr_auc = 0.0 if np.isnan(pr_auc_val) else pr_auc_val
        else:
            pr_auc = 0.0
    except (ValueError, TypeError):
        pr_auc = 0.0

    return {
        "tp": int(tp),
        "fn": int(fn),
        "fp": int(fp),
        "tn": int(tn),
        "tss": tss,
        "hss": hss,
        "recall": rec,
        "far": far,
        "pofd": pofd,
        "precision": prec,
        "f1": f1,
        "accuracy": acc,
        "brier": brier,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": float(threshold),
    }
