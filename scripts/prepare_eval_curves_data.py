"""
Extract compact curve data (ROC, PR, threshold sweep, calibration, and 48h case study)
from predictions so the notebook runs instantly without needing 160MB of raw prediction dumps.
"""
import os, sys, json
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix
from sklearn.calibration import calibration_curve

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

OUT_FILE = os.path.join(ROOT, "results", "eval_curves_data.json")

data = {}

# 1. ROC and PR curves for candidate models
print("Extracting ROC & PR curves...")
models = ["nb", "lr", "rf", "gb", "dt"]
data["roc_pr"] = {}

for m in models:
    csv_path = f"results/predictions_N30_{m}.csv"
    if not os.path.exists(csv_path):
        continue
    df = pd.read_csv(csv_path)
    fpr, tpr, _ = roc_curve(df["y_true"], df["proba"])
    prec, rec, _ = precision_recall_curve(df["y_true"], df["proba"])
    
    # Subsample to 100 points for compact size while preserving curve shape
    idx_roc = np.linspace(0, len(fpr)-1, min(100, len(fpr))).astype(int)
    idx_pr = np.linspace(0, len(prec)-1, min(100, len(prec))).astype(int)
    
    data["roc_pr"][m] = {
        "fpr": fpr[idx_roc].round(4).tolist(),
        "tpr": tpr[idx_roc].round(4).tolist(),
        "roc_auc": round(float(auc(fpr, tpr)), 4),
        "rec": rec[idx_pr].round(4).tolist(),
        "prec": prec[idx_pr].round(4).tolist(),
        "pr_auc": round(float(auc(rec, prec)), 4)
    }

# 2. Threshold sweep for Gaussian NB
print("Extracting threshold sweep for NB...")
df_nb = pd.read_csv("results/predictions_N30_nb.csv")
thresholds = np.linspace(0.05, 0.95, 30).round(4)
hss_vals, tss_vals, rec_vals, far_vals = [], [], [], []
y_true = df_nb["y_true"].values
proba = df_nb["proba"].values

for t in thresholds:
    yp = (proba >= t).astype(int)
    c = confusion_matrix(y_true, yp)
    tn, fp, fn, tp = c.ravel()
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    far = fp / (tp + fp) if (tp + fp) > 0 else 0
    tss = (tp / (tp + fn) - fp / (fp + tn)) if (tp + fn) > 0 and (fp + tn) > 0 else 0
    hss_den = ((tp + fn)*(fn + tn) + (tp + fp)*(fp + tn))
    hss = (2 * (tp * tn - fp * fn) / hss_den) if hss_den > 0 else 0
    hss_vals.append(round(float(hss), 4))
    tss_vals.append(round(float(tss), 4))
    rec_vals.append(round(float(rec), 4))
    far_vals.append(round(float(far), 4))

data["threshold_sweep"] = {
    "thresholds": thresholds.tolist(),
    "hss": hss_vals,
    "tss": tss_vals,
    "recall": rec_vals,
    "far": far_vals,
    "optimal_threshold": 0.4753
}

# 3. Reliability calibration curve
print("Extracting calibration curve for NB...")
prob_true, prob_pred = calibration_curve(df_nb["y_true"], df_nb["proba"], n_bins=10)
hist_counts, bin_edges = np.histogram(df_nb["proba"], bins=25)

data["calibration"] = {
    "prob_pred": prob_pred.round(4).tolist(),
    "prob_true": prob_true.round(4).tolist(),
    "hist_counts": hist_counts.tolist(),
    "bin_edges": bin_edges.round(4).tolist()
}

# 4. 48-Hour case study slice
print("Extracting 48h case study slice...")
df_case = df_nb.tail(2880).reset_index(drop=True)
data["case_study_48h"] = {
    "y_true": df_case["y_true"].astype(int).tolist(),
    "proba": df_case["proba"].round(4).tolist()
}

with open(OUT_FILE, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

size_kb = os.path.getsize(OUT_FILE) / 1024
print(f"Compact curves & case study data written to: {OUT_FILE} ({size_kb:.1f} KB)")
