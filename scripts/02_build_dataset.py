"""
Script 02: Build feature and label datasets for solar flare forecasting.
Owned by Agent A2 (Labels and Features).

Builds data/processed/dataset_N{N}.csv for horizons N = 30 first, then 5, 15, 60.
Strictly adheres to:
1. Target: SIGNIFICANT_FLARE (abs_class_simple in {C, M, X} and fclass_simple in {C, M, X}).
2. Skip condition: in-progress significant flare (tstart <= t <= tend).
3. Chronological split with 25h blackout gaps.
4. Strict No-Leakage guarantee.
"""

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np

# Ensure src/ is importable
sys.path.insert(0, os.path.abspath("src"))

from solarflare.labels import build_flare_labels, identify_in_progress_minutes
from solarflare.features import assemble_features
from solarflare.split import chronological_split


def build_datasets(
    flux_path: str = "data/processed/flux_1min.csv",
    flares_path: str = "data/processed/flares.csv",
    output_dir: str = "data/processed",
    horizons: list = [30, 5, 15, 60],
) -> None:
    """
    Build processed datasets for all prediction horizons.
    N = 30 is built and saved FIRST as mandated by the swarm execution contract.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"[{time.strftime('%H:%M:%S')}] Loading flux from {flux_path}...")
    flux_df = pd.read_csv(flux_path)
    print(f"[{time.strftime('%H:%M:%S')}] Loaded {len(flux_df):,} flux minutes.")

    print(f"[{time.strftime('%H:%M:%S')}] Loading significant flares from {flares_path}...")
    flares_df = pd.read_csv(flares_path)
    print(f"[{time.strftime('%H:%M:%S')}] Loaded {len(flares_df):,} significant flares.")

    # 1. Compute chronological split (60% Train, 25h gap, 15% Val, 25h gap, 25% Test)
    print(f"[{time.strftime('%H:%M:%S')}] Computing chronological split with 25h gaps...")
    split_df = chronological_split(
        flux_df[["time"]],
        train_ratio=0.60,
        val_ratio=0.15,
        test_ratio=0.25,
        gap_hours=25,
    )

    # 2. Identify in-progress flare minutes and mark as 'skip'
    print(f"[{time.strftime('%H:%M:%S')}] Identifying in-progress significant flare minutes...")
    in_progress = identify_in_progress_minutes(flux_df["time"], flares_df)
    n_skipped = int(in_progress.sum())
    pct_skipped = (n_skipped / len(flux_df)) * 100
    print(f"[{time.strftime('%H:%M:%S')}] In-progress flare minutes to skip: {n_skipped:,} ({pct_skipped:.2f}%)")

    # Apply skip mask to split column
    split_col = split_df["split"].copy()
    split_col[in_progress] = "skip"

    # 3. Calendar day column for day-block bootstrap
    day_col = pd.to_datetime(flux_df["time"]).dt.strftime("%Y-%m-%d")

    # 4. Compute all clues (G0, G1, G2, G3) once
    print(f"[{time.strftime('%H:%M:%S')}] Assembling clues G0, G1, G2, G3...")
    t0_feat = time.time()
    features_df = assemble_features(flux_df, flares_df)
    print(f"[{time.strftime('%H:%M:%S')}] Assembled {features_df.shape[1]} clues across {len(features_df):,} rows in {time.time()-t0_feat:.2f}s.")

    # 5. Build datasets for each horizon (N=30 first)
    for N in horizons:
        out_path = os.path.join(output_dir, f"dataset_N{N}.csv")
        print(f"\n[{time.strftime('%H:%M:%S')}] === Building dataset for horizon N = {N} min ===")
        t0_n = time.time()

        # Compute binary target y
        labels_df = build_flare_labels(flux_df["time"], flares_df, horizon_min=N)
        y = labels_df["y"].values

        # Assemble dataset table
        metadata = pd.DataFrame({
            "time": flux_df["time"].values,
            "y": y,
            "split": split_col.values,
            "day": day_col.values,
        })

        dataset_n = pd.concat([metadata, features_df], axis=1)

        # Print statistics
        valid_mask = dataset_n["split"].isin(["train", "val", "test"])
        n_valid = int(valid_mask.sum())
        n_pos = int(dataset_n.loc[valid_mask, "y"].sum())
        pos_rate = (n_pos / n_valid) * 100 if n_valid > 0 else 0.0

        train_mask = dataset_n["split"] == "train"
        val_mask = dataset_n["split"] == "val"
        test_mask = dataset_n["split"] == "test"

        pos_train = int(dataset_n.loc[train_mask, "y"].sum())
        pos_val = int(dataset_n.loc[val_mask, "y"].sum())
        pos_test = int(dataset_n.loc[test_mask, "y"].sum())

        print(f"  Valid rows: {n_valid:,} / {len(dataset_n):,} ({n_valid/len(dataset_n)*100:.2f}%)")
        print(f"  Total positive: {n_pos:,} ({pos_rate:.2f}%)")
        print(f"  Train: {pos_train:,} pos ({pos_train/train_mask.sum()*100:.2f}%)")
        print(f"  Val:   {pos_val:,} pos ({pos_val/val_mask.sum()*100:.2f}%)")
        print(f"  Test:  {pos_test:,} pos ({pos_test/test_mask.sum()*100:.2f}%)")

        print(f"  Saving to {out_path}...")
        dataset_n.to_csv(out_path, index=False)
        print(f"  Saved {out_path} ({os.path.getsize(out_path)/(1024*1024):.1f} MB) in {time.time()-t0_n:.2f}s.")

    print(f"\n[{time.strftime('%H:%M:%S')}] All datasets built successfully!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build feature and label datasets for solar flare prediction.")
    parser.add_argument("--flux", type=str, default="data/processed/flux_1min.csv", help="Path to cleaned flux CSV.")
    parser.add_argument("--flares", type=str, default="data/processed/flares.csv", help="Path to filtered flares CSV.")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Directory to save datasets.")
    parser.add_argument("--horizons", nargs="+", type=int, default=[30, 5, 15, 60], help="Horizons in minutes.")
    args = parser.parse_args()

    build_datasets(
        flux_path=args.flux,
        flares_path=args.flares,
        output_dir=args.output_dir,
        horizons=args.horizons,
    )
