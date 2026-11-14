"""
Script 01: Download, Verify, and Preprocess Raw GOES X-ray Flux & ASR Flare Catalog.
Owned by Agent A1 (Data Engineer).

Usage:
  python scripts/01_download_data.py [--years 2022 2023 2024]
"""

import os
import sys
import argparse
import time
import pandas as pd
import numpy as np

# Ensure src/ is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from solarflare.data import (
    load_raw_flares,
    filter_flares,
    download_goes_year,
    clean_flux_series,
)


def main():
    parser = argparse.ArgumentParser(description="Download and preprocess solar flare data.")
    parser.add_argument(
        "--years",
        nargs="+",
        type=int,
        default=[2022, 2023, 2024],
        help="Years of GOES flux series to process (default: 2022 2023 2024)",
    )
    parser.add_argument(
        "--raw-flare-path",
        type=str,
        default="data/raw/f_1995_2024.csv",
        help="Path to raw ASR flare catalog CSV",
    )
    parser.add_argument(
        "--out-flux-path",
        type=str,
        default="data/processed/flux_1min.csv",
        help="Path to save processed 1-minute flux series",
    )
    parser.add_argument(
        "--out-flares-path",
        type=str,
        default="data/processed/flares.csv",
        help="Path to save processed significant flare catalog",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("STEP 1: VERIFYING & DOWNLOADING RAW GOES FLUX DATA")
    print("=" * 70)

    raw_dfs = []
    total_raw_rows = 0

    for yr in sorted(args.years):
        print(f"Checking GOES X-ray series for year {yr}...")
        df_yr = download_goes_year(yr, dest_dir="data/raw")
        rows = len(df_yr)
        total_raw_rows += rows
        print(f"  Year {yr}: {rows:,} rows loaded.")
        raw_dfs.append(df_yr)

    print(f"\nTotal raw flux records loaded across {len(args.years)} years: {total_raw_rows:,}")

    print("\n" + "=" * 70)
    print("STEP 2: CLEANING FLUX SERIES (1-MIN GRID, 6H BG, NET FLUX)")
    print("=" * 70)
    t0 = time.time()
    df_raw_all = pd.concat(raw_dfs, ignore_index=True)
    
    print("Running clean_flux_series...")
    df_flux_clean = clean_flux_series(df_raw_all, bg_window_min=360)
    print(f"Cleaning complete in {time.time() - t0:.2f}s.")
    print(f"Cleaned 1-minute flux series shape: {df_flux_clean.shape}")
    print(f"Time span: {df_flux_clean['time'].min()} to {df_flux_clean['time'].max()}")

    # Quality breakdown
    quality_counts = df_flux_clean["quality_ok"].value_counts().to_dict()
    nominal_count = quality_counts.get(True, 0)
    degraded_count = quality_counts.get(False, 0)
    print(f"Nominal (quality_ok=True):   {nominal_count:,} ({nominal_count / len(df_flux_clean) * 100:.2f}%)")
    print(f"Degraded/Bad (quality_ok=False): {degraded_count:,} ({degraded_count / len(df_flux_clean) * 100:.2f}%)")

    # Save processed flux
    os.makedirs(os.path.dirname(args.out_flux_path), exist_ok=True)
    df_flux_clean.to_csv(args.out_flux_path, index=False)
    print(f"Saved processed flux to: {args.out_flux_path}")

    print("\n" + "=" * 70)
    print("STEP 3: FILTERING FLARE CATALOG (LOCKED: SIGNIFICANT_FLARE)")
    print("=" * 70)
    print(f"Loading raw flare catalog from: {args.raw_flare_path}")
    raw_flares = load_raw_flares(args.raw_flare_path)
    print(f"Total raw flare catalog rows: {len(raw_flares):,}")

    # Rationale documentation
    print("\n[LOCKED DECISION RATIONALE]")
    print("Target definition: SIGNIFICANT_FLARE")
    print("- Rule: abs_class_simple in {C, M, X} AND fclass_simple in {C, M, X}")
    print("- Solar max (2022-2024) baseline flux sits at C-level (~10^-6 W/m^2).")
    print("- As a result, ~85% of absolute C-class events are micro-bumps with A/B energy.")
    print("- SIGNIFICANT_FLARE ensures genuine flare energy injection of C-class or higher above background.")

    start_date = f"{min(args.years)}-01-01 00:00:00"
    end_date = f"{max(args.years)}-12-31 23:59:00"
    print(f"\nFiltering significant flares overlapping {start_date} to {end_date}...")
    flares_clean = filter_flares(
        raw_flares,
        min_classes=["C", "M", "X"],
        significant_only=True,
        start_time=start_date,
        end_time=end_date,
    )
    print(f"Filtered significant flares count: {len(flares_clean):,}")

    # Breakdown by class
    class_counts = flares_clean["class_letter"].value_counts().to_dict()
    print("\nFlare breakdown by NOAA class (abs_class_simple):")
    for cls in ["C", "M", "X"]:
        cnt = class_counts.get(cls, 0)
        pct = cnt / len(flares_clean) * 100 if len(flares_clean) > 0 else 0
        print(f"  Class {cls}: {cnt:,} flares ({pct:.2f}%)")

    # Save processed flares
    os.makedirs(os.path.dirname(args.out_flares_path), exist_ok=True)
    flares_clean.to_csv(args.out_flares_path, index=False)
    print(f"Saved processed flares to: {args.out_flares_path}")

    print("\n" + "=" * 70)
    print("STEP 4: SUMMARY & LABEL STATISTICS FROM results/label_stats.csv")
    print("=" * 70)
    stats_file = "results/label_stats.csv"
    if os.path.exists(stats_file):
        stats_df = pd.read_csv(stats_file)
        print(stats_df.to_string(index=False))
    else:
        print(f"Notice: {stats_file} not found yet (will be computed during split/label phase).")

    print("\n" + "=" * 70)
    print("DATA PREPROCESSING COMPLETE (SUCCESS)")
    print("=" * 70)


if __name__ == "__main__":
    main()
