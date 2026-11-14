"""
Generate synthetic GOES X-ray flux series and flare catalogue.
Used for development and pipeline testing before real data is available.

Schema matches:
  - data/processed/flux_1min.csv: time, flux, quality_ok, bg, net_flux
  - data/processed/flares.csv: flare_id, tstart, tpeak, tend, class_letter, peak_flux
"""

import os
import argparse
import numpy as np
import pandas as pd


def generate_synthetic_data(days: int = 30, seed: int = 42) -> None:
    """Generate synthetic 1-minute flux and flare list matching real data schemas."""
    np.random.seed(seed)
    
    start_time = pd.Timestamp("2024-01-01 00:00:00")
    total_minutes = days * 24 * 60
    time_index = pd.date_range(start=start_time, periods=total_minutes, freq="1min")
    
    # 1. Base solar flux with slow diurnal variation (~1e-7 to 3e-7 W/m2, typical B-class background)
    t_hours = np.arange(total_minutes) / 60.0
    slow_bg = 1.5e-7 + 0.5e-7 * np.sin(2 * np.pi * t_hours / 24.0) + 0.2e-7 * np.sin(2 * np.pi * t_hours / (24.0 * 7))
    noise = np.random.normal(0, 0.05e-7, size=total_minutes)
    flux = np.maximum(slow_bg + noise, 1e-8)
    
    # 2. Add synthetic solar flares
    # ~3 flares per day on average: mix of C, M, and rare X
    n_flares = int(days * 3.5)
    flare_starts = np.sort(np.random.choice(np.arange(120, total_minutes - 180), size=n_flares, replace=False))
    
    flares_list = []
    classes = ["C"] * int(0.85 * n_flares) + ["M"] * int(0.12 * n_flares) + ["X"] * max(1, int(0.03 * n_flares))
    np.random.shuffle(classes)
    
    for i, (idx, cls) in enumerate(zip(flare_starts, classes)):
        # Duration: rise 5-20 min, decay 15-60 min
        rise_m = int(np.random.uniform(5, 20))
        decay_m = int(np.random.uniform(15, 60))
        duration = rise_m + decay_m
        
        if cls == "C":
            peak_val = np.random.uniform(1.1e-6, 9.5e-6)
        elif cls == "M":
            peak_val = np.random.uniform(1.1e-5, 8.5e-5)
        else: # X class
            peak_val = np.random.uniform(1.1e-4, 3.5e-4)
            
        base_val = flux[idx]
        delta = peak_val - base_val
        
        # Flare profile: exponential rise, exponential decay
        rise_curve = delta * (1 - np.exp(-np.linspace(0, 3, rise_m)))
        decay_curve = delta * np.exp(-np.linspace(0, 3, decay_m))
        profile = np.concatenate([rise_curve, decay_curve])
        
        end_idx = min(idx + duration, total_minutes)
        prof_len = end_idx - idx
        flux[idx:end_idx] += profile[:prof_len]
        
        # Record flare
        tstart = time_index[idx]
        tpeak = time_index[min(idx + rise_m, total_minutes - 1)]
        tend = time_index[end_idx - 1]
        
        flares_list.append({
            "flare_id": f"syn_{i+1:04d}",
            "tstart": str(tstart),
            "tpeak": str(tpeak),
            "tend": str(tend),
            "class_letter": cls,
            "peak_flux": float(peak_val)
        })
        
    # 3. Add artificial quality flags and missing data gaps
    quality_ok = np.ones(total_minutes, dtype=bool)
    
    # Random bad quality bursts (instrument calibration or glitch)
    n_glitches = max(2, days // 5)
    for _ in range(n_glitches):
        g_idx = np.random.randint(60, total_minutes - 60)
        g_len = np.random.randint(5, 30)
        quality_ok[g_idx:g_idx + g_len] = False
        flux[g_idx:g_idx + g_len] *= np.random.uniform(0.1, 10.0) # Bad spike/drop
        
    # Compute rolling background (e.g. 6-hour rolling minimum) and net flux
    df_flux = pd.DataFrame({
        "time": time_index.astype(str),
        "flux": flux,
        "quality_ok": quality_ok
    })
    
    # 6-hour window = 360 minutes
    df_flux["bg"] = df_flux["flux"].rolling(window=360, min_periods=30).min().bfill()
    df_flux["net_flux"] = np.maximum(df_flux["flux"] - df_flux["bg"], 0.0)
    
    # Introduce 1 or 2 small gaps (missing rows)
    gap_starts = [int(total_minutes * 0.35), int(total_minutes * 0.75)]
    gap_indices = []
    for gs in gap_starts:
        gap_indices.extend(range(gs, gs + 45)) # 45-minute missing gap
    df_flux = df_flux.drop(index=gap_indices).reset_index(drop=True)
    
    df_flares = pd.DataFrame(flares_list)
    
    # Save processed files
    os.makedirs("data/processed", exist_ok=True)
    flux_path = "data/processed/flux_1min.csv"
    flares_path = "data/processed/flares.csv"
    
    df_flux.to_csv(flux_path, index=False)
    df_flares.to_csv(flares_path, index=False)
    
    print(f"Generated synthetic data successfully:")
    print(f"  Flux:   {flux_path} ({len(df_flux):,} rows, {days} days)")
    print(f"  Flares: {flares_path} ({len(df_flares):,} flares: {dict(df_flares['class_letter'].value_counts())})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic solar flux and flare dataset.")
    parser.add_argument("--days", type=int, default=30, help="Number of synthetic days to generate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    args = parser.parse_args()
    
    generate_synthetic_data(days=args.days, seed=args.seed)

