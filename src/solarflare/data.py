"""
Data loading and preprocessing module for GOES X-ray flux series and ASR flare catalog.
Owned by Agent A1 (Data Engineer).

Follows INTERFACES.md strictly:
- load_raw_flares
- filter_flares
- download_goes_year
- clean_flux_series
- load_processed_data
"""

from typing import Optional, List, Union
import os
import pathlib
import shutil
import pandas as pd
import numpy as np


def load_raw_flares(path: str = "data/raw/f_1995_2024.csv") -> pd.DataFrame:
    """
    Load raw flare catalog CSV.
    Verified columns: flare_id, tstart, tpeak, tend, peak_flux, BG_flux, flux_int,
                      ratio, peak_end_ratio, flux_int_corrected, delta_flux,
                      fclass_simple, fclass_full, abs_class_simple, abs_class_full.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Raw flare list not found at {path}")
    df = pd.read_csv(path)
    df["tstart"] = pd.to_datetime(df["tstart"], format="mixed")
    df["tpeak"] = pd.to_datetime(df["tpeak"], format="mixed")
    df["tend"] = pd.to_datetime(df["tend"], format="mixed")
    return df


def filter_flares(
    df_flares: pd.DataFrame,
    min_class: Optional[str] = "C",
    min_classes: Optional[List[str]] = None,
    significant_only: bool = True,
    start_time: Optional[Union[str, pd.Timestamp]] = None,
    end_time: Optional[Union[str, pd.Timestamp]] = None,
) -> pd.DataFrame:
    """
    Filter flares using NOAA standard class (abs_class_simple) and background-relative class.
    
    LOCKED DECISION:
    - Target: SIGNIFICANT_FLARE (abs_class_simple in {C, M, X} AND fclass_simple in {C, M, X}).
    - Rationale: In solar maximum (2022-2024), solar background sits at C-level,
      so ~85% of absolute C-flares are micro-bumps with A/B energy above background.
      SIGNIFICANT_FLARE requires genuine C-class or higher energy injection above background.

    Returns DataFrame matching schema:
      flare_id, tstart, tpeak, tend, class_letter, peak_flux
    """
    if min_classes is None:
        if min_class == "X":
            min_classes = ["X"]
        elif min_class == "M":
            min_classes = ["M", "X"]
        else:
            min_classes = ["C", "M", "X"]

    # Filter by class
    if significant_only and "fclass_simple" in df_flares.columns:
        mask = (
            df_flares["abs_class_simple"].isin(min_classes)
            & df_flares["fclass_simple"].isin(min_classes)
        )
    else:
        mask = df_flares["abs_class_simple"].isin(min_classes)

    # Optional temporal bounds
    if start_time is not None:
        t_start_ts = pd.to_datetime(start_time)
        mask = mask & (pd.to_datetime(df_flares["tend"]) >= t_start_ts)

    if end_time is not None:
        t_end_ts = pd.to_datetime(end_time)
        mask = mask & (pd.to_datetime(df_flares["tstart"]) <= t_end_ts)

    filtered = df_flares[mask].copy()

    flares_clean = pd.DataFrame({
        "flare_id": filtered["flare_id"].astype(str),
        "tstart": pd.to_datetime(filtered["tstart"]),
        "tpeak": pd.to_datetime(filtered["tpeak"]),
        "tend": pd.to_datetime(filtered["tend"]),
        "class_letter": filtered["abs_class_simple"].astype(str),
        "peak_flux": filtered["peak_flux"].astype(float),
    })

    return flares_clean.sort_values("tstart").reset_index(drop=True)


def download_goes_year(year: int, dest_dir: str = "data/raw") -> pd.DataFrame:
    """
    Download or load 1-minute GOES XRS data for a specified year.
    If the file dest_dir/goes_{year}.csv exists, loads and returns it.
    Otherwise uses the ASR goes_downloader_old function.
    """
    os.makedirs(dest_dir, exist_ok=True)
    out_file = os.path.join(dest_dir, f"goes_{year}.csv")
    if os.path.exists(out_file):
        return pd.read_csv(out_file)

    # Configure SunPy directories inside workspace to avoid permission issues
    cfg_dir = pathlib.Path("data/.sunpy").resolve()
    cfg_dir.mkdir(parents=True, exist_ok=True)
    mgr_dir = pathlib.Path("data/sunpy_manager").resolve()
    mgr_dir.mkdir(parents=True, exist_ok=True)
    dl_dir = pathlib.Path("data/downloads").resolve()
    dl_dir.mkdir(parents=True, exist_ok=True)

    sunpyrc = cfg_dir / "sunpyrc"
    with open(sunpyrc, "w") as f:
        f.write(f"""[downloads]
download_dir = {dl_dir.as_posix()}
remote_data_manager_dir = {mgr_dir.as_posix()}
sample_dir = {dl_dir.as_posix()}
""")

    os.environ["SUNPY_CONFIGDIR"] = str(cfg_dir)
    os.environ["SUNPY_DOWNLOADDIR"] = str(dl_dir)

    try:
        from asr.utils import goes_downloader_old
    except ImportError:
        import sys
        sys.path.insert(0, "third_party/ASR_cat")
        from asr.utils import goes_downloader_old

    df = goes_downloader_old(year)
    df.to_csv(out_file, index=False)

    if dl_dir.exists():
        shutil.rmtree(dl_dir, ignore_errors=True)

    return df


def clean_flux_series(df_raw: pd.DataFrame, bg_window_min: int = 360) -> pd.DataFrame:
    """
    Clean GOES 1-minute flux series:
    - Standardize column names (time, flux, quality_ok).
    - Parse timestamps and enforce regular 1-minute grid.
    - Remove duplicate timestamps.
    - Compute rolling-minimum background over bg_window_min (default 6h = 360 min)
      using only good-quality readings to avoid glitch corruption.
    - Compute net_flux = max(flux - bg, 0).
    """
    df = df_raw.copy()

    # Identify time column
    time_col = "time_tag" if "time_tag" in df.columns else "time"
    flux_col = "xl" if "xl" in df.columns else "flux"
    quality_col = "xl_quality" if "xl_quality" in df.columns else "quality_ok"

    df["time"] = pd.to_datetime(df[time_col])
    df = df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)

    # Reindex to strict 1-minute grid if gaps exist
    start_t = df["time"].min()
    end_t = df["time"].max()
    full_index = pd.date_range(start_t, end_t, freq="1min")
    if len(df) != len(full_index):
        df = df.set_index("time").reindex(full_index).reset_index()
        df.rename(columns={"index": "time"}, inplace=True)

    flux = df[flux_col].astype(float)

    # Quality flag evaluation
    if quality_col in df.columns and df[quality_col].notna().any():
        if df[quality_col].dtype == bool:
            quality_ok = df[quality_col].fillna(False) & flux.notna() & (flux > 0)
        else:
            quality_ok = (df[quality_col] == 0) & flux.notna() & (flux > 0)
    else:
        quality_ok = flux.notna() & (flux > 0)

    # Rolling-minimum background over 6 hours (360 minutes)
    # Computed strictly on valid points so telemetry drops (1e-9 / negative) don't distort background
    valid_flux = flux.where(quality_ok)
    bg = valid_flux.rolling(window=bg_window_min, min_periods=30).min()
    bg = bg.bfill().ffill()

    # Net flux: physical flare emission above local background
    net_flux = np.maximum(flux - bg, 0.0)

    df_clean = pd.DataFrame({
        "time": df["time"].dt.strftime("%Y-%m-%d %H:%M:%S"),
        "flux": flux,
        "quality_ok": quality_ok.astype(bool),
        "bg": bg,
        "net_flux": net_flux,
    })

    return df_clean


def load_processed_data(
    flux_path: str = "data/processed/flux_1min.csv",
    flares_path: str = "data/processed/flares.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load preprocessed 1-minute flux series and significant flare catalog.
    """
    if not os.path.exists(flux_path):
        raise FileNotFoundError(f"Processed flux file not found: {flux_path}")
    if not os.path.exists(flares_path):
        raise FileNotFoundError(f"Processed flares file not found: {flares_path}")

    flux_df = pd.read_csv(flux_path)
    flares_df = pd.read_csv(flares_path)

    flares_df["tstart"] = pd.to_datetime(flares_df["tstart"])
    flares_df["tpeak"] = pd.to_datetime(flares_df["tpeak"])
    flares_df["tend"] = pd.to_datetime(flares_df["tend"])

    return flux_df, flares_df
