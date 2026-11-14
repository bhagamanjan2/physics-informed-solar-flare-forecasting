"""
Label generation and skip condition handling for solar flare prediction.
Owned by Agent A2 (Labels and Features).

Rules implemented:
1. Target: y = 1 if a SIGNIFICANT_FLARE starts in (t, t + horizon_min], else 0.
2. Skip condition: in-progress flare (tstart <= t <= tend) or lookback touches bad quality.
"""

from typing import Optional
import numpy as np
import pandas as pd


def build_flare_labels(
    flux_times: pd.Series, flares_df: pd.DataFrame, horizon_min: int
) -> pd.DataFrame:
    """
    Assign binary target y for prediction horizon horizon_min.
    
    For each minute t in flux_times:
      y = 1 if any flare in flares_df has tstart in the interval (t, t + horizon_min]
      y = 0 otherwise
      
    Interval (t, t + horizon_min] is right-open at t and closed at t + horizon_min:
    - A flare starting at exactly t does NOT count (it is already starting).
    - A flare starting at t + horizon_min DOES count.
    
    Returns:
      DataFrame with columns ['time', 'y']
    """
    # Convert timestamps to sorted NumPy datetime64 arrays for fast binary search
    times = pd.to_datetime(flux_times).values
    flare_starts = np.sort(pd.to_datetime(flares_df["tstart"]).values)

    # Window end time: t + horizon_min
    horizon_delta = np.timedelta64(horizon_min, "m")
    times_plus_h = times + horizon_delta

    # np.searchsorted with side='right':
    # left_idx: number of flare starts <= t (strictly after t starts at left_idx)
    # right_idx: number of flare starts <= t + horizon_min
    left_idx = np.searchsorted(flare_starts, times, side="right")
    right_idx = np.searchsorted(flare_starts, times_plus_h, side="right")

    # If right_idx > left_idx, at least one flare starts in (t, t + horizon_min]
    y = ((right_idx - left_idx) > 0).astype(int)

    return pd.DataFrame({"time": flux_times.values, "y": y})


def identify_in_progress_minutes(
    flux_times: pd.Series, flares_df: pd.DataFrame
) -> pd.Series:
    """
    Identify minutes t where a significant flare is already in progress:
      tstart <= t <= tend
      
    Such minutes must be skipped from training and evaluation to prevent
    evaluating a model while inside an ongoing flare.
    
    Uses an interval prefix-sum sweep over the sorted time grid for O(N_flares + N_times) speed.
    """
    times = pd.to_datetime(flux_times).values
    n = len(times)
    if n == 0 or len(flares_df) == 0:
        return pd.Series(False, index=flux_times.index)

    tstarts = pd.to_datetime(flares_df["tstart"]).values
    tends = pd.to_datetime(flares_df["tend"]).values

    # Find integer index bounds on the sorted time grid
    idx_starts = np.searchsorted(times, tstarts, side="left")
    idx_ends = np.searchsorted(times, tends, side="right")

    # Filter flares that fall within the observation range
    valid = idx_starts < n
    idx_starts = idx_starts[valid]
    idx_ends = np.minimum(idx_ends[valid], n)

    # Difference array prefix-sum sweep
    diff = np.zeros(n + 1, dtype=np.int32)
    np.add.at(diff, idx_starts, 1)
    np.add.at(diff, idx_ends, -1)

    in_progress = np.cumsum(diff)[:-1] > 0
    return pd.Series(in_progress, index=flux_times.index)


def identify_skip_minutes(
    flux_df: pd.DataFrame, flares_df: pd.DataFrame, lookback_min: int = 60
) -> pd.Series:
    """
    Combine all skip criteria into a single boolean series:
    1. A significant flare is already in progress (tstart <= t <= tend).
    2. The lookback window [t - lookback_min, t] contains degraded/bad flux quality.
    
    Returns:
      Boolean Series (True = skip this minute).
    """
    flux_times = flux_df["time"]
    in_flare = identify_in_progress_minutes(flux_times, flares_df)

    # Check flux quality flags in lookback window if quality_ok is present
    if "quality_ok" in flux_df.columns:
        quality = flux_df["quality_ok"].astype(bool)
        # Any bad quality in the past lookback_min minutes flags the window
        bad_quality = (~quality).rolling(window=lookback_min, min_periods=1).max().astype(bool)
        skip = in_flare | bad_quality
    else:
        skip = in_flare

    return pd.Series(skip.values, index=flux_df.index)
