"""
Chronological split and blocked time-series cross-validation.
Owned by Agent A3 (Split and Models).
Enforces 25-hour blackout gaps between train, val, and test splits to prevent temporal leakage.
"""

from typing import Tuple, List
import pandas as pd
import numpy as np


def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.60,
    val_ratio: float = 0.15,
    test_ratio: float = 0.25,
    gap_hours: int = 25,
) -> pd.DataFrame:
    """
    Split time-ordered dataset chronologically:
    - Train: first train_ratio of total duration
    - Gap 1: gap_hours (25h) blackout
    - Val: next val_ratio of total duration
    - Gap 2: gap_hours (25h) blackout
    - Test: remaining duration (~test_ratio)

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'time' column.
    train_ratio : float
        Fraction of time span for training (default 0.60).
    val_ratio : float
        Fraction of time span for validation (default 0.15).
    test_ratio : float
        Fraction of time span for test (default 0.25).
    gap_hours : int
        Hours of blackout gap between partitions (default 25).

    Returns
    -------
    pd.DataFrame
        Copy of df with 'split' column containing: 'train', 'val', 'test', 'gap', or 'skip'.
    """
    df_out = df.copy()
    times = pd.to_datetime(df_out["time"])
    t_min = times.min()
    t_max = times.max()
    total_duration = t_max - t_min

    t_train_end = t_min + total_duration * train_ratio
    t_val_start = t_train_end + pd.Timedelta(hours=gap_hours)
    t_val_end = t_val_start + total_duration * val_ratio
    t_test_start = t_val_end + pd.Timedelta(hours=gap_hours)

    splits = np.full(len(df_out), "test", dtype=object)
    splits[times <= t_train_end] = "train"
    splits[(times > t_train_end) & (times < t_val_start)] = "gap"
    splits[(times >= t_val_start) & (times <= t_val_end)] = "val"
    splits[(times > t_val_end) & (times < t_test_start)] = "gap"

    # Preserve pre-existing 'skip' labels (e.g. flare in progress or bad flux)
    if "split" in df.columns:
        skip_mask = (df["split"] == "skip").values
        splits[skip_mask] = "skip"

    df_out["split"] = splits
    return df_out


def blocked_time_series_cv(
    df_train: pd.DataFrame, n_splits: int = 5, gap_hours: int = 25
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate blocked time-series cross-validation folds on train partition.
    Inserts a gap_hours buffer between train and validation folds to prevent leakage.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training partition DataFrame.
    n_splits : int
        Number of cross-validation folds (default 5).
    gap_hours : int
        Blackout buffer hours between train and val folds (default 25).

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        List of (train_indices, val_indices) as 0-based numpy arrays.
    """
    n = len(df_train)
    if n == 0:
        return []

    fold_size = n // (n_splits + 1)
    if fold_size == 0:
        fold_size = max(1, n // n_splits)

    has_time = "time" in df_train.columns
    if has_time:
        times = pd.to_datetime(df_train["time"]).values

    folds = []
    for i in range(1, n_splits + 1):
        train_end = i * fold_size
        train_idx = np.arange(0, train_end)

        if has_time:
            t_train_last = times[train_end - 1]
            t_gap_end = t_train_last + np.timedelta64(gap_hours, "h")
            val_start = int(np.searchsorted(times, t_gap_end, side="left"))
        else:
            gap_samples = int(gap_hours * 60)
            val_start = train_end + gap_samples

        if i < n_splits:
            val_end = min(max(val_start + 1, (i + 1) * fold_size), n)
        else:
            val_end = n

        # Fallback if gap pushes val_start past val_end or n
        if val_start >= n:
            val_start = max(0, n - max(1, fold_size // 2))
            val_end = n
            train_idx = np.arange(0, val_start)
        elif val_start >= val_end:
            val_end = min(val_start + max(1, fold_size // 2), n)

        val_idx = np.arange(val_start, val_end)
        if len(train_idx) > 0 and len(val_idx) > 0:
            folds.append((train_idx, val_idx))

    return folds
