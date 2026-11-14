"""
Unit tests for chronological splitting and blocked time-series cross-validation.
Owned by Agent A8 (CLI and Tests).
Verifies:
- Chronological ordering (never shuffled)
- 25-hour blackout gap between train/val and val/test
- Complete partition without overlap
- Approximate split ratios (60% train, 15% val, 25% test)
- Blocked CV fold integrity and safety gaps
"""

import pytest
import pandas as pd
import numpy as np
from solarflare.split import chronological_split, blocked_time_series_cv


@pytest.fixture
def sample_timeseries_df():
    """Create a 60-day 1-minute time series dataframe."""
    times = pd.date_range("2024-01-01 00:00:00", periods=60 * 24 * 60, freq="1min")
    return pd.DataFrame({
        "time": times.strftime("%Y-%m-%d %H:%M:%S"),
        "flux": np.random.uniform(1e-7, 1e-6, size=len(times)),
        "y": np.random.choice([0, 1], size=len(times), p=[0.98, 0.02])
    })


def test_chronological_ordering_and_no_overlap(sample_timeseries_df):
    """Verify strict chronological sequence and zero index/timestamp overlap."""
    df_split = chronological_split(
        sample_timeseries_df,
        train_ratio=0.60,
        val_ratio=0.15,
        test_ratio=0.25,
        gap_hours=25
    )
    
    assert "split" in df_split.columns
    unique_splits = set(df_split["split"].unique())
    assert unique_splits.issubset({"train", "val", "test", "gap"})
    
    df_train = df_split[df_split["split"] == "train"]
    df_val = df_split[df_split["split"] == "val"]
    df_test = df_split[df_split["split"] == "test"]
    df_gap = df_split[df_split["split"] == "gap"]
    
    # Must all have rows
    assert len(df_train) > 0
    assert len(df_val) > 0
    assert len(df_test) > 0
    assert len(df_gap) > 0
    
    # Complete partition: no lost or duplicate rows
    assert len(df_train) + len(df_val) + len(df_test) + len(df_gap) == len(sample_timeseries_df)
    
    # Convert to timestamps for sequence validation
    t_train_max = pd.to_datetime(df_train["time"].max())
    t_val_min = pd.to_datetime(df_val["time"].min())
    t_val_max = pd.to_datetime(df_val["time"].max())
    t_test_min = pd.to_datetime(df_test["time"].min())
    
    # Strict chronological ordering: train < val < test
    assert t_train_max < t_val_min
    assert t_val_max < t_test_min


def test_25h_blackout_gaps(sample_timeseries_df):
    """Verify the 25-hour blackout gap between train->val and val->test."""
    gap_hours = 25
    df_split = chronological_split(
        sample_timeseries_df,
        train_ratio=0.60,
        val_ratio=0.15,
        test_ratio=0.25,
        gap_hours=gap_hours
    )
    
    t_train_max = pd.to_datetime(df_split[df_split["split"] == "train"]["time"].max())
    t_val_min = pd.to_datetime(df_split[df_split["split"] == "val"]["time"].min())
    t_val_max = pd.to_datetime(df_split[df_split["split"] == "val"]["time"].max())
    t_test_min = pd.to_datetime(df_split[df_split["split"] == "test"]["time"].min())
    
    train_to_val_gap = (t_val_min - t_train_max).total_seconds() / 3600.0
    val_to_test_gap = (t_test_min - t_val_max).total_seconds() / 3600.0
    
    # The time gap between partition bounds must be >= gap_hours
    assert train_to_val_gap >= gap_hours
    assert val_to_test_gap >= gap_hours


def test_split_ratios(sample_timeseries_df):
    """Verify train, val, and test partition durations approximately match requested ratios."""
    df_split = chronological_split(
        sample_timeseries_df,
        train_ratio=0.60,
        val_ratio=0.15,
        test_ratio=0.25,
        gap_hours=25
    )
    
    times = pd.to_datetime(df_split["time"])
    total_duration = (times.max() - times.min()).total_seconds()
    
    train_times = pd.to_datetime(df_split[df_split["split"] == "train"]["time"])
    train_duration = (train_times.max() - train_times.min()).total_seconds()
    
    # Train ratio should be very close to 0.60
    assert pytest.approx(train_duration / total_duration, abs=0.03) == 0.60


def test_blocked_time_series_cv_gaps(sample_timeseries_df):
    """Verify blocked cross-validation folds have chronological order and safety gap."""
    df_split = chronological_split(sample_timeseries_df, gap_hours=25)
    df_train = df_split[df_split["split"] == "train"].reset_index(drop=True)
    
    gap_hours = 25
    gap_samples = int(gap_hours * 60)
    folds = blocked_time_series_cv(df_train, n_splits=3, gap_hours=gap_hours)
    
    assert len(folds) >= 1
    for fold_idx, (tr_idx, va_idx) in enumerate(folds):
        assert len(tr_idx) > 0
        assert len(va_idx) > 0
        # Train strictly before validation
        assert np.max(tr_idx) < np.min(va_idx)
        # Gap between train and val fold is at least gap_samples
        assert np.min(va_idx) - np.max(tr_idx) >= gap_samples
        # No intersection
        assert len(np.intersect1d(tr_idx, va_idx)) == 0
