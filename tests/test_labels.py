"""
Unit tests for target labels and skip conditions.
Owned by Agent A2 (Labels and Features) and Agent A8 (CLI and Tests).
"""

import pytest
import numpy as np
import pandas as pd
from solarflare.labels import (
    build_flare_labels,
    identify_in_progress_minutes,
    identify_skip_minutes,
)


def test_build_flare_labels_boundaries():
    """
    Test that flare at tstart is caught in (t, t + horizon]
    Specifically:
    - If flare starts at 10:20, and horizon is 30:
      - At 09:50 (t + 30m = 10:20): flare is at boundary t + h -> caught (y=1)
      - At 10:19 (t + 30m = 10:49): flare is inside -> caught (y=1)
      - At 10:20 (t + 30m = 10:50): flare starts at t, so NOT in (t, t+h] -> y=0
      - At 09:49 (t + 30m = 10:19): flare is after window -> y=0
    """
    times = pd.date_range("2024-01-01 09:45:00", "2024-01-01 10:30:00", freq="1min")
    time_series = pd.Series(times.strftime("%Y-%m-%d %H:%M:%S"))

    flares = pd.DataFrame([
        {
            "flare_id": "test_1",
            "tstart": "2024-01-01 10:20:00",
            "tend": "2024-01-01 10:40:00",
            "class_letter": "C",
            "peak_flux": 5.0e-6,
        }
    ])

    labels = build_flare_labels(time_series, flares, horizon_min=30)
    df_check = pd.DataFrame({"time": time_series, "y": labels["y"]})

    y_0949 = df_check.loc[df_check["time"] == "2024-01-01 09:49:00", "y"].values[0]
    y_0950 = df_check.loc[df_check["time"] == "2024-01-01 09:50:00", "y"].values[0]
    y_1019 = df_check.loc[df_check["time"] == "2024-01-01 10:19:00", "y"].values[0]
    y_1020 = df_check.loc[df_check["time"] == "2024-01-01 10:20:00", "y"].values[0]

    assert y_0949 == 0, "Minute 09:49 must have y=0 (flare at 10:20 is > 30m ahead)"
    assert y_0950 == 1, "Minute 09:50 must have y=1 (flare at 10:20 is exactly t + 30m)"
    assert y_1019 == 1, "Minute 10:19 must have y=1 (flare at 10:20 is in (t, t+30m])"
    assert y_1020 == 0, "Minute 10:20 must have y=0 (flare starts at t, not strictly in future)"


def test_identify_in_progress_minutes():
    """
    Test in-progress mask: True for tstart <= t <= tend.
    """
    times = pd.date_range("2024-01-01 10:00:00", "2024-01-01 11:00:00", freq="1min")
    time_series = pd.Series(times.strftime("%Y-%m-%d %H:%M:%S"))

    flares = pd.DataFrame([
        {
            "flare_id": "test_1",
            "tstart": "2024-01-01 10:15:00",
            "tend": "2024-01-01 10:35:00",
            "class_letter": "C",
            "peak_flux": 5.0e-6,
        }
    ])

    in_prog = identify_in_progress_minutes(time_series, flares)
    df_check = pd.DataFrame({"time": time_series, "in_prog": in_prog})

    assert df_check.loc[df_check["time"] == "2024-01-01 10:14:00", "in_prog"].values[0] == False
    assert df_check.loc[df_check["time"] == "2024-01-01 10:15:00", "in_prog"].values[0] == True
    assert df_check.loc[df_check["time"] == "2024-01-01 10:35:00", "in_prog"].values[0] == True
    assert df_check.loc[df_check["time"] == "2024-01-01 10:36:00", "in_prog"].values[0] == False
