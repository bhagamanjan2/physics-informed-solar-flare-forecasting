"""
Unit tests for data leakage prevention.
Owned by Agent A2 (Labels and Features) and Agent A8 (CLI and Tests).

Strict invariant:
Perturbing any flux or flare data at t_future > t must have ZERO effect on features at t.
"""

import pytest
import numpy as np
import pandas as pd
from solarflare.features import assemble_features


def test_no_future_leakage():
    """
    Verify that features at minute t depend ONLY on data <= t.
    Modifying flux and adding flares strictly after t_cutoff must produce
    identical features for all timestamps <= t_cutoff.
    """
    times = pd.date_range("2024-01-01 10:00:00", periods=180, freq="1min")
    np.random.seed(42)
    flux = 1.0e-7 + np.random.uniform(0.0, 5.0e-8, size=180)
    net_flux = np.maximum(flux - 1.0e-7, 0.0)
    quality_ok = np.ones(180, dtype=bool)

    df_flux_base = pd.DataFrame({
        "time": times.strftime("%Y-%m-%d %H:%M:%S"),
        "flux": flux.copy(),
        "net_flux": net_flux.copy(),
        "quality_ok": quality_ok.copy(),
    })

    df_flares_base = pd.DataFrame([
        {
            "flare_id": "f_past",
            "tstart": "2024-01-01 10:20:00",
            "tend": "2024-01-01 10:45:00",
            "peak_flux": 5.0e-6,
            "class_letter": "C",
        }
    ])

    features_base = assemble_features(df_flux_base, df_flares_base)

    # Cutoff index at t = 90 (11:30:00)
    cutoff_idx = 90

    # Perturbed future data strictly after cutoff_idx
    df_flux_perturbed = df_flux_base.copy()
    # Drastic future changes: multiply future flux by 1000 and inject large spikes
    df_flux_perturbed.loc[cutoff_idx + 1:, "flux"] = (
        df_flux_perturbed.loc[cutoff_idx + 1:, "flux"] * 1000.0 + 5.0e-5
    )
    df_flux_perturbed.loc[cutoff_idx + 1:, "net_flux"] = (
        df_flux_perturbed.loc[cutoff_idx + 1:, "net_flux"] * 1000.0 + 5.0e-5
    )

    # Add a massive future flare after cutoff_idx
    df_flares_perturbed = pd.DataFrame([
        {
            "flare_id": "f_past",
            "tstart": "2024-01-01 10:20:00",
            "tend": "2024-01-01 10:45:00",
            "peak_flux": 5.0e-6,
            "class_letter": "C",
        },
        {
            "flare_id": "f_future_X",
            "tstart": "2024-01-01 12:00:00",
            "tend": "2024-01-01 12:30:00",
            "peak_flux": 1.0e-3,
            "class_letter": "X",
        },
    ])

    features_perturbed = assemble_features(df_flux_perturbed, df_flares_perturbed)

    # Slices up to and including cutoff_idx must be bit-for-bit or numerically identical
    past_base = features_base.iloc[:cutoff_idx + 1].values
    past_perturbed = features_perturbed.iloc[:cutoff_idx + 1].values

    abs_diff = np.abs(past_base - past_perturbed)
    max_leakage = np.max(abs_diff)

    assert max_leakage == 0.0, f"Leakage detected! Max absolute difference: {max_leakage}"
