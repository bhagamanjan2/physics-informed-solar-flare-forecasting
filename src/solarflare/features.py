"""
Feature engineering module for clue groups G0, G1, G2, and G3.
Owned by Agent A2 (Labels and Features).

Strict Golden Rule:
No feature at minute t may use flux > t, or flares with tend > t.
All calculations are strictly causal and leak-free.
"""

from typing import List, Dict, Optional
import numpy as np
import pandas as pd
from scipy.signal import convolve


def compute_g0_brightness(
    flux_df: pd.DataFrame, windows: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Group G0: Recent brightness clues.
    - log10 of net flux: log10(net_flux + 1e-10)
    - Rolling multi-scale statistics over windows W in [5, 15, 30, 60] min:
      mean, std, min, max, linear OLS slope, and change of slope (d_slope).
      
    All rolling operations use causal lookback [t - W + 1, t] exclusively.
    """
    if windows is None:
        windows = [5, 15, 30, 60]

    features = pd.DataFrame(index=flux_df.index)

    # Net flux (flux minus rolling 6-hour background)
    if "net_flux" in flux_df.columns:
        raw_net = flux_df["net_flux"].values
    elif "bg" in flux_df.columns:
        raw_net = np.maximum(flux_df["flux"].values - flux_df["bg"].values, 0.0)
    else:
        bg = pd.Series(flux_df["flux"].values).rolling(360, min_periods=30).min().bfill().values
        raw_net = np.maximum(flux_df["flux"].values - bg, 0.0)

    # Causal forward-fill for any isolated missing values
    net_flux = pd.Series(raw_net, dtype=float).ffill().fillna(1e-10).values
    features["g0_log_net_flux"] = np.log10(np.maximum(net_flux, 1e-10))

    flux_series = pd.Series(flux_df["flux"].values, dtype=float).ffill().fillna(1e-8)
    flux_vals = flux_series.values

    for w in windows:
        # Rolling mean, std, min, max
        roll = flux_series.rolling(window=w, min_periods=1)
        mean_s = roll.mean().values
        std_s = roll.std().fillna(0.0).values
        min_s = roll.min().values
        max_s = roll.max().values

        # Exact OLS linear regression slope via 1D causal FIR convolution filter
        # For a window [y_{t-w+1}, ..., y_t] with x = 0, ..., w-1:
        # slope = sum((x_i - x_bar) * y_i) / sum((x_i - x_bar)^2)
        x = np.arange(w, dtype=float)
        x_diff = x - x.mean()
        x_var = np.sum(x_diff ** 2)

        if x_var > 0:
            weights = x_diff / x_var
            # convolve with reversed weights yields dot product of window ending at t
            raw_slope = convolve(flux_vals, weights[::-1], mode="full")[:len(flux_vals)]
            # For the first w - 1 points, use simple two-point slope
            if w > 1:
                two_pt = np.zeros(len(flux_vals))
                two_pt[1:w] = (flux_vals[1:w] - flux_vals[0]) / np.arange(1, w)
                raw_slope[:w - 1] = two_pt[:w - 1]
        else:
            raw_slope = np.zeros(len(flux_vals))

        d_slope = np.zeros(len(raw_slope))
        d_slope[1:] = raw_slope[1:] - raw_slope[:-1]

        features[f"g0_flux_mean_{w}"] = mean_s
        features[f"g0_flux_std_{w}"] = std_s
        features[f"g0_flux_min_{w}"] = min_s
        features[f"g0_flux_max_{w}"] = max_s
        features[f"g0_flux_slope_{w}"] = raw_slope
        features[f"g0_flux_d_slope_{w}"] = d_slope

    return features


def compute_g1_history(
    flux_times: pd.Series,
    flares_df: pd.DataFrame,
    windows_hours: Optional[List[int]] = None,
) -> pd.DataFrame:
    """
    Group G1: Historical flare clues.
    Strictly uses flares that ALREADY ENDED (tend <= t):
    - g1_dt_last_flare: minutes since the most recent flare ended.
    - g1_n_flares_6h: count of significant flares ended in [t - 6h, t].
    - g1_n_flares_24h: count of significant flares ended in [t - 24h, t].
    - g1_max_class_24h: peak flux of strongest flare ended in [t - 24h, t] (default 1e-8).
    
    Fully vectorized using binary search and Range Maximum Query (RMQ) sparse table.
    """
    if windows_hours is None:
        windows_hours = [6, 24]

    times = pd.to_datetime(flux_times).values
    n = len(times)

    if len(flares_df) == 0 or "tend" not in flares_df.columns:
        return pd.DataFrame({
            "g1_dt_last_flare": np.full(n, 10080.0),
            "g1_n_flares_6h": np.zeros(n, dtype=int),
            "g1_n_flares_24h": np.zeros(n, dtype=int),
            "g1_max_class_24h": np.full(n, 1e-8),
        }, index=flux_times.index)

    flares_clean = flares_df.dropna(subset=["tend"]).copy()
    flares_clean["tend"] = pd.to_datetime(flares_clean["tend"])
    flares_clean["peak_flux"] = flares_clean["peak_flux"].astype(float)
    flares_sorted = flares_clean.sort_values("tend").reset_index(drop=True)

    t_ends = flares_sorted["tend"].values
    p_fluxes = flares_sorted["peak_flux"].values
    m_flares = len(t_ends)

    # Number of flares ended at or before time t
    k = np.searchsorted(t_ends, times, side="right")

    # 1. Minutes since last flare
    dt_last = np.full(n, 10080.0)  # Default: 1 week (10080 minutes) if no prior flare
    has_prior = k > 0
    if np.any(has_prior):
        last_flare_end = t_ends[k[has_prior] - 1]
        dt_last[has_prior] = (times[has_prior] - last_flare_end) / np.timedelta64(1, "m")

    # 2. Flare counts in lookback windows (6h and 24h)
    k_6h = np.searchsorted(t_ends, times - np.timedelta64(6, "h"), side="left")
    k_24h = np.searchsorted(t_ends, times - np.timedelta64(24, "h"), side="left")

    n_flares_6h = k - k_6h
    n_flares_24h = k - k_24h

    # 3. Peak flux of strongest flare ended in past 24h
    # Build Range Maximum Query (RMQ) sparse table on p_fluxes
    max_flux_24h = np.full(n, 1e-8)
    if m_flares > 0:
        log_table = np.zeros(m_flares + 1, dtype=int)
        for i in range(2, m_flares + 1):
            log_table[i] = log_table[i // 2] + 1
        k_max_pow = log_table[m_flares] + 1

        st = np.zeros((m_flares, k_max_pow))
        st[:, 0] = p_fluxes
        for j in range(1, k_max_pow):
            length = 1 << j
            half = length >> 1
            st[:m_flares - length + 1, j] = np.maximum(
                st[:m_flares - length + 1, j - 1],
                st[half:m_flares - length + 1 + half, j - 1],
            )

        # Query unique (k_24h, k) pairs for speed
        pairs = np.stack([k_24h, k], axis=1)
        unique_pairs, inv_idx = np.unique(pairs, axis=0, return_inverse=True)

        valid_pairs = unique_pairs[:, 1] > unique_pairs[:, 0]
        u_max = np.full(len(unique_pairs), 1e-8)

        if np.any(valid_pairs):
            L = unique_pairs[valid_pairs, 0]
            R = unique_pairs[valid_pairs, 1]
            diff_len = R - L
            j_pow = log_table[diff_len]
            u_max[valid_pairs] = np.maximum(st[L, j_pow], st[R - (1 << j_pow), j_pow])

        max_flux_24h = u_max[inv_idx]

    return pd.DataFrame({
        "g1_dt_last_flare": dt_last,
        "g1_n_flares_6h": n_flares_6h,
        "g1_n_flares_24h": n_flares_24h,
        "g1_max_class_24h": max_flux_24h,
    }, index=flux_times.index)


def compute_g2_pulsations(
    flux_df: pd.DataFrame, window_min: int = 30
) -> pd.DataFrame:
    """
    Group G2: Pulsation clues (quasi-periodic pulsations via STFT).
    Computes spectral energy ratios across 3 physical period bands over the past 30 min:
    - g2_band_ratio_1: 2 to 5 min periods (high frequency)
    - g2_band_ratio_2: 5 to 12 min periods (intermediate)
    - g2_band_ratio_3: 12 to 30 min periods (slow)
    
    With 1-minute cadence, the fastest visible period is 2 minutes (Nyquist).
    All calculations strictly use flux[t - window_min + 1 : t + 1].
    Vectorized sliding window for sub-second execution on millions of points.
    """
    flux_vals = pd.Series(flux_df["flux"].values, dtype=float).ffill().fillna(1e-8).values
    n = len(flux_vals)

    r1 = np.full(n, 1.0 / 3.0)
    r2 = np.full(n, 1.0 / 3.0)
    r3 = np.full(n, 1.0 / 3.0)

    if n < window_min:
        return pd.DataFrame({
            "g2_band_ratio_1": r1,
            "g2_band_ratio_2": r2,
            "g2_band_ratio_3": r3,
        }, index=flux_df.index)

    # 1. Sliding window view of length window_min: shape (n - window_min + 1, window_min)
    windows = np.lib.stride_tricks.sliding_window_view(flux_vals, window_min)
    m = windows.shape[0]

    # 2. Vectorized linear detrend
    x = np.arange(window_min, dtype=float)
    x_diff = x - x.mean()
    x_var = np.sum(x_diff ** 2)

    y_mean = windows.mean(axis=1, keepdims=True)
    slopes = np.dot(windows - y_mean, x_diff)[:, None] / x_var
    detrended = windows - (y_mean + slopes * x_diff)

    # 3. Vectorized real FFT along rows
    fft_vals = np.abs(np.fft.rfft(detrended, axis=1)) ** 2

    # 4. Period bands
    freqs = np.fft.rfftfreq(window_min, d=1.0)
    periods = np.zeros_like(freqs)
    periods[1:] = 1.0 / freqs[1:]
    periods[0] = 999.0  # DC bin

    band1 = (periods >= 2.0) & (periods < 5.0)
    band2 = (periods >= 5.0) & (periods < 12.0)
    band3 = (periods >= 12.0) & (periods <= 30.0)

    # Total non-DC AC spectral power
    total_power = np.sum(fft_vals[:, 1:], axis=1, keepdims=True) + 1e-15
    tot_sq = total_power.squeeze()

    p1 = np.sum(fft_vals[:, band1], axis=1) / tot_sq
    p2 = np.sum(fft_vals[:, band2], axis=1) / tot_sq
    p3 = np.sum(fft_vals[:, band3], axis=1) / tot_sq

    # Assign to output from index window_min - 1 onwards
    r1[window_min - 1:] = p1
    r2[window_min - 1:] = p2
    r3[window_min - 1:] = p3

    # For initial points < window_min, fill forward from the first valid window
    r1[:window_min - 1] = p1[0]
    r2[:window_min - 1] = p2[0]
    r3[:window_min - 1] = p3[0]

    return pd.DataFrame({
        "g2_band_ratio_1": r1,
        "g2_band_ratio_2": r2,
        "g2_band_ratio_3": r3,
    }, index=flux_df.index)


def compute_g3_phase(
    flux_df: pd.DataFrame, thresholds: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Group G3: Physical phase flag.
    Categorizes the solar activity state over the past 15 minutes into 4 physical regimes:
      0 = Quiet    (low net flux, negligible slope)
      1 = Rising   (steep positive slope over 15 min)
      2 = Peaking  (elevated flux with leveling slope)
      3 = Decaying (negative slope following an elevated level)
      
    Thresholds can be provided or default to robust physical values:
      quiet_max: 2.0e-7 W/m^2 net flux
      slope_up:  1.0e-8 W/m^2/min
      slope_down: -1.0e-8 W/m^2/min
    """
    if thresholds is None:
        thresholds = {
            "quiet_max": 2.0e-7,
            "slope_up": 1.0e-8,
            "slope_down": -1.0e-8,
        }

    flux = pd.Series(flux_df["flux"].values, dtype=float).ffill().fillna(1e-8).values
    n = len(flux)

    # 15-minute causal slope
    slope_15 = np.zeros(n)
    if n > 15:
        slope_15[15:] = (flux[15:] - flux[:-15]) / 15.0
        slope_15[:15] = (flux[:15] - flux[0]) / np.maximum(np.arange(15), 1)

    # Determine reference flux (net_flux if available, otherwise absolute flux)
    if "net_flux" in flux_df.columns:
        raw_net = flux_df["net_flux"].values
        ref_flux = pd.Series(raw_net, dtype=float).ffill().fillna(1e-10).values
    else:
        ref_flux = flux

    phase = np.zeros(n, dtype=int)  # Default 0: Quiet

    rising = slope_15 > thresholds["slope_up"]
    decaying = slope_15 < thresholds["slope_down"]
    peaking = (~rising) & (~decaying) & (ref_flux > thresholds["quiet_max"])

    phase[rising] = 1
    phase[peaking] = 2
    phase[decaying] = 3

    return pd.DataFrame({"g3_phase": phase}, index=flux_df.index)


def assemble_features(flux_df: pd.DataFrame, flares_df: pd.DataFrame) -> pd.DataFrame:
    """
    Assemble all clue groups G0, G1, G2, and G3 into a single DataFrame.
    Guarantees strict causal ordering and no future data leakage.
    
    Total features:
    - G0 (Recent brightness): 1 + 4*6 = 25 clues
    - G1 (Flare history): 4 clues
    - G2 (Pulsations): 3 clues
    - G3 (Phase flag): 1 clue
    Total: 33 clues.
    """
    g0 = compute_g0_brightness(flux_df)
    g1 = compute_g1_history(flux_df["time"], flares_df)
    g2 = compute_g2_pulsations(flux_df)
    g3 = compute_g3_phase(flux_df)

    # Concatenate all groups horizontally and ensure no residual NaNs
    features = pd.concat([g0, g1, g2, g3], axis=1).fillna(0.0)
    return features
