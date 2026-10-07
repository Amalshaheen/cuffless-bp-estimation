"""
Feature Extraction Module for Cuffless Blood Pressure Estimation.
Part of the pipeline based on:
"Cuff-Less Blood Pressure Estimation from Photoplethysmogram Signals Using Deep Learning
and Cardiovascular Dynamics" (Sensors 2023, 23, 4145).

Extracts:
1. 7 Pulse Rate Variability (PRV) dynamics features from systolic peaks.
2. 7 mean Point-to-Point (mPTP) morphology features from cleaned pulse waves.
"""

from typing import List, Optional
import logging
import warnings
import numpy as np
import pandas as pd
import neurokit2 as nk

logger = logging.getLogger(__name__)

# Feature column definitions as per the paper (Sensors 2023, 23, 4145)
MORPHOLOGY_FEATURES: List[str] = [
    # 3 Timings
    "cardiac_period",
    "systolic_upstroke_time",
    "diastolic_time",
    # 6 Diastolic Widths
    "dias_w_10",
    "dias_w_25",
    "dias_w_33",
    "dias_w_50",
    "dias_w_66",
    "dias_w_75",
    # 6 Sum Widths
    "sum_w_10",
    "sum_w_25",
    "sum_w_33",
    "sum_w_50",
    "sum_w_66",
    "sum_w_75",
    # 6 Ratios
    "ratio_10",
    "ratio_25",
    "ratio_33",
    "ratio_50",
    "ratio_66",
    "ratio_75",
]

DYNAMICS_FEATURES: List[str] = [
    "SDNN",
    "PRVTi",
    "TINN",
    "LF",
    "HF",
    "a1",
    "a2",
]


def extract_prv_dynamics(
    peaks: np.ndarray,
    sampling_rate: int = 100
) -> pd.Series:
    """
    Extract the 7 PRV dynamics features specified in the paper from PPG peaks.

    Features:
    - SDNN: Standard deviation of normal-to-normal inter-beat intervals (HRV_SDNN)
    - PRVTi: Triangular index (HRV_HTI)
    - TINN: Baseline width of the IBI histogram (HRV_TINN)
    - LF: Low-frequency power, 0.04–0.15 Hz (HRV_LF)
    - HF: High-frequency power, 0.15–0.40 Hz (HRV_HF)
    - a1: Short-term DFA alpha1 exponent (HRV_DFA_alpha1)
    - a2: Long-term DFA alpha2 exponent (HRV_DFA_alpha2)

    Parameters
    ----------
    peaks : np.ndarray
        1D array of sample indices of detected systolic peaks.
    sampling_rate : int, optional
        Sampling frequency in Hertz (Hz). Default is 100 Hz.

    Returns
    -------
    pd.Series
        Pandas Series indexed by ['SDNN', 'PRVTi', 'TINN', 'LF', 'HF', 'a1', 'a2'].
    """
    peaks = np.asarray(peaks, dtype=np.int64).squeeze()

    # Minimum peaks check: frequency domain and DFA need sufficient beats (typically > 30 beats)
    num_peaks = len(peaks) if peaks.ndim == 1 else 0
    if num_peaks < 5:
        logger.warning(
            "Insufficient peaks (%d detected). Cannot compute PRV dynamics; returning NaNs.",
            num_peaks,
        )
        return pd.Series(
            data=[np.nan] * len(DYNAMICS_FEATURES),
            index=DYNAMICS_FEATURES,
            dtype=np.float64,
        )

    if num_peaks < 40:
        logger.warning(
            "Short peak series (%d peaks). Frequency domain (LF/HF) and DFA (a1/a2) "
            "metrics may evaluate to NaN or be noisy. Windows >= 120s are recommended.",
            num_peaks,
        )

    # Compute HRV/PRV metrics using neurokit2
    mapping = {
        "HRV_SDNN": "SDNN",
        "HRV_HTI": "PRVTi",
        "HRV_TINN": "TINN",
        "HRV_LF": "LF",
        "HRV_HF": "HF",
        "HRV_DFA_alpha1": "a1",
        "HRV_DFA_alpha2": "a2",
    }

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Compute only the relevant HRV submodules (time, frequency, nonlinear)
            # which is 4x faster than full nk.hrv while calculating the exact same metrics.
            prv_df = pd.concat([
                nk.hrv_time(peaks, sampling_rate=sampling_rate),
                nk.hrv_frequency(peaks, sampling_rate=sampling_rate),
                nk.hrv_nonlinear(peaks, sampling_rate=sampling_rate),
            ], axis=1)
    except Exception as exc:
        logger.warning("neurokit2 HRV extraction raised exception: %s. Returning NaNs.", exc)
        return pd.Series(
            data=[np.nan] * len(DYNAMICS_FEATURES),
            index=DYNAMICS_FEATURES,
            dtype=np.float64,
        )

    # Assemble the series with fallbacks for missing columns
    values = {}
    for nk_col, paper_col in mapping.items():
        if nk_col in prv_df.columns:
            val = prv_df[nk_col].iloc[0]
            values[paper_col] = float(val) if pd.notna(val) else np.nan
        else:
            values[paper_col] = np.nan

    series = pd.Series(values, index=DYNAMICS_FEATURES, dtype=np.float64)

    # Check for NaN in frequency and DFA features
    nan_cols = series[series.isna()].index.tolist()
    if nan_cols:
        logger.warning(
            "PRV features %s evaluated to NaN (likely due to short window length).",
            nan_cols,
        )

    return series


def extract_mptp_morphology(
    cleaned_signal: np.ndarray,
    peaks: np.ndarray,
    troughs: np.ndarray
) -> pd.Series:
    """
    Extract the full 21 pulse morphology indicators using the mean Point-to-Point (mPTP) algorithm.
    Based on Section 2.2.1 of Sensors 2023, 23, 4145.

    For each valid cardiac cycle delimited by consecutive troughs (troughs[i] to troughs[i+1])
    with exactly one intermediate systolic peak:
    - 3 Timings:
        1. cardiac_period: Total pulse duration (samples)
        2. systolic_upstroke_time: Duration from pulse start trough to systolic peak (samples)
        3. diastolic_time: Duration from systolic peak to pulse end trough (samples)
    - 6 Diastolic Widths:
        4-9. dias_w_10, dias_w_25, dias_w_33, dias_w_50, dias_w_66, dias_w_75:
             Width from systolic peak to downstroke crossing at 10%, 25%, 33%, 50%, 66%, 75%
             of peak amplitude (samples)
    - 6 Sum Widths:
        10-15. sum_w_10, sum_w_25, sum_w_33, sum_w_50, sum_w_66, sum_w_75:
               Total pulse width (systolic + diastolic) at 10%, 25%, 33%, 50%, 66%, 75%
               of peak amplitude (samples)
    - 6 Ratios:
        16-21. ratio_10, ratio_25, ratio_33, ratio_50, ratio_66, ratio_75:
               Ratio of diastolic width to systolic width at 10%, 25%, 33%, 50%, 66%, 75%
               of peak amplitude

    Parameters
    ----------
    cleaned_signal : np.ndarray
        1D array of the preprocessed PPG signal (e.g., 480-second window).
    peaks : np.ndarray
        1D array of sample indices for detected systolic peaks.
    troughs : np.ndarray
        1D array of sample indices for detected diastolic troughs.

    Returns
    -------
    pd.Series
        Pandas Series containing the mean feature vector indexed by MORPHOLOGY_FEATURES.
    """
    cleaned_signal = np.asarray(cleaned_signal, dtype=np.float64).squeeze()
    peaks = np.asarray(peaks, dtype=np.int64).squeeze()
    troughs = np.asarray(troughs, dtype=np.int64).squeeze()

    if len(troughs) < 2 or len(peaks) < 1:
        logger.warning("Insufficient peaks or troughs to extract morphology features.")
        return pd.Series(
            data=[np.nan] * len(MORPHOLOGY_FEATURES),
            index=MORPHOLOGY_FEATURES,
            dtype=np.float64,
        )

    morphology_records: List[List[float]] = []

    # Threshold percentages for widths and ratios
    threshold_specs = [
        ("10", 0.10),
        ("25", 0.25),
        ("33", 0.33),
        ("50", 0.50),
        ("66", 0.66),
        ("75", 0.75),
    ]

    def _find_crossings(sig: np.ndarray, p_idx: int, level: float):
        up = np.where(sig[:p_idx] <= level)[0]
        down = np.where(sig[p_idx:] <= level)[0]
        up_idx = int(up[-1]) if len(up) > 0 else 0
        down_idx = int(p_idx + down[0]) if len(down) > 0 else len(sig) - 1
        return up_idx, down_idx

    # Iterate through consecutive troughs defining pulse boundaries
    for i in range(len(troughs) - 1):
        t_start = int(troughs[i])
        t_end = int(troughs[i + 1])

        if t_end <= t_start or t_end > len(cleaned_signal):
            continue

        # Locate systolic peak(s) between these two troughs
        matching_peaks = [p for p in peaks if t_start < p < t_end]
        if len(matching_peaks) != 1:
            # Skip irregular pulses containing 0 or multiple peaks
            continue
        p_curr = int(matching_peaks[0])

        segment = cleaned_signal[t_start:t_end]
        local_p = p_curr - t_start

        y_min = float(np.min(segment))
        y_max = float(segment[local_p])
        amp = y_max - y_min

        # Skip inverted, flat, or degenerate cycles
        if amp <= 1e-6:
            continue

        # 3 Timings (in sample counts)
        cardiac_period = float(len(segment))
        systolic_upstroke_time = float(local_p)
        diastolic_time = float(len(segment) - local_p)

        dias_widths: List[float] = []
        sum_widths: List[float] = []
        ratios: List[float] = []

        for _, pct in threshold_specs:
            h = y_min + pct * amp
            u, d = _find_crossings(segment, local_p, h)
            dias_w = float(d - local_p)
            sum_w = float(d - u)
            sys_w = max(float(local_p - u), 1.0)
            dias_w_pos = max(dias_w, 1.0)
            ratio = float(dias_w_pos / sys_w)

            dias_widths.append(dias_w)
            sum_widths.append(sum_w)
            ratios.append(ratio)

        # Assemble the full 21 indicators for this cardiac cycle
        cycle_features = [
            cardiac_period,
            systolic_upstroke_time,
            diastolic_time,
            *dias_widths,
            *sum_widths,
            *ratios,
        ]
        morphology_records.append(cycle_features)

    if not morphology_records:
        logger.warning("No valid pulse cycles met criteria for mPTP morphology extraction.")
        return pd.Series(
            data=[np.nan] * len(MORPHOLOGY_FEATURES),
            index=MORPHOLOGY_FEATURES,
            dtype=np.float64,
        )

    # Compute arithmetic mean across all valid beats in the window (mPTP)
    morph_arr = np.array(morphology_records, dtype=np.float64)
    mptp_mean = np.mean(morph_arr, axis=0)

    logger.debug(
        "Extracted full 21 mPTP morphology features across %d valid cardiac cycles.",
        len(morphology_records),
    )

    return pd.Series(mptp_mean, index=MORPHOLOGY_FEATURES, dtype=np.float64)
