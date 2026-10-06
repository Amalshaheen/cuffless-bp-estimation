"""
Signal Preprocessing Module for Photoplethysmogram (PPG) Signals.
Part of the Cuffless Blood Pressure Estimation pipeline based on:
"Cuff-Less Blood Pressure Estimation from Photoplethysmogram Signals Using Deep Learning
and Cardiovascular Dynamics" (Sensors 2023, 23, 4145).
"""

from typing import Tuple
import logging
import numpy as np
import neurokit2 as nk

logger = logging.getLogger(__name__)


def preprocess_ppg(
    raw_signal: np.ndarray,
    sampling_rate: int = 100
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Clean raw PPG signal and detect systolic peaks and diastolic troughs.

    Preprocessing Steps:
    1. Scrub baseline wander and high-frequency noise using a 0.5–8 Hz
       Butterworth bandpass filter via `neurokit2.ppg_clean` (Elgendi method).
    2. Detect systolic peaks (local maxima) using `neurokit2.ppg_findpeaks`.
    3. Detect diastolic troughs (onsets/valleys) by inverting the cleaned signal
       and applying `neurokit2.ppg_findpeaks`.

    Parameters
    ----------
    raw_signal : np.ndarray
        1D array containing the raw PPG time-series amplitude values.
    sampling_rate : int, optional
        Sampling frequency in Hertz (Hz). Default is 100 Hz.

    Returns
    -------
    cleaned_signal : np.ndarray
        1D array of the bandpass-filtered, denoised PPG signal.
    peaks : np.ndarray
        1D array of sample indices corresponding to detected systolic peaks.
    troughs : np.ndarray
        1D array of sample indices corresponding to detected diastolic troughs.
    """
    # Ensure input is a 1D numpy array
    signal = np.asarray(raw_signal, dtype=np.float64).squeeze()
    if signal.ndim != 1:
        raise ValueError(f"raw_signal must be a 1D array, got shape {signal.shape}")

    # Check for empty or excessively short signal
    if len(signal) < sampling_rate:
        raise ValueError(
            f"Signal length ({len(signal)}) is shorter than 1 second at {sampling_rate} Hz."
        )

    # Impute NaNs or Infs if present
    if not np.all(np.isfinite(signal)):
        logger.warning("Non-finite values detected in raw PPG signal. Interpolating.")
        valid_mask = np.isfinite(signal)
        if not np.any(valid_mask):
            raise ValueError("All values in raw_signal are non-finite.")
        indices = np.arange(len(signal))
        signal = np.interp(indices, indices[valid_mask], signal[valid_mask])

    # 1. Clean the signal using Butterworth bandpass filter (0.5–8 Hz)
    # The default 'elgendi' method in neurokit2 applies a 0.5–8 Hz Butterworth bandpass filter
    cleaned_signal = nk.ppg_clean(signal, sampling_rate=sampling_rate, method="elgendi")
    cleaned_signal = np.asarray(cleaned_signal, dtype=np.float64)

    # 2. Detect systolic peaks
    peak_info = nk.ppg_findpeaks(cleaned_signal, sampling_rate=sampling_rate)
    peaks = np.asarray(peak_info.get("PPG_Peaks", []), dtype=np.int64)

    # 3. Detect diastolic troughs by inverting the cleaned signal
    inverted_signal = cleaned_signal * -1.0
    trough_info = nk.ppg_findpeaks(inverted_signal, sampling_rate=sampling_rate)
    troughs = np.asarray(trough_info.get("PPG_Peaks", []), dtype=np.int64)

    logger.debug(
        "Preprocessing completed: %d samples, %d peaks, %d troughs detected at fs=%d Hz",
        len(cleaned_signal),
        len(peaks),
        len(troughs),
        sampling_rate,
    )

    return cleaned_signal, peaks, troughs
