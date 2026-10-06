"""
Batch Dataset Pipeline for Cuffless Blood Pressure Estimation.
Replicating the dataset construction methodology from:
"Cuff-Less Blood Pressure Estimation from Photoplethysmogram Signals Using Deep Learning
and Cardiovascular Dynamics" (Sensors 2023, 23, 4145).

Extracts:
- 7 mPTP pulse morphology features (Stage 1 inputs)
- 7 PRV dynamics features (Stage 2 inputs)
- Target SBP and DBP labels (from ABP peaks/troughs or cuff references)
- Record/window identifiers

Saves processed feature matrix to:
`data/processed/feature_matrix.csv`
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import glob
import logging
import argparse
from typing import Dict, List, Optional, Tuple
import h5py
import numpy as np
import pandas as pd
import scipy.signal as signal

from src.preprocessing.signal_cleaner import preprocess_ppg
from src.features.extractors import (
    extract_prv_dynamics,
    extract_mptp_morphology,
    MORPHOLOGY_FEATURES,
    DYNAMICS_FEATURES,
)

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_dataset")

FEATURE_COLUMNS = MORPHOLOGY_FEATURES + DYNAMICS_FEATURES + [
    "target_sbp",
    "target_dbp",
    "record_id",
]


def extract_uci_ground_truth_bp(
    abp_segment: np.ndarray,
    sampling_rate: int = 125
) -> Tuple[float, float]:
    """
    Extract reference SBP and DBP from an Arterial Blood Pressure (ABP) segment.

    Calculates mean SBP from systolic peaks and mean DBP from diastolic troughs.
    Applies physiological plausibility bounds (60 <= SBP <= 220, 30 <= DBP <= 130).

    Parameters
    ----------
    abp_segment : np.ndarray
        1D array of ABP values (mmHg).
    sampling_rate : int
        Sampling frequency in Hz. Default is 125 Hz.

    Returns
    -------
    Tuple[float, float]
        (mean_sbp, mean_dbp) or (np.nan, np.nan) if segment is non-physiological.
    """
    if len(abp_segment) < sampling_rate:
        return np.nan, np.nan

    # Min distance between cardiac beats (~0.35s = ~170 BPM max)
    min_distance = int(sampling_rate * 0.35)

    # Detect systolic peaks in ABP
    abp_peaks, _ = signal.find_peaks(
        abp_segment,
        distance=min_distance,
        prominence=10.0,
    )

    # Detect diastolic troughs in ABP (by finding peaks of inverted signal)
    abp_troughs, _ = signal.find_peaks(
        -abp_segment,
        distance=min_distance,
        prominence=10.0,
    )

    if len(abp_peaks) < 5 or len(abp_troughs) < 5:
        return np.nan, np.nan

    sbp_vals = abp_segment[abp_peaks]
    dbp_vals = abp_segment[abp_troughs]

    mean_sbp = float(np.mean(sbp_vals))
    mean_dbp = float(np.mean(dbp_vals))

    # Physiological plausibility checks
    if not (60.0 <= mean_sbp <= 220.0 and 30.0 <= mean_dbp <= 130.0):
        return np.nan, np.nan

    if mean_sbp <= mean_dbp + 10.0:
        return np.nan, np.nan

    return mean_sbp, mean_dbp


def process_window(
    ppg_segment: np.ndarray,
    target_sbp: float,
    target_dbp: float,
    record_id: str,
    sampling_rate: int
) -> Optional[Dict[str, float]]:
    """
    Process a single segment of PPG to extract the 14 features and assemble row dict.

    Returns None if any step fails or features contain non-finite values.
    """
    if not (np.isfinite(target_sbp) and np.isfinite(target_dbp)):
        return None

    try:
        cleaned, peaks, troughs = preprocess_ppg(ppg_segment, sampling_rate=sampling_rate)
    except Exception as exc:
        logger.debug("Preprocessing failed for %s: %s", record_id, exc)
        return None

    if len(peaks) < 10 or len(troughs) < 10:
        return None

    morph_series = extract_mptp_morphology(cleaned, peaks, troughs)
    if morph_series.isna().any():
        return None

    dyn_series = extract_prv_dynamics(peaks, sampling_rate=sampling_rate)
    if dyn_series.isna().any():
        return None

    row: Dict[str, float] = {}
    for col in MORPHOLOGY_FEATURES:
        row[col] = float(morph_series[col])
    for col in DYNAMICS_FEATURES:
        row[col] = float(dyn_series[col])

    row["target_sbp"] = float(target_sbp)
    row["target_dbp"] = float(target_dbp)
    row["record_id"] = record_id

    return row


def process_uci_mimic(
    mat_path: str = "data/raw/uci_mimic/Part_1.mat",
    max_records: Optional[int] = None,
    window_seconds: int = 120,
    stride_seconds: Optional[int] = None,
    sampling_rate: int = 125,
) -> pd.DataFrame:
    """
    Extract features and ground truth BP from UCI MIMIC-II .mat files.

    In UCI MIMIC HDF5 files:
    - Channel 0: PPG (sampled at 125 Hz)
    - Channel 1: ABP (Arterial Blood Pressure in mmHg)
    - Channel 2: ECG (optional)
    """
    if not os.path.isfile(mat_path):
        raise FileNotFoundError(f"UCI dataset file not found at: {mat_path}")

    logger.info("Opening UCI MIMIC-II file: %s", mat_path)
    window_samples = int(window_seconds * sampling_rate)
    stride_samples = int((stride_seconds or window_seconds) * sampling_rate)

    records: List[Dict[str, float]] = []

    with h5py.File(mat_path, "r") as f:
        # Find cell array key (e.g., 'Part_1')
        main_key = None
        for k in f.keys():
            if k.startswith("Part_"):
                main_key = k
                break
        if not main_key:
            main_key = list(f.keys())[-1]

        dataset = f[main_key]
        total_cells = dataset.shape[0]
        num_cells = min(total_cells, max_records) if max_records else total_cells

        logger.info(
            "Found %d records in %s. Processing %d records (window=%ds, stride=%ds).",
            total_cells,
            main_key,
            num_cells,
            window_seconds,
            stride_seconds or window_seconds,
        )

        for cell_idx in range(num_cells):
            try:
                ref = dataset[cell_idx, 0]
                sample_data = np.array(f[ref])
            except Exception as e:
                logger.warning("Failed to read cell %d: %s", cell_idx, e)
                continue

            # In h5py, MATLAB (3, N) arrays are loaded transposed as (N, 3)
            if sample_data.ndim != 2:
                continue

            if sample_data.shape[1] >= 2 and sample_data.shape[0] > sample_data.shape[1]:
                ppg_all = sample_data[:, 0]
                abp_all = sample_data[:, 1]
            elif sample_data.shape[0] >= 2:
                ppg_all = sample_data[0, :]
                abp_all = sample_data[1, :]
            else:
                continue

            total_samples = len(ppg_all)
            if total_samples < window_samples:
                continue

            # Sliding windows over the record
            win_count = 0
            for start_idx in range(0, total_samples - window_samples + 1, stride_samples):
                end_idx = start_idx + window_samples
                ppg_win = ppg_all[start_idx:end_idx]
                abp_win = abp_all[start_idx:end_idx]

                mean_sbp, mean_dbp = extract_uci_ground_truth_bp(abp_win, sampling_rate)
                if not (np.isfinite(mean_sbp) and np.isfinite(mean_dbp)):
                    continue

                rec_id = f"uci_cell{cell_idx:04d}_win{win_count:02d}"
                row = process_window(
                    ppg_win,
                    target_sbp=mean_sbp,
                    target_dbp=mean_dbp,
                    record_id=rec_id,
                    sampling_rate=sampling_rate,
                )
                if row is not None:
                    records.append(row)
                    win_count += 1

            if (cell_idx + 1) % 10 == 0 or cell_idx == num_cells - 1:
                logger.info(
                    "Processed %d/%d records | Current valid feature rows: %d",
                    cell_idx + 1,
                    num_cells,
                    len(records),
                )

    df = pd.DataFrame(records, columns=FEATURE_COLUMNS)
    logger.info("UCI extraction complete. Total valid samples: %d", len(df))
    return df


def process_uq_dataset(
    uq_dir: str = "data/raw/uq_vital_signs/uqvitalsignsdata",
    max_cases: Optional[int] = None,
    window_seconds: int = 120,
    stride_seconds: Optional[int] = None,
    sampling_rate: int = 100,
) -> pd.DataFrame:
    """
    Extract features and ground truth BP from UQ Vital Signs dataset CSV files.

    UQ dataset uses 100 Hz sampling rate. PPG is found in the 'ECG' column,
    and reference NBP is recorded in 'NBP (Sys)' and 'NBP (Dia)'.
    """
    csv_pattern = os.path.join(uq_dir, "case*", "fulldata", "*.csv")
    csv_files = sorted(glob.glob(csv_pattern))
    if not csv_files:
        raise FileNotFoundError(f"No UQ CSV files found matching {csv_pattern}")

    target_files = csv_files[:max_cases] if max_cases else csv_files
    logger.info("Found %d UQ CSV files. Processing %d files.", len(csv_files), len(target_files))

    window_samples = int(window_seconds * sampling_rate)
    stride_samples = int((stride_seconds or window_seconds) * sampling_rate)
    records: List[Dict[str, float]] = []

    for f_idx, file_path in enumerate(target_files):
        case_name = os.path.basename(os.path.dirname(os.path.dirname(file_path)))
        try:
            df_case = pd.read_csv(file_path, on_bad_lines="skip", low_memory=False)
        except Exception as exc:
            logger.warning("Error reading %s: %s", file_path, exc)
            continue

        if "ECG" not in df_case.columns:
            continue

        ppg_col = pd.to_numeric(df_case["ECG"], errors="coerce").values

        # Determine reference BP source
        sbp_col = None
        dbp_col = None
        for s_cand in ["NBP (Sys)", "ART (Sys)"]:
            if s_cand in df_case.columns and df_case[s_cand].notna().sum() > 5:
                sbp_col = pd.to_numeric(df_case[s_cand], errors="coerce").values
                break
        for d_cand in ["NBP (Dia)", "ART (Dia)"]:
            if d_cand in df_case.columns and df_case[d_cand].notna().sum() > 5:
                dbp_col = pd.to_numeric(df_case[d_cand], errors="coerce").values
                break

        if sbp_col is None or dbp_col is None:
            continue

        total_samples = len(ppg_col)
        # Skip first 50 seconds (initialization noise)
        start_offset = 50 * sampling_rate
        win_count = 0

        for start_idx in range(start_offset, total_samples - window_samples + 1, stride_samples):
            end_idx = start_idx + window_samples
            ppg_win = ppg_col[start_idx:end_idx]

            sbp_win = sbp_col[start_idx:end_idx]
            dbp_win = dbp_col[start_idx:end_idx]
            valid_sbp = sbp_win[np.isfinite(sbp_win)]
            valid_dbp = dbp_win[np.isfinite(dbp_win)]

            if len(valid_sbp) == 0 or len(valid_dbp) == 0:
                continue

            ref_sbp = float(np.mean(valid_sbp))
            ref_dbp = float(np.mean(valid_dbp))

            # Handle sensor column inversion in some clinical recordings
            if ref_dbp > ref_sbp:
                ref_sbp, ref_dbp = ref_dbp, ref_sbp

            if not (50.0 <= ref_sbp <= 220.0 and 30.0 <= ref_dbp <= 130.0 and ref_sbp > ref_dbp + 5):
                continue

            rec_id = f"uq_{case_name}_win{win_count:02d}"
            row = process_window(
                ppg_win,
                target_sbp=ref_sbp,
                target_dbp=ref_dbp,
                record_id=rec_id,
                sampling_rate=sampling_rate,
            )
            if row is not None:
                records.append(row)
                win_count += 1

        logger.info(
            "Processed UQ file %d/%d (%s) | Total valid rows: %d",
            f_idx + 1,
            len(target_files),
            case_name,
            len(records),
        )

    df = pd.DataFrame(records, columns=FEATURE_COLUMNS)
    logger.info("UQ extraction complete. Total valid samples: %d", len(df))
    return df


def build_dataset(
    dataset_type: str = "uci",
    raw_data_path: Optional[str] = None,
    output_path: str = "data/processed/feature_matrix.csv",
    max_records: Optional[int] = None,
    window_seconds: int = 120,
    stride_seconds: Optional[int] = None,
) -> pd.DataFrame:
    """
    Main orchestration function to build the feature matrix and write to disk.

    Parameters
    ----------
    dataset_type : str
        'uci' for UCI MIMIC-II or 'uq' for University of Queensland dataset.
    raw_data_path : str, optional
        Path to file/directory. If None, uses default path for the chosen type.
    output_path : str
        Destination CSV file path for the processed feature matrix.
    max_records : int, optional
        Maximum number of raw records/cases to process (for batch dry runs).
    window_seconds : int
        Window duration in seconds (recommended >= 120s to ensure PRV validity).
    stride_seconds : int, optional
        Step between consecutive windows. Defaults to window_seconds (non-overlapping).

    Returns
    -------
    pd.DataFrame
        Constructed feature matrix DataFrame.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if dataset_type.lower() == "uci":
        path = raw_data_path or "data/raw/uci_mimic/Part_1.mat"
        df = process_uci_mimic(
            mat_path=path,
            max_records=max_records,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            sampling_rate=125,
        )
    elif dataset_type.lower() == "uq":
        path = raw_data_path or "data/raw/uq_vital_signs/uqvitalsignsdata"
        df = process_uq_dataset(
            uq_dir=path,
            max_cases=max_records,
            window_seconds=window_seconds,
            stride_seconds=stride_seconds,
            sampling_rate=100,
        )
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}. Choose 'uci' or 'uq'.")

    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info("Saved %d feature records to: %s", len(df), output_path)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Build batch feature dataset for cuffless blood pressure estimation."
    )
    parser.add_argument(
        "--dataset-type",
        type=str,
        default="uci",
        choices=["uci", "uq"],
        help="Target dataset: 'uci' (MIMIC-II) or 'uq' (Queensland).",
    )
    parser.add_argument(
        "--raw-path",
        type=str,
        default=None,
        help="Optional custom path to raw dataset file or directory.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/processed/feature_matrix.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Max records to process (for quick testing).",
    )
    parser.add_argument(
        "--window-sec",
        type=int,
        default=120,
        help="Window duration in seconds (default: 120s).",
    )
    parser.add_argument(
        "--stride-sec",
        type=int,
        default=None,
        help="Stride between windows in seconds (default: non-overlapping).",
    )

    args = parser.parse_args()
    build_dataset(
        dataset_type=args.dataset_type,
        raw_data_path=args.raw_path,
        output_path=args.output_path,
        max_records=args.max_records,
        window_seconds=args.window_sec,
        stride_seconds=args.stride_sec,
    )


if __name__ == "__main__":
    main()
