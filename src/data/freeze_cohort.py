"""
Cohort Freezing Script for Cuffless Blood Pressure Estimation.
Replicating the experimental train/test patient split methodology from:
"Cuff-Less Blood Pressure Estimation from Photoplethysmogram Signals Using Deep Learning
and Cardiovascular Dynamics" (Sensors 2023, 23, 4145).

Splits the 225-subject cohort into:
- 25 unseen test patients satisfying the exact paper benchmark bounds:
    105.40 <= target_sbp <= 151.70 mmHg
    53.85  <= target_dbp <= 72.93 mmHg
- 200 development/training patients (for train & validation)

Saves:
- `data/processed/uci_test_25_unseen.csv`
- `data/processed/uci_train_200_dev.csv`
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import argparse
import logging
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("freeze_cohort")


def freeze_cohort(
    cohort_path: str = "data/processed/cohort_225.csv",
    test_out_path: str = "data/processed/uci_test_25_unseen.csv",
    train_out_path: str = "data/processed/uci_train_200_dev.csv",
    num_test_patients: int = 25,
    random_state: int = 42,
):
    if not os.path.isfile(cohort_path):
        raise FileNotFoundError(f"Cohort CSV not found at: {cohort_path}")

    df = pd.read_csv(cohort_path)
    logger.info("Loaded cohort data from %s with %d subjects.", cohort_path, len(df))

    # Paper screening criteria for test subjects:
    # SBP range: [105.40, 151.70] mmHg
    # DBP range: [53.85, 72.93] mmHg
    sbp_min, sbp_max = 105.40, 151.70
    dbp_min, dbp_max = 53.85, 72.93

    criteria_mask = (
        (df["target_sbp"] >= sbp_min)
        & (df["target_sbp"] <= sbp_max)
        & (df["target_dbp"] >= dbp_min)
        & (df["target_dbp"] <= dbp_max)
    )

    candidates = df[criteria_mask]
    logger.info(
        "Screened %d candidates meeting paper test criteria (SBP: [%.2f, %.2f], DBP: [%.2f, %.2f]).",
        len(candidates),
        sbp_min,
        sbp_max,
        dbp_min,
        dbp_max,
    )

    if len(candidates) < num_test_patients:
        raise ValueError(
            f"Only {len(candidates)} candidates met the criteria, but {num_test_patients} are required."
        )

    # Sample exactly 25 distinct subjects using fixed random seed
    test_df = candidates.sample(n=num_test_patients, random_state=random_state)
    dev_df = df.drop(test_df.index)

    # If the remaining pool is larger than 200, retain exactly 200 (or all if exactly 200)
    if len(dev_df) > 200:
        dev_df = dev_df.iloc[:200]

    os.makedirs(os.path.dirname(test_out_path), exist_ok=True)
    os.makedirs(os.path.dirname(train_out_path), exist_ok=True)

    test_df.to_csv(test_out_path, index=False)
    dev_df.to_csv(train_out_path, index=False)

    logger.info("Saved %d test subjects to: %s", len(test_df), test_out_path)
    logger.info("Saved %d training/dev subjects to: %s", len(dev_df), train_out_path)

    # Print summary statistics
    print("\n" + "=" * 70)
    print("COHORT SPLIT STATISTICS (SENSORS 2023 BENCHMARK)")
    print("=" * 70)

    def _print_stats(label: str, sub_df: pd.DataFrame):
        sbp = sub_df["target_sbp"]
        dbp = sub_df["target_dbp"]
        print(f"\n{label} (N = {len(sub_df)}):")
        print(
            f"  SBP: Mean = {sbp.mean():6.2f} mmHg | Min = {sbp.min():6.2f} mmHg | Max = {sbp.max():6.2f} mmHg | Std = {sbp.std():5.2f}"
        )
        print(
            f"  DBP: Mean = {dbp.mean():6.2f} mmHg | Min = {dbp.min():6.2f} mmHg | Max = {dbp.max():6.2f} mmHg | Std = {dbp.std():5.2f}"
        )

    _print_stats("Unseen Test Set (uci_test_25_unseen.csv)", test_df)
    _print_stats("Development/Training Set (uci_train_200_dev.csv)", dev_df)
    print("=" * 70 + "\n")

    return test_df, dev_df


def main():
    parser = argparse.ArgumentParser(description="Freeze cohort into 25 test and 200 train subjects.")
    parser.add_argument(
        "--cohort-path",
        type=str,
        default="data/processed/cohort_225.csv",
        help="Path to full cohort CSV.",
    )
    parser.add_argument(
        "--test-out",
        type=str,
        default="data/processed/uci_test_25_unseen.csv",
        help="Path for unseen test CSV.",
    )
    parser.add_argument(
        "--train-out",
        type=str,
        default="data/processed/uci_train_200_dev.csv",
        help="Path for dev/train CSV.",
    )
    parser.add_argument(
        "--num-test",
        type=int,
        default=25,
        help="Number of test subjects (default: 25).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling test subjects (default: 42).",
    )

    args = parser.parse_args()
    freeze_cohort(
        cohort_path=args.cohort_path,
        test_out_path=args.test_out,
        train_out_path=args.train_out,
        num_test_patients=args.num_test,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
