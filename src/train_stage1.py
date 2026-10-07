"""
Training and Evaluation Pipeline for Stage 1 Morphology Deep Neural Network (DNN).
Replicating Stage 1 from:
"Cuff-Less Blood Pressure Estimation from Photoplethysmogram Signals Using Deep Learning
and Cardiovascular Dynamics" (Sensors 2023, 23, 4145).

Stage 1 estimates preliminary SBP and DBP directly from the full 21 mPTP morphology features.
Paper benchmarks for Stage 1:
- SBP MAE: ~11.24 mmHg
- DBP MAE: ~4.75 mmHg
"""

import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import copy
import logging
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.features.extractors import MORPHOLOGY_FEATURES
from src.models.cascaded_bpe_net import MorphologyDNN

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train_stage1")

TARGET_COLUMNS = ["target_sbp", "target_dbp"]

PAPER_STAGE1_SBP_MAE = 11.24
PAPER_STAGE1_DBP_MAE = 4.75


def prepare_datasets(
    train_csv: str = "data/processed/uci_train_200_dev.csv",
    test_csv: str = "data/processed/uci_test_25_unseen.csv",
    val_ratio: float = 0.15,
    random_state: int = 3,
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], StandardScaler]:
    """
    Load frozen development and unseen test sets.
    Splits development set into Train (85%) and Validation (15%).
    Standardizes features using a scaler fitted exclusively on the Train partition.
    """
    if not os.path.isfile(train_csv):
        raise FileNotFoundError(f"Training dataset not found: {train_csv}")
    if not os.path.isfile(test_csv):
        raise FileNotFoundError(f"Test dataset not found: {test_csv}")

    train_df = pd.read_csv(train_csv).dropna(subset=MORPHOLOGY_FEATURES + TARGET_COLUMNS)
    test_df = pd.read_csv(test_csv).dropna(subset=MORPHOLOGY_FEATURES + TARGET_COLUMNS)

    logger.info("Loaded %d dev samples and %d unseen test samples.", len(train_df), len(test_df))

    X_dev = train_df[MORPHOLOGY_FEATURES].values.astype(np.float32)
    y_dev = train_df[TARGET_COLUMNS].values.astype(np.float32)

    X_test = test_df[MORPHOLOGY_FEATURES].values.astype(np.float32)
    y_test = test_df[TARGET_COLUMNS].values.astype(np.float32)

    # 85% Train, 15% Validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=val_ratio, random_state=random_state, shuffle=True
    )

    logger.info(
        "Split Partition Counts: Train=%d (%.1f%%) | Val=%d (%.1f%%) | Unseen Test=%d",
        len(X_train),
        100.0 * len(X_train) / len(X_dev),
        len(X_val),
        100.0 * len(X_val) / len(X_dev),
        len(X_test),
    )

    # Standardize features using training partition statistics only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    return (
        (X_train_scaled, y_train),
        (X_val_scaled, y_val),
        (X_test_scaled, y_test),
        scaler,
    )


def cross_validate_stage1(
    train_csv: str = "data/processed/uci_train_200_dev.csv",
    n_splits: int = 5,
    lr: float = 0.005,
    weight_decay: float = 1e-4,
    batch_size: int = 16,
    max_epochs: int = 300,
    patience: int = 35,
    seed: int = 3,
) -> Dict[str, float]:
    """
    Run K-Fold Cross-Validation across the 200 development subjects.
    Emulates the patient partition validation described in Section 2.3 of the paper.
    """
    if not os.path.isfile(train_csv):
        raise FileNotFoundError(f"Development CSV not found at: {train_csv}")

    dev_df = pd.read_csv(train_csv).dropna(subset=MORPHOLOGY_FEATURES + TARGET_COLUMNS)
    X_dev = dev_df[MORPHOLOGY_FEATURES].values.astype(np.float32)
    y_dev = dev_df[TARGET_COLUMNS].values.astype(np.float32)

    logger.info(
        "Running %d-Fold Cross-Validation across %d development subjects (Morphology Dim: %d)...",
        n_splits,
        len(dev_df),
        X_dev.shape[1],
    )

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof_predictions = np.zeros_like(y_dev)

    print("\n" + "=" * 70)
    print(f"STAGE 1 MORPHOLOGY DNN - {n_splits}-FOLD CROSS-VALIDATION (200 DEV SUBJECTS)")
    print("=" * 70)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_dev), start=1):
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_dev[train_idx]).astype(np.float32)
        X_va = scaler.transform(X_dev[val_idx]).astype(np.float32)
        y_tr = y_dev[train_idx]
        y_va = y_dev[val_idx]

        torch.manual_seed(seed + fold * 10)
        np.random.seed(seed + fold * 10)

        model = MorphologyDNN(in_features=X_dev.shape[1], out_features=2)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
            batch_size=batch_size,
            shuffle=True,
        )

        best_val_loss = float("inf")
        best_weights = copy.deepcopy(model.state_dict())
        pat_counter = 0

        for epoch in range(1, max_epochs + 1):
            model.train()
            for bx, by in train_loader:
                optimizer.zero_grad()
                loss = criterion(model(bx), by)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                val_preds = model(torch.from_numpy(X_va))
                val_loss = criterion(val_preds, torch.from_numpy(y_va)).item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_weights = copy.deepcopy(model.state_dict())
                pat_counter = 0
            else:
                pat_counter += 1
                if pat_counter >= patience:
                    break

        model.load_state_dict(best_weights)
        model.eval()
        with torch.no_grad():
            fold_preds = model(torch.from_numpy(X_va)).numpy()
            oof_predictions[val_idx] = fold_preds

        fold_sbp_mae = np.mean(np.abs(fold_preds[:, 0] - y_va[:, 0]))
        fold_dbp_mae = np.mean(np.abs(fold_preds[:, 1] - y_va[:, 1]))
        print(f"Fold {fold:02d}/{n_splits:02d} | Val SBP MAE: {fold_sbp_mae:5.2f} mmHg | Val DBP MAE: {fold_dbp_mae:5.2f} mmHg")

    # Compute Out-Of-Fold Performance Metrics
    sbp_err = oof_predictions[:, 0] - y_dev[:, 0]
    dbp_err = oof_predictions[:, 1] - y_dev[:, 1]

    cv_sbp_mae = float(np.mean(np.abs(sbp_err)))
    cv_dbp_mae = float(np.mean(np.abs(dbp_err)))
    cv_sbp_rmse = float(np.sqrt(np.mean(sbp_err**2)))
    cv_dbp_rmse = float(np.sqrt(np.mean(dbp_err**2)))
    cv_sbp_me = float(np.mean(sbp_err))
    cv_dbp_me = float(np.mean(dbp_err))
    cv_sbp_sde = float(np.std(sbp_err))
    cv_dbp_sde = float(np.std(dbp_err))

    print("-" * 70)
    print(f"{'Target':<8} | {'CV MAE':<12} | {'CV RMSE':<12} | {'CV Bias (ME)':<14} | {'Paper Table 5 Benchmark'}")
    print("-" * 70)
    print(f"{'SBP':<8} | {cv_sbp_mae:6.2f} mmHg    | {cv_sbp_rmse:6.2f} mmHg    | {cv_sbp_me:+6.2f} mmHg     | ~{PAPER_STAGE1_SBP_MAE:.2f} mmHg (MAE)")
    print(f"{'DBP':<8} | {cv_dbp_mae:6.2f} mmHg    | {cv_dbp_rmse:6.2f} mmHg    | {cv_dbp_me:+6.2f} mmHg     | ~{PAPER_STAGE1_DBP_MAE:.2f} mmHg (MAE)")
    print("=" * 70 + "\n")

    return {
        "cv_sbp_mae": cv_sbp_mae,
        "cv_dbp_mae": cv_dbp_mae,
        "cv_sbp_rmse": cv_sbp_rmse,
        "cv_dbp_rmse": cv_dbp_rmse,
        "cv_sbp_me": cv_sbp_me,
        "cv_dbp_me": cv_dbp_me,
        "cv_sbp_sde": cv_sbp_sde,
        "cv_dbp_sde": cv_dbp_sde,
    }


def train_stage1(
    train_csv: str = "data/processed/uci_train_200_dev.csv",
    test_csv: str = "data/processed/uci_test_25_unseen.csv",
    save_path: str = "models/stage1_morphology_dnn_21feat.pth",
    lr: float = 0.005,
    weight_decay: float = 1e-4,
    batch_size: int = 16,
    max_epochs: int = 300,
    patience: int = 35,
    seed: int = 3,
    cv_folds: int = 0,
) -> Tuple[nn.Module, Dict]:
    """
    Train and evaluate MorphologyDNN for Stage 1 using the full 21 mPTP features.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    cv_results = {}
    if cv_folds > 1:
        cv_results = cross_validate_stage1(
            train_csv=train_csv,
            n_splits=cv_folds,
            lr=lr,
            weight_decay=weight_decay,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            seed=seed,
        )

    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = prepare_datasets(
        train_csv=train_csv, test_csv=test_csv, val_ratio=0.15, random_state=seed
    )

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_features = len(MORPHOLOGY_FEATURES)
    # Instantiate Stage 1 MorphologyDNN with 21 inputs
    model = MorphologyDNN(in_features=num_features, out_features=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    logger.info("MorphologyDNN Architecture (Input: %d features):\n%s", num_features, model)
    logger.info(
        "Hyperparameters: Adam(lr=%.4f, weight_decay=%.1e), batch_size=%d, patience=%d, seed=%d",
        lr,
        weight_decay,
        batch_size,
        patience,
        seed,
    )

    # Training Loop with Early Stopping
    best_val_loss = float("inf")
    best_model_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0
    best_epoch = 0

    print("\n" + "=" * 70)
    print(f"Training Stage 1 Morphology Deep Neural Network ({num_features} Features)")
    print("=" * 70)

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(batch_x)

        train_loss /= len(train_dataset)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item() * len(batch_x)

        val_loss /= len(val_dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 10 == 0 or epoch == 1 or patience_counter == 0:
            print(
                f"Epoch [{epoch:03d}/{max_epochs}] | "
                f"Train MSE: {train_loss:8.3f} | "
                f"Val MSE: {val_loss:8.3f} | "
                f"Best Val MSE: {best_val_loss:8.3f} (Epoch {best_epoch:03d})"
            )

        if patience_counter >= patience:
            logger.info("Early stopping triggered at epoch %d (patience=%d).", epoch, patience)
            break

    # Evaluation on 25 Unseen Test Subjects
    model.load_state_dict(best_model_weights)
    model.eval()

    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            preds = model(batch_x)
            all_preds.append(preds.numpy())
            all_targets.append(batch_y.numpy())

    y_pred = np.vstack(all_preds)
    y_true = np.vstack(all_targets)

    # Compute SBP and DBP performance metrics
    sbp_err = y_pred[:, 0] - y_true[:, 0]
    dbp_err = y_pred[:, 1] - y_true[:, 1]

    sbp_mae = float(np.mean(np.abs(sbp_err)))
    dbp_mae = float(np.mean(np.abs(dbp_err)))
    sbp_rmse = float(np.sqrt(np.mean(sbp_err**2)))
    dbp_rmse = float(np.sqrt(np.mean(dbp_err**2)))
    sbp_me = float(np.mean(sbp_err))
    dbp_me = float(np.mean(dbp_err))
    sbp_sde = float(np.std(sbp_err))
    dbp_sde = float(np.std(dbp_err))

    print("\n" + "=" * 70)
    print("STAGE 1 MORPHOLOGY DNN - 25 UNSEEN TEST SUBJECTS EVALUATION")
    print("=" * 70)
    print(f"Number of unseen test subjects: {len(y_true)}")
    print(f"Input features count:           {num_features} (Full mPTP set)")
    print("-" * 70)
    print(f"{'Target':<8} | {'Test MAE':<12} | {'Test RMSE':<12} | {'Bias (ME)':<12} | {'Paper Stage 1 Benchmark'}")
    print("-" * 70)
    print(f"{'SBP':<8} | {sbp_mae:6.2f} mmHg    | {sbp_rmse:6.2f} mmHg    | {sbp_me:+6.2f} mmHg  | ~{PAPER_STAGE1_SBP_MAE:.2f} mmHg (MAE)")
    print(f"{'DBP':<8} | {dbp_mae:6.2f} mmHg    | {dbp_rmse:6.2f} mmHg    | {dbp_me:+6.2f} mmHg  | ~{PAPER_STAGE1_DBP_MAE:.2f} mmHg (MAE)")
    print("=" * 70)

    # Save Checkpoint & Scaler
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    checkpoint = {
        "model_state_dict": best_model_weights,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "morphology_features": MORPHOLOGY_FEATURES,
        "target_columns": TARGET_COLUMNS,
        "best_epoch": best_epoch,
        "best_val_mse": best_val_loss,
        "test_sbp_mae": sbp_mae,
        "test_dbp_mae": dbp_mae,
        "test_sbp_rmse": sbp_rmse,
        "test_dbp_rmse": dbp_rmse,
        "test_sbp_me": sbp_me,
        "test_dbp_me": dbp_me,
        "test_sbp_sde": sbp_sde,
        "test_dbp_sde": dbp_sde,
        "cv_results": cv_results,
    }
    torch.save(checkpoint, save_path)
    logger.info("Saved best model checkpoint and scaler to: %s", save_path)

    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train Stage 1 Morphology DNN on Frozen Cohort.")
    parser.add_argument(
        "--train-csv",
        type=str,
        default="data/processed/uci_train_200_dev.csv",
        help="Path to development/training CSV.",
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        default="data/processed/uci_test_25_unseen.csv",
        help="Path to unseen test CSV.",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/stage1_morphology_dnn_21feat.pth",
        help="Path to save trained PyTorch model checkpoint.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of K-Fold CV folds across dev set (default: 5; 0 to skip).",
    )
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate (default: 0.005).")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay L2 penalty (default: 1e-4).",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16).")
    parser.add_argument("--epochs", type=int, default=300, help="Max epochs (default: 300).")
    parser.add_argument(
        "--patience",
        type=int,
        default=35,
        help="Early stopping patience (default: 35).",
    )
    parser.add_argument("--seed", type=int, default=3, help="Random seed (default: 3).")

    args = parser.parse_args()
    train_stage1(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        save_path=args.save_path,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
        cv_folds=args.cv_folds,
    )


if __name__ == "__main__":
    main()
