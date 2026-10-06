"""
Training and Evaluation Pipeline for Stage 1 Morphology Deep Neural Network (DNN).
Replicating Stage 1 from:
"Cuff-Less Blood Pressure Estimation from Photoplethysmogram Signals Using Deep Learning
and Cardiovascular Dynamics" (Sensors 2023, 23, 4145).

Stage 1 estimates preliminary SBP and DBP directly from 7 mPTP morphology features.
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
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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


def load_and_preprocess_data(
    csv_path: str = "data/processed/feature_matrix.csv",
    test_size: float = 0.10,
    val_size: float = 0.10,
    random_state: int = 42,
):
    """
    Load feature matrix, isolate morphology features and BP targets, clean NaNs,
    and split into train (80%), validation (10%), and test (10%) splits.
    Scales features with StandardScaler fitted exclusively on training set.
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Feature matrix file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info("Loaded feature matrix with %d rows and %d columns.", len(df), len(df.columns))

    required_cols = MORPHOLOGY_FEATURES + TARGET_COLUMNS
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in dataset: {missing_cols}")

    # Clean out any rows containing NaNs in the 9 columns
    clean_df = df[required_cols].dropna().copy()
    logger.info("Valid complete samples after NaN filtering: %d", len(clean_df))

    if len(clean_df) < 10:
        raise ValueError(
            f"Insufficient clean samples ({len(clean_df)}). Please run build_dataset to generate more records."
        )

    X = clean_df[MORPHOLOGY_FEATURES].values.astype(np.float32)
    y = clean_df[TARGET_COLUMNS].values.astype(np.float32)

    # 80% Train, 10% Val, 10% Test
    temp_size = val_size + test_size  # e.g. 0.20
    test_prop = test_size / temp_size  # e.g. 0.5 of temp => 0.10 of total

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_size, random_state=random_state, shuffle=True
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=test_prop, random_state=random_state, shuffle=True
    )

    logger.info(
        "Partition split counts -> Train: %d (%.1f%%) | Val: %d (%.1f%%) | Test: %d (%.1f%%)",
        len(X_train),
        100.0 * len(X_train) / len(X),
        len(X_val),
        100.0 * len(X_val) / len(X),
        len(X_test),
        100.0 * len(X_test) / len(X),
    )

    # Standardize features using training statistics only
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


def train_stage1(
    csv_path: str = "data/processed/feature_matrix.csv",
    save_path: str = "models/stage1_morphology_dnn.pth",
    lr: float = 0.005,
    weight_decay: float = 1e-4,
    batch_size: int = 16,
    max_epochs: int = 250,
    patience: int = 20,
    seed: int = 42,
):
    """
    Train and evaluate MorphologyDNN for Stage 1.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Load, filter, split, scale data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler = load_and_preprocess_data(
        csv_path=csv_path, random_state=seed
    )

    train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_dataset = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    test_dataset = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 2. Instantiate Stage 1 MorphologyDNN
    model = MorphologyDNN(in_features=7, out_features=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    logger.info("MorphologyDNN Architecture:\n%s", model)
    logger.info(
        "Hyperparameters: Optimizer=Adam, lr=%.4f, weight_decay=%.1e, batch_size=%d, patience=%d",
        lr,
        weight_decay,
        batch_size,
        patience,
    )

    # 3. Training Loop with Early Stopping
    best_val_loss = float("inf")
    best_model_weights = copy.deepcopy(model.state_dict())
    patience_counter = 0
    best_epoch = 0

    print("\n" + "=" * 65)
    print("Beginning Training: Stage 1 Morphology Deep Neural Network (DNN)")
    print("=" * 65)

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

    # 4. Evaluation on Held-Out Test Set
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

    # Calculate metrics
    sbp_err = np.abs(y_pred[:, 0] - y_true[:, 0])
    dbp_err = np.abs(y_pred[:, 1] - y_true[:, 1])

    sbp_mae = float(np.mean(sbp_err))
    dbp_mae = float(np.mean(dbp_err))
    sbp_rmse = float(np.sqrt(np.mean((y_pred[:, 0] - y_true[:, 0]) ** 2)))
    dbp_rmse = float(np.sqrt(np.mean((y_pred[:, 1] - y_true[:, 1]) ** 2)))
    sbp_std = float(np.std(sbp_err))
    dbp_std = float(np.std(dbp_err))

    # Paper Benchmarks for Stage 1 (from Sensors 2023, 23, 4145)
    PAPER_STAGE1_SBP_MAE = 11.24
    PAPER_STAGE1_DBP_MAE = 4.75

    print("\n" + "=" * 65)
    print("STAGE 1 MORPHOLOGY DNN - HELD-OUT TEST EVALUATION")
    print("=" * 65)
    print(f"Number of test samples: {len(y_true)}")
    print("-" * 65)
    print(f"{'Target':<10} | {'Test MAE':<12} | {'Test RMSE':<12} | {'Paper Stage 1 Benchmark'}")
    print("-" * 65)
    print(f"{'SBP':<10} | {sbp_mae:6.2f} mmHg    | {sbp_rmse:6.2f} mmHg    | ~{PAPER_STAGE1_SBP_MAE:.2f} mmHg (MAE)")
    print(f"{'DBP':<10} | {dbp_mae:6.2f} mmHg    | {dbp_rmse:6.2f} mmHg    | ~{PAPER_STAGE1_DBP_MAE:.2f} mmHg (MAE)")
    print("=" * 65)

    # 5. Save Checkpoint & Scaler
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
    }
    torch.save(checkpoint, save_path)
    logger.info("Saved best model checkpoint and scaler to: %s", save_path)

    return model, checkpoint


def main():
    parser = argparse.ArgumentParser(description="Train Stage 1 Morphology DNN for BP Estimation.")
    parser.add_argument(
        "--csv-path",
        type=str,
        default="data/processed/feature_matrix.csv",
        help="Path to feature_matrix.csv",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="models/stage1_morphology_dnn.pth",
        help="Path to save trained PyTorch model checkpoint.",
    )
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate (default: 0.005).")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay L2 penalty (default: 1e-4).",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size (default: 16).")
    parser.add_argument("--epochs", type=int, default=250, help="Max epochs (default: 250).")
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Early stopping patience (default: 20).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    args = parser.parse_args()
    train_stage1(
        csv_path=args.csv_path,
        save_path=args.save_path,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.epochs,
        patience=args.patience,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
