#!/usr/bin/env python3
"""Train realtime GOI+HOI frame-window MLPs for two-stage step prediction."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


NON_FEATURE_COLUMNS = {
    "participant_id",
    "task_name",
    "full_id",
    "video_frame_idx",
    "step_label",
    "step_start_frame",
    "step_end_frame",
    "step_start_s",
    "step_end_s",
    "hand",
    "object",
    "target",
    "confidence",
    "notes",
    "task_step",
    "label_start_s",
    "label_end_s",
    "label_start_frame",
    "label_end_frame",
    "label_task",
    "label_step",
    "label_hand",
    "label_object",
    "label_target",
    "label_confidence",
    "label_notes",
    "task_label_from_annotation",
}

TASK_NAME_NORMALIZATION = {
    "Tee_kochen1": "Teekochen1",
    "Tee_kochen2": "Teekochen2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train causal flattened GOI+HOI window MLPs for realtime step prediction."
    )
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--target-task-col", default="task_name")
    parser.add_argument("--target-step-col", default="step_label")
    parser.add_argument("--participant-col", default="participant_id")
    parser.add_argument("--frame-col", default="video_frame_idx")
    parser.add_argument("--full-id-col", default="full_id")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--window-sec", type=float, default=0.5)
    parser.add_argument(
        "--split-mode",
        default="participant",
        choices=["participant", "row"],
        help=(
            "participant: hold out whole participants; row: random split over "
            "supervised frame rows."
        ),
    )
    parser.add_argument(
        "--model-mode",
        default="two_stage",
        choices=["two_stage", "direct_step"],
        help=(
            "two_stage: train task classifier plus task-specific step classifiers; "
            "direct_step: train one global step classifier only."
        ),
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable CUDA AMP mixed precision training.",
    )
    parser.add_argument(
        "--log-every-batches",
        type=int,
        default=50,
        help="Print training/scaler progress every N batches. Use 0 to disable batch logs.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_arg)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def normalize_task_names(df: pd.DataFrame, task_col: str) -> pd.DataFrame:
    result = df.copy()
    task_values = result[task_col].astype("string")
    before_counts = task_values.value_counts(dropna=True)
    result[task_col] = task_values.replace(TASK_NAME_NORMALIZATION)
    after_counts = result[task_col].astype("string").value_counts(dropna=True)

    changed = {
        source: {
            "target": target,
            "rows": int(before_counts.get(source, 0)),
        }
        for source, target in TASK_NAME_NORMALIZATION.items()
        if int(before_counts.get(source, 0)) > 0
    }
    if changed:
        print("Normalized task names:")
        for source, info in changed.items():
            print(f"  {source} -> {info['target']}: {info['rows']} rows")
        affected_targets = sorted({info["target"] for info in changed.values()})
        print("Task counts after normalization:")
        print(after_counts.loc[after_counts.index.isin(affected_targets)])
    return result


def load_and_prepare_data(
    input_csv: str,
    participant_col: str,
    task_col: str,
    step_col: str,
    full_id_col: str,
    frame_col: str,
) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(input_csv, low_memory=False)
    required = [participant_col, task_col, full_id_col, frame_col, step_col]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    frame_values = pd.to_numeric(df[frame_col], errors="coerce")
    invalid_frames = int(frame_values.isna().sum())
    if invalid_frames:
        print(f"Warning: dropping {invalid_frames} rows with invalid {frame_col}.")
    df = df.loc[frame_values.notna()].copy()
    df[frame_col] = frame_values.loc[frame_values.notna()].round().astype(int)
    df = normalize_task_names(df, task_col)

    df = df.sort_values([full_id_col, frame_col]).reset_index(drop=True)
    non_features = set(NON_FEATURE_COLUMNS) | {
        participant_col,
        task_col,
        step_col,
        full_id_col,
        frame_col,
    }
    candidate_cols = [column for column in df.columns if column not in non_features]

    numeric_features: list[str] = []
    for column in candidate_cols:
        numeric = pd.to_numeric(df[column], errors="coerce")
        if numeric.notna().any():
            df[column] = numeric.fillna(0.0).astype(np.float32)
            numeric_features.append(column)

    if not numeric_features:
        raise ValueError("No numeric frame-level feature columns were found.")

    print(f"Frame rows kept for causal windows: {len(df)}")
    print(f"Numeric frame feature columns: {len(numeric_features)}")
    return df, numeric_features


def build_flatten_lag_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    full_id_col: str,
    frame_col: str,
    participant_col: str,
    task_col: str,
    step_col: str,
    window_frames: int,
    output_dir: Path,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    if window_frames < 1:
        raise ValueError("window_frames must be >= 1")

    meta_cols = [participant_col, task_col, full_id_col, frame_col, step_col]
    meta_chunks: list[pd.DataFrame] = []
    output_shape = (len(df), window_frames * len(feature_cols))
    x_path = output_dir / "X_flat.npy"
    X_flat = np.lib.format.open_memmap(
        x_path,
        mode="w+",
        dtype=np.float32,
        shape=output_shape,
    )
    row_offset = 0

    for _, group in df.sort_values([full_id_col, frame_col]).groupby(full_id_col, sort=False):
        group = group.sort_values(frame_col)
        values = group[feature_cols].to_numpy(dtype=np.float32, copy=True)
        num_frames, num_features = values.shape
        windows = np.zeros((num_frames, window_frames, num_features), dtype=np.float32)

        for i in range(num_frames):
            start = max(0, i - window_frames + 1)
            valid = values[start : i + 1]
            windows[i, window_frames - len(valid) :, :] = valid

        X_flat[row_offset : row_offset + num_frames] = windows.reshape(
            num_frames,
            window_frames * num_features,
        )
        row_offset += num_frames
        meta_chunks.append(group[meta_cols].copy())

    X_flat.flush()
    meta_df = pd.concat(meta_chunks, ignore_index=True)
    flattened_feature_names = [
        f"{feature}_lag_{lag}"
        for lag in range(window_frames - 1, -1, -1)
        for feature in feature_cols
    ]

    print(f"Flattened X shape: {X_flat.shape}")
    print(f"Approx X memory: {X_flat.nbytes / 1024**3:.3f} GB")
    print(f"Flattened X memmap: {x_path}")
    return X_flat, meta_df, flattened_feature_names


def split_by_participant(
    meta_df: pd.DataFrame,
    candidate_idx: np.ndarray,
    participant_col: str,
    task_col: str,
    step_col: str,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    candidate_meta = meta_df.iloc[candidate_idx]
    participants = np.array(sorted(candidate_meta[participant_col].astype(str).unique()))
    if len(participants) < 2:
        raise ValueError(
            "Need at least two participants with non-empty task/step labels and "
            "non-negative frame indices for participant-level split."
        )
    rng = np.random.default_rng(random_state)
    rng.shuffle(participants)
    test_count = max(1, int(round(len(participants) * test_size)))
    if len(participants) > 1:
        test_count = min(test_count, len(participants) - 1)
    test_participants = sorted(participants[:test_count].tolist())
    train_participants = sorted(participants[test_count:].tolist())

    participant_values = meta_df[participant_col].astype(str)
    candidate_participants = participant_values.iloc[candidate_idx]
    train_mask = candidate_participants.isin(train_participants).to_numpy()
    test_mask = candidate_participants.isin(test_participants).to_numpy()
    train_idx = candidate_idx[train_mask]
    test_idx = candidate_idx[test_mask]

    print(f"Train participants ({len(train_participants)}): {train_participants}")
    print(f"Test participants ({len(test_participants)}): {test_participants}")
    print(f"Train rows: {len(train_idx)}")
    print(f"Test rows: {len(test_idx)}")
    print("Train task distribution:")
    print(meta_df.iloc[train_idx][task_col].value_counts())
    print("Test task distribution:")
    print(meta_df.iloc[test_idx][task_col].value_counts())
    print("Train step distribution:")
    print(meta_df.iloc[train_idx][step_col].value_counts())
    print("Test step distribution:")
    print(meta_df.iloc[test_idx][step_col].value_counts())

    return train_idx, test_idx, train_participants, test_participants


def split_by_row(
    meta_df: pd.DataFrame,
    candidate_idx: np.ndarray,
    task_col: str,
    step_col: str,
    test_size: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    candidate_meta = meta_df.iloc[candidate_idx]
    stratify_labels = (
        candidate_meta[task_col].astype(str) + "||" + candidate_meta[step_col].astype(str)
    ).to_numpy()
    _, counts = np.unique(stratify_labels, return_counts=True)
    stratify = stratify_labels if counts.min() >= 2 else None
    if stratify is None:
        print("Warning: cannot stratify row split because some task/step classes have <2 rows.")

    train_idx, test_idx = train_test_split(
        candidate_idx,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
        stratify=stratify,
    )
    train_idx = np.asarray(train_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)

    print("Split mode: row")
    print(f"Train rows: {len(train_idx)}")
    print(f"Test rows: {len(test_idx)}")
    print("Train task distribution:")
    print(meta_df.iloc[train_idx][task_col].value_counts())
    print("Test task distribution:")
    print(meta_df.iloc[test_idx][task_col].value_counts())
    print("Train step distribution:")
    print(meta_df.iloc[train_idx][step_col].value_counts())
    print("Test step distribution:")
    print(meta_df.iloc[test_idx][step_col].value_counts())
    return train_idx, test_idx


def select_supervised_sample_indices(
    meta_df: pd.DataFrame,
    task_col: str,
    step_col: str,
    frame_col: str,
) -> np.ndarray:
    task_valid = meta_df[task_col].notna() & (meta_df[task_col].astype(str).str.strip() != "")
    step_valid = meta_df[step_col].notna() & (meta_df[step_col].astype(str).str.strip() != "")
    frame_valid = pd.to_numeric(meta_df[frame_col], errors="coerce") >= 0
    valid_mask = (task_valid & step_valid & frame_valid).to_numpy()
    supervised_idx = np.flatnonzero(valid_mask)
    dropped = len(meta_df) - len(supervised_idx)
    print(f"Supervised rows selected after lag construction: {len(supervised_idx)}")
    print(f"Rows kept only as unlabeled/invalid-frame context: {dropped}")
    if len(supervised_idx) == 0:
        raise ValueError("No supervised rows remain after task/step/frame filtering.")
    return supervised_idx


class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


class IndexedScaledDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        row_indices: np.ndarray,
        y: np.ndarray,
        scaler: StandardScaler,
    ):
        self.X = X
        self.row_indices = np.asarray(row_indices, dtype=np.int64)
        self.y = np.asarray(y, dtype=np.int64)
        self.mean = scaler.mean_.astype(np.float32)
        self.scale = scaler.scale_.astype(np.float32)
        self.scale[self.scale == 0] = 1.0

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = np.asarray(self.X[self.row_indices[idx]], dtype=np.float32)
        row = (row - self.mean) / self.scale
        return torch.from_numpy(row), torch.tensor(self.y[idx], dtype=torch.long)


class MLPClassifierTorch(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list[int],
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        current_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def can_stratify(y: np.ndarray, val_size: float) -> bool:
    classes, counts = np.unique(y, return_counts=True)
    val_count = int(np.ceil(len(y) * val_size))
    train_count = len(y) - val_count
    return bool(
        len(classes) > 1
        and counts.min() >= 2
        and val_count >= len(classes)
        and train_count >= len(classes)
    )


def compute_balanced_class_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    total = float(counts.sum())
    weights = np.zeros(num_classes, dtype=np.float32)
    nonzero = counts > 0
    if total > 0 and num_classes > 0:
        weights[nonzero] = total / (num_classes * counts[nonzero])
    return weights


def print_step_distribution_and_weights(
    model_name: str,
    label_encoder: LabelEncoder,
    y: np.ndarray,
    class_weights: np.ndarray,
) -> None:
    counts = np.bincount(y, minlength=len(label_encoder.classes_))
    print(f"{model_name}: step label distribution and class weights:")
    for idx, label in enumerate(label_encoder.classes_):
        print(
            f"  {label}: count={int(counts[idx])}, "
            f"weight={float(class_weights[idx]):.6f}"
        )


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    num_workers: int,
) -> DataLoader:
    effective_batch_size = min(batch_size, max(1, len(y)))
    remainder = len(y) % effective_batch_size
    if shuffle and remainder == 1 and effective_batch_size > 2:
        effective_batch_size -= 1
    drop_last = shuffle and len(y) > effective_batch_size and len(y) % effective_batch_size == 1
    return DataLoader(
        TabularDataset(X, y),
        batch_size=effective_batch_size,
        shuffle=shuffle,
        pin_memory=device.type == "cuda",
        drop_last=drop_last,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )


def make_indexed_scaled_loader(
    X: np.ndarray,
    row_indices: np.ndarray,
    y: np.ndarray,
    scaler: StandardScaler,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
    num_workers: int,
) -> DataLoader:
    effective_batch_size = min(batch_size, max(1, len(y)))
    remainder = len(y) % effective_batch_size
    if shuffle and remainder == 1 and effective_batch_size > 2:
        effective_batch_size -= 1
    drop_last = shuffle and len(y) > effective_batch_size and len(y) % effective_batch_size == 1
    return DataLoader(
        IndexedScaledDataset(X, row_indices, y, scaler),
        batch_size=effective_batch_size,
        shuffle=shuffle,
        pin_memory=device.type == "cuda",
        drop_last=drop_last,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
    )


def fit_standard_scaler_batches(
    X: np.ndarray,
    row_indices: np.ndarray,
    batch_size: int,
    model_name: str,
    log_every_batches: int,
) -> StandardScaler:
    scaler = StandardScaler()
    row_indices = np.asarray(row_indices, dtype=np.int64)
    scaler_batch_size = max(batch_size, 2048)
    total_batches = int(np.ceil(len(row_indices) / scaler_batch_size))
    print(
        f"{model_name}: fitting StandardScaler on {len(row_indices)} rows "
        f"in {total_batches} batches."
    )
    for batch_no, start in enumerate(range(0, len(row_indices), scaler_batch_size), start=1):
        batch_idx = row_indices[start : start + scaler_batch_size]
        scaler.partial_fit(np.asarray(X[batch_idx], dtype=np.float32))
        if log_every_batches > 0 and (
            batch_no == 1 or batch_no == total_batches or batch_no % log_every_batches == 0
        ):
            print(f"{model_name}: scaler batch {batch_no}/{total_batches}")
    return scaler


def train_torch_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    num_classes: int,
    hidden_dims: list[int],
    output_dir: Path,
    model_name: str,
    device: torch.device,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    random_state: int,
    num_workers: int,
    amp: bool,
    log_every_batches: int = 0,
) -> MLPClassifierTorch | None:
    if num_classes == 1:
        print(f"{model_name}: one class only; skipping MLP training.")
        return None
    if len(y_train) < 4:
        raise ValueError(f"{model_name}: not enough samples to train/validate.")

    indices = np.arange(len(y_train))
    stratify = y_train if can_stratify(y_train, 0.15) else None
    train_idx, val_idx = train_test_split(
        indices,
        test_size=0.15,
        random_state=random_state,
        shuffle=True,
        stratify=stratify,
    )

    model = MLPClassifierTorch(X_train.shape[1], num_classes, hidden_dims).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = make_loader(
        X_train[train_idx],
        y_train[train_idx],
        batch_size,
        True,
        device,
        num_workers,
    )
    val_loader = make_loader(
        X_train[val_idx],
        y_train[val_idx],
        batch_size,
        False,
        device,
        num_workers,
    )
    use_amp = amp and device.type == "cuda"
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_state: dict[str, torch.Tensor] | None = None
    best_f1 = -1.0
    best_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_seen = 0
        total_train_batches = len(train_loader)
        for batch_no, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            train_loss_sum += loss.item() * len(yb)
            train_seen += len(yb)
            if log_every_batches > 0 and (
                batch_no == 1
                or batch_no == total_train_batches
                or batch_no % log_every_batches == 0
            ):
                running_loss = train_loss_sum / max(train_seen, 1)
                print(
                    f"{model_name} epoch {epoch:03d} "
                    f"batch {batch_no}/{total_train_batches}: "
                    f"train_loss={running_loss:.6f}"
                )

        model.eval()
        val_loss_sum = 0.0
        val_seen = 0
        all_true: list[np.ndarray] = []
        all_pred: list[np.ndarray] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                pred = logits.argmax(dim=1)
                val_loss_sum += loss.item() * len(yb)
                val_seen += len(yb)
                all_true.append(yb.cpu().numpy())
                all_pred.append(pred.cpu().numpy())

        y_val_true = np.concatenate(all_true)
        y_val_pred = np.concatenate(all_pred)
        train_loss = train_loss_sum / max(train_seen, 1)
        val_loss = val_loss_sum / max(val_seen, 1)
        val_acc = accuracy_score(y_val_true, y_val_pred)
        val_f1 = f1_score(y_val_true, y_val_pred, average="macro", zero_division=0)
        print(
            f"{model_name} epoch {epoch:03d}: train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_acc={val_acc:.6f} val_macro_f1={val_f1:.6f}"
        )

        improved = val_f1 > best_f1 or (np.isclose(val_f1, best_f1) and val_loss < best_loss)
        if improved:
            best_f1 = val_f1
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(best_state, output_dir / f"{model_name}.pt")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"{model_name}: early stopping at epoch {epoch}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_torch_classifier_indexed(
    X: np.ndarray,
    row_indices: np.ndarray,
    y_train: np.ndarray,
    scaler: StandardScaler,
    num_classes: int,
    hidden_dims: list[int],
    dropout: float,
    output_dir: Path,
    model_name: str,
    device: torch.device,
    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
    random_state: int,
    num_workers: int,
    amp: bool,
    log_every_batches: int = 0,
    training_history: list[dict[str, Any]] | None = None,
    training_summary: dict[str, Any] | None = None,
    class_weights: np.ndarray | None = None,
) -> MLPClassifierTorch | None:
    if num_classes == 1:
        print(f"{model_name}: one class only; skipping MLP training.")
        if training_summary is not None:
            training_summary[model_name] = {
                "type": "constant",
                "num_classes": int(num_classes),
                "train_rows": int(len(y_train)),
            }
        return None
    if len(y_train) < 4:
        raise ValueError(f"{model_name}: not enough samples to train/validate.")

    positions = np.arange(len(y_train))
    stratify = y_train if can_stratify(y_train, 0.15) else None
    train_pos, val_pos = train_test_split(
        positions,
        test_size=0.15,
        random_state=random_state,
        shuffle=True,
        stratify=stratify,
    )

    print(
        f"{model_name}: input_dim={X.shape[1]} hidden_dims={hidden_dims} "
        f"dropout={dropout} lr={lr} weight_decay={weight_decay}"
    )
    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.as_tensor(class_weights, dtype=torch.float32, device=device)
        print(f"{model_name}: using class-weighted CrossEntropyLoss")

    model = MLPClassifierTorch(X.shape[1], num_classes, hidden_dims, dropout=dropout).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader = make_indexed_scaled_loader(
        X,
        row_indices[train_pos],
        y_train[train_pos],
        scaler,
        batch_size,
        True,
        device,
        num_workers,
    )
    val_loader = make_indexed_scaled_loader(
        X,
        row_indices[val_pos],
        y_train[val_pos],
        scaler,
        batch_size,
        False,
        device,
        num_workers,
    )
    use_amp = amp and device.type == "cuda"
    grad_scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_state: dict[str, torch.Tensor] | None = None
    best_f1 = -1.0
    best_acc = -1.0
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    stopped_epoch = epochs

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_seen = 0
        total_train_batches = len(train_loader)
        for batch_no, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = criterion(model(xb), yb)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            train_loss_sum += loss.item() * len(yb)
            train_seen += len(yb)
            if log_every_batches > 0 and (
                batch_no == 1
                or batch_no == total_train_batches
                or batch_no % log_every_batches == 0
            ):
                running_loss = train_loss_sum / max(train_seen, 1)
                print(
                    f"{model_name} epoch {epoch:03d} "
                    f"batch {batch_no}/{total_train_batches}: "
                    f"train_loss={running_loss:.6f}"
                )

        model.eval()
        val_loss_sum = 0.0
        val_seen = 0
        all_true: list[np.ndarray] = []
        all_pred: list[np.ndarray] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                pred = logits.argmax(dim=1)
                val_loss_sum += loss.item() * len(yb)
                val_seen += len(yb)
                all_true.append(yb.cpu().numpy())
                all_pred.append(pred.cpu().numpy())

        y_val_true = np.concatenate(all_true)
        y_val_pred = np.concatenate(all_pred)
        train_loss = train_loss_sum / max(train_seen, 1)
        val_loss = val_loss_sum / max(val_seen, 1)
        val_acc = accuracy_score(y_val_true, y_val_pred)
        val_f1 = f1_score(y_val_true, y_val_pred, average="macro", zero_division=0)
        print(
            f"{model_name} epoch {epoch:03d}: train_loss={train_loss:.6f} "
            f"val_loss={val_loss:.6f} val_acc={val_acc:.6f} val_macro_f1={val_f1:.6f}"
        )
        if training_history is not None:
            training_history.append(
                {
                    "model_name": model_name,
                    "epoch": int(epoch),
                    "train_loss": float(train_loss),
                    "val_loss": float(val_loss),
                    "val_accuracy": float(val_acc),
                    "val_macro_f1": float(val_f1),
                    "train_rows": int(len(train_pos)),
                    "val_rows": int(len(val_pos)),
                    "num_classes": int(num_classes),
                    "hidden_dims": list(hidden_dims),
                    "dropout": float(dropout),
                    "lr": float(lr),
                    "weight_decay": float(weight_decay),
                    "class_weighted_loss": class_weights is not None,
                    "is_best": False,
                }
            )

        improved = val_f1 > best_f1 or (np.isclose(val_f1, best_f1) and val_loss < best_loss)
        if improved:
            best_f1 = val_f1
            best_acc = val_acc
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            torch.save(best_state, output_dir / f"{model_name}.pt")
            if training_history is not None:
                training_history[-1]["is_best"] = True
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"{model_name}: early stopping at epoch {epoch}.")
                stopped_epoch = epoch
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if training_summary is not None:
        training_summary[model_name] = {
            "type": "torch",
            "num_classes": int(num_classes),
            "train_rows": int(len(train_pos)),
            "val_rows": int(len(val_pos)),
            "input_dim": int(X.shape[1]),
            "hidden_dims": list(hidden_dims),
            "dropout": float(dropout),
            "lr": float(lr),
            "weight_decay": float(weight_decay),
            "class_weighted_loss": class_weights is not None,
            "class_weights": class_weights.astype(float).tolist()
            if class_weights is not None
            else None,
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_loss),
            "best_val_accuracy": float(best_acc),
            "best_val_macro_f1": float(best_f1),
            "stopped_epoch": int(stopped_epoch),
        }
    return model


def predict_torch_classifier(
    model: MLPClassifierTorch,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 4096,
    num_workers: int = 0,
) -> np.ndarray:
    dummy_y = np.zeros(len(X), dtype=np.int64)
    loader = make_loader(X, dummy_y, batch_size, False, device, num_workers)
    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            preds.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def predict_torch_classifier_indexed(
    model: MLPClassifierTorch,
    X: np.ndarray,
    row_indices: np.ndarray,
    scaler: StandardScaler,
    device: torch.device,
    batch_size: int = 4096,
    num_workers: int = 0,
) -> np.ndarray:
    dummy_y = np.zeros(len(row_indices), dtype=np.int64)
    loader = make_indexed_scaled_loader(
        X,
        row_indices,
        dummy_y,
        scaler,
        batch_size,
        False,
        device,
        num_workers,
    )
    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device, non_blocking=True)
            preds.append(model(xb).argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def train_task_classifier(
    X: np.ndarray,
    train_idx: np.ndarray,
    y_task_train: np.ndarray,
    output_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
    training_history: list[dict[str, Any]],
    training_summary: dict[str, Any],
) -> dict[str, Any]:
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_task_train)
    scaler = fit_standard_scaler_batches(
        X,
        train_idx,
        args.batch_size,
        "task_model",
        args.log_every_batches,
    )
    joblib.dump(scaler, output_dir / "task_scaler.joblib")
    joblib.dump(label_encoder, output_dir / "task_label_encoder.joblib")

    model = train_torch_classifier_indexed(
        X,
        train_idx,
        y,
        scaler,
        len(label_encoder.classes_),
        [256, 128],
        0.3,
        output_dir,
        "task_model",
        device,
        args.batch_size,
        args.epochs,
        args.lr,
        args.weight_decay,
        args.patience,
        args.random_state,
        args.num_workers,
        args.amp,
        args.log_every_batches,
        training_history,
        training_summary,
    )
    return {
        "type": "constant" if model is None else "torch",
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "constant_task": label_encoder.classes_[0] if model is None else None,
    }


def train_step_classifiers(
    X: np.ndarray,
    train_idx: np.ndarray,
    meta_train: pd.DataFrame,
    task_col: str,
    step_col: str,
    output_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
    training_history: list[dict[str, Any]],
    training_summary: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    step_models: dict[str, dict[str, Any]] = {}
    for task in sorted(meta_train[task_col].astype(str).unique()):
        task_mask = meta_train[task_col].astype(str).to_numpy() == task
        task_row_indices = train_idx[task_mask]
        y_step = meta_train.loc[task_mask, step_col].astype(str).to_numpy()
        step_counts = pd.Series(y_step).value_counts()
        print(f"Training step classifier for task={task!r}: rows={len(y_step)}, steps={len(step_counts)}")
        print(f"step_model_{safe_name(task)}: raw step label distribution:")
        print(step_counts)

        if len(step_counts) == 1:
            constant_step = str(step_counts.index[0])
            print(
                f"step_model_{safe_name(task)}: constant classifier; "
                "class weights are not used."
            )
            training_summary[f"step_model_{safe_name(task)}"] = {
                "type": "constant",
                "task": task,
                "constant_step": constant_step,
                "train_rows": int(len(y_step)),
                "num_classes": 1,
            }
            step_models[task] = {
                "type": "constant",
                "constant_step": constant_step,
                "class_names": [constant_step],
            }
            continue

        if len(y_step) < 20:
            constant_step = str(step_counts.index[0])
            print(
                f"Warning: task={task!r} has fewer than 20 samples; "
                f"using constant step={constant_step!r}."
            )
            print(
                f"step_model_{safe_name(task)}: small-sample constant fallback; "
                "class weights are not used."
            )
            training_summary[f"step_model_{safe_name(task)}"] = {
                "type": "constant_small_sample",
                "task": task,
                "constant_step": constant_step,
                "train_rows": int(len(y_step)),
                "num_classes": int(len(step_counts)),
            }
            step_models[task] = {
                "type": "constant",
                "constant_step": constant_step,
                "class_names": step_counts.index.astype(str).tolist(),
            }
            continue

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_step)
        model_name = f"step_model_{safe_name(task)}"
        class_weights = compute_balanced_class_weights(y, len(label_encoder.classes_))
        print_step_distribution_and_weights(
            model_name,
            label_encoder,
            y,
            class_weights,
        )
        scaler = fit_standard_scaler_batches(
            X,
            task_row_indices,
            args.batch_size,
            model_name,
            args.log_every_batches,
        )
        model = train_torch_classifier_indexed(
            X,
            task_row_indices,
            y,
            scaler,
            len(label_encoder.classes_),
            [128, 64],
            0.3,
            output_dir,
            model_name,
            device,
            args.batch_size,
            args.epochs,
            args.lr,
            args.weight_decay,
            args.patience,
            args.random_state,
            args.num_workers,
            args.amp,
            args.log_every_batches,
            training_history,
            training_summary,
            class_weights,
        )

        if model is None:
            step_models[task] = {
                "type": "constant",
                "constant_step": str(label_encoder.classes_[0]),
                "class_names": label_encoder.classes_.astype(str).tolist(),
            }
        else:
            step_models[task] = {
                "type": "torch",
                "model": model,
                "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "input_dim": X.shape[1],
                "hidden_dims": [128, 64],
                "dropout": 0.3,
                "class_weights": class_weights.astype(float).tolist(),
                "scaler": scaler,
                "label_encoder": label_encoder,
                "class_names": label_encoder.classes_.astype(str).tolist(),
            }

    saveable = {}
    for task, entry in step_models.items():
        save_entry = {key: value for key, value in entry.items() if key != "model"}
        saveable[task] = save_entry
    joblib.dump(saveable, output_dir / "step_models.joblib")
    return step_models


def train_direct_step_classifier(
    X: np.ndarray,
    train_idx: np.ndarray,
    meta_train: pd.DataFrame,
    step_col: str,
    output_dir: Path,
    device: torch.device,
    args: argparse.Namespace,
    training_history: list[dict[str, Any]],
    training_summary: dict[str, Any],
) -> dict[str, Any]:
    y_step = meta_train[step_col].astype(str).to_numpy()
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_step)
    class_weights = compute_balanced_class_weights(y, len(label_encoder.classes_))
    model_name = "direct_step_model"

    print("Training direct global step classifier.")
    print_step_distribution_and_weights(model_name, label_encoder, y, class_weights)
    scaler = fit_standard_scaler_batches(
        X,
        train_idx,
        args.batch_size,
        model_name,
        args.log_every_batches,
    )
    joblib.dump(scaler, output_dir / "direct_step_scaler.joblib")
    joblib.dump(label_encoder, output_dir / "direct_step_label_encoder.joblib")

    model = train_torch_classifier_indexed(
        X,
        train_idx,
        y,
        scaler,
        len(label_encoder.classes_),
        [128, 64],
        0.3,
        output_dir,
        model_name,
        device,
        args.batch_size,
        args.epochs,
        args.lr,
        args.weight_decay,
        args.patience,
        args.random_state,
        args.num_workers,
        args.amp,
        args.log_every_batches,
        training_history,
        training_summary,
        class_weights,
    )
    if model is None:
        return {
            "type": "constant",
            "constant_step": str(label_encoder.classes_[0]),
            "label_encoder": label_encoder,
            "scaler": scaler,
            "class_names": label_encoder.classes_.astype(str).tolist(),
        }

    joblib.dump(
        {
            "type": "torch",
            "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            "input_dim": X.shape[1],
            "hidden_dims": [128, 64],
            "dropout": 0.3,
            "class_weights": class_weights.astype(float).tolist(),
            "scaler": scaler,
            "label_encoder": label_encoder,
            "class_names": label_encoder.classes_.astype(str).tolist(),
        },
        output_dir / "direct_step_model.joblib",
    )
    return {
        "type": "torch",
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "class_names": label_encoder.classes_.astype(str).tolist(),
    }


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)[:120]


def predict_step_entry(
    entry: dict[str, Any],
    X: np.ndarray,
    row_index: int,
    device: torch.device,
) -> str:
    if entry["type"] == "constant":
        return str(entry["constant_step"])
    scaler: StandardScaler = entry["scaler"]
    label_encoder: LabelEncoder = entry["label_encoder"]
    model: MLPClassifierTorch = entry["model"]
    row = np.asarray(X[row_index], dtype=np.float32)
    mean = scaler.mean_.astype(np.float32)
    scale = scaler.scale_.astype(np.float32)
    scale[scale == 0] = 1.0
    row = (row - mean) / scale
    model.eval()
    with torch.no_grad():
        xb = torch.from_numpy(row).unsqueeze(0).to(device)
        pred_idx = int(model(xb).argmax(dim=1).cpu().numpy()[0])
    return str(label_encoder.inverse_transform([pred_idx])[0])


def predict_two_stage(
    X: np.ndarray,
    test_idx: np.ndarray,
    meta_test: pd.DataFrame,
    task_model_info: dict[str, Any],
    step_models: dict[str, dict[str, Any]],
    global_fallback_step: str,
    device: torch.device,
    participant_col: str,
    full_id_col: str,
    frame_col: str,
    task_col: str,
    step_col: str,
    num_workers: int,
) -> pd.DataFrame:
    if task_model_info["type"] == "constant":
        pred_task = np.full(len(test_idx), task_model_info["constant_task"], dtype=object)
    else:
        pred_task_idx = predict_torch_classifier_indexed(
            task_model_info["model"],
            X,
            test_idx,
            task_model_info["scaler"],
            device,
            num_workers=num_workers,
        )
        pred_task = task_model_info["label_encoder"].inverse_transform(pred_task_idx)

    pred_step_two_stage: list[str] = []
    pred_step_oracle: list[str] = []
    true_tasks = meta_test[task_col].astype(str).to_numpy()

    for i, predicted_task in enumerate(pred_task):
        row_index = int(test_idx[i])
        entry = step_models.get(str(predicted_task))
        pred_step_two_stage.append(
            predict_step_entry(entry, X, row_index, device) if entry else str(global_fallback_step)
        )

        oracle_entry = step_models.get(str(true_tasks[i]))
        pred_step_oracle.append(
            predict_step_entry(oracle_entry, X, row_index, device) if oracle_entry else str(global_fallback_step)
        )

    result = pd.DataFrame(
        {
            "participant_id": meta_test[participant_col].to_numpy(),
            "full_id": meta_test[full_id_col].to_numpy(),
            "video_frame_idx": meta_test[frame_col].to_numpy(),
            "true_task": true_tasks,
            "pred_task": pred_task.astype(str),
            "true_step": meta_test[step_col].astype(str).to_numpy(),
            "pred_step_two_stage": pred_step_two_stage,
            "pred_step_oracle_task": pred_step_oracle,
        }
    )
    result["task_correct"] = result["true_task"] == result["pred_task"]
    result["step_correct_two_stage"] = result["true_step"] == result["pred_step_two_stage"]
    result["step_correct_oracle"] = result["true_step"] == result["pred_step_oracle_task"]
    return result


def predict_direct_step(
    X: np.ndarray,
    test_idx: np.ndarray,
    meta_test: pd.DataFrame,
    direct_step_info: dict[str, Any],
    device: torch.device,
    participant_col: str,
    full_id_col: str,
    frame_col: str,
    task_col: str,
    step_col: str,
    num_workers: int,
) -> pd.DataFrame:
    if direct_step_info["type"] == "constant":
        pred_step = np.full(len(test_idx), direct_step_info["constant_step"], dtype=object)
    else:
        pred_idx = predict_torch_classifier_indexed(
            direct_step_info["model"],
            X,
            test_idx,
            direct_step_info["scaler"],
            device,
            num_workers=num_workers,
        )
        pred_step = direct_step_info["label_encoder"].inverse_transform(pred_idx)

    result = pd.DataFrame(
        {
            "participant_id": meta_test[participant_col].to_numpy(),
            "full_id": meta_test[full_id_col].to_numpy(),
            "video_frame_idx": meta_test[frame_col].to_numpy(),
            "true_task": meta_test[task_col].astype(str).to_numpy(),
            "true_step": meta_test[step_col].astype(str).to_numpy(),
            "pred_step_direct": pred_step.astype(str),
        }
    )
    result["step_correct_direct"] = result["true_step"] == result["pred_step_direct"]
    return result


def metric_block(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def save_confusion_matrix(
    y_true: pd.Series,
    y_pred: pd.Series,
    output_path: Path,
) -> None:
    labels = sorted(set(y_true.astype(str)) | set(y_pred.astype(str)))
    matrix = confusion_matrix(y_true.astype(str), y_pred.astype(str), labels=labels)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(output_path)


def evaluate_direct_step_and_save(
    predictions: pd.DataFrame,
    output_dir: Path,
    class_names: list[str],
) -> None:
    metrics: dict[str, Any] = {
        "direct_step": metric_block(predictions["true_step"], predictions["pred_step_direct"]),
        "step_classes": class_names,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "direct_step_classification_report.txt").write_text(
        classification_report(
            predictions["true_step"],
            predictions["pred_step_direct"],
            zero_division=0,
        ),
        encoding="utf-8",
    )
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    save_confusion_matrix(
        predictions["true_step"],
        predictions["pred_step_direct"],
        output_dir / "direct_step_confusion_matrix.csv",
    )


def save_training_logs(
    output_dir: Path,
    training_history: list[dict[str, Any]],
    training_summary: dict[str, Any],
) -> None:
    history_path = output_dir / "training_history.csv"
    if training_history:
        pd.DataFrame(training_history).to_csv(history_path, index=False)
    else:
        pd.DataFrame(
            columns=[
                "model_name",
                "epoch",
                "train_loss",
                "val_loss",
                "val_accuracy",
                "val_macro_f1",
                "train_rows",
                "val_rows",
                "num_classes",
                "is_best",
            ]
        ).to_csv(history_path, index=False)

    (output_dir / "training_history.json").write_text(
        json.dumps(training_history, indent=2),
        encoding="utf-8",
    )
    (output_dir / "training_summary.json").write_text(
        json.dumps(training_summary, indent=2),
        encoding="utf-8",
    )


def evaluate_and_save(
    predictions: pd.DataFrame,
    output_dir: Path,
    task_label_encoder: LabelEncoder,
) -> None:
    metrics: dict[str, Any] = {
        "task": metric_block(predictions["true_task"], predictions["pred_task"]),
        "final_step_two_stage": metric_block(
            predictions["true_step"], predictions["pred_step_two_stage"]
        ),
        "oracle_task_step": metric_block(
            predictions["true_step"], predictions["pred_step_oracle_task"]
        ),
        "task_step_accuracy": float(
            (
                (predictions["true_task"] == predictions["pred_task"])
                & (predictions["true_step"] == predictions["pred_step_two_stage"])
            ).mean()
        ),
        "per_task_oracle_step": {},
        "task_classes": task_label_encoder.classes_.astype(str).tolist(),
    }

    for task, group in predictions.groupby("true_task", sort=True):
        metrics["per_task_oracle_step"][str(task)] = {
            "rows": int(len(group)),
            "accuracy": float(accuracy_score(group["true_step"], group["pred_step_oracle_task"])),
            "macro_f1": float(
                f1_score(
                    group["true_step"],
                    group["pred_step_oracle_task"],
                    average="macro",
                    zero_division=0,
                )
            ),
        }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "task_classification_report.txt").write_text(
        classification_report(
            predictions["true_task"],
            predictions["pred_task"],
            zero_division=0,
        ),
        encoding="utf-8",
    )
    (output_dir / "final_step_classification_report.txt").write_text(
        classification_report(
            predictions["true_step"],
            predictions["pred_step_two_stage"],
            zero_division=0,
        ),
        encoding="utf-8",
    )
    (output_dir / "oracle_step_classification_report.txt").write_text(
        classification_report(
            predictions["true_step"],
            predictions["pred_step_oracle_task"],
            zero_division=0,
        ),
        encoding="utf-8",
    )
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    save_confusion_matrix(predictions["true_task"], predictions["pred_task"], output_dir / "task_confusion_matrix.csv")
    save_confusion_matrix(
        predictions["true_step"],
        predictions["pred_step_two_stage"],
        output_dir / "final_step_confusion_matrix.csv",
    )
    save_confusion_matrix(
        predictions["true_step"],
        predictions["pred_step_oracle_task"],
        output_dir / "oracle_step_confusion_matrix.csv",
    )


def main() -> None:
    args = parse_args()
    set_seed(args.random_state)
    device = get_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    window_frames = int(round(args.fps * args.window_sec))
    print(f"window_frames: {window_frames}")

    df, feature_cols = load_and_prepare_data(
        args.input_csv,
        args.participant_col,
        args.target_task_col,
        args.target_step_col,
        args.full_id_col,
        args.frame_col,
    )
    X_flat, meta_df, flattened_feature_names = build_flatten_lag_features(
        df,
        feature_cols,
        args.full_id_col,
        args.frame_col,
        args.participant_col,
        args.target_task_col,
        args.target_step_col,
        window_frames,
        output_dir,
    )
    (output_dir / "flattened_feature_columns.json").write_text(
        json.dumps(flattened_feature_names, indent=2), encoding="utf-8"
    )
    (output_dir / "original_frame_feature_columns.json").write_text(
        json.dumps(feature_cols, indent=2), encoding="utf-8"
    )
    (output_dir / "task_name_normalization.json").write_text(
        json.dumps(TASK_NAME_NORMALIZATION, indent=2),
        encoding="utf-8",
    )

    supervised_idx = select_supervised_sample_indices(
        meta_df,
        args.target_task_col,
        args.target_step_col,
        args.frame_col,
    )
    split_info: dict[str, Any] = {"split_mode": args.split_mode}
    if args.split_mode == "participant":
        train_idx, test_idx, train_participants, test_participants = split_by_participant(
            meta_df,
            supervised_idx,
            args.participant_col,
            args.target_task_col,
            args.target_step_col,
            args.test_size,
            args.random_state,
        )
        split_info["train_participants"] = train_participants
        split_info["test_participants"] = test_participants
    else:
        train_idx, test_idx = split_by_row(
            meta_df,
            supervised_idx,
            args.target_task_col,
            args.target_step_col,
            args.test_size,
            args.random_state,
        )
        split_info["train_participants"] = sorted(
            meta_df.iloc[train_idx][args.participant_col].astype(str).unique().tolist()
        )
        split_info["test_participants"] = sorted(
            meta_df.iloc[test_idx][args.participant_col].astype(str).unique().tolist()
        )

    (output_dir / "train_test_participants.json").write_text(
        json.dumps(split_info, indent=2),
        encoding="utf-8",
    )

    meta_train = meta_df.iloc[train_idx].reset_index(drop=True)
    meta_test = meta_df.iloc[test_idx].reset_index(drop=True)
    training_history: list[dict[str, Any]] = []
    training_summary: dict[str, Any] = {
        "task_name_normalization": TASK_NAME_NORMALIZATION,
        "split_mode": args.split_mode,
        "model_mode": args.model_mode,
    }

    if args.model_mode == "direct_step":
        direct_step_info = train_direct_step_classifier(
            X_flat,
            train_idx,
            meta_train,
            args.target_step_col,
            output_dir,
            device,
            args,
            training_history,
            training_summary,
        )
        save_training_logs(output_dir, training_history, training_summary)
        predictions = predict_direct_step(
            X_flat,
            test_idx,
            meta_test,
            direct_step_info,
            device,
            args.participant_col,
            args.full_id_col,
            args.frame_col,
            args.target_task_col,
            args.target_step_col,
            args.num_workers,
        )
        evaluate_direct_step_and_save(
            predictions,
            output_dir,
            direct_step_info["class_names"],
        )
        print(f"Done. Outputs saved to: {output_dir}")
        return

    task_model_info = train_task_classifier(
        X_flat,
        train_idx,
        meta_train[args.target_task_col].astype(str).to_numpy(),
        output_dir,
        device,
        args,
        training_history,
        training_summary,
    )
    step_models = train_step_classifiers(
        X_flat,
        train_idx,
        meta_train,
        args.target_task_col,
        args.target_step_col,
        output_dir,
        device,
        args,
        training_history,
        training_summary,
    )
    save_training_logs(output_dir, training_history, training_summary)

    global_fallback_step = str(meta_train[args.target_step_col].mode().iloc[0])
    predictions = predict_two_stage(
        X_flat,
        test_idx,
        meta_test,
        task_model_info,
        step_models,
        global_fallback_step,
        device,
        args.participant_col,
        args.full_id_col,
        args.frame_col,
        args.target_task_col,
        args.target_step_col,
        args.num_workers,
    )
    evaluate_and_save(
        predictions,
        output_dir,
        task_model_info["label_encoder"],
    )
    print(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
