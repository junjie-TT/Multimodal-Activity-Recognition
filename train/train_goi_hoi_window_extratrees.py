#!/usr/bin/env python3
"""Train ExtraTrees GOI+HOI causal window classifiers for step prediction."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


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
    "Chaos_several_Task_combine1": "Chaos_several_Tasks_combine1",
    "Chaos_several_Task_combine2": "Chaos_several_Tasks_combine2",
    "Chaos_several_Task_combine3": "Chaos_several_Tasks_combine3",
    "Chaos_several_Task_combine4": "Chaos_several_Tasks_combine4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ExtraTrees on causal flattened GOI+HOI window features."
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
        default="row",
        choices=["participant", "row"],
        help="participant: hold out participants; row: random supervised row split.",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--min-samples-leaf", type=int, default=10)
    parser.add_argument("--max-features", default="sqrt")
    parser.add_argument("--class-weight", default="balanced", choices=["balanced", "none"])
    parser.add_argument("--n-jobs", type=int, default=2)
    parser.add_argument(
        "--max-train-rows-per-model",
        type=int,
        default=0,
        help="If >0, cap training rows used by each model. 0 means unlimited.",
    )
    parser.add_argument(
        "--max-task-train-rows",
        type=int,
        default=0,
        help="If >0, cap training rows used by the task classifier. 0 means unlimited.",
    )
    parser.add_argument(
        "--max-step-train-rows",
        type=int,
        default=0,
        help="If >0, cap training rows used by each step classifier. 0 means unlimited.",
    )
    parser.add_argument(
        "--predict-batch-size",
        type=int,
        default=4096,
        help="Rows copied from memmap per sklearn predict batch.",
    )
    parser.add_argument(
        "--resume-evaluate",
        action="store_true",
        help=(
            "Skip feature/model training and evaluate existing X_flat.npy, "
            "task_model.joblib, task_label_encoder.joblib, and step_models.joblib "
            "from output-dir."
        ),
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def normalize_task_names(df: pd.DataFrame, task_col: str) -> pd.DataFrame:
    result = df.copy()
    task_values = result[task_col].astype("string")
    before_counts = task_values.value_counts(dropna=True)
    result[task_col] = task_values.replace(TASK_NAME_NORMALIZATION)
    after_counts = result[task_col].astype("string").value_counts(dropna=True)

    changed = {
        source: {"target": target, "rows": int(before_counts.get(source, 0))}
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


def build_meta_df_only(
    df: pd.DataFrame,
    full_id_col: str,
    frame_col: str,
    participant_col: str,
    task_col: str,
    step_col: str,
) -> pd.DataFrame:
    meta_cols = [participant_col, task_col, full_id_col, frame_col, step_col]
    meta_chunks = []
    for _, group in df.sort_values([full_id_col, frame_col]).groupby(full_id_col, sort=False):
        meta_chunks.append(group.sort_values(frame_col)[meta_cols].copy())
    return pd.concat(meta_chunks, ignore_index=True)


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
        raise ValueError("Need at least two participants for participant split.")
    rng = np.random.default_rng(random_state)
    rng.shuffle(participants)
    test_count = max(1, int(round(len(participants) * test_size)))
    test_count = min(test_count, len(participants) - 1)
    test_participants = sorted(participants[:test_count].tolist())
    train_participants = sorted(participants[test_count:].tolist())

    candidate_participants = meta_df[participant_col].astype(str).iloc[candidate_idx]
    train_idx = candidate_idx[candidate_participants.isin(train_participants).to_numpy()]
    test_idx = candidate_idx[candidate_participants.isin(test_participants).to_numpy()]

    print(f"Train participants ({len(train_participants)}): {train_participants}")
    print(f"Test participants ({len(test_participants)}): {test_participants}")
    print_split_distributions(meta_df, train_idx, test_idx, task_col, step_col)
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
    print_split_distributions(meta_df, train_idx, test_idx, task_col, step_col)
    return train_idx, test_idx


def print_split_distributions(
    meta_df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    task_col: str,
    step_col: str,
) -> None:
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


def select_supervised_sample_indices(
    meta_df: pd.DataFrame,
    task_col: str,
    step_col: str,
    frame_col: str,
) -> np.ndarray:
    task_valid = meta_df[task_col].notna() & (meta_df[task_col].astype(str).str.strip() != "")
    step_valid = meta_df[step_col].notna() & (meta_df[step_col].astype(str).str.strip() != "")
    frame_valid = pd.to_numeric(meta_df[frame_col], errors="coerce") >= 0
    supervised_idx = np.flatnonzero((task_valid & step_valid & frame_valid).to_numpy())
    print(f"Supervised rows selected after lag construction: {len(supervised_idx)}")
    print(f"Rows kept only as unlabeled/invalid-frame context: {len(meta_df) - len(supervised_idx)}")
    if len(supervised_idx) == 0:
        raise ValueError("No supervised rows remain after task/step/frame filtering.")
    return supervised_idx


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value)[:120]


def extratrees_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": args.max_features,
        "class_weight": None if args.class_weight == "none" else args.class_weight,
        "n_jobs": args.n_jobs,
        "random_state": args.random_state,
    }


def make_extratrees(args: argparse.Namespace) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(**extratrees_params(args))


def rows_as_float32(X: np.ndarray, row_indices: np.ndarray) -> np.ndarray:
    return np.asarray(X[row_indices], dtype=np.float32)


def _random_subsample_positions(
    num_rows: int,
    max_rows: int,
    random_state: int,
) -> np.ndarray:
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(num_rows, size=max_rows, replace=False)).astype(np.int64)


def _subsample_positions(
    row_indices: np.ndarray,
    y: np.ndarray | None = None,
    max_rows: int = 0,
    random_state: int = 42,
) -> np.ndarray:
    num_rows = len(row_indices)
    if max_rows <= 0 or num_rows <= max_rows:
        return np.arange(num_rows, dtype=np.int64)

    if y is None:
        return _random_subsample_positions(num_rows, max_rows, random_state)

    try:
        positions = np.arange(num_rows, dtype=np.int64)
        sampled_positions, _ = train_test_split(
            positions,
            train_size=max_rows,
            random_state=random_state,
            shuffle=True,
            stratify=y,
        )
        return np.sort(np.asarray(sampled_positions, dtype=np.int64))
    except ValueError as exc:
        print(f"Warning: stratified subsampling failed ({exc}); using random sampling.")
        return _random_subsample_positions(num_rows, max_rows, random_state)


def subsample_indices(
    row_indices: np.ndarray,
    y: np.ndarray | None = None,
    max_rows: int = 0,
    random_state: int = 42,
) -> np.ndarray:
    row_indices = np.asarray(row_indices, dtype=np.int64)
    if max_rows <= 0 or len(row_indices) <= max_rows:
        print(f"Subsample rows: before={len(row_indices)}, after={len(row_indices)}")
        return row_indices
    positions = _subsample_positions(row_indices, y, max_rows, random_state)
    sampled = row_indices[positions].astype(np.int64, copy=False)
    print(f"Subsample rows: before={len(row_indices)}, after={len(sampled)}")
    return sampled


def subsample_indices_with_y(
    row_indices: np.ndarray,
    y: np.ndarray,
    max_rows: int = 0,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    row_indices = np.asarray(row_indices, dtype=np.int64)
    y = np.asarray(y)
    if len(row_indices) != len(y):
        raise ValueError(
            f"row_indices and y must have the same length; got {len(row_indices)} and {len(y)}."
        )
    if max_rows <= 0 or len(row_indices) <= max_rows:
        print(f"Subsample rows: before={len(row_indices)}, after={len(row_indices)}")
        return row_indices, y
    positions = _subsample_positions(row_indices, y, max_rows, random_state)
    sampled_indices = row_indices[positions].astype(np.int64, copy=False)
    sampled_y = y[positions]
    print(f"Subsample rows: before={len(row_indices)}, after={len(sampled_indices)}")
    return sampled_indices, sampled_y


def predict_sklearn_in_batches(
    model: Any,
    X: np.ndarray,
    row_indices: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0.")

    row_indices = np.asarray(row_indices, dtype=np.int64)
    predictions = []
    for start in range(0, len(row_indices), batch_size):
        batch_idx = row_indices[start : start + batch_size]
        batch_X = np.asarray(X[batch_idx], dtype=np.float32)
        predictions.append(model.predict(batch_X))
    if not predictions:
        return np.array([], dtype=np.int64)
    return np.concatenate(predictions)


def train_task_classifier_extratrees(
    X: np.ndarray,
    train_idx: np.ndarray,
    y_task_train: np.ndarray,
    output_dir: Path,
    args: argparse.Namespace,
    training_summary: dict[str, Any],
) -> dict[str, Any]:
    label_encoder = LabelEncoder()
    label_encoder.fit(y_task_train)
    joblib.dump(label_encoder, output_dir / "task_label_encoder.joblib")

    summary = {
        "type": "constant" if len(label_encoder.classes_) == 1 else "sklearn",
        "num_classes": int(len(label_encoder.classes_)),
        "train_rows": int(len(train_idx)),
        "original_train_rows": int(len(train_idx)),
        "used_train_rows": int(len(train_idx)),
        "sampled": False,
        "class_weight": args.class_weight,
    }
    training_summary["task_model"] = summary

    if len(label_encoder.classes_) == 1:
        constant_task = str(label_encoder.classes_[0])
        joblib.dump(
            {"type": "constant", "constant_task": constant_task},
            output_dir / "task_model.joblib",
        )
        return {
            "type": "constant",
            "model": None,
            "label_encoder": label_encoder,
            "constant_task": constant_task,
        }

    limit = args.max_task_train_rows or args.max_train_rows_per_model
    used_idx, used_y_task = subsample_indices_with_y(
        train_idx,
        y_task_train,
        limit,
        args.random_state,
    )
    y = label_encoder.transform(used_y_task)
    summary["used_train_rows"] = int(len(used_idx))
    summary["sampled"] = bool(len(used_idx) != len(train_idx))

    print(
        f"Training task ExtraTrees: rows={len(used_idx)} "
        f"(original={len(train_idx)}), classes={len(label_encoder.classes_)}"
    )
    model = make_extratrees(args)
    model.fit(rows_as_float32(X, used_idx), y)
    joblib.dump(model, output_dir / "task_model.joblib")
    return {
        "type": "sklearn",
        "model": model,
        "label_encoder": label_encoder,
        "constant_task": None,
    }


def train_step_classifiers_extratrees(
    X: np.ndarray,
    train_idx: np.ndarray,
    meta_train: pd.DataFrame,
    task_col: str,
    step_col: str,
    output_dir: Path,
    args: argparse.Namespace,
    training_summary: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    step_models: dict[str, dict[str, Any]] = {}
    training_summary["step_models"] = {}

    for task in sorted(meta_train[task_col].astype(str).unique()):
        model_name = f"step_model_{safe_name(task)}"
        task_mask = meta_train[task_col].astype(str).to_numpy() == task
        task_row_indices = train_idx[task_mask]
        y_step = meta_train.loc[task_mask, step_col].astype(str).to_numpy()
        step_distribution = pd.Series(y_step).value_counts().to_dict()
        print(f"Training step ExtraTrees for task={task!r}: rows={len(y_step)}")
        print("Step label distribution:")
        print(pd.Series(step_distribution))

        summary_entry = {
            "task": task,
            "train_rows": int(len(y_step)),
            "original_train_rows": int(len(y_step)),
            "used_train_rows": int(len(y_step)),
            "sampled": False,
            "step_distribution": {str(k): int(v) for k, v in step_distribution.items()},
            "class_weight": args.class_weight,
        }

        if len(step_distribution) == 1:
            constant_step = str(next(iter(step_distribution.keys())))
            summary_entry.update(
                {"type": "constant", "num_classes": 1, "constant_step": constant_step}
            )
            step_models[task] = {
                "type": "constant",
                "constant_step": constant_step,
                "class_names": [constant_step],
                "model_params": extratrees_params(args),
            }
            training_summary["step_models"][model_name] = summary_entry
            continue

        if len(y_step) < 20:
            constant_step = str(pd.Series(y_step).value_counts().idxmax())
            print(
                f"Warning: task={task!r} has fewer than 20 samples; "
                f"using constant step={constant_step!r}."
            )
            summary_entry.update(
                {
                    "type": "constant_small_sample",
                    "num_classes": int(len(step_distribution)),
                    "constant_step": constant_step,
                }
            )
            step_models[task] = {
                "type": "constant",
                "constant_step": constant_step,
                "class_names": [str(label) for label in step_distribution.keys()],
                "model_params": extratrees_params(args),
            }
            training_summary["step_models"][model_name] = summary_entry
            continue

        label_encoder = LabelEncoder()
        label_encoder.fit(y_step)
        limit = args.max_step_train_rows or args.max_train_rows_per_model
        used_task_row_indices, used_y_step = subsample_indices_with_y(
            task_row_indices,
            y_step,
            limit,
            args.random_state,
        )
        y = label_encoder.transform(used_y_step)
        summary_entry["used_train_rows"] = int(len(used_task_row_indices))
        summary_entry["sampled"] = bool(len(used_task_row_indices) != len(task_row_indices))
        print(
            f"Fitting step ExtraTrees for task={task!r}: rows={len(used_task_row_indices)} "
            f"(original={len(task_row_indices)})"
        )
        model = make_extratrees(args)
        model.fit(rows_as_float32(X, used_task_row_indices), y)
        class_names = label_encoder.classes_.astype(str).tolist()
        step_models[task] = {
            "type": "sklearn",
            "model": model,
            "label_encoder": label_encoder,
            "class_names": class_names,
            "model_params": extratrees_params(args),
        }
        summary_entry.update({"type": "sklearn", "num_classes": int(len(class_names))})
        training_summary["step_models"][model_name] = summary_entry

    joblib.dump(step_models, output_dir / "step_models.joblib")
    return step_models


def predict_task_extratrees(
    X: np.ndarray,
    test_idx: np.ndarray,
    task_model_info: dict[str, Any],
    batch_size: int,
) -> np.ndarray:
    if task_model_info["type"] == "constant":
        return np.full(len(test_idx), task_model_info["constant_task"], dtype=object)
    pred_idx = predict_sklearn_in_batches(task_model_info["model"], X, test_idx, batch_size)
    return task_model_info["label_encoder"].inverse_transform(pred_idx)


def predict_grouped_steps_extratrees(
    X: np.ndarray,
    row_indices: np.ndarray,
    group_tasks: np.ndarray,
    step_models: dict[str, dict[str, Any]],
    global_fallback_step: str,
    batch_size: int,
    label: str,
) -> np.ndarray:
    row_indices = np.asarray(row_indices, dtype=np.int64)
    group_tasks = np.asarray(group_tasks).astype(str)
    predictions = np.full(len(row_indices), str(global_fallback_step), dtype=object)
    unique_tasks = sorted(pd.unique(group_tasks).tolist())

    print(f"Predicting {label} step models by task group: {len(unique_tasks)} groups.")
    for task in unique_tasks:
        positions = np.flatnonzero(group_tasks == task)
        entry = step_models.get(str(task))
        if entry is None:
            print(
                f"  task={task!r}: no step model for {len(positions)} rows; "
                f"using fallback={global_fallback_step!r}."
            )
            continue
        if entry["type"] == "constant":
            predictions[positions] = str(entry["constant_step"])
            continue

        pred_idx = predict_sklearn_in_batches(
            entry["model"],
            X,
            row_indices[positions],
            batch_size,
        )
        predictions[positions] = entry["label_encoder"].inverse_transform(pred_idx).astype(str)
    return predictions


def predict_two_stage_extratrees(
    X: np.ndarray,
    test_idx: np.ndarray,
    meta_test: pd.DataFrame,
    task_model_info: dict[str, Any],
    step_models: dict[str, dict[str, Any]],
    global_fallback_step: str,
    participant_col: str,
    full_id_col: str,
    frame_col: str,
    task_col: str,
    step_col: str,
    batch_size: int,
) -> pd.DataFrame:
    print(f"Predicting task for {len(test_idx)} test rows.")
    pred_task = predict_task_extratrees(X, test_idx, task_model_info, batch_size)
    true_tasks = meta_test[task_col].astype(str).to_numpy()
    pred_step_two_stage = predict_grouped_steps_extratrees(
        X,
        test_idx,
        pred_task,
        step_models,
        global_fallback_step,
        batch_size,
        "two-stage",
    )
    pred_step_oracle = predict_grouped_steps_extratrees(
        X,
        test_idx,
        true_tasks,
        step_models,
        global_fallback_step,
        batch_size,
        "oracle-task",
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
        classification_report(predictions["true_task"], predictions["pred_task"], zero_division=0),
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
    save_confusion_matrix(
        predictions["true_task"],
        predictions["pred_task"],
        output_dir / "task_confusion_matrix.csv",
    )
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


def save_model_config(args: argparse.Namespace, output_dir: Path, window_frames: int) -> None:
    config = {
        "model": "ExtraTreesClassifier",
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": args.max_features,
        "class_weight": args.class_weight,
        "n_jobs": args.n_jobs,
        "split_mode": args.split_mode,
        "window_sec": args.window_sec,
        "window_frames": window_frames,
        "fps": args.fps,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "max_train_rows_per_model": args.max_train_rows_per_model,
        "max_task_train_rows": args.max_task_train_rows,
        "max_step_train_rows": args.max_step_train_rows,
        "predict_batch_size": args.predict_batch_size,
    }
    (output_dir / "model_config.json").write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )


def load_task_model_info(output_dir: Path) -> dict[str, Any]:
    label_encoder = joblib.load(output_dir / "task_label_encoder.joblib")
    task_model = joblib.load(output_dir / "task_model.joblib")
    if isinstance(task_model, dict) and task_model.get("type") == "constant":
        return {
            "type": "constant",
            "model": None,
            "label_encoder": label_encoder,
            "constant_task": task_model["constant_task"],
        }
    return {
        "type": "sklearn",
        "model": task_model,
        "label_encoder": label_encoder,
        "constant_task": None,
    }


def set_loaded_model_n_jobs(
    task_model_info: dict[str, Any],
    step_models: dict[str, dict[str, Any]],
    n_jobs: int,
) -> None:
    if task_model_info["type"] == "sklearn" and hasattr(task_model_info["model"], "n_jobs"):
        task_model_info["model"].n_jobs = n_jobs
    for entry in step_models.values():
        if entry.get("type") == "sklearn" and hasattr(entry.get("model"), "n_jobs"):
            entry["model"].n_jobs = n_jobs


def split_supervised_indices(
    meta_df: pd.DataFrame,
    supervised_idx: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
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
    return train_idx, test_idx, split_info


def main() -> None:
    args = parse_args()
    set_seed(args.random_state)
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
    if args.resume_evaluate:
        x_path = output_dir / "X_flat.npy"
        if not x_path.exists():
            raise FileNotFoundError(f"Missing existing memmap: {x_path}")
        print(f"Resume evaluate: loading existing memmap {x_path}")
        X_flat = np.load(x_path, mmap_mode="r")
        meta_df = build_meta_df_only(
            df,
            args.full_id_col,
            args.frame_col,
            args.participant_col,
            args.target_task_col,
            args.target_step_col,
        )
        if len(meta_df) != X_flat.shape[0]:
            raise ValueError(
                f"meta rows ({len(meta_df)}) do not match X rows ({X_flat.shape[0]})."
            )
    else:
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
            json.dumps(flattened_feature_names, indent=2),
            encoding="utf-8",
        )
        (output_dir / "original_frame_feature_columns.json").write_text(
            json.dumps(feature_cols, indent=2),
            encoding="utf-8",
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
    train_idx, test_idx, split_info = split_supervised_indices(meta_df, supervised_idx, args)

    (output_dir / "train_test_participants.json").write_text(
        json.dumps(split_info, indent=2),
        encoding="utf-8",
    )

    meta_train = meta_df.iloc[train_idx].reset_index(drop=True)
    meta_test = meta_df.iloc[test_idx].reset_index(drop=True)
    training_summary: dict[str, Any] = {
        "model": "ExtraTreesClassifier",
        "split_mode": args.split_mode,
        "task_name_normalization": TASK_NAME_NORMALIZATION,
        "class_weight": args.class_weight,
    }

    if args.resume_evaluate:
        print("Resume evaluate: loading existing task and step models.")
        task_model_info = load_task_model_info(output_dir)
        step_models = joblib.load(output_dir / "step_models.joblib")
        set_loaded_model_n_jobs(task_model_info, step_models, args.n_jobs)
        print(f"Resume evaluate: loaded models; set sklearn n_jobs={args.n_jobs}.")
    else:
        task_model_info = train_task_classifier_extratrees(
            X_flat,
            train_idx,
            meta_train[args.target_task_col].astype(str).to_numpy(),
            output_dir,
            args,
            training_summary,
        )
        step_models = train_step_classifiers_extratrees(
            X_flat,
            train_idx,
            meta_train,
            args.target_task_col,
            args.target_step_col,
            output_dir,
            args,
            training_summary,
        )

    global_fallback_step = str(meta_train[args.target_step_col].mode().iloc[0])
    predictions = predict_two_stage_extratrees(
        X_flat,
        test_idx,
        meta_test,
        task_model_info,
        step_models,
        global_fallback_step,
        args.participant_col,
        args.full_id_col,
        args.frame_col,
        args.target_task_col,
        args.target_step_col,
        args.predict_batch_size,
    )
    evaluate_and_save(predictions, output_dir, task_model_info["label_encoder"])
    save_model_config(args, output_dir, window_frames)
    (output_dir / "training_summary.json").write_text(
        json.dumps(training_summary, indent=2),
        encoding="utf-8",
    )
    print(f"Done. Outputs saved to: {output_dir}")


if __name__ == "__main__":
    main()
