#!/usr/bin/env python3
"""Train and evaluate an LSTM current-action baseline on causal frame windows."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder

from common import DEFAULT_FPS, FRAME_LEVEL_PATH, RESULTS_DIR, TARGET_MODES, configure_logging, read_dataframe


BASELINE_DIR = Path(__file__).resolve().parent


def load_helper_module(filename: str, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, BASELINE_DIR / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load helper module: {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


window_helpers = load_helper_module("02_build_window_features.py", "window_feature_helpers")
svm_helpers = load_helper_module("03_train_svm_baseline.py", "svm_baseline_helpers")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LSTM baseline on causal frame-level ADL windows.")
    parser.add_argument("--input", type=Path, default=FRAME_LEVEL_PATH, help="Frame-level multimodal dataframe.")
    parser.add_argument("--target_mode", choices=TARGET_MODES, default="task_step")
    parser.add_argument(
        "--classifier_mode",
        choices=("auto", "single", "hierarchical"),
        default="auto",
        help="auto uses hierarchical for target_mode=task_step and single otherwise.",
    )
    parser.add_argument("--split_mode", choices=("video", "participant"), default="participant")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--window_sec", type=float, default=1.5)
    parser.add_argument("--stride_sec", type=float, default=0.5)
    parser.add_argument("--label_tail_sec", type=float, default=0.5)
    parser.add_argument("--min_label_purity", type=float, default=0.6)
    parser.add_argument("--keep_low_purity", action="store_true")
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience on validation macro F1.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap for training windows only.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def effective_classifier_mode(args: argparse.Namespace) -> str:
    if args.classifier_mode == "auto":
        return "hierarchical" if args.target_mode == "task_step" else "single"
    return args.classifier_mode


def result_directory(args: argparse.Namespace) -> Path:
    mode = effective_classifier_mode(args)
    window_tag = f"{args.window_sec:g}".replace(".", "p")
    stride_tag = f"{args.stride_sec:g}".replace(".", "p")
    return args.results_dir / f"lstm_{args.target_mode}_{mode}_{window_tag}s_stride{stride_tag}s_{args.split_mode}"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:  # noqa: BLE001
        pass


def build_sequence_windows(args: argparse.Namespace) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    if not args.input.exists():
        raise SystemExit(f"Input frame-level table does not exist: {args.input}")
    df = read_dataframe(args.input)
    if "frame_id" not in df.columns:
        raise SystemExit("Input frame-level table must contain frame_id.")

    target_col = window_helpers.LABEL_COLUMNS[args.target_mode]
    if target_col not in df.columns:
        raise SystemExit(f"Input frame-level table must contain {target_col}.")

    feature_cols, _ = window_helpers.feature_columns(df)
    if not feature_cols:
        raise SystemExit("No numeric feature columns found after excluding metadata/labels.")

    window_frames = int(round(args.window_sec * args.fps))
    stride_frames = int(round(args.stride_sec * args.fps))
    label_tail_frames = int(round(args.label_tail_sec * args.fps))
    if min(window_frames, stride_frames, label_tail_frames) <= 0:
        raise SystemExit("window_sec, stride_sec, and label_tail_sec must be positive.")

    rows: list[dict[str, object]] = []
    sequences: list[np.ndarray] = []
    group_cols = window_helpers.sequence_group_columns(df)
    logging.info("Building LSTM windows from %d frame features, grouped by %s", len(feature_cols), group_cols)

    for sequence_count, (_, seq) in enumerate(df.groupby(group_cols, dropna=False, sort=False), start=1):
        seq = seq.sort_values("frame_id", kind="mergesort").copy()
        seq["frame_id"] = pd.to_numeric(seq["frame_id"], errors="coerce")
        seq = seq[seq["frame_id"].notna()]
        if seq.empty:
            continue

        seq["frame_id"] = seq["frame_id"].round().astype("int64")
        seq = seq.drop_duplicates(subset=["frame_id"], keep="first")
        min_frame = int(seq["frame_id"].min())
        max_frame = int(seq["frame_id"].max())
        if max_frame - min_frame + 1 < window_frames:
            continue

        frame_index = pd.RangeIndex(min_frame, max_frame + 1, name="frame_id")
        indexed = seq.set_index("frame_id").reindex(frame_index)
        numeric = indexed[feature_cols].apply(pd.to_numeric, errors="coerce")
        labels = indexed[target_col] if target_col in indexed else pd.Series(index=frame_index, dtype="object")

        meta = {
            "participant_id": window_helpers.first_non_null(seq["participant_id"]) if "participant_id" in seq else pd.NA,
            "task_name": window_helpers.first_non_null(seq["task_name"]) if "task_name" in seq else pd.NA,
            "video_id": window_helpers.first_non_null(seq["video_id"]) if "video_id" in seq else pd.NA,
        }
        values = numeric.to_numpy(dtype=np.float32, copy=True)
        first_end = min_frame + window_frames - 1
        for end_frame in np.arange(first_end, max_frame + 1, stride_frames, dtype=np.int64):
            label_start = int(end_frame - label_tail_frames + 1)
            tail = labels.loc[label_start:int(end_frame)]
            label, purity, valid_frames = window_helpers.majority_label(tail)
            if label is None:
                continue
            if purity < args.min_label_purity and not args.keep_low_purity:
                continue
            start_frame = int(end_frame - window_frames + 1)
            start_idx = start_frame - min_frame
            end_idx = end_frame - min_frame + 1
            sequences.append(values[start_idx:end_idx])
            rows.append(
                {
                    **meta,
                    "window_start_frame": start_frame,
                    "window_end_frame": int(end_frame),
                    "window_start_time": float(start_frame / args.fps),
                    "window_end_time": float(end_frame / args.fps),
                    "target_label": label,
                    "label_purity": float(purity),
                    "num_valid_label_frames": int(valid_frames),
                }
            )
        if sequence_count % 25 == 0:
            logging.info("Processed %d sequences, accumulated %d windows", sequence_count, len(sequences))

    if not sequences:
        raise SystemExit("No valid LSTM windows were built.")
    return np.stack(sequences).astype(np.float32), pd.DataFrame(rows), feature_cols


def split_task_step_label(label: object) -> tuple[str | None, str | None]:
    return svm_helpers.split_task_step_label(label)


def add_task_step_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    parsed = df["target_label"].map(split_task_step_label)
    df["task_target"] = parsed.map(lambda item: item[0])
    df["step_target"] = parsed.map(lambda item: item[1])
    return df[df["task_target"].notna() & df["step_target"].notna()].copy()


def sample_training_indices(meta: pd.DataFrame, max_samples: int | None, random_state: int) -> np.ndarray:
    idx = meta.index[meta["split"] == "train"].to_numpy()
    if max_samples is None or len(idx) <= max_samples:
        return idx
    rng = np.random.default_rng(random_state)
    return np.sort(rng.choice(idx, size=max_samples, replace=False))


def fit_scaler(x_train: np.ndarray) -> dict[str, np.ndarray]:
    flat = x_train.reshape(-1, x_train.shape[-1])
    median = np.nanmedian(flat, axis=0).astype(np.float32)
    median = np.where(np.isfinite(median), median, 0.0).astype(np.float32)
    filled = np.where(np.isnan(flat), median, flat)
    mean = filled.mean(axis=0).astype(np.float32)
    std = filled.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return {"median": median, "mean": mean, "std": std}


def transform_sequences(x: np.ndarray, scaler: dict[str, np.ndarray]) -> np.ndarray:
    filled = np.where(np.isnan(x), scaler["median"], x)
    return ((filled - scaler["mean"]) / scaler["std"]).astype(np.float32)


def evaluate_label_columns(df: pd.DataFrame, true_col: str, pred_col: str, split_name: str) -> dict[str, object]:
    y_true = df[true_col].astype(str)
    y_pred = df[pred_col].astype(str)
    return {
        "split": split_name,
        "n_samples": int(len(df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def import_torch() -> Any:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is required for LSTM training. Install torch in this environment "
            "or run the script from an environment that already has PyTorch."
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def make_lstm_class(torch: Any, nn: Any) -> type:
    class LSTMClassifier(nn.Module):
        def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int,
            num_classes: int,
            dropout: float,
        ) -> None:
            super().__init__()
            lstm_dropout = dropout if num_layers > 1 else 0.0
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=lstm_dropout,
            )
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_size, num_classes)

        def forward(self, x: Any) -> Any:
            _, (hidden, _) = self.lstm(x)
            return self.classifier(self.dropout(hidden[-1]))

    return LSTMClassifier


def compute_class_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    return weights.astype(np.float32)


def train_label_model(
    x: np.ndarray,
    meta: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    label_col: str,
    args: argparse.Namespace,
    device: str,
) -> dict[str, object]:
    labels = meta.loc[train_idx, label_col].astype(str)
    unique_labels = sorted(labels.unique())
    if len(unique_labels) == 1:
        return {"kind": "constant", "label": unique_labels[0], "classes": unique_labels, "train_rows": int(len(train_idx))}

    torch, nn, DataLoader, TensorDataset = import_torch()
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(labels)
    val_known = meta.loc[val_idx, label_col].astype(str).isin(label_encoder.classes_).to_numpy()
    usable_val_idx = val_idx[val_known]
    y_val = label_encoder.transform(meta.loc[usable_val_idx, label_col].astype(str)) if len(usable_val_idx) else np.asarray([], dtype=np.int64)

    train_ds = TensorDataset(torch.from_numpy(x[train_idx]), torch.from_numpy(y_train.astype(np.int64)))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = None
    if len(usable_val_idx):
        val_ds = TensorDataset(torch.from_numpy(x[usable_val_idx]), torch.from_numpy(y_val.astype(np.int64)))
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model_cls = make_lstm_class(torch, nn)
    model = model_cls(
        input_size=x.shape[-1],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_classes=len(label_encoder.classes_),
        dropout=args.dropout,
    ).to(device)
    class_weights = torch.from_numpy(compute_class_weights(y_train, len(label_encoder.classes_))).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    best_score = -float("inf")
    best_state = None
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(xb)
            total_seen += len(xb)

        train_loss = total_loss / max(total_seen, 1)
        if val_loader is None:
            score = -train_loss
            val_macro_f1 = np.nan
        else:
            pred = predict_encoded(model, val_loader, torch, device)
            val_macro_f1 = float(f1_score(y_val, pred, average="macro", zero_division=0))
            score = val_macro_f1
        history.append({"epoch": float(epoch), "train_loss": float(train_loss), "val_macro_f1": float(val_macro_f1)})

        if score > best_score:
            best_score = score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {
        "kind": "lstm",
        "model": model,
        "label_encoder": label_encoder,
        "classes": label_encoder.classes_.tolist(),
        "train_rows": int(len(train_idx)),
        "history": history,
        "best_score": float(best_score),
    }


def predict_encoded(model: Any, data_loader: Any, torch: Any, device: str) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for xb, _ in data_loader:
            logits = model(xb.to(device))
            outputs.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(outputs) if outputs else np.asarray([], dtype=np.int64)


def predict_label_model(model_info: dict[str, object], x: np.ndarray, indices: np.ndarray, device: str) -> np.ndarray:
    if len(indices) == 0:
        return np.asarray([], dtype=object)
    if model_info["kind"] == "constant":
        return np.asarray([model_info["label"]] * len(indices), dtype=object)

    torch, _, DataLoader, TensorDataset = import_torch()
    dummy_y = np.zeros(len(indices), dtype=np.int64)
    ds = TensorDataset(torch.from_numpy(x[indices]), torch.from_numpy(dummy_y))
    loader = DataLoader(ds, batch_size=512, shuffle=False)
    pred = predict_encoded(model_info["model"], loader, torch, device)
    return model_info["label_encoder"].inverse_transform(pred)


def evaluate_predictions(meta: pd.DataFrame, split_name: str) -> dict[str, object]:
    return evaluate_label_columns(meta[meta["split"] == split_name], "target_label", "pred_label", split_name)


def train_hierarchical_models(
    x: np.ndarray,
    meta: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args: argparse.Namespace,
    device: str,
) -> dict[str, object]:
    task_model = train_label_model(x, meta, train_idx, val_idx, "task_target", args, device)
    step_models: dict[str, dict[str, object]] = {}
    for task_label, task_df in meta.loc[train_idx].groupby("task_target", sort=True):
        task_train_idx = task_df.index.to_numpy()
        task_val_idx = val_idx[meta.loc[val_idx, "task_target"].astype(str).to_numpy() == str(task_label)]
        step_models[str(task_label)] = train_label_model(x, meta, task_train_idx, task_val_idx, "step_target", args, device)
    return {"task_model": task_model, "step_models": step_models}


def predict_hierarchical(models: dict[str, object], x: np.ndarray, meta: pd.DataFrame, indices: np.ndarray, device: str) -> pd.DataFrame:
    predictions = pd.DataFrame(index=indices)
    pred_task = predict_label_model(models["task_model"], x, indices, device).astype(str)
    predictions["pred_task"] = pred_task
    predictions["pred_step"] = pd.NA
    for task_label in sorted(set(pred_task)):
        local_mask = pred_task == task_label
        task_indices = indices[local_mask]
        step_model = models["step_models"].get(task_label)
        if step_model is None:
            continue
        predictions.loc[task_indices, "pred_step"] = predict_label_model(step_model, x, task_indices, device)
    predictions["pred_label"] = predictions["pred_task"].astype(str) + "__" + predictions["pred_step"].astype(str)
    return predictions


def evaluate_hierarchical(meta: pd.DataFrame, split_name: str) -> dict[str, object]:
    eval_df = meta[meta["split"] == split_name]
    out = {
        "task": evaluate_label_columns(eval_df, "task_target", "pred_task", split_name),
        "step": evaluate_label_columns(eval_df, "step_target", "pred_step", split_name),
        "task_step": evaluate_label_columns(eval_df, "target_label", "pred_label", split_name),
    }
    correct_task = eval_df["task_target"].astype(str) == eval_df["pred_task"].astype(str)
    out["step_given_correct_task"] = (
        evaluate_label_columns(eval_df[correct_task], "step_target", "pred_step", split_name)
        if correct_task.any()
        else None
    )
    return out


def write_confusion_matrix_png(cm: np.ndarray, labels: list[str], path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        logging.warning("Could not import matplotlib; skipping confusion matrix PNG: %s", exc)
        return

    size = max(8, min(36, 0.35 * len(labels) + 6))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="LSTM Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def serializable_model_info(model_info: dict[str, object]) -> dict[str, object]:
    return {
        "kind": model_info["kind"],
        "classes": model_info["classes"],
        "train_rows": model_info["train_rows"],
        "history": model_info.get("history"),
        "best_score": model_info.get("best_score"),
    }


def torch_payload_model_info(model_info: dict[str, object]) -> dict[str, object]:
    if model_info["kind"] == "constant":
        return serializable_model_info(model_info)
    return {
        **serializable_model_info(model_info),
        "state_dict": {
            key: value.detach().cpu()
            for key, value in model_info["model"].state_dict().items()
        },
    }


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    set_seed(args.random_state)
    classifier_mode = effective_classifier_mode(args)
    if classifier_mode == "hierarchical" and args.target_mode != "task_step":
        raise SystemExit("--classifier_mode hierarchical requires --target_mode task_step.")

    torch, _, _, _ = import_torch()
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logging.info("Using device=%s", device)

    x_raw, meta, feature_cols = build_sequence_windows(args)
    if classifier_mode == "hierarchical":
        meta = add_task_step_targets(meta)
        x_raw = x_raw[meta.index.to_numpy()]
        meta = meta.reset_index(drop=True)

    group_split_table = svm_helpers.build_group_split_table(
        meta,
        args.split_mode,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.random_state,
    )
    group_col = "video_id" if args.split_mode == "video" else "participant_id"
    split_lookup = dict(zip(group_split_table[group_col], group_split_table["split"]))
    meta["split"] = meta[group_col].astype(str).map(split_lookup).fillna("test").astype("object")

    train_idx = sample_training_indices(meta, args.max_samples, args.random_state)
    val_idx = meta.index[meta["split"] == "val"].to_numpy()
    test_idx = meta.index[meta["split"] == "test"].to_numpy()
    if classifier_mode == "single" and meta.loc[train_idx, "target_label"].nunique() < 2:
        raise SystemExit("Training split has fewer than two classes; LSTM cannot be trained.")
    if classifier_mode == "hierarchical" and meta.loc[train_idx, "task_target"].nunique() < 2:
        raise SystemExit("Training split has fewer than two task classes; hierarchical task LSTM cannot be trained.")

    scaler = fit_scaler(x_raw[train_idx])
    x = transform_sequences(x_raw, scaler)
    logging.info(
        "Training windows=%d, val=%d, test=%d, sequence_shape=%s, target classes=%d",
        len(train_idx),
        len(val_idx),
        len(test_idx),
        tuple(x.shape[1:]),
        meta.loc[train_idx, "target_label"].nunique(),
    )

    out_dir = result_directory(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    group_split_table.to_csv(out_dir / "split_groups.csv", index=False)
    eval_idx = np.concatenate([val_idx, test_idx])

    metrics: dict[str, object] = {
        "input": str(args.input),
        "target_mode": args.target_mode,
        "classifier_mode": classifier_mode,
        "requested_classifier_mode": args.classifier_mode,
        "split_mode": args.split_mode,
        "random_state": args.random_state,
        "device": device,
        "window_sec": args.window_sec,
        "stride_sec": args.stride_sec,
        "label_tail_sec": args.label_tail_sec,
        "feature_count": len(feature_cols),
        "sequence_length": int(x.shape[1]),
        "train_rows": int(len(train_idx)),
        "val_rows": int(len(val_idx)),
        "test_rows": int(len(test_idx)),
        "train_groups": int(meta.loc[meta["split"] == "train", group_col].nunique()),
        "val_groups": int(meta.loc[meta["split"] == "val", group_col].nunique()),
        "test_groups": int(meta.loc[meta["split"] == "test", group_col].nunique()),
        "split_groups_csv": str(out_dir / "split_groups.csv"),
        "shuffled_groups": group_split_table.to_dict(orient="records"),
        "lstm": {
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
        },
    }

    if classifier_mode == "single":
        model_info = train_label_model(x, meta, train_idx, val_idx, "target_label", args, device)
        meta.loc[eval_idx, "pred_label"] = predict_label_model(model_info, x, eval_idx, device)
        metrics.update(
            {
                "classes": model_info["classes"],
                "model": serializable_model_info(model_info),
                "val": evaluate_predictions(meta, "val") if len(val_idx) else None,
                "test": evaluate_predictions(meta, "test") if len(test_idx) else None,
            }
        )
        model_payload = {"model": torch_payload_model_info(model_info)}
        prediction_cols = [
            "participant_id",
            "task_name",
            "video_id",
            "window_start_time",
            "window_end_time",
            "target_label",
            "pred_label",
            "label_purity",
            "split",
        ]
    else:
        hierarchical_models = train_hierarchical_models(x, meta, train_idx, val_idx, args, device)
        pred_df = predict_hierarchical(hierarchical_models, x, meta, eval_idx, device)
        for col in pred_df.columns:
            meta.loc[pred_df.index, col] = pred_df[col]
        metrics.update(
            {
                "task_model": serializable_model_info(hierarchical_models["task_model"]),
                "step_models": {
                    task: serializable_model_info(model_info)
                    for task, model_info in hierarchical_models["step_models"].items()
                },
                "val": evaluate_hierarchical(meta, "val") if len(val_idx) else None,
                "test": evaluate_hierarchical(meta, "test") if len(test_idx) else None,
            }
        )
        model_payload = {
            "hierarchical_models": {
                "task_model": torch_payload_model_info(hierarchical_models["task_model"]),
                "step_models": {
                    task: torch_payload_model_info(model_info)
                    for task, model_info in hierarchical_models["step_models"].items()
                },
            }
        }
        prediction_cols = [
            "participant_id",
            "task_name",
            "video_id",
            "window_start_time",
            "window_end_time",
            "task_target",
            "step_target",
            "target_label",
            "pred_task",
            "pred_step",
            "pred_label",
            "label_purity",
            "split",
        ]

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    meta.loc[eval_idx, prediction_cols].to_csv(out_dir / "predictions.csv", index=False)

    eval_meta = meta.loc[eval_idx].copy()
    labels = sorted(pd.concat([eval_meta["target_label"], eval_meta["pred_label"]]).dropna().astype(str).unique())
    cm = confusion_matrix(eval_meta["target_label"].astype(str), eval_meta["pred_label"].astype(str), labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out_dir / "confusion_matrix.csv")
    write_confusion_matrix_png(cm, labels, out_dir / "confusion_matrix.png")
    report = classification_report(
        eval_meta["target_label"].astype(str),
        eval_meta["pred_label"].astype(str),
        labels=labels,
        zero_division=0,
    )
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")

    torch.save(
        {
            **model_payload,
            "feature_columns": feature_cols,
            "scaler": scaler,
            "args": vars(args),
            "metrics": metrics,
        },
        out_dir / "model.pt",
    )
    logging.info("Wrote LSTM results: %s", out_dir)


if __name__ == "__main__":
    main()
