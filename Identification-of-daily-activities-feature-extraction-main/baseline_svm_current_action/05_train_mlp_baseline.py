#!/usr/bin/env python3
"""Train and evaluate an MLP current-action baseline on window statistics."""

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
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from common import RESULTS_DIR, TARGET_MODES, configure_logging, read_dataframe, window_feature_path


BASELINE_DIR = Path(__file__).resolve().parent


def load_svm_helpers() -> Any:
    spec = importlib.util.spec_from_file_location("svm_baseline_helpers", BASELINE_DIR / "03_train_svm_baseline.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load 03_train_svm_baseline.py helpers.")
    module = importlib.util.module_from_spec(spec)
    sys.modules["svm_baseline_helpers"] = module
    spec.loader.exec_module(module)
    return module


svm_helpers = load_svm_helpers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MLP baseline on window-level ADL features.")
    parser.add_argument("--input", type=Path, default=None, help="Window-level dataframe. Defaults from target_mode.")
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
    parser.add_argument("--hidden_sizes", default="512,256", help="Comma-separated MLP hidden layer sizes.")
    parser.add_argument("--dropout", type=float, default=0.30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience on validation macro F1.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap for training rows only.")
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
    return args.results_dir / f"mlp_{args.target_mode}_{mode}_1p5s_stride0p5s_{args.split_mode}"


def parse_hidden_sizes(value: str) -> list[int]:
    sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not sizes:
        raise SystemExit("--hidden_sizes must contain at least one layer size.")
    if any(size <= 0 for size in sizes):
        raise SystemExit("--hidden_sizes must contain positive integers.")
    return sizes


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


def import_torch() -> Any:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "PyTorch is required for MLP training. Install torch in this environment "
            "or run the script from an environment that already has PyTorch."
        ) from exc
    return torch, nn, DataLoader, TensorDataset


def add_task_step_targets(df: pd.DataFrame) -> pd.DataFrame:
    return svm_helpers.add_task_step_targets(df)


def sample_training(df: pd.DataFrame, max_samples: int | None, random_state: int) -> pd.DataFrame:
    return svm_helpers.sample_training(df, max_samples, random_state)


def make_mlp_class(nn: Any) -> type:
    class MLPClassifier(nn.Module):
        def __init__(self, input_size: int, hidden_sizes: list[int], num_classes: int, dropout: float) -> None:
            super().__init__()
            layers: list[Any] = []
            previous = input_size
            for hidden_size in hidden_sizes:
                layers.extend(
                    [
                        nn.Linear(previous, hidden_size),
                        nn.BatchNorm1d(hidden_size),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ]
                )
                previous = hidden_size
            layers.append(nn.Linear(previous, num_classes))
            self.net = nn.Sequential(*layers)

        def forward(self, x: Any) -> Any:
            return self.net(x)

    return MLPClassifier


def compute_class_weights(y: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    return (counts.sum() / (num_classes * counts)).astype(np.float32)


def fit_preprocessor(train_df: pd.DataFrame, feature_cols: list[str]) -> tuple[SimpleImputer, StandardScaler, np.ndarray]:
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    x_train = imputer.fit_transform(train_df[feature_cols])
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    return imputer, scaler, x_train


def transform_features(df: pd.DataFrame, feature_cols: list[str], imputer: SimpleImputer, scaler: StandardScaler) -> np.ndarray:
    return scaler.transform(imputer.transform(df[feature_cols])).astype(np.float32)


def predict_encoded(model: Any, loader: Any, torch: Any, device: str) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for xb, _ in loader:
            logits = model(xb.to(device))
            outputs.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(outputs) if outputs else np.asarray([], dtype=np.int64)


def train_label_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    label_col: str,
    feature_cols: list[str],
    args: argparse.Namespace,
    hidden_sizes: list[int],
    device: str,
) -> dict[str, object]:
    labels = train_df[label_col].astype(str)
    unique_labels = sorted(labels.unique())
    if len(unique_labels) == 1:
        return {"kind": "constant", "label": unique_labels[0], "classes": unique_labels, "train_rows": int(len(train_df))}

    torch, nn, DataLoader, TensorDataset = import_torch()
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(labels).astype(np.int64)
    imputer, scaler, x_train = fit_preprocessor(train_df, feature_cols)

    known_val = val_df[label_col].astype(str).isin(label_encoder.classes_)
    usable_val_df = val_df[known_val]
    x_val = transform_features(usable_val_df, feature_cols, imputer, scaler) if not usable_val_df.empty else np.empty((0, len(feature_cols)), dtype=np.float32)
    y_val = (
        label_encoder.transform(usable_val_df[label_col].astype(str)).astype(np.int64)
        if not usable_val_df.empty
        else np.asarray([], dtype=np.int64)
    )

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = None
    if len(y_val):
        val_ds = TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val))
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model_cls = make_mlp_class(nn)
    model = model_cls(
        input_size=len(feature_cols),
        hidden_sizes=hidden_sizes,
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
        "kind": "mlp",
        "model": model,
        "label_encoder": label_encoder,
        "imputer": imputer,
        "scaler": scaler,
        "classes": label_encoder.classes_.tolist(),
        "train_rows": int(len(train_df)),
        "history": history,
        "best_score": float(best_score),
    }


def predict_label_model(model_info: dict[str, object], df: pd.DataFrame, feature_cols: list[str], device: str) -> np.ndarray:
    if df.empty:
        return np.asarray([], dtype=object)
    if model_info["kind"] == "constant":
        return np.asarray([model_info["label"]] * len(df), dtype=object)

    torch, _, DataLoader, TensorDataset = import_torch()
    x = transform_features(df, feature_cols, model_info["imputer"], model_info["scaler"])
    dummy_y = np.zeros(len(df), dtype=np.int64)
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(dummy_y))
    loader = DataLoader(ds, batch_size=512, shuffle=False)
    pred = predict_encoded(model_info["model"], loader, torch, device)
    return model_info["label_encoder"].inverse_transform(pred)


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


def train_hierarchical_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    args: argparse.Namespace,
    hidden_sizes: list[int],
    device: str,
) -> dict[str, object]:
    task_model = train_label_model(train_df, val_df, "task_target", feature_cols, args, hidden_sizes, device)
    step_models: dict[str, dict[str, object]] = {}
    for task_label, task_df in train_df.groupby("task_target", sort=True):
        task_val_df = val_df[val_df["task_target"].astype(str) == str(task_label)]
        step_models[str(task_label)] = train_label_model(task_df, task_val_df, "step_target", feature_cols, args, hidden_sizes, device)
    return {"task_model": task_model, "step_models": step_models}


def predict_hierarchical(models: dict[str, object], df: pd.DataFrame, feature_cols: list[str], device: str) -> pd.DataFrame:
    predictions = pd.DataFrame(index=df.index)
    pred_task = predict_label_model(models["task_model"], df, feature_cols, device).astype(str)
    predictions["pred_task"] = pred_task
    predictions["pred_step"] = pd.NA
    for task_label in sorted(set(pred_task)):
        mask = predictions["pred_task"] == task_label
        step_model = models["step_models"].get(task_label)
        if step_model is None:
            continue
        predictions.loc[mask, "pred_step"] = predict_label_model(step_model, df.loc[mask], feature_cols, device)
    predictions["pred_label"] = predictions["pred_task"].astype(str) + "__" + predictions["pred_step"].astype(str)
    return predictions


def evaluate_hierarchical_predictions(eval_df: pd.DataFrame, split_name: str) -> dict[str, object]:
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
        title="MLP Confusion Matrix",
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
        "state_dict": {key: value.detach().cpu() for key, value in model_info["model"].state_dict().items()},
        "label_encoder": model_info["label_encoder"],
        "imputer": model_info["imputer"],
        "scaler": model_info["scaler"],
    }


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    set_seed(args.random_state)
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    classifier_mode = effective_classifier_mode(args)
    if classifier_mode == "hierarchical" and args.target_mode != "task_step":
        raise SystemExit("--classifier_mode hierarchical requires --target_mode task_step.")

    torch, _, _, _ = import_torch()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    logging.info("Using device=%s", device)

    input_path = args.input or window_feature_path(args.target_mode)
    if not input_path.exists():
        raise SystemExit(f"Input window-level dataframe does not exist: {input_path}")
    df = read_dataframe(input_path)
    if "target_label" not in df.columns:
        raise SystemExit("Input window-level table must contain target_label.")
    df = df[df["target_label"].notna()].copy()
    if classifier_mode == "hierarchical":
        df = add_task_step_targets(df)

    feature_cols = svm_helpers.numeric_feature_columns(df)
    if not feature_cols:
        raise SystemExit("No numeric MLP feature columns found.")

    group_split_table = svm_helpers.build_group_split_table(
        df,
        args.split_mode,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio,
        args.random_state,
    )
    group_col = "video_id" if args.split_mode == "video" else "participant_id"
    split_lookup = dict(zip(group_split_table[group_col], group_split_table["split"]))
    df["split"] = df[group_col].astype(str).map(split_lookup).fillna("test").astype("object")

    train_df = sample_training(df[df["split"] == "train"], args.max_samples, args.random_state)
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    if classifier_mode == "single" and train_df["target_label"].nunique() < 2:
        raise SystemExit("Training split has fewer than two classes; MLP cannot be trained.")
    if classifier_mode == "hierarchical" and train_df["task_target"].nunique() < 2:
        raise SystemExit("Training split has fewer than two task classes; hierarchical task MLP cannot be trained.")

    logging.info(
        "Training rows=%d, val rows=%d, test rows=%d, features=%d, target classes=%d",
        len(train_df),
        len(val_df),
        len(test_df),
        len(feature_cols),
        train_df["target_label"].nunique(),
    )

    out_dir = result_directory(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    group_split_table.to_csv(out_dir / "split_groups.csv", index=False)
    eval_df = pd.concat([val_df, test_df], ignore_index=True)

    metrics: dict[str, object] = {
        "input": str(input_path),
        "target_mode": args.target_mode,
        "classifier_mode": classifier_mode,
        "requested_classifier_mode": args.classifier_mode,
        "split_mode": args.split_mode,
        "random_state": args.random_state,
        "device": device,
        "feature_count": len(feature_cols),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_groups": int(df.loc[df["split"] == "train", group_col].nunique()),
        "val_groups": int(df.loc[df["split"] == "val", group_col].nunique()),
        "test_groups": int(df.loc[df["split"] == "test", group_col].nunique()),
        "split_groups_csv": str(out_dir / "split_groups.csv"),
        "shuffled_groups": group_split_table.to_dict(orient="records"),
        "mlp": {
            "hidden_sizes": hidden_sizes,
            "dropout": args.dropout,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "patience": args.patience,
        },
    }

    if classifier_mode == "single":
        model_info = train_label_model(train_df, val_df, "target_label", feature_cols, args, hidden_sizes, device)
        eval_df["pred_label"] = predict_label_model(model_info, eval_df, feature_cols, device) if not eval_df.empty else []
        metrics.update(
            {
                "classes": model_info["classes"],
                "model": serializable_model_info(model_info),
                "val": evaluate_label_columns(eval_df[eval_df["split"] == "val"], "target_label", "pred_label", "val")
                if not val_df.empty
                else None,
                "test": evaluate_label_columns(eval_df[eval_df["split"] == "test"], "target_label", "pred_label", "test")
                if not test_df.empty
                else None,
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
        hierarchical_models = train_hierarchical_models(train_df, val_df, feature_cols, args, hidden_sizes, device)
        pred_df = predict_hierarchical(hierarchical_models, eval_df, feature_cols, device)
        eval_df = pd.concat([eval_df, pred_df.reset_index(drop=True)], axis=1)
        metrics.update(
            {
                "task_model": serializable_model_info(hierarchical_models["task_model"]),
                "step_models": {
                    task: serializable_model_info(model_info)
                    for task, model_info in hierarchical_models["step_models"].items()
                },
                "val": evaluate_hierarchical_predictions(eval_df[eval_df["split"] == "val"], "val")
                if not val_df.empty
                else None,
                "test": evaluate_hierarchical_predictions(eval_df[eval_df["split"] == "test"], "test")
                if not test_df.empty
                else None,
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
    eval_df[prediction_cols].to_csv(out_dir / "predictions.csv", index=False)

    labels = sorted(pd.concat([eval_df["target_label"], eval_df["pred_label"]]).dropna().astype(str).unique())
    cm = confusion_matrix(eval_df["target_label"].astype(str), eval_df["pred_label"].astype(str), labels=labels)
    pd.DataFrame(cm, index=labels, columns=labels).to_csv(out_dir / "confusion_matrix.csv")
    write_confusion_matrix_png(cm, labels, out_dir / "confusion_matrix.png")
    report = classification_report(
        eval_df["target_label"].astype(str),
        eval_df["pred_label"].astype(str),
        labels=labels,
        zero_division=0,
    )
    (out_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    torch.save(
        {
            **model_payload,
            "feature_columns": feature_cols,
            "args": vars(args),
            "metrics": metrics,
        },
        out_dir / "model.pt",
    )
    logging.info("Wrote MLP results: %s", out_dir)


if __name__ == "__main__":
    main()
