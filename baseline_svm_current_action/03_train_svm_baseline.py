#!/usr/bin/env python3
"""Train and evaluate the SVM current-action baseline."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC as SklearnSVC

from common import RESULTS_DIR, TARGET_MODES, configure_logging, read_dataframe, window_feature_path


EXCLUDE_COLUMNS = {
    "participant_id",
    "task_name",
    "video_id",
    "window_start_frame",
    "window_end_frame",
    "window_start_time",
    "window_end_time",
    "target_label",
    "task_target",
    "step_target",
    "label_purity",
    "num_valid_label_frames",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an SVM baseline on window-level ADL features.")
    parser.add_argument("--input", type=Path, default=None, help="Window-level parquet. Defaults from target_mode.")
    parser.add_argument("--target_mode", choices=TARGET_MODES, default="task_step")
    parser.add_argument(
        "--classifier_mode",
        choices=("auto", "single", "hierarchical"),
        default="auto",
        help=(
            "auto uses hierarchical for target_mode=task_step and single otherwise. "
            "single trains one SVM on target_label. hierarchical first predicts task, "
            "then predicts step with a task-specific SVM."
        ),
    )
    parser.add_argument("--split_mode", choices=("video", "participant"), default="participant")
    parser.add_argument("--train_ratio", type=float, default=0.70)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--svm_kernel", choices=("rbf", "linear"), default="rbf")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--gamma", default="scale")
    parser.add_argument(
        "--backend",
        choices=("sklearn", "cuml"),
        default="sklearn",
        help="sklearn uses CPU SVC; cuml uses RAPIDS GPU SVC.",
    )
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap for training rows only.")
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in EXCLUDE_COLUMNS and is_numeric_dtype(df[col])]


def split_by_group(
    df: pd.DataFrame,
    mode: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
) -> pd.Series:
    group_table = build_group_split_table(
        df,
        mode,
        train_ratio,
        val_ratio,
        test_ratio,
        random_state,
    )
    group_col = "participant_id" if mode == "participant" else "video_id"
    split_lookup = dict(zip(group_table[group_col], group_table["split"]))

    return df[group_col].astype(str).map(split_lookup).fillna("test").astype("object")


def build_group_split_table(
    df: pd.DataFrame,
    mode: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_state: int,
) -> pd.DataFrame:
    group_col = "participant_id" if mode == "participant" else "video_id"
    if group_col not in df.columns:
        raise SystemExit(f"Cannot use split_mode={mode}; missing column {group_col}.")
    ratio_sum = train_ratio + val_ratio + test_ratio
    if ratio_sum <= 0:
        raise SystemExit("Split ratios must sum to a positive value.")

    groups = np.asarray(sorted(df[group_col].dropna().astype(str).unique()), dtype=object)
    if len(groups) < 3:
        raise SystemExit(f"Need at least 3 unique {group_col} groups for train/val/test split; found {len(groups)}.")

    rng = np.random.default_rng(random_state)
    groups = rng.permutation(groups)
    n_train = max(1, int(round(len(groups) * train_ratio / ratio_sum)))
    n_val = max(1, int(round(len(groups) * val_ratio / ratio_sum)))
    if n_train + n_val >= len(groups):
        n_train = max(1, len(groups) - 2)
        n_val = 1

    rows = []
    for order, group in enumerate(groups):
        if order < n_train:
            split = "train"
        elif order < n_train + n_val:
            split = "val"
        else:
            split = "test"
        rows.append({group_col: str(group), "split": split, "shuffle_order": int(order)})
    return pd.DataFrame(rows)


def sample_training(df: pd.DataFrame, max_samples: int | None, random_state: int) -> pd.DataFrame:
    if max_samples is None or len(df) <= max_samples:
        return df
    return df.sample(n=max_samples, random_state=random_state)


def make_svc(args: argparse.Namespace):
    if args.backend == "sklearn":
        return SklearnSVC(
            kernel=args.svm_kernel,
            C=args.C,
            gamma=args.gamma,
            class_weight="balanced",
        )

    try:
        from cuml.svm import SVC as CuMLSVC
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(
            "RAPIDS cuML is not available. Install cudf/cuml in the object_detection env "
            "or rerun with --backend sklearn."
        ) from exc

    try:
        return CuMLSVC(
            kernel=args.svm_kernel,
            C=args.C,
            gamma=args.gamma,
            class_weight="balanced",
        )
    except TypeError:
        logging.warning("cuML SVC in this environment does not accept class_weight; continuing without it.")
        return CuMLSVC(kernel=args.svm_kernel, C=args.C, gamma=args.gamma)


def make_model(args: argparse.Namespace) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("svc", make_svc(args)),
        ]
    )


def split_task_step_label(label: object) -> tuple[str | None, str | None]:
    if label is None or pd.isna(label):
        return None, None
    text = str(label)
    if "__" not in text:
        return text, None
    task, step = text.split("__", 1)
    return task, step


def add_task_step_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    parsed = df["target_label"].map(split_task_step_label)
    df["task_target"] = parsed.map(lambda item: item[0])
    df["step_target"] = parsed.map(lambda item: item[1])
    return df[df["task_target"].notna() & df["step_target"].notna()].copy()


def predictions_to_numpy(predictions: object) -> np.ndarray:
    if hasattr(predictions, "to_numpy"):
        return np.asarray(predictions.to_numpy())
    try:
        import cupy as cp

        if isinstance(predictions, cp.ndarray):
            return cp.asnumpy(predictions)
    except Exception:  # noqa: BLE001
        pass
    return np.asarray(predictions)


def predict_labels(model: Pipeline, label_encoder: LabelEncoder, df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    pred_encoded = predictions_to_numpy(model.predict(df[feature_cols]))
    pred_encoded = np.rint(pred_encoded).astype(int)
    return label_encoder.inverse_transform(pred_encoded)


def train_label_model(
    train_df: pd.DataFrame,
    label_col: str,
    feature_cols: list[str],
    args: argparse.Namespace,
) -> dict[str, object]:
    labels = train_df[label_col].astype(str)
    unique_labels = sorted(labels.unique())
    if len(unique_labels) == 1:
        return {
            "kind": "constant",
            "label": unique_labels[0],
            "classes": unique_labels,
            "train_rows": int(len(train_df)),
        }

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(labels)
    model = make_model(args)
    model.fit(train_df[feature_cols], y_train)
    return {
        "kind": "svm",
        "model": model,
        "label_encoder": label_encoder,
        "classes": label_encoder.classes_.tolist(),
        "train_rows": int(len(train_df)),
    }


def predict_label_model(model_info: dict[str, object], df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    if df.empty:
        return np.asarray([], dtype=object)
    if model_info["kind"] == "constant":
        return np.asarray([model_info["label"]] * len(df), dtype=object)
    return predict_labels(
        model_info["model"],
        model_info["label_encoder"],
        df,
        feature_cols,
    )


def evaluate(
    model: Pipeline,
    label_encoder: LabelEncoder,
    df: pd.DataFrame,
    feature_cols: list[str],
    split_name: str,
) -> dict[str, object]:
    y_true = df["target_label"].astype(str)
    y_pred = predict_labels(model, label_encoder, df, feature_cols)
    return {
        "split": split_name,
        "n_samples": int(len(df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


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
    feature_cols: list[str],
    args: argparse.Namespace,
) -> dict[str, object]:
    task_model = train_label_model(train_df, "task_target", feature_cols, args)
    step_models: dict[str, dict[str, object]] = {}
    for task_label, task_df in train_df.groupby("task_target", sort=True):
        step_models[str(task_label)] = train_label_model(task_df, "step_target", feature_cols, args)
    return {
        "task_model": task_model,
        "step_models": step_models,
    }


def predict_hierarchical(
    models: dict[str, object],
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    predictions = pd.DataFrame(index=df.index)
    if df.empty:
        predictions["pred_task"] = []
        predictions["pred_step"] = []
        predictions["pred_label"] = []
        return predictions

    pred_task = predict_label_model(models["task_model"], df, feature_cols).astype(str)
    predictions["pred_task"] = pred_task
    predictions["pred_step"] = pd.NA
    step_models: dict[str, dict[str, object]] = models["step_models"]

    for task_label in sorted(set(pred_task)):
        mask = predictions["pred_task"] == task_label
        step_model = step_models.get(task_label)
        if step_model is None:
            predictions.loc[mask, "pred_step"] = pd.NA
            continue
        predictions.loc[mask, "pred_step"] = predict_label_model(step_model, df.loc[mask], feature_cols)

    predictions["pred_label"] = (
        predictions["pred_task"].astype(str) + "__" + predictions["pred_step"].astype(str)
    )
    return predictions


def evaluate_hierarchical_predictions(eval_df: pd.DataFrame, split_name: str) -> dict[str, object]:
    out = {
        "task": evaluate_label_columns(eval_df, "task_target", "pred_task", split_name),
        "step": evaluate_label_columns(eval_df, "step_target", "pred_step", split_name),
        "task_step": evaluate_label_columns(eval_df, "target_label", "pred_label", split_name),
    }
    correct_task = eval_df["task_target"].astype(str) == eval_df["pred_task"].astype(str)
    if correct_task.any():
        out["step_given_correct_task"] = evaluate_label_columns(
            eval_df[correct_task],
            "step_target",
            "pred_step",
            split_name,
        )
    else:
        out["step_given_correct_task"] = None
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
        title="SVM Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def result_directory(args: argparse.Namespace) -> Path:
    mode = effective_classifier_mode(args)
    return args.results_dir / f"svm_{args.target_mode}_{mode}_1p5s_stride0p5s_{args.split_mode}_{args.backend}"


def effective_classifier_mode(args: argparse.Namespace) -> str:
    if args.classifier_mode == "auto":
        return "hierarchical" if args.target_mode == "task_step" else "single"
    return args.classifier_mode


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    input_path = args.input or window_feature_path(args.target_mode)
    if not input_path.exists():
        raise SystemExit(f"Input window-level parquet does not exist: {input_path}")
    classifier_mode = effective_classifier_mode(args)
    if classifier_mode == "hierarchical" and args.target_mode != "task_step":
        raise SystemExit("--classifier_mode hierarchical requires --target_mode task_step.")

    df = read_dataframe(input_path)
    if "target_label" not in df.columns:
        raise SystemExit("Input window-level table must contain target_label.")
    df = df[df["target_label"].notna()].copy()
    if classifier_mode == "hierarchical":
        df = add_task_step_targets(df)
    feature_cols = numeric_feature_columns(df)
    if not feature_cols:
        raise SystemExit("No numeric SVM feature columns found.")

    group_split_table = build_group_split_table(
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
        raise SystemExit("Training split has fewer than two classes; SVM cannot be trained.")
    if classifier_mode == "hierarchical" and train_df["task_target"].nunique() < 2:
        raise SystemExit("Training split has fewer than two task classes; hierarchical task SVM cannot be trained.")

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
    metrics = {
        "input": str(input_path),
        "target_mode": args.target_mode,
        "classifier_mode": classifier_mode,
        "requested_classifier_mode": args.classifier_mode,
        "split_mode": args.split_mode,
        "backend": args.backend,
        "svm_kernel": args.svm_kernel,
        "C": args.C,
        "gamma": args.gamma,
        "random_state": args.random_state,
        "feature_count": len(feature_cols),
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_groups": int(df.loc[df["split"] == "train", group_col].nunique()),
        "val_groups": int(df.loc[df["split"] == "val", group_col].nunique()),
        "test_groups": int(df.loc[df["split"] == "test", group_col].nunique()),
        "split_groups_csv": str(out_dir / "split_groups.csv"),
        "shuffled_groups": group_split_table.to_dict(orient="records"),
    }
    eval_df = pd.concat([val_df, test_df], ignore_index=True)

    if classifier_mode == "single":
        label_encoder = LabelEncoder()
        y_train = label_encoder.fit_transform(train_df["target_label"].astype(str))
        model = make_model(args)
        model.fit(train_df[feature_cols], y_train)
        metrics.update(
            {
                "classes": label_encoder.classes_.tolist(),
                "val": evaluate(model, label_encoder, val_df, feature_cols, "val") if not val_df.empty else None,
                "test": evaluate(model, label_encoder, test_df, feature_cols, "test") if not test_df.empty else None,
            }
        )
        eval_df["pred_label"] = predict_labels(model, label_encoder, eval_df, feature_cols) if not eval_df.empty else []
        model_payload = {
            "model": model,
            "label_encoder": label_encoder,
            "feature_columns": feature_cols,
            "args": vars(args),
        }
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
        hierarchical_models = train_hierarchical_models(train_df, feature_cols, args)
        pred_df = predict_hierarchical(hierarchical_models, eval_df, feature_cols)
        eval_df = pd.concat([eval_df, pred_df.reset_index(drop=True)], axis=1)
        metrics.update(
            {
                "task_classes": hierarchical_models["task_model"]["classes"],
                "step_models": {
                    task: {
                        "kind": model_info["kind"],
                        "classes": model_info["classes"],
                        "train_rows": model_info["train_rows"],
                    }
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
            "hierarchical_models": hierarchical_models,
            "feature_columns": feature_cols,
            "args": vars(args),
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
    joblib.dump(model_payload, out_dir / "model.joblib")
    logging.info("Wrote SVM baseline results to %s", out_dir)


if __name__ == "__main__":
    main()
