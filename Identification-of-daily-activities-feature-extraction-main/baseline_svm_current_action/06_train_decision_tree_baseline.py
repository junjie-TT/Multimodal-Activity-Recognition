#!/usr/bin/env python3
"""Train and evaluate a decision-tree current-action baseline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

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
    parser = argparse.ArgumentParser(description="Train a decision-tree baseline on window-level ADL features.")
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
    parser.add_argument("--criterion", choices=("gini", "entropy", "log_loss"), default="gini")
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--min_samples_split", type=int, default=2)
    parser.add_argument("--min_samples_leaf", type=int, default=1)
    parser.add_argument("--max_features", default=None, help="None, sqrt, log2, or a float/int accepted by sklearn.")
    parser.add_argument("--ccp_alpha", type=float, default=0.0)
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap for training rows only.")
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def effective_classifier_mode(args: argparse.Namespace) -> str:
    if args.classifier_mode == "auto":
        return "hierarchical" if args.target_mode == "task_step" else "single"
    return args.classifier_mode


def result_directory(args: argparse.Namespace) -> Path:
    mode = effective_classifier_mode(args)
    return args.results_dir / f"decision_tree_{args.target_mode}_{mode}_1p5s_stride0p5s_{args.split_mode}"


def parse_max_features(value: str | None) -> str | int | float | None:
    if value is None or str(value).lower() in {"none", "null"}:
        return None
    text = str(value)
    if text in {"sqrt", "log2"}:
        return text
    try:
        number = float(text)
    except ValueError as exc:
        raise SystemExit("--max_features must be None, sqrt, log2, an int, or a float.") from exc
    if number.is_integer() and number >= 1:
        return int(number)
    return number


def make_model(args: argparse.Namespace) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "tree",
                DecisionTreeClassifier(
                    criterion=args.criterion,
                    max_depth=args.max_depth,
                    min_samples_split=args.min_samples_split,
                    min_samples_leaf=args.min_samples_leaf,
                    max_features=parse_max_features(args.max_features),
                    ccp_alpha=args.ccp_alpha,
                    class_weight="balanced",
                    random_state=args.random_state,
                ),
            ),
        ]
    )


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


def train_label_model(
    train_df: pd.DataFrame,
    label_col: str,
    feature_cols: list[str],
    args: argparse.Namespace,
) -> dict[str, object]:
    labels = train_df[label_col].astype(str)
    unique_labels = sorted(labels.unique())
    if len(unique_labels) == 1:
        return {"kind": "constant", "label": unique_labels[0], "classes": unique_labels, "train_rows": int(len(train_df))}

    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(labels)
    model = make_model(args)
    model.fit(train_df[feature_cols], y_train)
    tree = model.named_steps["tree"]
    return {
        "kind": "decision_tree",
        "model": model,
        "label_encoder": label_encoder,
        "classes": label_encoder.classes_.tolist(),
        "train_rows": int(len(train_df)),
        "tree_depth": int(tree.get_depth()),
        "tree_leaves": int(tree.get_n_leaves()),
    }


def predict_label_model(model_info: dict[str, object], df: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    if df.empty:
        return np.asarray([], dtype=object)
    if model_info["kind"] == "constant":
        return np.asarray([model_info["label"]] * len(df), dtype=object)
    pred = model_info["model"].predict(df[feature_cols])
    return model_info["label_encoder"].inverse_transform(np.asarray(pred).astype(int))


def evaluate(model_info: dict[str, object], df: pd.DataFrame, feature_cols: list[str], split_name: str) -> dict[str, object]:
    eval_df = df.copy()
    eval_df["pred_label"] = predict_label_model(model_info, eval_df, feature_cols)
    return evaluate_label_columns(eval_df, "target_label", "pred_label", split_name)


def train_hierarchical_models(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    args: argparse.Namespace,
) -> dict[str, object]:
    task_model = train_label_model(train_df, "task_target", feature_cols, args)
    step_models: dict[str, dict[str, object]] = {}
    for task_label, task_df in train_df.groupby("task_target", sort=True):
        step_models[str(task_label)] = train_label_model(task_df, "step_target", feature_cols, args)
    return {"task_model": task_model, "step_models": step_models}


def predict_hierarchical(models: dict[str, object], df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    predictions = pd.DataFrame(index=df.index)
    pred_task = predict_label_model(models["task_model"], df, feature_cols).astype(str)
    predictions["pred_task"] = pred_task
    predictions["pred_step"] = pd.NA
    for task_label in sorted(set(pred_task)):
        mask = predictions["pred_task"] == task_label
        step_model = models["step_models"].get(task_label)
        if step_model is None:
            continue
        predictions.loc[mask, "pred_step"] = predict_label_model(step_model, df.loc[mask], feature_cols)
    predictions["pred_label"] = predictions["pred_task"].astype(str) + "__" + predictions["pred_step"].astype(str)
    return predictions


def evaluate_hierarchical_predictions(eval_df: pd.DataFrame, split_name: str) -> dict[str, object]:
    correct_task = eval_df["task_target"].astype(str) == eval_df["pred_task"].astype(str)
    return {
        "task": evaluate_label_columns(eval_df, "task_target", "pred_task", split_name),
        "step": evaluate_label_columns(eval_df, "step_target", "pred_step", split_name),
        "task_step": evaluate_label_columns(eval_df, "target_label", "pred_label", split_name),
        "step_given_correct_task": evaluate_label_columns(eval_df[correct_task], "step_target", "pred_step", split_name)
        if correct_task.any()
        else None,
    }


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
        title="Decision Tree Confusion Matrix",
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
        "tree_depth": model_info.get("tree_depth"),
        "tree_leaves": model_info.get("tree_leaves"),
    }


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    classifier_mode = effective_classifier_mode(args)
    if classifier_mode == "hierarchical" and args.target_mode != "task_step":
        raise SystemExit("--classifier_mode hierarchical requires --target_mode task_step.")

    input_path = args.input or window_feature_path(args.target_mode)
    if not input_path.exists():
        raise SystemExit(f"Input window-level dataframe does not exist: {input_path}")
    df = read_dataframe(input_path)
    if "target_label" not in df.columns:
        raise SystemExit("Input window-level table must contain target_label.")
    df = df[df["target_label"].notna()].copy()
    if classifier_mode == "hierarchical":
        df = svm_helpers.add_task_step_targets(df)

    feature_cols = svm_helpers.numeric_feature_columns(df)
    if not feature_cols:
        raise SystemExit("No numeric decision-tree feature columns found.")

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

    train_df = svm_helpers.sample_training(df[df["split"] == "train"], args.max_samples, args.random_state)
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    if classifier_mode == "single" and train_df["target_label"].nunique() < 2:
        raise SystemExit("Training split has fewer than two classes; decision tree cannot be trained.")
    if classifier_mode == "hierarchical" and train_df["task_target"].nunique() < 2:
        raise SystemExit("Training split has fewer than two task classes; hierarchical task tree cannot be trained.")

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
        "criterion": args.criterion,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": parse_max_features(args.max_features),
        "ccp_alpha": args.ccp_alpha,
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

    if classifier_mode == "single":
        model_info = train_label_model(train_df, "target_label", feature_cols, args)
        eval_df["pred_label"] = predict_label_model(model_info, eval_df, feature_cols) if not eval_df.empty else []
        metrics.update(
            {
                "classes": model_info["classes"],
                "model": serializable_model_info(model_info),
                "val": evaluate(model_info, val_df, feature_cols, "val") if not val_df.empty else None,
                "test": evaluate(model_info, test_df, feature_cols, "test") if not test_df.empty else None,
            }
        )
        model_payload = {"model": model_info}
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
        model_payload = {"hierarchical_models": hierarchical_models}
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

    joblib.dump(
        {
            **model_payload,
            "feature_columns": feature_cols,
            "args": vars(args),
            "metrics": metrics,
        },
        out_dir / "model.joblib",
    )
    logging.info("Wrote decision-tree results: %s", out_dir)


if __name__ == "__main__":
    main()
