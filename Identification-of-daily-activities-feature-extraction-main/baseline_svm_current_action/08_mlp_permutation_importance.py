#!/usr/bin/env python3
"""Compute permutation feature importance for the MLP current-action baseline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from common import RESULTS_DIR, TARGET_MODES, configure_logging, read_dataframe, window_feature_path


BASELINE_DIR = Path(__file__).resolve().parent


def load_mlp_helpers() -> Any:
    spec = importlib.util.spec_from_file_location("mlp_baseline_helpers", BASELINE_DIR / "05_train_mlp_baseline.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load 05_train_mlp_baseline.py helpers.")
    module = importlib.util.module_from_spec(spec)
    sys.modules["mlp_baseline_helpers"] = module
    spec.loader.exec_module(module)
    return module


mlp_helpers = load_mlp_helpers()
svm_helpers = mlp_helpers.svm_helpers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run permutation importance for the MLP baseline.")
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
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap for training rows only.")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:0.")
    parser.add_argument("--results_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--model_path", type=Path, default=None, help="Existing MLP model.pt. Defaults to the MLP result dir.")
    parser.add_argument("--output_dir", type=Path, default=None, help="Directory for importance outputs.")
    parser.add_argument("--eval_split", choices=("val", "test", "val_test"), default="test")
    parser.add_argument("--n_repeats", type=int, default=5)
    parser.add_argument("--max_eval_samples", type=int, default=None, help="Optional cap for permutation evaluation rows.")
    parser.add_argument("--importance_metric", choices=("accuracy", "macro_f1", "weighted_f1"), default="accuracy")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True, help="Show a stderr progress bar.")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def safe_torch_load(path: Path, device: str) -> dict[str, object]:
    torch, _, _, _ = mlp_helpers.import_torch()
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def restore_mlp_model(
    model_info: dict[str, object],
    feature_count: int,
    device: str,
    hidden_sizes: list[int] | None = None,
) -> dict[str, object]:
    if model_info["kind"] == "constant":
        return model_info

    torch, nn, _, _ = mlp_helpers.import_torch()
    classes = list(model_info["classes"])
    state_dict = model_info["state_dict"]
    if hidden_sizes is None:
        linear_weights = [value for key, value in state_dict.items() if key.endswith(".weight") and value.ndim == 2]
        hidden_sizes = [int(value.shape[0]) for value in linear_weights[:-1]]

    model_cls = mlp_helpers.make_mlp_class(nn)
    model = model_cls(feature_count, hidden_sizes, len(classes), dropout=0.0).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return {
        **model_info,
        "model": model,
        "label_encoder": model_info["label_encoder"],
        "imputer": model_info["imputer"],
        "scaler": model_info["scaler"],
    }


def restore_models(payload: dict[str, object], feature_count: int, classifier_mode: str, device: str) -> object:
    saved_args = payload.get("args", {})
    hidden_sizes = None
    if isinstance(saved_args, dict) and saved_args.get("hidden_sizes"):
        hidden_sizes = mlp_helpers.parse_hidden_sizes(str(saved_args["hidden_sizes"]))
    if classifier_mode == "single":
        return restore_mlp_model(payload["model"], feature_count, device, hidden_sizes)
    saved_models = payload["hierarchical_models"]
    return {
        "task_model": restore_mlp_model(saved_models["task_model"], feature_count, device, hidden_sizes),
        "step_models": {
            task: restore_mlp_model(model_info, feature_count, device, hidden_sizes)
            for task, model_info in saved_models["step_models"].items()
        },
    }


def prepare_dataframe(args: argparse.Namespace, classifier_mode: str) -> tuple[pd.DataFrame, list[str], str, Path]:
    input_path = args.input or window_feature_path(args.target_mode)
    if not input_path.exists():
        raise SystemExit(f"Input window-level dataframe does not exist: {input_path}")
    df = read_dataframe(input_path)
    if "target_label" not in df.columns:
        raise SystemExit("Input window-level table must contain target_label.")
    df = df[df["target_label"].notna()].copy()
    if classifier_mode == "hierarchical":
        df = mlp_helpers.add_task_step_targets(df)

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
    return df, feature_cols, group_col, input_path


def train_models(args: argparse.Namespace, df: pd.DataFrame, feature_cols: list[str], classifier_mode: str, device: str) -> object:
    hidden_sizes = mlp_helpers.parse_hidden_sizes(args.hidden_sizes)
    train_df = mlp_helpers.sample_training(df[df["split"] == "train"], args.max_samples, args.random_state)
    val_df = df[df["split"] == "val"]
    if classifier_mode == "single":
        return mlp_helpers.train_label_model(train_df, val_df, "target_label", feature_cols, args, hidden_sizes, device)
    return mlp_helpers.train_hierarchical_models(train_df, val_df, feature_cols, args, hidden_sizes, device)


def predict(models: object, df: pd.DataFrame, feature_cols: list[str], classifier_mode: str, device: str) -> pd.DataFrame:
    out = df.copy()
    if classifier_mode == "single":
        out["pred_label"] = mlp_helpers.predict_label_model(models, out, feature_cols, device) if not out.empty else []
    else:
        pred_df = mlp_helpers.predict_hierarchical(models, out, feature_cols, device)
        out = pd.concat([out.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
    return out


def score_predictions(eval_df: pd.DataFrame, classifier_mode: str, split_name: str) -> dict[str, object]:
    if classifier_mode == "single":
        return mlp_helpers.evaluate_label_columns(eval_df, "target_label", "pred_label", split_name)
    return mlp_helpers.evaluate_hierarchical_predictions(eval_df, split_name)["task_step"]


def feature_family(feature: str) -> str:
    if "__" in feature:
        return feature.split("__", 1)[0]
    if "_" in feature:
        return feature.split("_", 1)[0]
    return feature


def render_progress(completed: int, total: int, label: str) -> None:
    width = 32
    fraction = completed / max(total, 1)
    filled = int(round(width * fraction))
    bar = "#" * filled + "-" * (width - filled)
    label = label[:70]
    sys.stderr.write(f"\rPermutation Importance [{bar}] {completed}/{total} {fraction:6.1%} {label:<70}")
    sys.stderr.flush()
    if completed >= total:
        sys.stderr.write("\n")
        sys.stderr.flush()


def permutation_importance(
    models: object,
    eval_df: pd.DataFrame,
    feature_cols: list[str],
    classifier_mode: str,
    device: str,
    metric: str,
    n_repeats: int,
    random_state: int,
    progress: bool,
) -> tuple[dict[str, object], pd.DataFrame]:
    baseline_eval = predict(models, eval_df, feature_cols, classifier_mode, device)
    baseline_scores = score_predictions(baseline_eval, classifier_mode, "baseline")
    baseline_value = float(baseline_scores[metric])
    rng = np.random.default_rng(random_state)
    rows = []
    total_steps = len(feature_cols) * n_repeats
    completed_steps = 0

    for feature_index, feature in enumerate(feature_cols, start=1):
        repeat_scores = []
        repeat_drops = []
        for repeat in range(n_repeats):
            permuted = eval_df.copy()
            permuted[feature] = rng.permutation(permuted[feature].to_numpy(copy=True))
            pred_df = predict(models, permuted, feature_cols, classifier_mode, device)
            score = float(score_predictions(pred_df, classifier_mode, f"{feature}_repeat{repeat + 1}")[metric])
            repeat_scores.append(score)
            repeat_drops.append(baseline_value - score)
            completed_steps += 1
            if progress:
                render_progress(completed_steps, total_steps, f"{feature} repeat {repeat + 1}/{n_repeats}")
        logging.info(
            "Permutation %d/%d %s: mean_%s_drop=%.6f",
            feature_index,
            len(feature_cols),
            feature,
            metric,
            float(np.mean(repeat_drops)),
        )
        rows.append(
            {
                "feature": feature,
                "feature_family": feature_family(feature),
                "baseline_score": baseline_value,
                "permuted_score_mean": float(np.mean(repeat_scores)),
                "permuted_score_std": float(np.std(repeat_scores, ddof=1)) if n_repeats > 1 else 0.0,
                "importance_mean": float(np.mean(repeat_drops)),
                "importance_std": float(np.std(repeat_drops, ddof=1)) if n_repeats > 1 else 0.0,
                "n_repeats": int(n_repeats),
            }
        )

    importance = pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)
    return baseline_scores, importance


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    if args.n_repeats < 1:
        raise SystemExit("--n_repeats must be at least 1.")

    classifier_mode = mlp_helpers.effective_classifier_mode(args)
    if classifier_mode == "hierarchical" and args.target_mode != "task_step":
        raise SystemExit("--classifier_mode hierarchical requires --target_mode task_step.")

    mlp_helpers.set_seed(args.random_state)
    torch, _, _, _ = mlp_helpers.import_torch()
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    logging.info("Using device=%s", device)

    df, feature_cols, group_col, input_path = prepare_dataframe(args, classifier_mode)
    out_dir = args.output_dir or (mlp_helpers.result_directory(args) / "permutation_importance")
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = args.model_path or (mlp_helpers.result_directory(args) / "model.pt")
    if model_path.exists():
        logging.info("Loading MLP model: %s", model_path)
        payload = safe_torch_load(model_path, device)
        saved_features = list(payload.get("feature_columns", feature_cols))
        missing = [feature for feature in saved_features if feature not in df.columns]
        if missing:
            raise SystemExit(f"Saved model expects missing features: {missing[:5]}")
        feature_cols = saved_features
        models = restore_models(payload, len(feature_cols), classifier_mode, device)
        model_source = str(model_path)
    else:
        logging.info("No existing model found at %s; training MLP before permutation importance.", model_path)
        models = train_models(args, df, feature_cols, classifier_mode, device)
        model_source = "trained_in_importance_run"

    eval_df = df[df["split"].isin(["val", "test"] if args.eval_split == "val_test" else [args.eval_split])].copy()
    if eval_df.empty:
        raise SystemExit(f"No rows found for eval_split={args.eval_split}.")
    if args.max_eval_samples is not None and len(eval_df) > args.max_eval_samples:
        eval_df = eval_df.sample(n=args.max_eval_samples, random_state=args.random_state).copy()

    logging.info(
        "Running permutation importance: rows=%d, features=%d, repeats=%d, metric=%s",
        len(eval_df),
        len(feature_cols),
        args.n_repeats,
        args.importance_metric,
    )
    baseline_scores, importance = permutation_importance(
        models,
        eval_df,
        feature_cols,
        classifier_mode,
        device,
        args.importance_metric,
        args.n_repeats,
        args.random_state,
        args.progress,
    )

    importance.to_csv(out_dir / "permutation_importance.csv", index=False)
    family_importance = (
        importance.groupby("feature_family", as_index=False)
        .agg(
            importance_mean_sum=("importance_mean", "sum"),
            importance_mean_avg=("importance_mean", "mean"),
            top_feature_importance=("importance_mean", "max"),
            feature_count=("feature", "count"),
        )
        .sort_values("importance_mean_sum", ascending=False)
    )
    family_importance.to_csv(out_dir / "permutation_importance_by_family.csv", index=False)

    summary = {
        "input": str(input_path),
        "model_source": model_source,
        "target_mode": args.target_mode,
        "classifier_mode": classifier_mode,
        "split_mode": args.split_mode,
        "eval_split": args.eval_split,
        "eval_rows": int(len(eval_df)),
        "eval_groups": int(eval_df[group_col].nunique()),
        "feature_count": int(len(feature_cols)),
        "n_repeats": int(args.n_repeats),
        "importance_metric": args.importance_metric,
        "baseline": baseline_scores,
        "top_20_features": importance.head(20).to_dict(orient="records"),
        "outputs": {
            "feature_importance_csv": str(out_dir / "permutation_importance.csv"),
            "family_importance_csv": str(out_dir / "permutation_importance_by_family.csv"),
        },
    }
    with (out_dir / "permutation_importance_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    logging.info("Wrote MLP permutation importance: %s", out_dir)


if __name__ == "__main__":
    main()
