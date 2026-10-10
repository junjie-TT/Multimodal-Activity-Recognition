#!/usr/bin/env python3
"""Feature importance and source ablation with ExtraTrees for step prediction."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


LABEL_FILE_PATH = Path(
    "/home/julius/datasets/seminar_object_detection/input_train/"
    "goi_hoi_label_merged_features.csv"
)

SOURCE_CONFIGS = {
    "skeleton": {
        "path": Path(
            "/home/julius/datasets/seminar_object_detection/input_train/"
            "skeleton_upper_limb_key_raw_features_video_frame_mean.csv"
        ),
        "prefix": "skeleton__",
    },
    "imu_joint": {
        "path": Path(
            "/home/julius/datasets/seminar_object_detection/input_train/"
            "imu_joint_kinematics_key_raw_features_video_frame_mean.csv"
        ),
        "prefix": "imu_joint__",
    },
    "head_imu": {
        "path": Path(
            "/home/julius/datasets/seminar_object_detection/input_train/"
            "head_imu_key_raw_features_video_frame_mean.csv"
        ),
        "prefix": "head_imu__",
    },
    "hand_object": {
        "path": Path(
            "/home/julius/datasets/seminar_object_detection/input_train/"
            "hand_object_compact_train_features.csv"
        ),
        "prefix": "hand_object__",
    },
    "gaze_object": {
        "path": Path(
            "/home/julius/datasets/seminar_object_detection/input_train/"
            "gaze_object_compact_train_features.csv"
        ),
        "prefix": "gaze_object__",
    },
    "depth_scene": {
        "path": Path(
            "/home/julius/datasets/seminar_object_detection/input_train/"
            "depth_scene_key_raw_features_video_frame_mean.csv"
        ),
        "prefix": "depth_scene__",
    },
}

KEY_COLUMNS = ["participant_id", "task_name", "full_id", "video_frame_idx"]

EXPLICIT_NON_FEATURE_COLUMNS = {
    "participant_id",
    "task_name",
    "full_id",
    "body_id",
    "frame_idx",
    "video_frame_idx",
    "time_s",
    "video_time_s",
    "step_label",
    "label",
    "action_label",
    "task_label",
    "goi_label",
    "hoi_label",
    "task_label_encoded",
    "step_label_encoded",
    "window_start_time_s",
    "window_duration_s",
    "source_csv_path",
    "frame_feature_csv",
}

LEAKAGE_KEYWORDS = ("label", "encoded", "target", "annotation")


class NumericCleaner(BaseEstimator, TransformerMixin):
    """Coerce values to numeric, replace infinities, and fill NaN with train medians."""

    def fit(self, X: Any, y: Any = None) -> "NumericCleaner":
        frame = self._to_frame(X)
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        medians = numeric.median(axis=0, skipna=True).replace([np.inf, -np.inf], np.nan)
        self.fill_values_ = medians.fillna(0.0).astype(float)
        return self

    def transform(self, X: Any) -> np.ndarray:
        frame = self._to_frame(X)
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        numeric = numeric.fillna(self.fill_values_)
        return numeric.to_numpy(dtype=np.float32, copy=False)

    @staticmethod
    def _to_frame(X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        return pd.DataFrame(X)


class CategoricalCleaner(BaseEstimator, TransformerMixin):
    """Fill categorical missing values and keep validation-only categories harmless."""

    def fit(self, X: Any, y: Any = None) -> "CategoricalCleaner":
        return self

    def transform(self, X: Any) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            frame = X.copy()
        else:
            frame = pd.DataFrame(X)
        frame = frame.astype("object").where(pd.notna(frame), "MISSING")
        return frame.astype(str).to_numpy(dtype=object, copy=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train ExtraTrees step classifiers with six feature sources, then save "
            "permutation importance and source-level ablation results."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="/home/julius/datasets/feature_importance_ablation_results",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--max-depth",
        type=int,
        default=20,
        help="Maximum tree depth. Use 0 for sklearn's unlimited-depth default.",
    )
    parser.add_argument("--min-samples-leaf", type=int, default=5)
    parser.add_argument("--max-features", default="sqrt")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="If >0, stratified subsample this many training rows for each model.",
    )
    parser.add_argument("--permutation-repeats", type=int, default=5)
    parser.add_argument("--permutation-scoring", default="macro_f1")
    parser.add_argument(
        "--evaluation-mode",
        choices=["known_task", "direct_step"],
        default="known_task",
        help=(
            "known_task trains one step classifier per true task_name and evaluates "
            "with the true task. direct_step trains one global step classifier."
        ),
    )
    return parser.parse_args()


def ensure_video_frame_idx(df: pd.DataFrame, name: str) -> pd.DataFrame:
    result = df.copy()
    if "video_frame_idx" not in result.columns and "frame_idx" in result.columns:
        result["video_frame_idx"] = result["frame_idx"]
        print(f"{name}: created video_frame_idx from frame_idx.")
    return result


def read_csv_with_frame_idx(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} file not found: {path}")
    df = pd.read_csv(path, low_memory=False)
    df = ensure_video_frame_idx(df, name)
    missing_keys = [column for column in KEY_COLUMNS if column not in df.columns]
    if missing_keys:
        raise ValueError(f"{name} is missing key columns: {missing_keys}")
    return df


def first_non_null(series: pd.Series) -> Any:
    non_null = series.dropna()
    if non_null.empty:
        return np.nan
    return non_null.iloc[0]


def deduplicate_label_file(df: pd.DataFrame) -> pd.DataFrame:
    duplicate_count = int(df.duplicated(KEY_COLUMNS).sum())
    if duplicate_count:
        print(
            f"Warning: label file has {duplicate_count} duplicated key rows; "
            "keeping the first row for each key."
        )
        df = df.drop_duplicates(KEY_COLUMNS, keep="first").copy()
    return df


def deduplicate_feature_source(df: pd.DataFrame, source_name: str) -> pd.DataFrame:
    raw_shape = df.shape
    numeric_cols = []
    for column in df.columns:
        if column in KEY_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[column]) or pd.api.types.is_bool_dtype(df[column]):
            numeric_cols.append(column)
            continue
        coerced = pd.to_numeric(df[column], errors="coerce")
        non_missing = df[column].notna()
        if non_missing.any() and coerced[non_missing].notna().all():
            df[column] = coerced
            numeric_cols.append(column)
    other_cols = [
        column for column in df.columns if column not in KEY_COLUMNS and column not in numeric_cols
    ]
    agg: dict[str, Any] = {column: "mean" for column in numeric_cols}
    agg.update({column: first_non_null for column in other_cols})
    if agg:
        deduped = df.groupby(KEY_COLUMNS, dropna=False, as_index=False).agg(agg)
    else:
        deduped = df.drop_duplicates(KEY_COLUMNS, keep="first").copy()
    print(f"{source_name}: raw shape {raw_shape}, deduplicated shape {deduped.shape}")
    return deduped


def detect_label_column(df: pd.DataFrame) -> str:
    for column in ("step_label", "label", "action_label"):
        if column in df.columns:
            return column
    raise ValueError("No step label column found. Tried: step_label, label, action_label.")


def prefix_feature_columns(df: pd.DataFrame, prefix: str, label_column: str) -> pd.DataFrame:
    rename_map = {}
    label_like = {"step_label", "label", "action_label"}
    for column in df.columns:
        if column in KEY_COLUMNS or column == label_column or column in label_like:
            continue
        rename_map[column] = f"{prefix}{column}"
    renamed = df.rename(columns=rename_map)
    prefixed_columns = [column for column in renamed.columns if column.startswith(prefix)]
    return renamed[KEY_COLUMNS + prefixed_columns].copy()


def load_and_merge_sources() -> tuple[pd.DataFrame, str, dict[str, Any]]:
    label_df = read_csv_with_frame_idx(LABEL_FILE_PATH, "label_file")
    label_shape = label_df.shape
    print(f"Label file shape: {label_shape}")
    label_df = deduplicate_label_file(label_df)
    label_column = detect_label_column(label_df)
    print(f"Using label column: {label_column}")
    print("Label distribution:")
    label_distribution = label_df[label_column].value_counts(dropna=False).to_dict()
    print(pd.Series(label_distribution))

    base_columns = KEY_COLUMNS + [label_column]
    merged = label_df[base_columns].copy()
    source_summaries: list[dict[str, Any]] = []

    for source_name, config in SOURCE_CONFIGS.items():
        path = config["path"]
        prefix = config["prefix"]
        source_df = read_csv_with_frame_idx(path, source_name)
        raw_shape = source_df.shape
        print(f"{source_name}: loaded shape {raw_shape}")
        source_df = deduplicate_feature_source(source_df, source_name)
        deduplicated_shape = source_df.shape
        source_df = prefix_feature_columns(source_df, prefix, label_column)
        prefixed_feature_columns = [
            column for column in source_df.columns if column.startswith(prefix)
        ]
        before_merge_rows = len(merged)
        merged = merged.merge(source_df, on=KEY_COLUMNS, how="left", validate="many_to_one")
        matched_mask = merged[prefixed_feature_columns].notna().any(axis=1)
        matched_rows = int(matched_mask.sum())
        missing_rate = 1.0 - (matched_rows / before_merge_rows if before_merge_rows else 0.0)
        print(f"After merging {source_name}: shape {merged.shape}")
        source_summaries.append(
            {
                "source_name": source_name,
                "path": str(path),
                "raw_shape": list(raw_shape),
                "deduplicated_shape": list(deduplicated_shape),
                "num_prefixed_feature_columns": int(len(prefixed_feature_columns)),
                "num_matched_rows_after_merge": matched_rows,
                "missing_rate_after_merge": float(missing_rate),
            }
        )

    before_dropna = len(merged)
    label_valid = merged[label_column].notna() & (merged[label_column].astype(str).str.strip() != "")
    merged = merged.loc[label_valid].reset_index(drop=True)
    after_dropna = len(merged)
    print(f"Merged rows before dropping missing labels: {before_dropna}")
    print(f"Merged rows after dropping missing labels: {after_dropna}")
    print(f"Final merged shape: {merged.shape}")

    summary = {
        "label_file_path": str(LABEL_FILE_PATH),
        "label_file_shape": list(label_shape),
        "label_column": label_column,
        "label_distribution": {str(k): int(v) for k, v in label_distribution.items()},
        "merged_rows_before_dropna_label": int(before_dropna),
        "merged_rows_after_dropna_label": int(after_dropna),
        "sources": source_summaries,
    }
    return merged, label_column, summary


def source_for_feature(feature: str) -> str:
    for source_name, config in SOURCE_CONFIGS.items():
        if feature.startswith(config["prefix"]):
            return source_name
    return "unknown"


def is_leakage_column(column: str) -> bool:
    raw_name = column
    for config in SOURCE_CONFIGS.values():
        prefix = config["prefix"]
        if raw_name.startswith(prefix):
            raw_name = raw_name[len(prefix) :]
            break
    lowered = raw_name.lower()
    return any(keyword in lowered for keyword in LEAKAGE_KEYWORDS)


def select_candidate_features(
    df: pd.DataFrame,
    label_column: str,
) -> tuple[list[str], list[str], list[str], dict[str, int]]:
    excluded = []
    features = []
    for column in df.columns:
        raw_column = column
        for config in SOURCE_CONFIGS.values():
            prefix = config["prefix"]
            if raw_column.startswith(prefix):
                raw_column = raw_column[len(prefix) :]
                break
        if (
            column == label_column
            or raw_column in EXPLICIT_NON_FEATURE_COLUMNS
            or column in EXPLICIT_NON_FEATURE_COLUMNS
            or is_leakage_column(column)
        ):
            excluded.append(column)
            continue
        if source_for_feature(column) == "unknown":
            excluded.append(column)
            continue
        features.append(column)

    dropped_all_nan = []
    kept_features = []
    for column in features:
        numeric = pd.to_numeric(df[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if numeric.isna().all() and df[column].isna().all():
            dropped_all_nan.append(column)
            continue
        kept_features.append(column)

    source_counts = {
        source_name: int(sum(source_for_feature(column) == source_name for column in kept_features))
        for source_name in SOURCE_CONFIGS
    }
    for source_name, count in source_counts.items():
        if count == 0:
            print(f"Warning: source {source_name} has no usable feature columns.")
    print(f"Selected feature columns: {len(kept_features)}")
    print(f"Excluded columns: {len(excluded)}")
    print(f"Dropped all-NaN columns: {len(dropped_all_nan)}")
    return kept_features, excluded, dropped_all_nan, source_counts


def split_train_validation(
    X: pd.DataFrame,
    y: pd.Series,
    task: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series, bool]:
    stratify_labels = task.astype(str) + "||" + y.astype(str)
    counts = stratify_labels.value_counts()
    stratify = stratify_labels if len(counts) > 1 and counts.min() >= 2 else None
    used_stratify = stratify is not None
    if not used_stratify:
        print(
            "Warning: cannot stratify split by task/step because at least one "
            "task-step class has fewer than 2 rows."
        )
    try:
        X_train, X_val, y_train, y_val, task_train, task_val = train_test_split(
            X,
            y,
            task,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError as exc:
        print(f"Warning: stratified split failed ({exc}); falling back to random split.")
        X_train, X_val, y_train, y_val, task_train, task_val = train_test_split(
            X,
            y,
            task,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=None,
        )
        used_stratify = False
    print(f"Train samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Used stratify: {used_stratify}")
    return X_train, X_val, y_train, y_val, task_train, task_val, used_stratify


def infer_column_types(df: pd.DataFrame, columns: list[str]) -> tuple[list[str], list[str]]:
    numeric_columns = []
    categorical_columns = []
    for column in columns:
        series = df[column]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(column)
            continue
        coerced = pd.to_numeric(series, errors="coerce")
        non_missing = series.notna()
        if non_missing.any() and coerced[non_missing].notna().all():
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)
    return numeric_columns, categorical_columns


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def make_model(
    numeric_columns: list[str],
    categorical_columns: list[str],
    args: argparse.Namespace,
) -> Pipeline:
    transformers = []
    if numeric_columns:
        transformers.append(("numeric", NumericCleaner(), numeric_columns))
    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                ("cleaner", CategoricalCleaner()),
                ("onehot", make_one_hot_encoder()),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,
    )
    classifier = ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        max_depth=None if args.max_depth == 0 else args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
        class_weight="balanced",
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", classifier)])


def evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def subsample_training_rows(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.Series]:
    if args.max_train_samples <= 0 or len(y_train) <= args.max_train_samples:
        return X_train, y_train

    counts = y_train.value_counts()
    stratify = y_train if len(counts) > 1 and counts.min() >= 2 else None
    if args.max_train_samples < len(counts):
        print(
            "Warning: --max-train-samples is smaller than the number of classes; "
            "using random subsampling."
        )
        stratify = None
    try:
        sampled_X, _, sampled_y, _ = train_test_split(
            X_train,
            y_train,
            train_size=args.max_train_samples,
            random_state=args.random_state,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError as exc:
        print(f"Warning: stratified training subsample failed ({exc}); using random subsampling.")
        sampled_X, _, sampled_y, _ = train_test_split(
            X_train,
            y_train,
            train_size=args.max_train_samples,
            random_state=args.random_state,
            shuffle=True,
            stratify=None,
        )
    print(f"Training subsample: before={len(y_train)}, after={len(sampled_y)}")
    return sampled_X, sampled_y


def train_known_task_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    task_train: pd.Series,
    selected_features: list[str],
    numeric_columns: list[str],
    categorical_columns: list[str],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], int]:
    models: dict[str, Any] = {}
    used_rows = 0
    train_frame = pd.DataFrame(
        {
            "task_name": task_train.astype(str),
            "step_label": y_train.astype(str),
        },
        index=X_train.index,
    )
    for task_name in sorted(train_frame["task_name"].unique()):
        task_mask = train_frame["task_name"] == task_name
        task_X = X_train.loc[task_mask, selected_features]
        task_y = train_frame.loc[task_mask, "step_label"]
        distribution = task_y.value_counts()
        print(
            f"  known-task model task={task_name!r}: rows={len(task_y)}, "
            f"classes={len(distribution)}"
        )
        if len(distribution) == 1:
            constant_label = str(distribution.index[0])
            models[task_name] = {
                "type": "constant",
                "constant_label": constant_label,
                "train_rows": int(len(task_y)),
                "used_train_rows": int(len(task_y)),
            }
            used_rows += int(len(task_y))
            continue

        X_fit, y_fit = subsample_training_rows(task_X, task_y, args)
        model = make_model(numeric_columns, categorical_columns, args)
        model.fit(X_fit, y_fit)
        models[task_name] = {
            "type": "sklearn",
            "model": model,
            "train_rows": int(len(task_y)),
            "used_train_rows": int(len(y_fit)),
        }
        used_rows += int(len(y_fit))

    fallback_label = str(y_train.astype(str).value_counts().idxmax())
    return {"mode": "known_task", "models": models, "fallback_label": fallback_label}, used_rows


def predict_known_task_models(
    known_task_model: dict[str, Any],
    X_val: pd.DataFrame,
    task_val: pd.Series,
    selected_features: list[str],
) -> np.ndarray:
    predictions = pd.Series(
        known_task_model["fallback_label"],
        index=X_val.index,
        dtype="object",
    )
    task_values = task_val.astype(str)
    for task_name, entry in known_task_model["models"].items():
        task_mask = task_values == task_name
        if not task_mask.any():
            continue
        if entry["type"] == "constant":
            predictions.loc[task_mask] = entry["constant_label"]
        else:
            predictions.loc[task_mask] = entry["model"].predict(
                X_val.loc[task_mask, selected_features]
            )
    return predictions.to_numpy(dtype=object)


def train_and_evaluate_setting(
    setting: str,
    included_sources: list[str],
    removed_sources: list[str],
    feature_columns: list[str],
    X_train_all: pd.DataFrame,
    X_val_all: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    task_train: pd.Series,
    task_val: pd.Series,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Any | None]:
    selected_features = [
        column for column in feature_columns if source_for_feature(column) in included_sources
    ]
    result: dict[str, Any] = {
        "setting": setting,
        "included_sources": ",".join(included_sources),
        "removed_sources": ",".join(removed_sources),
        "accuracy": np.nan,
        "macro_f1": np.nan,
        "weighted_f1": np.nan,
        "number_of_features": int(len(selected_features)),
        "number_of_train_samples": int(len(y_train)),
        "original_number_of_train_samples": int(len(y_train)),
        "number_of_val_samples": int(len(y_val)),
    }
    if not selected_features:
        print(f"Warning: skipping {setting}; no usable features.")
        return result, None

    numeric_columns, categorical_columns = infer_column_types(X_train_all, selected_features)
    print(
        f"{setting}: numeric_columns={len(numeric_columns)}, "
        f"categorical_columns={len(categorical_columns)}"
    )
    if args.evaluation_mode == "known_task":
        print(f"{setting}: evaluation_mode=known_task/oracle_task")
        model, used_rows = train_known_task_models(
            X_train_all,
            y_train,
            task_train,
            selected_features,
            numeric_columns,
            categorical_columns,
            args,
        )
        result["number_of_train_samples"] = int(used_rows)
        y_pred = predict_known_task_models(model, X_val_all, task_val, selected_features)
    else:
        print(f"{setting}: evaluation_mode=direct_step")
        model = make_model(numeric_columns, categorical_columns, args)
        X_fit, y_fit = subsample_training_rows(X_train_all[selected_features], y_train, args)
        result["number_of_train_samples"] = int(len(y_fit))
        model.fit(X_fit, y_fit)
        y_pred = model.predict(X_val_all[selected_features])
    result.update(evaluate_predictions(y_val, y_pred))
    print(
        f"{setting}: accuracy={result['accuracy']:.4f}, "
        f"macro_f1={result['macro_f1']:.4f}, weighted_f1={result['weighted_f1']:.4f}, "
        f"features={len(selected_features)}"
    )
    return result, model


def ablation_settings(source_column_counts: dict[str, int]) -> list[tuple[str, list[str], list[str]]]:
    available_sources = [
        source_name for source_name in SOURCE_CONFIGS if source_column_counts.get(source_name, 0) > 0
    ]
    settings = [("all_sources", available_sources, [])]
    for source_name in SOURCE_CONFIGS:
        if source_name not in available_sources:
            print(f"Warning: remove_{source_name} will be recorded as NaN; source has no features.")
            settings.append((f"remove_{source_name}", [], [source_name]))
        else:
            included = [source for source in available_sources if source != source_name]
            settings.append((f"remove_{source_name}", included, [source_name]))
    for source_name in SOURCE_CONFIGS:
        if source_name not in available_sources:
            print(f"Warning: only_{source_name} will be recorded as NaN; source has no features.")
            settings.append((f"only_{source_name}", [], []))
        else:
            removed = [source for source in available_sources if source != source_name]
            settings.append((f"only_{source_name}", [source_name], removed))
    return settings


def predict_model(
    model: Any,
    X: pd.DataFrame,
    task: pd.Series,
    feature_columns: list[str],
    args: argparse.Namespace,
) -> np.ndarray:
    if args.evaluation_mode == "known_task":
        return predict_known_task_models(model, X, task, feature_columns)
    return model.predict(X[feature_columns])


def score_predictions(y_true: pd.Series, y_pred: np.ndarray, scoring: str) -> float:
    if scoring in {"macro_f1", "f1_macro"}:
        return float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    if scoring == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if scoring in {"weighted_f1", "f1_weighted"}:
        return float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    raise ValueError(
        f"Unsupported permutation scoring {scoring!r}. "
        "Use macro_f1, weighted_f1, or accuracy."
    )


def save_permutation_importance(
    model: Any,
    feature_columns: list[str],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    task_val: pd.Series,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[Path, Path]:
    print(
        "Computing permutation importance on validation set with "
        f"scoring={args.permutation_scoring}."
    )
    rng = np.random.default_rng(args.random_state)
    baseline_pred = predict_model(model, X_val, task_val, feature_columns, args)
    baseline_score = score_predictions(y_val, baseline_pred, args.permutation_scoring)
    importances = np.zeros((len(feature_columns), args.permutation_repeats), dtype=np.float64)
    for feature_idx, feature in enumerate(feature_columns):
        for repeat_idx in range(args.permutation_repeats):
            X_permuted = X_val[feature_columns].copy()
            X_permuted[feature] = rng.permutation(X_permuted[feature].to_numpy())
            permuted_pred = predict_model(model, X_permuted, task_val, feature_columns, args)
            permuted_score = score_predictions(y_val, permuted_pred, args.permutation_scoring)
            importances[feature_idx, repeat_idx] = baseline_score - permuted_score
        if (feature_idx + 1) % 25 == 0 or feature_idx + 1 == len(feature_columns):
            print(f"  permutation importance: {feature_idx + 1}/{len(feature_columns)} features")
    importance_mean = importances.mean(axis=1)
    importance_std = importances.std(axis=1)
    top_df = pd.DataFrame(
        {
            "feature": feature_columns,
            "source": [source_for_feature(column) for column in feature_columns],
            "importance_mean": importance_mean,
            "importance_std": importance_std,
        }
    ).sort_values("importance_mean", ascending=False)
    top_path = output_dir / "top_feature_permutation_importance.csv"
    top_df.to_csv(top_path, index=False)

    source_df = (
        top_df.groupby("source", as_index=False)
        .agg(
            importance_mean_sum=("importance_mean", "sum"),
            importance_mean_abs_sum=("importance_mean", lambda values: values.abs().sum()),
            importance_mean_mean=("importance_mean", "mean"),
            importance_std_mean=("importance_std", "mean"),
            num_features=("feature", "count"),
        )
        .sort_values("importance_mean_abs_sum", ascending=False)
    )
    source_path = output_dir / "source_permutation_importance.csv"
    source_df.to_csv(source_path, index=False)
    print(f"Saved top feature permutation importance: {top_path}")
    print(f"Saved source permutation importance: {source_path}")
    return top_path, source_path


def save_baseline_reports(
    model: Any,
    feature_columns: list[str],
    X_val: pd.DataFrame,
    y_val: pd.Series,
    task_val: pd.Series,
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[Path, Path]:
    y_pred = predict_model(model, X_val, task_val, feature_columns, args)
    labels = sorted(pd.Series(y_val).astype(str).unique().tolist())
    confusion = confusion_matrix(y_val.astype(str), pd.Series(y_pred).astype(str), labels=labels)
    confusion_df = pd.DataFrame(confusion, index=labels, columns=labels)
    confusion_path = output_dir / "confusion_matrix_all_sources.csv"
    confusion_df.to_csv(confusion_path)

    report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
    report_path = output_dir / "classification_report_all_sources.csv"
    pd.DataFrame(report).transpose().to_csv(report_path)
    print(f"Saved all_sources confusion matrix: {confusion_path}")
    print(f"Saved all_sources classification report: {report_path}")
    return confusion_path, report_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    merged, label_column, summary = load_and_merge_sources()
    feature_columns, excluded_columns, dropped_all_nan, source_column_counts = (
        select_candidate_features(merged, label_column)
    )
    if not feature_columns:
        raise ValueError("No usable feature columns remain after exclusions.")

    X = merged[feature_columns].replace([np.inf, -np.inf], np.nan)
    y = merged[label_column].astype(str)
    task = merged["task_name"].astype(str)
    print(f"Evaluation mode: {args.evaluation_mode}")
    if args.evaluation_mode == "known_task":
        print("Known-task mode: true task_name is used only to route step classifiers.")
    X_train, X_val, y_train, y_val, task_train, task_val, used_stratify = split_train_validation(
        X,
        y,
        task,
        args.test_size,
        args.random_state,
    )

    baseline_features = [
        column for column in feature_columns if source_for_feature(column) in SOURCE_CONFIGS
    ]
    settings = ablation_settings(source_column_counts)
    baseline_setting, baseline_included, baseline_removed = settings[0]
    baseline_row, baseline_model = train_and_evaluate_setting(
        baseline_setting,
        baseline_included,
        baseline_removed,
        feature_columns,
        X_train,
        X_val,
        y_train,
        y_val,
        task_train,
        task_val,
        args,
    )
    if baseline_model is None:
        raise ValueError("all_sources baseline could not be trained because no source features exist.")
    print(f"Baseline result: {baseline_row}")

    save_permutation_importance(
        baseline_model,
        baseline_features,
        X_val,
        y_val,
        task_val,
        args,
        output_dir,
    )
    save_baseline_reports(baseline_model, baseline_features, X_val, y_val, task_val, args, output_dir)
    del baseline_model
    gc.collect()

    ablation_rows = [baseline_row]
    for setting, included_sources, removed_sources in settings[1:]:
        row, model = train_and_evaluate_setting(
            setting,
            included_sources,
            removed_sources,
            feature_columns,
            X_train,
            X_val,
            y_train,
            y_val,
            task_train,
            task_val,
            args,
        )
        ablation_rows.append(row)
        del model
        gc.collect()

    ablation_path = output_dir / "source_ablation_results.csv"
    pd.DataFrame(ablation_rows).to_csv(ablation_path, index=False)
    print(f"Saved source ablation results: {ablation_path}")

    summary.update(
        {
            "merged_shape": list(merged.shape),
            "train_samples": int(len(X_train)),
            "val_samples": int(len(X_val)),
            "test_size": float(args.test_size),
            "random_state": int(args.random_state),
            "used_stratify": bool(used_stratify),
            "evaluation_mode": args.evaluation_mode,
            "output_dir": str(output_dir),
            "n_estimators": int(args.n_estimators),
            "max_depth": None if args.max_depth == 0 else int(args.max_depth),
            "min_samples_leaf": int(args.min_samples_leaf),
            "max_features": str(args.max_features),
            "max_train_samples": int(args.max_train_samples),
            "dropped_all_nan_columns": dropped_all_nan,
            "excluded_columns": excluded_columns,
            "source_column_counts": source_column_counts,
        }
    )
    summary_path = output_dir / "merged_feature_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved merged feature summary: {summary_path}")

    print("All output files:")
    for path in (
        ablation_path,
        output_dir / "source_permutation_importance.csv",
        output_dir / "top_feature_permutation_importance.csv",
        output_dir / "confusion_matrix_all_sources.csv",
        output_dir / "classification_report_all_sources.csv",
        summary_path,
    ):
        print(f"  {path}")


if __name__ == "__main__":
    main()
