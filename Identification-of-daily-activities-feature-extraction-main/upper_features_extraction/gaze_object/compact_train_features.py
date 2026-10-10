#!/usr/bin/env python3
"""Build compact gaze-object training features from probabilistic heatmap output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from upper_features_extraction import config

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


MERGE_KEY_COLUMNS = [
    "participant_id",
    "task_name",
    "full_id",
    "video_frame_idx",
]
SOURCE_FEATURE_COLUMNS = [
    "valid_gaze_sample_count",
    "yolo_detection_count",
    "best_class",
    "best_object_probability",
    "best_gaze_object_dx_norm",
    "best_gaze_object_dy_norm",
    "best_gaze_object_dist_norm",
]
NUMERIC_FEATURE_COLUMNS = [
    "best_object_probability",
    "best_gaze_object_dx_norm",
    "best_gaze_object_dy_norm",
    "best_gaze_object_dist_norm",
]
NO_VALID_GAZE_OBJECT_CLASS = "NO_VALID_GAZE_OBJECT"


def read_probabilistic_heatmap_index(index_path: Path) -> pd.DataFrame:
    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(
            f"Missing probabilistic heatmap frame index: {index_path}. "
            "Run upper_features_extraction.gaze_object.probabilistic_heatmap first."
        )

    index_df = pd.read_csv(index_path)
    required_columns = ["participant_id", "task_name", "full_id", "output_csv_path"]
    missing = [col for col in required_columns if col not in index_df.columns]
    if missing:
        raise ValueError(
            f"Probabilistic heatmap frame index is missing required columns: {missing}"
        )
    return index_df


def normalize_class_value(value: object) -> str:
    if pd.isna(value):
        return NO_VALID_GAZE_OBJECT_CLASS
    text = str(value).strip()
    if text == "":
        return NO_VALID_GAZE_OBJECT_CLASS
    try:
        numeric = float(text)
    except ValueError:
        return text
    if np.isfinite(numeric) and numeric.is_integer():
        return str(int(numeric))
    return text


def load_record_features(record: dict) -> pd.DataFrame:
    frame_path = Path(record["output_csv_path"])
    if not frame_path.exists():
        raise FileNotFoundError(f"Missing frame-level probabilistic heatmap CSV: {frame_path}")

    frame_df = pd.read_csv(frame_path)
    required_columns = ["video_frame_idx", *SOURCE_FEATURE_COLUMNS]
    missing = [col for col in required_columns if col not in frame_df.columns]
    if missing:
        raise ValueError(f"{frame_path} is missing required columns: {missing}")

    out = frame_df[["video_frame_idx", *SOURCE_FEATURE_COLUMNS]].copy()
    out.insert(0, "full_id", record["full_id"])
    out.insert(0, "task_name", record["task_name"])
    out.insert(0, "participant_id", record["participant_id"])
    return out


def build_compact_train_features(index_df: pd.DataFrame) -> pd.DataFrame:
    records = list(index_df.to_dict("records"))
    iterator = records
    if tqdm is not None:
        iterator = tqdm(records, desc="Compact gaze-object train features", unit="recording")

    frames = [load_record_features(record) for record in iterator]
    if not frames:
        return pd.DataFrame(columns=MERGE_KEY_COLUMNS)

    features = pd.concat(frames, ignore_index=True, sort=False)
    features["has_valid_gaze_object"] = (
        (pd.to_numeric(features["valid_gaze_sample_count"], errors="coerce").fillna(0) > 0)
        & (pd.to_numeric(features["yolo_detection_count"], errors="coerce").fillna(0) > 0)
    ).astype("int8")

    for col in NUMERIC_FEATURE_COLUMNS:
        features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0.0)

    features["best_object_class"] = features["best_class"].map(normalize_class_value)
    features.loc[
        features["has_valid_gaze_object"] == 0,
        "best_object_class",
    ] = NO_VALID_GAZE_OBJECT_CLASS
    class_dummies = pd.get_dummies(
        features["best_object_class"],
        prefix="best_object_class",
        dtype="int8",
    )

    output = pd.concat(
        [
            features[MERGE_KEY_COLUMNS],
            class_dummies,
            features[
                [
                    "best_object_probability",
                    "best_gaze_object_dx_norm",
                    "best_gaze_object_dy_norm",
                    "best_gaze_object_dist_norm",
                    "has_valid_gaze_object",
                ]
            ],
        ],
        axis=1,
    )
    return output


def write_compact_train_features(
    index_path: Path = config.PROBABILISTIC_HEATMAP_FRAME_INDEX,
    output_csv: Path = config.GAZE_OBJECT_COMPACT_TRAIN_FEATURES_CSV,
) -> tuple[pd.DataFrame, dict]:
    index_df = read_probabilistic_heatmap_index(index_path)
    features = build_compact_train_features(index_df)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_csv, index=False)

    summary = {
        "index_path": str(index_path),
        "output_csv": str(output_csv),
        "recording_count": int(len(index_df)),
        "row_count": int(len(features)),
        "column_count": int(len(features.columns)),
        "one_hot_class_column_count": int(
            sum(col.startswith("best_object_class_") for col in features.columns)
        ),
    }
    return features, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build compact gaze-object training features from probabilistic heatmap "
            "frame-level outputs."
        )
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=config.PROBABILISTIC_HEATMAP_FRAME_INDEX,
        help="Path to probabilistic_heatmap_frame_index.csv.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=config.GAZE_OBJECT_COMPACT_TRAIN_FEATURES_CSV,
        help="Output compact gaze-object training feature CSV.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    features, summary = write_compact_train_features(
        index_path=args.index_path,
        output_csv=args.output_csv,
    )

    print("Compact gaze-object training features completed.")
    print(json.dumps(summary, indent=2))
    if not features.empty:
        print("\nOutput columns:")
        print(", ".join(features.columns))


if __name__ == "__main__":
    main()
