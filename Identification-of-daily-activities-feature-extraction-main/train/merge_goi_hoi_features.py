#!/usr/bin/env python3
"""Merge frame-level GOI, HOI, and annotation labels with a base table."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


KEYS = ["participant_id", "task_name", "full_id", "video_frame_idx"]
DEFAULT_ANNOTATIONS_DIR = (
    "/home/julius/datasets/seminar_object_detection/"
    "iAssistADL_dataset_export_2026-05-18/annotations"
)
SEGMENT_LABEL_COLUMNS = {
    "start_s": "label_start_s",
    "end_s": "label_end_s",
    "start_frame": "label_start_frame",
    "end_frame": "label_end_frame",
    "task": "label_task",
    "step": "label_step",
    "hand": "label_hand",
    "object": "label_object",
    "target": "label_target",
    "confidence": "label_confidence",
    "notes": "label_notes",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge frame-level base, GOI, HOI, and annotation label tables by "
            "shared keys."
        )
    )
    parser.add_argument(
        "--base-csv",
        help=(
            "Optional frame-level base CSV path. If omitted or missing, a base "
            "table is built from the union of GOI/HOI keys."
        ),
    )
    parser.add_argument("--goi-csv", required=True, help="Frame-level GOI feature CSV path.")
    parser.add_argument("--hoi-csv", required=True, help="Frame-level HOI feature CSV path.")
    parser.add_argument("--output-csv", required=True, help="Output merged CSV path.")
    parser.add_argument(
        "--annotations-dir",
        default=DEFAULT_ANNOTATIONS_DIR,
        help=(
            "Directory containing participant subfolders with *.segments.csv label "
            f"files. Default: {DEFAULT_ANNOTATIONS_DIR}"
        ),
    )
    return parser.parse_args()


def check_keys(df: pd.DataFrame, table_name: str, keys: Iterable[str] = KEYS) -> None:
    missing = [key for key in keys if key not in df.columns]
    if missing:
        raise ValueError(f"{table_name} is missing key columns: {missing}")


def normalize_video_frame_idx(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    result = df.copy()
    numeric_idx = pd.to_numeric(result["video_frame_idx"], errors="coerce")
    invalid_count = int(numeric_idx.isna().sum())

    if invalid_count:
        print(
            f"Warning: {table_name} has {invalid_count} rows with invalid "
            "video_frame_idx; dropping them."
        )

    result = result.loc[numeric_idx.notna()].copy()
    result["video_frame_idx"] = numeric_idx.loc[numeric_idx.notna()].round().astype(int)
    return result


def drop_duplicate_keys(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    duplicate_count = int(df.duplicated(KEYS).sum())
    if duplicate_count > 0:
        print(
            f"Warning: {table_name} has {duplicate_count} duplicate key rows; "
            "keeping the first occurrence."
        )
        return df.drop_duplicates(KEYS, keep="first").copy()
    return df


def load_and_prepare_csv(path: str, table_name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    check_keys(df, table_name)
    df = normalize_video_frame_idx(df, table_name)
    df = drop_duplicate_keys(df, table_name)
    return df


def build_base_from_feature_keys(goi: pd.DataFrame, hoi: pd.DataFrame) -> pd.DataFrame:
    base = pd.concat([goi[KEYS], hoi[KEYS]], ignore_index=True)
    base = drop_duplicate_keys(base, "auto_base")
    return base


def load_or_build_base(base_csv: str | None, goi: pd.DataFrame, hoi: pd.DataFrame) -> pd.DataFrame:
    if base_csv:
        base_path = Path(base_csv)
        if base_path.exists():
            return load_and_prepare_csv(base_csv, "base_csv")
        print(
            f"Warning: base_csv does not exist: {base_path}. "
            "Building base table from GOI/HOI keys."
        )
    else:
        print("Warning: --base-csv was not provided. Building base table from GOI/HOI keys.")

    return build_base_from_feature_keys(goi, hoi)


def prefix_feature_columns(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    rename_map = {
        column: f"{prefix}{column}"
        for column in df.columns
        if column not in KEYS
    }
    return df.rename(columns=rename_map)


def full_id_to_task_name(full_id: str, participant_id: str) -> str:
    prefix = f"{participant_id}_"
    if full_id.startswith(prefix):
        return full_id[len(prefix):]
    return full_id


def load_annotation_labels(annotations_dir: str) -> pd.DataFrame:
    root = Path(annotations_dir)
    if not root.exists():
        raise ValueError(f"annotations_dir does not exist: {root}")

    rows = []
    segment_paths = sorted(root.glob("*/*.segments.csv"))
    if not segment_paths:
        print(f"Warning: no *.segments.csv files found under {root}")
        return pd.DataFrame(columns=KEYS + list(SEGMENT_LABEL_COLUMNS.values()))

    for segment_path in segment_paths:
        participant_id = segment_path.parent.name
        full_id = segment_path.name.removesuffix(".segments.csv")
        task_name = full_id_to_task_name(full_id, participant_id)

        segments = pd.read_csv(segment_path)
        required = ["start_frame", "end_frame"]
        missing = [column for column in required if column not in segments.columns]
        if missing:
            raise ValueError(f"{segment_path} is missing columns: {missing}")

        segments["start_frame"] = pd.to_numeric(
            segments["start_frame"], errors="coerce"
        ).round()
        segments["end_frame"] = pd.to_numeric(
            segments["end_frame"], errors="coerce"
        ).round()
        segments = segments.dropna(subset=["start_frame", "end_frame"]).copy()
        segments["start_frame"] = segments["start_frame"].astype(int)
        segments["end_frame"] = segments["end_frame"].astype(int)

        for segment in segments.to_dict("records"):
            start_frame = segment["start_frame"]
            end_frame = segment["end_frame"]
            if end_frame <= start_frame:
                continue

            label_values = {
                output_column: segment.get(input_column)
                for input_column, output_column in SEGMENT_LABEL_COLUMNS.items()
            }
            for frame_idx in range(start_frame, end_frame):
                rows.append(
                    {
                        "participant_id": participant_id,
                        "task_name": task_name,
                        "full_id": full_id,
                        "video_frame_idx": frame_idx,
                        **label_values,
                    }
                )

    labels = pd.DataFrame(rows)
    if labels.empty:
        return pd.DataFrame(columns=KEYS + list(SEGMENT_LABEL_COLUMNS.values()))

    labels = drop_duplicate_keys(labels, "annotation_labels")
    return labels


def missing_ratio(df: pd.DataFrame, columns: list[str]) -> float:
    if not columns:
        return 0.0
    missing_cells = int(df[columns].isna().sum().sum())
    total_cells = len(df) * len(columns)
    if total_cells == 0:
        return 0.0
    return missing_cells / total_cells


def fill_numeric_feature_nans(
    df: pd.DataFrame, feature_columns: Iterable[str]
) -> pd.DataFrame:
    result = df.copy()
    numeric_columns = set(result.select_dtypes(include="number").columns.tolist())
    fill_columns = [
        column
        for column in feature_columns
        if column in result.columns and column in numeric_columns
    ]
    if fill_columns:
        result[fill_columns] = result[fill_columns].fillna(0)
    return result


def print_summary(
    base: pd.DataFrame,
    goi: pd.DataFrame,
    hoi: pd.DataFrame,
    labels: pd.DataFrame,
    merged: pd.DataFrame,
    goi_missing_ratio: float,
    hoi_missing_ratio: float,
    label_missing_ratio: float,
    output_path: Path,
) -> None:
    labeled_frame_count = 0
    unlabeled_frame_count = len(merged)
    unique_step_label_count = 0
    step_label_top30 = pd.Series(dtype="int64")

    if "step_label" in merged.columns:
        step_label = merged["step_label"]
        labeled_frame_count = int(step_label.notna().sum())
        unlabeled_frame_count = int(step_label.isna().sum())
        unique_step_label_count = int(step_label.nunique(dropna=True))
        step_label_top30 = step_label.value_counts(dropna=True).head(30)

    print("Summary:")
    print(f"  base rows/cols: {base.shape[0]}/{base.shape[1]}")
    print(f"  GOI rows/cols: {goi.shape[0]}/{goi.shape[1]}")
    print(f"  HOI rows/cols: {hoi.shape[0]}/{hoi.shape[1]}")
    print(f"  label rows/cols: {labels.shape[0]}/{labels.shape[1]}")
    print(f"  output rows/cols: {merged.shape[0]}/{merged.shape[1]}")
    print(f"  GOI feature missing ratio: {goi_missing_ratio:.6f}")
    print(f"  HOI feature missing ratio: {hoi_missing_ratio:.6f}")
    print(f"  label missing ratio: {label_missing_ratio:.6f}")
    print(f"  labeled frame count: {labeled_frame_count}")
    print(f"  unlabeled frame count: {unlabeled_frame_count}")
    print(f"  unique step_label count: {unique_step_label_count}")
    print("  step_label distribution top 30:")
    if step_label_top30.empty:
        print("    <empty>")
    else:
        for label, count in step_label_top30.items():
            print(f"    {label}: {count}")
    print(f"  output path: {output_path}")
    print(f"  first 40 column names: {merged.columns[:40].tolist()}")


def main() -> None:
    args = parse_args()

    goi = load_and_prepare_csv(args.goi_csv, "goi_csv")
    hoi = load_and_prepare_csv(args.hoi_csv, "hoi_csv")
    base = load_or_build_base(args.base_csv, goi, hoi)
    labels = load_annotation_labels(args.annotations_dir)

    goi = prefix_feature_columns(goi, "goi_")
    hoi = prefix_feature_columns(hoi, "hoi_")

    goi_feature_columns = [column for column in goi.columns if column not in KEYS]
    hoi_feature_columns = [column for column in hoi.columns if column not in KEYS]
    label_columns = [column for column in labels.columns if column not in KEYS]

    merged = base.merge(goi, on=KEYS, how="left")
    merged = merged.merge(hoi, on=KEYS, how="left")
    merged = merged.merge(labels, on=KEYS, how="left")
    merged = merged.loc[:, ~merged.columns.duplicated()].copy()

    if "label_step" in merged.columns and "step_label" not in merged.columns:
        merged["step_label"] = merged["label_step"]

    if (
        "label_task" in merged.columns
        and "task_label_from_annotation" not in merged.columns
    ):
        merged["task_label_from_annotation"] = merged["label_task"]

    goi_missing = missing_ratio(merged, goi_feature_columns)
    hoi_missing = missing_ratio(merged, hoi_feature_columns)
    label_missing = missing_ratio(merged, label_columns)

    merged = fill_numeric_feature_nans(
        merged, goi_feature_columns + hoi_feature_columns
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)

    print_summary(
        base,
        goi,
        hoi,
        labels,
        merged,
        goi_missing,
        hoi_missing,
        label_missing,
        output_path,
    )


if __name__ == "__main__":
    main()
