#!/usr/bin/env python3
"""Aggregate causal frame-level features into SVM-ready window statistics."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from common import (
    DEFAULT_FPS,
    FRAME_LEVEL_PATH,
    INTERMEDIATE_DIR,
    TARGET_MODES,
    configure_logging,
    ensure_parent,
    read_dataframe,
    write_dataframe,
    window_feature_path,
)


EXCLUDE_COLUMNS = {
    "participant_id",
    "task_name",
    "video_id",
    "video_path",
    "filename",
    "frame_id",
    "timestamp",
    "task_label",
    "step_label",
    "task_step_label",
    "is_labeled",
    "start_time",
    "end_time",
}
LABEL_COLUMNS = {
    "task": "task_label",
    "step": "step_label",
    "task_step": "task_step_label",
}
FEATURE_EXCLUDE_TOKENS = (
    "_source_timestamp",
    "timestamp",
    "frame_idx",
    "frame_id",
    "video_frame",
    "detection_idx",
    "body_id",
    "class_id",
    "object_id",
)
FEATURE_EXCLUDE_SUFFIXES = (
    "_time_s",
    "_time_sec",
    "_class",
    "_id",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build causal 1.5s window-level statistical features for SVM."
    )
    parser.add_argument("--input", type=Path, default=FRAME_LEVEL_PATH)
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS)
    parser.add_argument("--window_sec", type=float, default=1.5)
    parser.add_argument("--stride_sec", type=float, default=0.5)
    parser.add_argument("--label_tail_sec", type=float, default=0.5)
    parser.add_argument("--min_label_purity", type=float, default=0.6)
    parser.add_argument("--target_mode", choices=TARGET_MODES, default="task_step")
    parser.add_argument("--keep_low_purity", action="store_true")
    parser.add_argument(
        "--aggregation_mode",
        choices=("fast", "loop"),
        default="fast",
        help="fast uses vectorized rolling statistics; loop keeps the exact original implementation.",
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def is_binary_or_mask(series: pd.Series, name: str) -> bool:
    lname = name.lower()
    if any(token in lname for token in ("mask", "valid", "is_", "has_", "active")):
        return True
    values = series.dropna().unique()
    if len(values) == 0:
        return False
    if len(values) <= 2 and set(pd.Series(values).astype(float).round(6)).issubset({0.0, 1.0}):
        return True
    return False


def is_leaky_or_identifier_feature(col: str) -> bool:
    name = col.lower()
    if any(token in name for token in FEATURE_EXCLUDE_TOKENS):
        return True
    if any(name.endswith(suffix) for suffix in FEATURE_EXCLUDE_SUFFIXES):
        return True
    return False


def feature_columns(df: pd.DataFrame) -> tuple[list[str], set[str]]:
    cols: list[str] = []
    binary_cols: set[str] = set()
    for col in df.columns:
        if col in EXCLUDE_COLUMNS:
            continue
        if is_leaky_or_identifier_feature(col):
            continue
        if not is_numeric_dtype(df[col]):
            continue
        cols.append(col)
        if is_binary_or_mask(df[col], col):
            binary_cols.add(col)
    return cols, binary_cols


def aggregate_window(window: pd.DataFrame, cols: list[str], binary_cols: set[str]) -> dict[str, float]:
    features: dict[str, float] = {}
    for col in cols:
        values = pd.to_numeric(window[col], errors="coerce")
        if col in binary_cols:
            features[f"{col}__mean"] = float(values.mean()) if values.notna().any() else np.nan
            features[f"{col}__sum"] = float(values.sum()) if values.notna().any() else np.nan
            features[f"{col}__last"] = float(values.dropna().iloc[-1]) if values.notna().any() else np.nan
            continue
        clean = values.dropna()
        if clean.empty:
            for stat in ("mean", "std", "min", "max", "median", "first", "last", "delta", "range"):
                features[f"{col}__{stat}"] = np.nan
            continue
        first = clean.iloc[0]
        last = clean.iloc[-1]
        min_value = clean.min()
        max_value = clean.max()
        features[f"{col}__mean"] = float(clean.mean())
        features[f"{col}__std"] = float(clean.std(ddof=0))
        features[f"{col}__min"] = float(min_value)
        features[f"{col}__max"] = float(max_value)
        features[f"{col}__median"] = float(clean.median())
        features[f"{col}__first"] = float(first)
        features[f"{col}__last"] = float(last)
        features[f"{col}__delta"] = float(last - first)
        features[f"{col}__range"] = float(max_value - min_value)
    return features


def majority_label(labels: pd.Series) -> tuple[str | None, float, int]:
    valid = labels.dropna().astype(str)
    valid = valid[valid.str.len() > 0]
    if valid.empty:
        return None, 0.0, 0
    counts = valid.value_counts()
    label = counts.index[0]
    count = int(counts.iloc[0])
    return label, count / len(valid), len(valid)


def sequence_group_columns(df: pd.DataFrame) -> list[str]:
    if "video_id" in df.columns:
        return ["video_id"]
    cols = [col for col in ("participant_id", "task_name") if col in df.columns]
    return cols or ["task_name"]


def first_non_null(series: pd.Series) -> object:
    clean = series.dropna()
    return clean.iloc[0] if not clean.empty else pd.NA


def build_sequence_windows_fast(
    seq: pd.DataFrame,
    cols: list[str],
    binary_cols: set[str],
    target_col: str,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    seq = seq.sort_values("frame_id", kind="mergesort").copy()
    seq["frame_id"] = pd.to_numeric(seq["frame_id"], errors="coerce")
    seq = seq[seq["frame_id"].notna()]
    if seq.empty:
        return []

    seq["frame_id"] = seq["frame_id"].round().astype("int64")
    seq = seq.drop_duplicates(subset=["frame_id"], keep="first")
    min_frame = int(seq["frame_id"].min())
    max_frame = int(seq["frame_id"].max())
    window_frames = int(round(args.window_sec * args.fps))
    stride_frames = int(round(args.stride_sec * args.fps))
    label_tail_frames = int(round(args.label_tail_sec * args.fps))
    if max_frame - min_frame + 1 < window_frames:
        return []

    frame_index = pd.RangeIndex(min_frame, max_frame + 1, name="frame_id")
    indexed = seq.set_index("frame_id").reindex(frame_index)
    numeric = indexed[cols].apply(pd.to_numeric, errors="coerce")
    continuous_cols = [col for col in cols if col not in binary_cols]
    binary_list = [col for col in cols if col in binary_cols]

    rolling = numeric.rolling(window=window_frames, min_periods=1)
    stat_frames: list[pd.DataFrame] = []
    if continuous_cols:
        cont = numeric[continuous_cols]
        cont_roll = cont.rolling(window=window_frames, min_periods=1)
        stat_frames.extend(
            [
                cont_roll.mean().add_suffix("__mean"),
                cont_roll.std(ddof=0).add_suffix("__std"),
                cont_roll.min().add_suffix("__min"),
                cont_roll.max().add_suffix("__max"),
                cont_roll.median().add_suffix("__median"),
                cont.shift(window_frames - 1).add_suffix("__first"),
                cont.add_suffix("__last"),
                cont.sub(cont.shift(window_frames - 1)).add_suffix("__delta"),
                cont_roll.max().sub(cont_roll.min()).add_suffix("__range"),
            ]
        )
    if binary_list:
        binary = numeric[binary_list]
        binary_roll = binary.rolling(window=window_frames, min_periods=1)
        stat_frames.extend(
            [
                binary_roll.mean().add_suffix("__mean"),
                binary_roll.sum().add_suffix("__sum"),
                binary.add_suffix("__last"),
            ]
        )
    if not stat_frames:
        return []
    feature_stats = pd.concat(stat_frames, axis=1)

    first_end = min_frame + window_frames - 1
    end_frames = np.arange(first_end, max_frame + 1, stride_frames, dtype=np.int64)
    labels = indexed[target_col] if target_col in indexed else pd.Series(index=frame_index, dtype="object")
    meta = {
        "participant_id": first_non_null(seq["participant_id"]) if "participant_id" in seq else pd.NA,
        "task_name": first_non_null(seq["task_name"]) if "task_name" in seq else pd.NA,
        "video_id": first_non_null(seq["video_id"]) if "video_id" in seq else pd.NA,
    }

    rows: list[dict[str, object]] = []
    for end_frame in end_frames:
        label_start = int(end_frame - label_tail_frames + 1)
        tail = labels.loc[label_start:int(end_frame)]
        label, purity, valid_frames = majority_label(tail)
        if label is None:
            continue
        if purity < args.min_label_purity and not args.keep_low_purity:
            continue
        start_frame = int(end_frame - window_frames + 1)
        row = {
            **meta,
            "window_start_frame": start_frame,
            "window_end_frame": int(end_frame),
            "window_start_time": float(start_frame / args.fps),
            "window_end_time": float(end_frame / args.fps),
            "target_label": label,
            "label_purity": float(purity),
            "num_valid_label_frames": int(valid_frames),
        }
        row.update(feature_stats.loc[int(end_frame)].to_dict())
        rows.append(row)
    return rows


def build_windows_loop(args: argparse.Namespace, df: pd.DataFrame, cols: list[str], binary_cols: set[str]) -> pd.DataFrame:
    target_col = LABEL_COLUMNS[args.target_mode]
    window_frames = int(round(args.window_sec * args.fps))
    stride_frames = int(round(args.stride_sec * args.fps))
    label_tail_frames = int(round(args.label_tail_sec * args.fps))
    rows: list[dict[str, object]] = []
    group_cols = sequence_group_columns(df)

    for _, seq in df.groupby(group_cols, dropna=False, sort=False):
        seq = seq.sort_values("frame_id", kind="mergesort").copy()
        seq["frame_id"] = pd.to_numeric(seq["frame_id"], errors="coerce")
        seq = seq[seq["frame_id"].notna()]
        if seq.empty:
            continue
        min_frame = int(seq["frame_id"].min())
        max_frame = int(seq["frame_id"].max())
        first_end = min_frame + window_frames - 1
        for end_frame in range(first_end, max_frame + 1, stride_frames):
            start_frame = end_frame - window_frames + 1
            label_start = end_frame - label_tail_frames + 1
            window = seq[(seq["frame_id"] >= start_frame) & (seq["frame_id"] <= end_frame)]
            tail = seq[(seq["frame_id"] >= label_start) & (seq["frame_id"] <= end_frame)]
            label, purity, valid_frames = majority_label(tail[target_col])
            if label is None:
                continue
            if purity < args.min_label_purity and not args.keep_low_purity:
                continue
            meta = {
                "participant_id": window["participant_id"].dropna().iloc[0] if "participant_id" in window and window["participant_id"].notna().any() else pd.NA,
                "task_name": window["task_name"].dropna().iloc[0] if "task_name" in window and window["task_name"].notna().any() else pd.NA,
                "video_id": window["video_id"].dropna().iloc[0] if "video_id" in window and window["video_id"].notna().any() else pd.NA,
                "window_start_frame": int(start_frame),
                "window_end_frame": int(end_frame),
                "window_start_time": float(start_frame / args.fps),
                "window_end_time": float(end_frame / args.fps),
                "target_label": label,
                "label_purity": float(purity),
                "num_valid_label_frames": int(valid_frames),
            }
            meta.update(aggregate_window(window, cols, binary_cols))
            rows.append(meta)

    return pd.DataFrame(rows)


def build_windows(args: argparse.Namespace) -> pd.DataFrame:
    if not args.input.exists():
        raise SystemExit(f"Input frame-level table does not exist: {args.input}")
    df = read_dataframe(args.input)
    if "frame_id" not in df.columns:
        raise SystemExit("Input frame-level table must contain frame_id.")
    target_col = LABEL_COLUMNS[args.target_mode]
    if target_col not in df.columns:
        raise SystemExit(f"Input frame-level table must contain {target_col}.")
    if "timestamp" not in df.columns:
        df["timestamp"] = pd.to_numeric(df["frame_id"], errors="coerce") / args.fps

    cols, binary_cols = feature_columns(df)
    if not cols:
        raise SystemExit("No numeric feature columns found after excluding metadata/labels.")
    logging.info(
        "Using %d numeric feature columns (%d binary/mask columns), aggregation_mode=%s",
        len(cols),
        len(binary_cols),
        args.aggregation_mode,
    )
    if args.aggregation_mode == "loop":
        return build_windows_loop(args, df, cols, binary_cols)

    group_cols = sequence_group_columns(df)
    rows: list[dict[str, object]] = []
    sequence_count = 0
    for _, seq in df.groupby(group_cols, dropna=False, sort=False):
        sequence_count += 1
        rows.extend(build_sequence_windows_fast(seq, cols, binary_cols, target_col, args))
        if sequence_count % 25 == 0:
            logging.info("Processed %d sequences, accumulated %d windows", sequence_count, len(rows))

    return pd.DataFrame(rows)


def write_label_stats(windows: pd.DataFrame, target_mode: str) -> None:
    path = INTERMEDIATE_DIR / f"window_label_stats_{target_mode}.csv"
    ensure_parent(path)
    if windows.empty:
        pd.DataFrame(columns=["target_label", "window_count", "percentage"]).to_csv(path, index=False)
        return
    counts = windows["target_label"].value_counts()
    stats = pd.DataFrame(
        {
            "target_label": counts.index,
            "window_count": counts.values,
            "percentage": counts.values / counts.values.sum(),
        }
    )
    stats.to_csv(path, index=False)
    logging.info("Wrote window label stats: %s", path)


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    windows = build_windows(args)
    output = args.output or window_feature_path(args.target_mode, args.window_sec, args.stride_sec)
    write_dataframe(windows, output)
    write_label_stats(windows, args.target_mode)
    logging.info("Wrote window features: %s (%d rows, %d columns)", output, len(windows), len(windows.columns))


if __name__ == "__main__":
    main()
