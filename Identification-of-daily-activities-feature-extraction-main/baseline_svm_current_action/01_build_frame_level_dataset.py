#!/usr/bin/env python3
"""Build a unified frame-level multimodal feature table with action labels."""

from __future__ import annotations

import argparse
import json
import logging
from functools import reduce
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

from common import (
    CALIBRATION_TASKS,
    DEFAULT_ANNOTATIONS_DIR,
    DEFAULT_DEPTH_DIR,
    DEFAULT_FPS,
    DEFAULT_GAZE_DIR,
    DEFAULT_HAND_DIR,
    DEFAULT_HEAD_IMU_DIR,
    DEFAULT_IMU_DIR,
    DEFAULT_SKEL_DIR,
    FRAME_ALIASES,
    FRAME_LEVEL_PATH,
    INTERMEDIATE_DIR,
    KEY_COLUMNS,
    TIMESTAMP_ALIASES,
    configure_logging,
    ensure_parent,
    existing_dir,
    infer_participant_id,
    infer_recording_name,
    iter_feature_files,
    normalize_label,
    read_table,
    write_dataframe,
)


MERGE_KEY_CANDIDATES = (
    ("participant_id", "task_name", "video_id", "frame_id"),
    ("video_id", "frame_id"),
    ("video_path", "frame_id"),
    ("filename", "frame_id"),
)
PRIMARY_MERGE_KEYS = ("participant_id", "task_name", "video_id", "frame_id")
MODALITY_FRAME_STEM_SUFFIXES = {
    "gaze": ("_gaze_yolo_probabilistic_heatmap_frame",),
    "hand": ("_hand_object_frame_features", "_hoi_v2_frame_topk"),
    "depth": ("_depth_scene_frame_features",),
    "headimu": ("_head_imu_frame_features",),
    "imu": ("_imu_joint_kinematics_frame_features",),
    "skel": ("_skeleton_upper_limb_kinematics_frame_features",),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge frame-level multimodal features and assign ADL action labels."
    )
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Video FPS used for frame/time conversion.")
    parser.add_argument("--gaze_dir", type=existing_dir, default=DEFAULT_GAZE_DIR)
    parser.add_argument("--hand_dir", type=existing_dir, default=DEFAULT_HAND_DIR)
    parser.add_argument("--depth_dir", type=existing_dir, default=DEFAULT_DEPTH_DIR)
    parser.add_argument("--headimu_dir", type=existing_dir, default=DEFAULT_HEAD_IMU_DIR)
    parser.add_argument("--imu_dir", type=existing_dir, default=DEFAULT_IMU_DIR)
    parser.add_argument("--skel_dir", type=existing_dir, default=DEFAULT_SKEL_DIR)
    parser.add_argument("--annotations_dir", type=existing_dir, default=DEFAULT_ANNOTATIONS_DIR)
    parser.add_argument("--output", type=Path, default=FRAME_LEVEL_PATH)
    parser.add_argument("--sample_rows", type=int, default=1000)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def first_existing(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    return next((name for name in candidates if name in columns), None)


def normalize_feature_table(df: pd.DataFrame, path: Path, prefix: str, fps: float) -> pd.DataFrame | None:
    if df.empty:
        logging.warning("Skipping empty feature file: %s", path)
        return None

    df = df.copy()
    participant = infer_participant_id(path, df["participant_id"].iloc[0] if "participant_id" in df else None)
    recording = None
    if "task_name" in df.columns and df["task_name"].notna().any():
        recording = str(df["task_name"].dropna().iloc[0])
    recording = recording or infer_recording_name(path, participant)

    if "participant_id" not in df.columns:
        df["participant_id"] = participant
    elif participant:
        df["participant_id"] = df["participant_id"].fillna(participant)
    if "task_name" not in df.columns:
        df["task_name"] = recording
    elif recording:
        df["task_name"] = df["task_name"].fillna(recording)
    if "video_id" not in df.columns:
        if "full_id" in df.columns and df["full_id"].notna().any():
            df["video_id"] = df["full_id"].astype(str)
        elif participant and recording:
            df["video_id"] = f"{participant}_{recording}"
        else:
            df["video_id"] = path.stem
    elif "full_id" in df.columns:
        df["video_id"] = df["video_id"].fillna(df["full_id"].astype(str))
    elif participant and recording:
        df["video_id"] = df["video_id"].fillna(f"{participant}_{recording}")
    if "filename" not in df.columns:
        df["filename"] = path.name

    timestamp_col = first_existing(df.columns, TIMESTAMP_ALIASES)
    frame_col = None
    if "frame_id" in df.columns:
        frame_col = "frame_id"
    elif "video_frame_idx" in df.columns:
        frame_col = "video_frame_idx"
    elif timestamp_col:
        df["frame_id"] = np.rint(pd.to_numeric(df[timestamp_col], errors="coerce") * fps)
        frame_col = "frame_id"
    else:
        frame_col = first_existing(df.columns, FRAME_ALIASES)

    if frame_col is None:
        logging.warning("Skipping feature file without usable frame/timestamp column: %s", path)
        return None
    if frame_col != "frame_id":
        df["frame_id"] = df[frame_col]
    df["frame_id"] = pd.to_numeric(df["frame_id"], errors="coerce")
    df = df[df["frame_id"].notna()].copy()
    if df.empty:
        logging.warning("Skipping feature file after dropping invalid frame_id rows: %s", path)
        return None
    df["frame_id"] = df["frame_id"].round().astype("int64")

    if timestamp_col and timestamp_col in df.columns:
        df["_source_timestamp"] = pd.to_numeric(df[timestamp_col], errors="coerce")
    else:
        df["_source_timestamp"] = df["frame_id"] / fps

    if "video_path" not in df.columns:
        df["video_path"] = pd.NA

    keep_unprefixed = KEY_COLUMNS
    rename = {
        col: f"{prefix}{col}"
        for col in df.columns
        if col not in keep_unprefixed
    }
    df = df.rename(columns=rename)
    preferred_cols = [col for col in ("participant_id", "task_name", "video_id", "video_path", "filename", "frame_id") if col in df]
    other_cols = [col for col in df.columns if col not in preferred_cols]
    df = df[preferred_cols + other_cols]
    keys = [col for col in PRIMARY_MERGE_KEYS if col in df.columns]
    if len(keys) == len(PRIMARY_MERGE_KEYS):
        df = coalesce_duplicate_keys(df, keys, f"{prefix.rstrip('_')} file {path.name}")
    return df


def is_modality_frame_file(path: Path, modality_name: str) -> bool:
    suffixes = MODALITY_FRAME_STEM_SUFFIXES.get(modality_name)
    if suffixes is None:
        return True
    return any(path.stem.endswith(suffix) for suffix in suffixes)


def load_modality(name: str, root: Path, prefix: str, fps: float) -> pd.DataFrame | None:
    frames: list[pd.DataFrame] = []
    files = [path for path in iter_feature_files(root) if is_modality_frame_file(path, name)]
    if not files:
        logging.warning("No frame-level CSV/Parquet files found for modality %s in %s", name, root)
        return None

    for path in files:
        try:
            raw = read_table(path)
            normalized = normalize_feature_table(raw, path, prefix, fps)
            if normalized is not None:
                frames.append(normalized)
        except Exception as exc:  # noqa: BLE001
            logging.warning("Skipping %s feature file %s: %s", name, path, exc)

    if not frames:
        logging.warning("No usable files remained for modality %s", name)
        return None
    out = pd.concat(frames, ignore_index=True, sort=False)
    out = coalesce_duplicate_frame_keys(out, name)
    logging.info("Loaded %s modality: %d rows, %d columns", name, len(out), len(out.columns))
    return out


def coalesce_duplicate_frame_keys(df: pd.DataFrame, modality_name: str) -> pd.DataFrame:
    keys = [col for col in PRIMARY_MERGE_KEYS if col in df.columns]
    if len(keys) != len(PRIMARY_MERGE_KEYS):
        return df
    return coalesce_duplicate_keys(df, keys, f"{modality_name} modality")


def should_average_duplicate_column(col: str, series: pd.Series) -> bool:
    if not is_numeric_dtype(series):
        return False
    name = col.lower()
    index_tokens = ("frame_idx", "frame_id", "timestamp", "time_s", "detection_idx", "body_id", "class_id", "object_id")
    if any(token in name for token in index_tokens):
        return False
    if name.endswith("_class") or name.endswith("_id") or name == "id":
        return False
    return True


def first_non_null(series: pd.Series) -> object:
    values = series.dropna()
    if values.empty:
        return pd.NA
    return values.iloc[0]


def coalesce_duplicate_keys(df: pd.DataFrame, keys: list[str], context: str) -> pd.DataFrame:
    duplicate_mask = df.duplicated(keys, keep=False)
    if not duplicate_mask.any():
        return df

    duplicate_rows = int(duplicate_mask.sum())
    duplicate_groups = int(df.loc[duplicate_mask, keys].drop_duplicates().shape[0])
    logging.info(
        "Coalescing %d duplicate rows in %s across %d frame keys",
        duplicate_rows,
        context,
        duplicate_groups,
    )
    aggregations = {
        col: ("mean" if should_average_duplicate_column(col, df[col]) else first_non_null)
        for col in df.columns
        if col not in keys
    }
    return df.groupby(keys, as_index=False, sort=False, dropna=False).agg(aggregations)


def choose_merge_keys(left: pd.DataFrame, right: pd.DataFrame) -> list[str]:
    for candidate in MERGE_KEY_CANDIDATES:
        if all(col in left.columns and col in right.columns for col in candidate):
            return list(candidate)
    raise ValueError("No compatible merge keys found")


def outer_merge_modalities(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    def merge_two(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
        keys = choose_merge_keys(left, right)
        logging.info("Outer merging with keys: %s", ", ".join(keys))
        merged = left.merge(right, on=keys, how="outer", suffixes=("", "_dup"), validate="one_to_one")
        duplicate_cols = [col for col in merged.columns if col.endswith("_dup")]
        if duplicate_cols:
            merged = merged.drop(columns=duplicate_cols)
        return merged

    merged = reduce(merge_two, dfs)
    return merged


def as_number(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def first_number(*values: Any) -> float | None:
    for value in values:
        number = as_number(value)
        if number is not None:
            return number
    return None


def get_nested_number(record: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        if key in record:
            value = as_number(record[key])
            if value is not None:
                return value
    segment = record.get("segment")
    if isinstance(segment, dict):
        for key in keys:
            if key in segment:
                value = as_number(segment[key])
                if value is not None:
                    return value
    return None


def maybe_segment_from_record(
    record: dict[str, Any],
    path: Path,
    context_task: str | None,
    fps: float,
) -> dict[str, Any] | None:
    participant = (
        record.get("participant_id")
        or record.get("participant")
        or record.get("subject")
        or infer_participant_id(path)
    )
    recording = (
        record.get("recording")
        or record.get("recording_name")
        or record.get("video_id")
        or infer_recording_name(path, participant)
    )
    task = record.get("task") or record.get("task_name") or record.get("action") or context_task
    step = record.get("step") or record.get("step_name") or record.get("label") or record.get("name")

    start_time = get_nested_number(record, ("start_s", "start_time", "start", "begin_s", "begin_time"))
    end_time = get_nested_number(record, ("end_s", "end_time", "end", "stop_s", "stop_time"))
    start_frame = get_nested_number(record, ("start_frame", "start_frame_id", "start_idx"))
    end_frame = get_nested_number(record, ("end_frame", "end_frame_id", "end_idx"))
    if start_time is None and start_frame is not None:
        start_time = start_frame / fps
    if end_time is None and end_frame is not None:
        end_time = end_frame / fps

    if not (participant and recording and task and step and start_time is not None and end_time is not None):
        return None
    if end_time <= start_time:
        return None
    return {
        "participant_id": str(participant),
        "recording_name": str(recording),
        "task_name": normalize_label(task),
        "step_name": normalize_label(step),
        "start_time": float(start_time),
        "end_time": float(end_time),
    }


def walk_annotation_json(obj: Any, path: Path, fps: float, context_task: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if isinstance(obj, dict):
        segment = maybe_segment_from_record(obj, path, context_task, fps)
        if segment:
            rows.append(segment)
        for key, value in obj.items():
            if key.lower() in {"metadata", "segment"}:
                next_context = context_task
            elif isinstance(value, list) and key.lower() not in {"annotations", "segments"}:
                next_context = key
            else:
                next_context = context_task
            rows.extend(walk_annotation_json(value, path, fps, next_context))
    elif isinstance(obj, list):
        for item in obj:
            rows.extend(walk_annotation_json(item, path, fps, context_task))
    return rows


def load_annotation_json(path: Path, fps: float) -> list[dict[str, Any]]:
    if "alignment" in path.name.lower() or "objects" in path.name.lower():
        return []
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
    rows = walk_annotation_json(data, path, fps)
    for row in rows:
        row["participant_id"] = str(metadata.get("participant") or row["participant_id"])
        row["recording_name"] = str(metadata.get("recording") or row["recording_name"])
    return rows


def load_segments_csv(path: Path, fps: float) -> list[dict[str, Any]]:
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        logging.warning("Could not read segment CSV %s: %s", path, exc)
        return []
    participant = infer_participant_id(path)
    recording = infer_recording_name(path, participant)
    rows: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        task = row.get("task") or row.get("task_name")
        step = row.get("step") or row.get("step_name")
        start_time = first_number(row.get("start_s"), row.get("start_time"))
        end_time = first_number(row.get("end_s"), row.get("end_time"))
        if start_time is None and as_number(row.get("start_frame")) is not None:
            start_time = as_number(row.get("start_frame")) / fps
        if end_time is None and as_number(row.get("end_frame")) is not None:
            end_time = as_number(row.get("end_frame")) / fps
        if participant and recording and task and step and start_time is not None and end_time is not None:
            rows.append(
                {
                    "participant_id": participant,
                    "recording_name": recording,
                    "task_name": normalize_label(task),
                    "step_name": normalize_label(step),
                    "start_time": float(start_time),
                    "end_time": float(end_time),
                }
            )
    return rows


def load_annotations(root: Path, fps: float) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        try:
            rows.extend(load_annotation_json(path, fps))
        except Exception as exc:  # noqa: BLE001
            logging.warning("Skipping annotation JSON %s: %s", path, exc)
    for path in sorted(root.rglob("*.segments.csv")):
        rows.extend(load_segments_csv(path, fps))

    if not rows:
        logging.warning("No usable annotation segments found in %s", root)
        return pd.DataFrame(
            columns=["participant_id", "recording_name", "task_name", "step_name", "start_time", "end_time"]
        )
    ann = pd.DataFrame(rows).drop_duplicates()
    ann = ann[ann["task_name"].notna() & ann["step_name"].notna()]
    logging.info("Loaded %d annotation segments", len(ann))
    return ann


def assign_labels(frames: pd.DataFrame, annotations: pd.DataFrame, fps: float) -> pd.DataFrame:
    out = frames.copy()
    out["timestamp"] = pd.to_numeric(out["frame_id"], errors="coerce") / fps
    out["task_label"] = pd.NA
    out["step_label"] = pd.NA
    if annotations.empty:
        out["task_step_label"] = pd.NA
        out["is_labeled"] = False
        return out

    grouped_annotations = annotations.groupby(["participant_id", "recording_name"], dropna=False)
    for (participant, recording), group in grouped_annotations:
        mask = (out["participant_id"].astype(str) == str(participant)) & (out["task_name"].astype(str) == str(recording))
        if not mask.any():
            continue
        seq_index = out.index[mask]
        seq_time = out.loc[seq_index, "timestamp"]
        for _, seg in group.sort_values("start_time").iterrows():
            seg_mask = (seq_time >= seg["start_time"]) & (seq_time < seg["end_time"])
            target_idx = seq_index[seg_mask.to_numpy()]
            out.loc[target_idx, "task_label"] = seg["task_name"]
            out.loc[target_idx, "step_label"] = seg["step_name"]

    out["is_labeled"] = out["task_label"].notna() & out["step_label"].notna()
    out["task_step_label"] = pd.NA
    labeled = out["is_labeled"]
    out.loc[labeled, "task_step_label"] = (
        out.loc[labeled, "task_label"].astype(str) + "__" + out.loc[labeled, "step_label"].astype(str)
    )
    return out


def write_label_stats(df: pd.DataFrame, fps: float) -> None:
    rows = []
    labeled = df[df["is_labeled"]]
    total = max(len(labeled), 1)
    for target_col in ("task_label", "step_label", "task_step_label"):
        counts = labeled[target_col].value_counts(dropna=True)
        for label, count in counts.items():
            rows.append(
                {
                    "target": target_col,
                    "label": label,
                    "frame_count": int(count),
                    "duration_seconds": float(count) / fps,
                    "percentage": float(count) / total,
                }
            )
    stats = pd.DataFrame(rows)
    path = INTERMEDIATE_DIR / "frame_level_label_stats.csv"
    ensure_parent(path)
    stats.to_csv(path, index=False)
    logging.info("Wrote label stats: %s", path)


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    modality_dirs = {
        "gaze": (args.gaze_dir, "gaze_"),
        "hand": (args.hand_dir, "hand_"),
        "depth": (args.depth_dir, "depth_"),
        "headimu": (args.headimu_dir, "headimu_"),
        "imu": (args.imu_dir, "imu_"),
        "skel": (args.skel_dir, "skel_"),
    }

    modality_frames = []
    for name, (root, prefix) in modality_dirs.items():
        df = load_modality(name, root, prefix, args.fps)
        if df is not None:
            modality_frames.append(df)
    if not modality_frames:
        raise SystemExit("No usable modality feature files were found.")

    merged = outer_merge_modalities(modality_frames)
    if "task_name" in merged.columns:
        merged = merged[~merged["task_name"].isin(CALIBRATION_TASKS)].copy()
    annotations = load_annotations(args.annotations_dir, args.fps)
    merged = assign_labels(merged, annotations, args.fps)
    merged = merged.sort_values(["participant_id", "task_name", "video_id", "frame_id"], kind="mergesort")

    write_dataframe(merged, args.output)
    sample_path = INTERMEDIATE_DIR / "frame_level_multimodal_features_sample.csv"
    merged.head(args.sample_rows).to_csv(sample_path, index=False)
    write_label_stats(merged, args.fps)
    logging.info("Wrote frame-level features: %s (%d rows, %d columns)", args.output, len(merged), len(merged.columns))
    logging.info("Wrote sample CSV: %s", sample_path)


if __name__ == "__main__":
    main()
