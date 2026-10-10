#!/usr/bin/env python3
"""Simple conservative hand-object interaction v2 pipeline.

Design goals:
- keep all valid hand-object pairs per frame
- avoid early object filtering
- only drop object class 0 and objects covering more than half the video frame
- write frame-level top-k summaries separately from pair-level rows
- write coverage diagnostics for frame/object/hand loss checks
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DATASET_ROOT = Path("/home/julius/datasets/seminar_object_detection")
DEFAULT_OBJECT_ROOT = DATASET_ROOT / "lower_feature_outcome" / "yolo11s_coco_object_outcome_csv"
DEFAULT_HAND_ROOT = DATASET_ROOT / "lower_feature_outcome" / "yolo11s_left_right_hand_outcome_csv"
DEFAULT_VIDEO_ROOT = DATASET_ROOT / "iAssistADL_dataset_export_2026-05-18" / "world_camera_videos"
DEFAULT_OUTPUT_ROOT = DATASET_ROOT / "upper_feature_outcome" / "hand_object_interaction_v2_simple"

WORLD_CAMERA_SUFFIX = "_world_camera"
PAIR_SUFFIX = "_hoi_v2_pair_features"
FRAME_SUFFIX = "_hoi_v2_frame_topk"
REQUIRED_COLUMNS = ("frame_id", "class_id", "confidence", "x1", "y1", "x2", "y2")
EPS = 1e-9

COCO_CLASS_NAMES = {
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}


def full_id_from_path(path: Path) -> str:
    stem = path.stem
    return stem[: -len(WORLD_CAMERA_SUFFIX)] if stem.endswith(WORLD_CAMERA_SUFFIX) else stem


def participant_from_full_id(full_id: str) -> str:
    return "_".join(full_id.split("_")[:4])


def task_from_full_id(full_id: str) -> str:
    participant = participant_from_full_id(full_id)
    prefix = f"{participant}_"
    return full_id[len(prefix) :] if full_id.startswith(prefix) else full_id


def world_camera_csvs(root: Path) -> dict[str, Path]:
    return {full_id_from_path(path): path for path in sorted(root.glob("*_world_camera.csv"))}


def video_path_for(full_id: str, video_root: Path) -> Path:
    return video_root / participant_from_full_id(full_id) / f"{full_id}{WORLD_CAMERA_SUFFIX}.mp4"


def video_metadata(full_id: str, video_root: Path) -> dict:
    path = video_path_for(full_id, video_root)
    if not path.exists():
        return {"video_path": str(path), "video_exists": False}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"video_path": str(path), "video_exists": True, "video_opened": False}
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return {
        "video_path": str(path),
        "video_exists": True,
        "video_opened": True,
        "video_width": width,
        "video_height": height,
        "video_frame_count": frame_count,
        "video_fps": fps,
    }


def require_columns(path: Path) -> None:
    header = pd.read_csv(path, nrows=0).columns
    missing = sorted(set(REQUIRED_COLUMNS) - set(header))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def numeric_detection_columns(df: pd.DataFrame) -> pd.DataFrame:
    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def hand_side_from_row(row: pd.Series) -> str:
    class_name = str(row.get("class_name", "")).lower()
    if "left" in class_name:
        return "left"
    if "right" in class_name:
        return "right"
    class_id = row.get("class_id")
    if pd.isna(class_id):
        return "unknown"
    class_id = int(class_id)
    if class_id == 0:
        return "left"
    if class_id == 1:
        return "right"
    return "unknown"


def read_objects(path: Path, full_id: str, image_area: float | None) -> tuple[pd.DataFrame, dict]:
    require_columns(path)
    raw = numeric_detection_columns(pd.read_csv(path))
    raw_rows = int(len(raw))
    raw = raw.dropna(subset=list(REQUIRED_COLUMNS)).copy()
    raw["frame_id"] = raw["frame_id"].round().astype(int)
    raw["class_id"] = raw["class_id"].round().astype(int)
    raw["object_detection_id"] = raw.groupby("frame_id").cumcount()
    raw["object_width"] = (raw["x2"] - raw["x1"]).clip(lower=0.0)
    raw["object_height"] = (raw["y2"] - raw["y1"]).clip(lower=0.0)
    raw["object_area"] = raw["object_width"] * raw["object_height"]
    raw["object_cx"] = (raw["x1"] + raw["x2"]) / 2.0
    raw["object_cy"] = (raw["y1"] + raw["y2"]) / 2.0
    if "class_name" not in raw.columns:
        raw["class_name"] = raw["class_id"].map(lambda value: COCO_CLASS_NAMES.get(int(value), f"class_{int(value)}"))
    raw["class_name"] = raw["class_name"].fillna("").astype(str)

    if image_area is None or image_area <= 0:
        max_x = float(raw["x2"].max()) if len(raw) else 0.0
        max_y = float(raw["y2"].max()) if len(raw) else 0.0
        image_area = max(max_x * max_y, EPS)
    raw["object_area_ratio"] = raw["object_area"] / max(float(image_area), EPS)
    invalid_box = (raw["object_width"] <= 0.0) | (raw["object_height"] <= 0.0)
    ignored_class0 = raw["class_id"].eq(0)
    ignored_oversized = raw["object_area_ratio"].gt(0.5)
    keep = ~(invalid_box | ignored_class0 | ignored_oversized)
    objects = raw.loc[keep].copy()
    objects = objects.rename(
        columns={
            "class_id": "object_class_id",
            "class_name": "object_class",
            "confidence": "object_conf",
            "x1": "object_x1",
            "y1": "object_y1",
            "x2": "object_x2",
            "y2": "object_y2",
        }
    )
    diagnostics = {
        "object_rows_raw_file": raw_rows,
        "object_rows_after_numeric": int(len(raw)),
        "object_rows_kept": int(len(objects)),
        "object_rows_dropped_invalid_box": int(invalid_box.sum()),
        "object_rows_ignored_class0": int(ignored_class0.sum()),
        "object_rows_ignored_oversized_gt_half_frame": int((ignored_oversized & ~ignored_class0 & ~invalid_box).sum()),
        "object_frames_raw": int(raw["frame_id"].nunique()) if len(raw) else 0,
        "object_frames_kept": int(objects["frame_id"].nunique()) if len(objects) else 0,
    }
    wanted = [
        "frame_id",
        "object_detection_id",
        "object_class_id",
        "object_class",
        "object_conf",
        "object_x1",
        "object_y1",
        "object_x2",
        "object_y2",
        "object_cx",
        "object_cy",
        "object_width",
        "object_height",
        "object_area",
        "object_area_ratio",
    ]
    return objects[wanted], diagnostics


def read_hands(path: Path) -> tuple[pd.DataFrame, dict]:
    require_columns(path)
    raw = numeric_detection_columns(pd.read_csv(path))
    raw_rows = int(len(raw))
    raw = raw.dropna(subset=list(REQUIRED_COLUMNS)).copy()
    raw["frame_id"] = raw["frame_id"].round().astype(int)
    raw["class_id"] = raw["class_id"].round().astype(int)
    raw["hand_detection_id"] = raw.groupby("frame_id").cumcount()
    raw["hand_side"] = raw.apply(hand_side_from_row, axis=1)
    raw["hand_width"] = (raw["x2"] - raw["x1"]).clip(lower=0.0)
    raw["hand_height"] = (raw["y2"] - raw["y1"]).clip(lower=0.0)
    raw["hand_area"] = raw["hand_width"] * raw["hand_height"]
    raw["hand_cx"] = (raw["x1"] + raw["x2"]) / 2.0
    raw["hand_cy"] = (raw["y1"] + raw["y2"]) / 2.0
    invalid_box = (raw["hand_width"] <= 0.0) | (raw["hand_height"] <= 0.0)
    hands = raw.loc[~invalid_box].copy()
    hands = hands.rename(
        columns={
            "class_id": "hand_class_id",
            "confidence": "hand_conf",
            "x1": "hand_x1",
            "y1": "hand_y1",
            "x2": "hand_x2",
            "y2": "hand_y2",
        }
    )
    diagnostics = {
        "hand_rows_raw_file": raw_rows,
        "hand_rows_after_numeric": int(len(raw)),
        "hand_rows_kept": int(len(hands)),
        "hand_rows_dropped_invalid_box": int(invalid_box.sum()),
        "hand_frames_raw": int(raw["frame_id"].nunique()) if len(raw) else 0,
        "hand_frames_kept": int(hands["frame_id"].nunique()) if len(hands) else 0,
        "left_hand_rows_kept": int(hands["hand_side"].eq("left").sum()) if len(hands) else 0,
        "right_hand_rows_kept": int(hands["hand_side"].eq("right").sum()) if len(hands) else 0,
        "unknown_hand_rows_kept": int(hands["hand_side"].eq("unknown").sum()) if len(hands) else 0,
    }
    wanted = [
        "frame_id",
        "hand_detection_id",
        "hand_side",
        "hand_class_id",
        "hand_conf",
        "hand_x1",
        "hand_y1",
        "hand_x2",
        "hand_y2",
        "hand_cx",
        "hand_cy",
        "hand_width",
        "hand_height",
        "hand_area",
    ]
    return hands[wanted], diagnostics


def bbox_iou(left: pd.DataFrame) -> pd.Series:
    ix1 = np.maximum(left["hand_x1"], left["object_x1"])
    iy1 = np.maximum(left["hand_y1"], left["object_y1"])
    ix2 = np.minimum(left["hand_x2"], left["object_x2"])
    iy2 = np.minimum(left["hand_y2"], left["object_y2"])
    inter = np.maximum(0.0, ix2 - ix1) * np.maximum(0.0, iy2 - iy1)
    union = left["hand_area"] + left["object_area"] - inter
    return inter / np.maximum(union, EPS)


def bbox_gap(df: pd.DataFrame) -> pd.Series:
    dx = np.maximum.reduce(
        [
            (df["object_x1"] - df["hand_x2"]).to_numpy(dtype=float),
            (df["hand_x1"] - df["object_x2"]).to_numpy(dtype=float),
            np.zeros(len(df), dtype=float),
        ]
    )
    dy = np.maximum.reduce(
        [
            (df["object_y1"] - df["hand_y2"]).to_numpy(dtype=float),
            (df["hand_y1"] - df["object_y2"]).to_numpy(dtype=float),
            np.zeros(len(df), dtype=float),
        ]
    )
    return pd.Series(np.hypot(dx, dy), index=df.index)


def build_pairs(
    full_id: str,
    objects: pd.DataFrame,
    hands: pd.DataFrame,
    image_width: float,
    image_height: float,
    fps: float,
) -> pd.DataFrame:
    if objects.empty or hands.empty:
        return pd.DataFrame()
    pairs = hands.merge(objects, on="frame_id", how="inner", sort=False)
    if pairs.empty:
        return pairs

    pairs.insert(0, "participant_id", participant_from_full_id(full_id))
    pairs.insert(1, "task_name", task_from_full_id(full_id))
    pairs.insert(2, "full_id", full_id)
    pairs["timestamp_s"] = pairs["frame_id"] / max(float(fps), EPS)
    pairs["center_distance"] = np.hypot(pairs["hand_cx"] - pairs["object_cx"], pairs["hand_cy"] - pairs["object_cy"])
    image_diagonal = max(float(np.hypot(image_width, image_height)), EPS)
    pairs["center_distance_norm"] = pairs["center_distance"] / image_diagonal
    pairs["bbox_iou"] = bbox_iou(pairs)
    pairs["bbox_gap"] = bbox_gap(pairs)
    object_size = np.sqrt(np.maximum(pairs["object_area"], EPS))
    hand_size = np.sqrt(np.maximum(pairs["hand_area"], EPS))
    pairs["bbox_gap_object_norm"] = pairs["bbox_gap"] / object_size
    pairs["center_distance_object_norm"] = pairs["center_distance"] / object_size
    pairs["center_distance_hand_norm"] = pairs["center_distance"] / hand_size
    pairs["hand_center_in_object_bbox"] = (
        pairs["hand_cx"].between(pairs["object_x1"], pairs["object_x2"])
        & pairs["hand_cy"].between(pairs["object_y1"], pairs["object_y2"])
    ).astype(int)
    pairs["object_center_in_hand_bbox"] = (
        pairs["object_cx"].between(pairs["hand_x1"], pairs["hand_x2"])
        & pairs["object_cy"].between(pairs["hand_y1"], pairs["hand_y2"])
    ).astype(int)
    gap_score = np.exp(-pairs["bbox_gap_object_norm"].clip(lower=0.0, upper=10.0))
    distance_score = np.exp(-pairs["center_distance_object_norm"].clip(lower=0.0, upper=10.0))
    pairs["simple_interaction_score"] = (
        0.35 * gap_score
        + 0.25 * pairs["bbox_iou"].clip(0.0, 1.0)
        + 0.15 * distance_score
        + 0.10 * pairs["hand_center_in_object_bbox"]
        + 0.10 * pairs["object_center_in_hand_bbox"]
        + 0.05 * pairs["object_conf"].clip(0.0, 1.0)
    ).clip(0.0, 1.0)
    pairs["contact_like"] = ((pairs["bbox_gap_object_norm"] <= 0.35) | (pairs["bbox_iou"] > 0.0)).astype(int)
    pairs["pair_id_in_frame"] = pairs.groupby("frame_id").cumcount()
    sort_cols = ["frame_id", "hand_side", "hand_detection_id", "simple_interaction_score", "object_conf"]
    return pairs.sort_values(sort_cols, ascending=[True, True, True, False, False]).reset_index(drop=True)


def empty_frame_summary(full_id: str, frames: list[int]) -> pd.DataFrame:
    rows = []
    for frame_id in frames:
        rows.append(
            {
                "participant_id": participant_from_full_id(full_id),
                "task_name": task_from_full_id(full_id),
                "full_id": full_id,
                "frame_id": int(frame_id),
                "hand_count": 0,
                "object_count": 0,
                "pair_count": 0,
            }
        )
    return pd.DataFrame(rows)


def add_topk_columns(row: dict, candidates: pd.DataFrame, prefix: str, top_k: int) -> dict:
    ordered = (
        candidates.sort_values(
            ["simple_interaction_score", "bbox_iou", "object_conf"],
            ascending=[False, False, False],
        )
        if not candidates.empty
        else pd.DataFrame()
    )
    for rank in range(1, top_k + 1):
        if len(ordered) < rank:
            row[f"{prefix}_top{rank}_hand_side"] = ""
            row[f"{prefix}_top{rank}_hand_detection_id"] = np.nan
            row[f"{prefix}_top{rank}_object_detection_id"] = np.nan
            row[f"{prefix}_top{rank}_object_class_id"] = np.nan
            row[f"{prefix}_top{rank}_object"] = ""
            row[f"{prefix}_top{rank}_score"] = np.nan
            row[f"{prefix}_top{rank}_gap_object_norm"] = np.nan
            row[f"{prefix}_top{rank}_bbox_iou"] = np.nan
            continue
        pair = ordered.iloc[rank - 1]
        row[f"{prefix}_top{rank}_hand_side"] = pair["hand_side"]
        row[f"{prefix}_top{rank}_hand_detection_id"] = int(pair["hand_detection_id"])
        row[f"{prefix}_top{rank}_object_detection_id"] = int(pair["object_detection_id"])
        row[f"{prefix}_top{rank}_object_class_id"] = int(pair["object_class_id"])
        row[f"{prefix}_top{rank}_object"] = pair["object_class"]
        row[f"{prefix}_top{rank}_score"] = float(pair["simple_interaction_score"])
        row[f"{prefix}_top{rank}_gap_object_norm"] = float(pair["bbox_gap_object_norm"])
        row[f"{prefix}_top{rank}_bbox_iou"] = float(pair["bbox_iou"])
    return row


def build_frame_summary(
    full_id: str,
    pairs: pd.DataFrame,
    objects: pd.DataFrame,
    hands: pd.DataFrame,
    top_k: int,
) -> pd.DataFrame:
    frames = sorted(set(objects["frame_id"].tolist()) | set(hands["frame_id"].tolist()))
    if not frames:
        return empty_frame_summary(full_id, [])
    object_counts = objects.groupby("frame_id").size()
    hand_counts = hands.groupby("frame_id").size()
    pair_counts = pairs.groupby("frame_id").size() if not pairs.empty else pd.Series(dtype=int)
    pair_groups = {frame_id: group for frame_id, group in pairs.groupby("frame_id", sort=False)} if not pairs.empty else {}
    rows = []
    for frame_id in frames:
        frame_pairs = pair_groups.get(frame_id, pd.DataFrame())
        row = {
            "participant_id": participant_from_full_id(full_id),
            "task_name": task_from_full_id(full_id),
            "full_id": full_id,
            "frame_id": int(frame_id),
            "hand_count": int(hand_counts.get(frame_id, 0)),
            "object_count": int(object_counts.get(frame_id, 0)),
            "pair_count": int(pair_counts.get(frame_id, 0)),
        }
        row = add_topk_columns(row, frame_pairs, "all", top_k)
        for side in ("left", "right", "unknown"):
            side_pairs = frame_pairs[frame_pairs["hand_side"].eq(side)] if not frame_pairs.empty else frame_pairs
            row = add_topk_columns(row, side_pairs, side, top_k)
        rows.append(row)
    return pd.DataFrame(rows)


def diagnostics_for_recording(
    full_id: str,
    metadata: dict,
    object_diag: dict,
    hand_diag: dict,
    objects: pd.DataFrame,
    hands: pd.DataFrame,
    pairs: pd.DataFrame,
    frame_summary: pd.DataFrame,
) -> dict:
    video_frame_count = int(metadata.get("video_frame_count") or 0)
    object_frames = set(objects["frame_id"].tolist())
    hand_frames = set(hands["frame_id"].tolist())
    pair_frames = set(pairs["frame_id"].tolist()) if not pairs.empty else set()
    union_frames = object_frames | hand_frames
    return {
        "participant_id": participant_from_full_id(full_id),
        "task_name": task_from_full_id(full_id),
        "full_id": full_id,
        **metadata,
        **object_diag,
        **hand_diag,
        "pair_rows": int(len(pairs)),
        "pair_frames": int(len(pair_frames)),
        "frame_summary_rows": int(len(frame_summary)),
        "object_and_hand_common_frames": int(len(object_frames & hand_frames)),
        "object_without_hand_frames": int(len(object_frames - hand_frames)),
        "hand_without_object_frames": int(len(hand_frames - object_frames)),
        "union_detection_frames": int(len(union_frames)),
        "video_frames_without_kept_object": max(video_frame_count - len(object_frames), 0) if video_frame_count else np.nan,
        "video_frames_without_kept_hand": max(video_frame_count - len(hand_frames), 0) if video_frame_count else np.nan,
        "video_frames_without_pair": max(video_frame_count - len(pair_frames), 0) if video_frame_count else np.nan,
        "object_frame_coverage_of_video": len(object_frames) / video_frame_count if video_frame_count else np.nan,
        "hand_frame_coverage_of_video": len(hand_frames) / video_frame_count if video_frame_count else np.nan,
        "pair_frame_coverage_of_video": len(pair_frames) / video_frame_count if video_frame_count else np.nan,
    }


def process_recording(full_id: str, object_path: Path, hand_path: Path, args: argparse.Namespace) -> dict:
    metadata = video_metadata(full_id, args.video_root)
    image_width = float(args.image_width or metadata.get("video_width") or 0.0)
    image_height = float(args.image_height or metadata.get("video_height") or 0.0)
    image_area = image_width * image_height if image_width > 0 and image_height > 0 else None
    fps = float(args.fps or metadata.get("video_fps") or 30.0)

    objects, object_diag = read_objects(object_path, full_id, image_area)
    hands, hand_diag = read_hands(hand_path)
    if image_area is None:
        max_x = max(objects["object_x2"].max() if len(objects) else 0.0, hands["hand_x2"].max() if len(hands) else 0.0)
        max_y = max(objects["object_y2"].max() if len(objects) else 0.0, hands["hand_y2"].max() if len(hands) else 0.0)
        image_width = float(max_x)
        image_height = float(max_y)

    pairs = build_pairs(full_id, objects, hands, image_width=image_width, image_height=image_height, fps=fps)
    frame_summary = build_frame_summary(full_id, pairs, objects, hands, top_k=max(1, args.top_k))

    participant = participant_from_full_id(full_id)
    pair_path = args.output_root / "pair_features" / participant / f"{full_id}{PAIR_SUFFIX}.csv"
    frame_path = args.output_root / "frame_features" / participant / f"{full_id}{FRAME_SUFFIX}.csv"
    pair_path.parent.mkdir(parents=True, exist_ok=True)
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(pair_path, index=False)
    frame_summary.to_csv(frame_path, index=False)

    diagnostics = diagnostics_for_recording(
        full_id,
        metadata,
        object_diag,
        hand_diag,
        objects,
        hands,
        pairs,
        frame_summary,
    )
    return {
        "participant_id": participant,
        "task_name": task_from_full_id(full_id),
        "full_id": full_id,
        "object_path": str(object_path),
        "hand_path": str(hand_path),
        "pair_path": str(pair_path),
        "frame_path": str(frame_path),
        "pair_rows": int(len(pairs)),
        "frame_rows": int(len(frame_summary)),
        "status": "ok",
        "diagnostics": diagnostics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-root", type=Path, default=DEFAULT_OBJECT_ROOT)
    parser.add_argument("--hand-root", type=Path, default=DEFAULT_HAND_ROOT)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--participant")
    parser.add_argument("--full-id", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--image-width", type=float)
    parser.add_argument("--image-height", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    object_paths = world_camera_csvs(args.object_root)
    hand_paths = world_camera_csvs(args.hand_root)
    full_ids = sorted(set(object_paths) & set(hand_paths))
    if args.participant:
        full_ids = [full_id for full_id in full_ids if full_id.startswith(f"{args.participant}_")]
    if args.full_id:
        wanted = set(args.full_id)
        full_ids = [full_id for full_id in full_ids if full_id in wanted]
    if args.limit is not None:
        full_ids = full_ids[: args.limit]

    missing = [
        {"full_id": full_id, "missing_side": "hand", "object_path": str(object_paths[full_id])}
        for full_id in sorted(set(object_paths) - set(hand_paths))
    ]
    missing.extend(
        {"full_id": full_id, "missing_side": "object", "hand_path": str(hand_paths[full_id])}
        for full_id in sorted(set(hand_paths) - set(object_paths))
    )

    records = []
    diagnostics = []
    iterator = tqdm(full_ids, desc="HOI v2 simple") if tqdm is not None else full_ids
    for full_id in iterator:
        participant = participant_from_full_id(full_id)
        pair_path = args.output_root / "pair_features" / participant / f"{full_id}{PAIR_SUFFIX}.csv"
        frame_path = args.output_root / "frame_features" / participant / f"{full_id}{FRAME_SUFFIX}.csv"
        if pair_path.exists() and frame_path.exists() and not args.overwrite:
            records.append(
                {
                    "participant_id": participant,
                    "task_name": task_from_full_id(full_id),
                    "full_id": full_id,
                    "object_path": str(object_paths[full_id]),
                    "hand_path": str(hand_paths[full_id]),
                    "pair_path": str(pair_path),
                    "frame_path": str(frame_path),
                    "status": "skipped_exists",
                }
            )
            continue
        record = process_recording(full_id, object_paths[full_id], hand_paths[full_id], args)
        diagnostics.append(record.pop("diagnostics"))
        records.append(record)

    args.output_root.mkdir(parents=True, exist_ok=True)
    index_path = args.output_root / "hand_object_interaction_v2_index.csv"
    diagnostics_path = args.output_root / "hand_object_interaction_v2_coverage_diagnostics.csv"
    missing_path = args.output_root / "hand_object_interaction_v2_missing_inputs.csv"
    summary_path = args.output_root / "hand_object_interaction_v2_summary.json"

    pd.DataFrame(records).to_csv(index_path, index=False)
    pd.DataFrame(diagnostics).to_csv(diagnostics_path, index=False)
    pd.DataFrame(missing).to_csv(missing_path, index=False)
    index_df = pd.DataFrame(records)
    summary = {
        "matched_recording_count": len(full_ids),
        "processed_recording_count": int(sum(record.get("status") == "ok" for record in records)),
        "skipped_recording_count": int(sum(record.get("status") == "skipped_exists" for record in records)),
        "missing_input_count": len(missing),
        "pair_rows": int(pd.to_numeric(index_df.get("pair_rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
        "frame_rows": int(pd.to_numeric(index_df.get("frame_rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
        "object_root": str(args.object_root),
        "hand_root": str(args.hand_root),
        "video_root": str(args.video_root),
        "output_root": str(args.output_root),
        "top_k": max(1, args.top_k),
        "object_filter_policy": "drop class_id==0 and object_area_ratio>0.5 only, plus invalid boxes",
        "phase_policy": "no phase labels in v2 simple pipeline",
        "index_path": str(index_path),
        "diagnostics_path": str(diagnostics_path),
        "missing_path": str(missing_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
