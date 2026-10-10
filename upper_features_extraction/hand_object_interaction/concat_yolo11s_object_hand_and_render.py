#!/usr/bin/env python3
"""Concatenate YOLO11s object/hand detections and render three demo videos."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


DATASET_ROOT = Path("/home/julius/datasets/seminar_object_detection")
DEFAULT_OBJECT_ROOT = DATASET_ROOT / "lower_feature_outcome" / "yolo11s_coco_object_outcome_csv"
DEFAULT_HAND_ROOT = DATASET_ROOT / "lower_feature_outcome" / "yolo11s_left_right_hand_outcome_csv"
DEFAULT_CONCAT_ROOT = DATASET_ROOT / "lower_feature_outcome" / "yolo11s_object_hand_concat_csv"
DEFAULT_VIDEO_ROOT = DATASET_ROOT / "iAssistADL_dataset_export_2026-05-18" / "world_camera_videos"
DEFAULT_DEMO_ROOT = DATASET_ROOT / "video_demonstration" / "yolo11s_object_hand_concat"

WORLD_CAMERA_SUFFIX = "_world_camera"
REQUIRED_COLUMNS = ("frame_id", "class_id", "confidence", "x1", "y1", "x2", "y2")
HAND_COLORS = {"left": (40, 220, 255), "right": (80, 160, 255), "unknown": (220, 220, 220)}
OBJECT_COLOR = (80, 255, 120)
TEXT_BG = (20, 20, 20)


def full_id_from_path(path: Path) -> str:
    stem = path.stem
    return stem[: -len(WORLD_CAMERA_SUFFIX)] if stem.endswith(WORLD_CAMERA_SUFFIX) else stem


def participant_from_full_id(full_id: str) -> str:
    return "_".join(full_id.split("_")[:4])


def task_name_from_full_id(full_id: str) -> str:
    participant = participant_from_full_id(full_id)
    prefix = f"{participant}_"
    return full_id[len(prefix) :] if full_id.startswith(prefix) else full_id


def action_family(full_id: str) -> str:
    family = re.sub(r"\d+$", "", task_name_from_full_id(full_id)).strip("_").lower()
    family = family.replace("several_tasks", "several_task")
    return family


def world_camera_csvs(root: Path) -> dict[str, Path]:
    return {full_id_from_path(path): path for path in sorted(root.glob("*_world_camera.csv"))}


def require_columns(path: Path, columns: pd.Index) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def hand_side_from_row(row: pd.Series) -> str:
    class_name = str(row.get("class_name", "")).lower()
    if "left" in class_name:
        return "left"
    if "right" in class_name:
        return "right"
    class_id = row.get("class_id")
    if pd.isna(class_id):
        return "unknown"
    return "left" if int(class_id) == 0 else "right" if int(class_id) == 1 else "unknown"


def normalized_detection_table(path: Path, source: str, full_id: str) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns
    require_columns(path, header)
    df = pd.read_csv(path)
    for column in REQUIRED_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=list(REQUIRED_COLUMNS)).copy()
    df["frame_id"] = df["frame_id"].round().astype(int)
    df["class_id"] = df["class_id"].round().astype(int)
    df["detection_source"] = source
    df["full_id"] = full_id
    df["participant_id"] = participant_from_full_id(full_id)
    df["task_name"] = task_name_from_full_id(full_id)
    df["hand_side"] = ""
    if source == "hand":
        df["hand_side"] = df.apply(hand_side_from_row, axis=1)
    if "class_name" not in df.columns:
        df["class_name"] = ""
    wanted = [
        "full_id",
        "participant_id",
        "task_name",
        "frame_id",
        "detection_source",
        "hand_side",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
    ]
    return df[wanted].sort_values(["frame_id", "detection_source", "confidence"], ascending=[True, True, False])


def concat_recording(full_id: str, object_path: Path, hand_path: Path, output_path: Path) -> dict:
    object_df = normalized_detection_table(object_path, "object", full_id)
    hand_df = normalized_detection_table(hand_path, "hand", full_id)
    concat_df = pd.concat([object_df, hand_df], ignore_index=True, sort=False)
    concat_df = concat_df.sort_values(
        ["frame_id", "detection_source", "hand_side", "confidence"],
        ascending=[True, True, True, False],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    concat_df.to_csv(output_path, index=False)

    object_frames = set(object_df["frame_id"].unique())
    hand_frames = set(hand_df["frame_id"].unique())
    return {
        "full_id": full_id,
        "participant_id": participant_from_full_id(full_id),
        "task_name": task_name_from_full_id(full_id),
        "object_path": str(object_path),
        "hand_path": str(hand_path),
        "concat_path": str(output_path),
        "object_rows": int(len(object_df)),
        "hand_rows": int(len(hand_df)),
        "concat_rows": int(len(concat_df)),
        "object_frame_count": int(len(object_frames)),
        "hand_frame_count": int(len(hand_frames)),
        "common_frame_count": int(len(object_frames & hand_frames)),
    }


def put_text(frame, text: str, x: int, y: int, color, scale=0.48) -> None:
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(frame, (x - 3, y - h - baseline - 4), (x + w + 3, y + baseline + 3), TEXT_BG, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_row(frame, row: pd.Series) -> None:
    x1, y1, x2, y2 = [int(round(float(row[col]))) for col in ("x1", "y1", "x2", "y2")]
    if x2 <= x1 or y2 <= y1:
        return
    source = row["detection_source"]
    if source == "hand":
        side = row.get("hand_side") or "unknown"
        color = HAND_COLORS.get(side, HAND_COLORS["unknown"])
        label = f"{side} hand {float(row['confidence']):.2f}"
    else:
        color = OBJECT_COLOR
        class_name = str(row.get("class_name") or "").strip()
        class_label = class_name if class_name else f"obj {int(row['class_id'])}"
        label = f"{class_label} {float(row['confidence']):.2f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    put_text(frame, label, x1, max(18, y1 - 6), color)


def load_concat_by_frame(concat_path: Path) -> dict[int, pd.DataFrame]:
    df = pd.read_csv(concat_path)
    df["frame_id"] = pd.to_numeric(df["frame_id"], errors="coerce")
    df = df.dropna(subset=["frame_id"]).copy()
    df["frame_id"] = df["frame_id"].astype(int)
    return {int(frame_id): group.copy() for frame_id, group in df.groupby("frame_id", sort=False)}


def render_video(full_id: str, concat_path: Path, video_root: Path, output_root: Path, overwrite: bool) -> dict:
    participant = participant_from_full_id(full_id)
    video_path = video_root / participant / f"{full_id}{WORLD_CAMERA_SUFFIX}.mp4"
    output_path = output_root / participant / f"{full_id}_yolo11s_object_hand_concat.mp4"
    if output_path.exists() and not overwrite:
        return {"full_id": full_id, "status": "skipped_exists", "output_path": str(output_path)}
    if not video_path.exists():
        return {"full_id": full_id, "status": "missing_video", "video_path": str(video_path)}
    if not concat_path.exists():
        return {"full_id": full_id, "status": "missing_concat", "concat_path": str(concat_path)}

    by_frame = load_concat_by_frame(concat_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"full_id": full_id, "status": "cannot_open_video", "video_path": str(video_path)}

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return {"full_id": full_id, "status": "cannot_open_writer", "output_path": str(output_path)}

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rows = by_frame.get(frame_idx)
        if rows is not None:
            for _, row in rows.sort_values(["detection_source", "confidence"], ascending=[True, False]).iterrows():
                draw_row(frame, row)
        put_text(frame, f"{full_id}  frame {frame_idx}", 12, 26, (255, 255, 255), scale=0.58)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return {
        "full_id": full_id,
        "status": "ok",
        "video_path": str(video_path),
        "concat_path": str(concat_path),
        "output_path": str(output_path),
        "written_frame_count": frame_idx,
    }


def select_three_examples(records: list[dict]) -> list[str]:
    best_by_action: dict[str, tuple[int, str]] = {}
    for record in records:
        score = int(record["common_frame_count"])
        family = action_family(record["full_id"])
        previous = best_by_action.get(family)
        if previous is None or score > previous[0]:
            best_by_action[family] = (score, record["full_id"])
    return [full_id for _, full_id in sorted(best_by_action.values(), reverse=True)[:3]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-root", type=Path, default=DEFAULT_OBJECT_ROOT)
    parser.add_argument("--hand-root", type=Path, default=DEFAULT_HAND_ROOT)
    parser.add_argument("--concat-root", type=Path, default=DEFAULT_CONCAT_ROOT)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--demo-root", type=Path, default=DEFAULT_DEMO_ROOT)
    parser.add_argument("--participant")
    parser.add_argument("--full-id", nargs="+")
    parser.add_argument("--video-full-id", nargs="+", help="Optional exact recordings to render; at most 3 are used.")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite-videos", action="store_true")
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

    iterator = tqdm(full_ids, desc="concat yolo11s object+hand") if tqdm is not None else full_ids
    records = []
    for full_id in iterator:
        participant = participant_from_full_id(full_id)
        output_path = args.concat_root / participant / f"{full_id}{WORLD_CAMERA_SUFFIX}.csv"
        if output_path.exists() and not args.overwrite:
            records.append({"full_id": full_id, "participant_id": participant, "concat_path": str(output_path), "status": "skipped_exists"})
            continue
        record = concat_recording(full_id, object_paths[full_id], hand_paths[full_id], output_path)
        record["status"] = "ok"
        records.append(record)

    render_ids = (args.video_full_id or select_three_examples([record for record in records if "common_frame_count" in record]))[:3]
    render_records = []
    for full_id in render_ids:
        participant = participant_from_full_id(full_id)
        concat_path = args.concat_root / participant / f"{full_id}{WORLD_CAMERA_SUFFIX}.csv"
        render_records.append(render_video(full_id, concat_path, args.video_root, args.demo_root, args.overwrite_videos))

    args.concat_root.mkdir(parents=True, exist_ok=True)
    args.demo_root.mkdir(parents=True, exist_ok=True)
    index_path = args.concat_root / "yolo11s_object_hand_concat_index.csv"
    missing_path = args.concat_root / "yolo11s_object_hand_concat_missing.csv"
    video_index_path = args.demo_root / "yolo11s_object_hand_concat_video_index.csv"
    summary_path = args.concat_root / "yolo11s_object_hand_concat_summary.json"
    pd.DataFrame(records).to_csv(index_path, index=False)
    pd.DataFrame(missing).to_csv(missing_path, index=False)
    pd.DataFrame(render_records).to_csv(video_index_path, index=False)
    summary = {
        "matched_recording_count": len(full_ids),
        "processed_recording_count": int(sum(record.get("status") == "ok" for record in records)),
        "missing_recording_count": len(missing),
        "rendered_video_count": int(sum(record.get("status") == "ok" for record in render_records)),
        "object_root": str(args.object_root),
        "hand_root": str(args.hand_root),
        "concat_root": str(args.concat_root),
        "demo_root": str(args.demo_root),
        "index_path": str(index_path),
        "missing_path": str(missing_path),
        "video_index_path": str(video_index_path),
        "rendered_videos": [record.get("output_path") for record in render_records if record.get("status") == "ok"],
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
