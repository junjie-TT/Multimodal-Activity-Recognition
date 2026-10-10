#!/usr/bin/env python3
"""Render short demo videos from simple conservative HOI v2 outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import cv2


DATASET_ROOT = Path("/home/julius/datasets/seminar_object_detection")
DEFAULT_HOI_ROOT = DATASET_ROOT / "upper_feature_outcome" / "hand_object_interaction_v2_simple"
DEFAULT_VIDEO_ROOT = DATASET_ROOT / "iAssistADL_dataset_export_2026-05-18" / "world_camera_videos"
DEFAULT_OUTPUT_ROOT = DATASET_ROOT / "video_demonstration" / "hand_object_interaction_v2_simple"

WORLD_CAMERA_SUFFIX = "_world_camera"
PAIR_SUFFIX = "_hoi_v2_pair_features"
TARGET_ACTIONS = ("brief_schreiben", "tasche_packen", "tee_kochen")
SIDE_COLORS = {
    "left": (50, 220, 255),
    "right": (80, 160, 255),
    "unknown": (220, 220, 220),
}
OBJECT_COLORS = {
    "left": (80, 255, 130),
    "right": (255, 170, 80),
    "unknown": (200, 200, 200),
}
TEXT_BG = (20, 20, 20)


def participant_from_full_id(full_id: str) -> str:
    return "_".join(full_id.split("_")[:4])


def task_from_full_id(full_id: str) -> str:
    participant = participant_from_full_id(full_id)
    prefix = f"{participant}_"
    return full_id[len(prefix) :] if full_id.startswith(prefix) else full_id


def action_family(full_id: str) -> str:
    task = re.sub(r"\d+$", "", task_from_full_id(full_id)).strip("_").lower()
    return task.replace("teekochen", "tee_kochen").replace("several_tasks", "several_task")


def pair_path_for(full_id: str, hoi_root: Path) -> Path:
    return hoi_root / "pair_features" / participant_from_full_id(full_id) / f"{full_id}{PAIR_SUFFIX}.csv"


def video_path_for(full_id: str, video_root: Path) -> Path:
    return video_root / participant_from_full_id(full_id) / f"{full_id}{WORLD_CAMERA_SUFFIX}.mp4"


def as_float(value: str | None, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except ValueError:
        return default


def as_int(value: str | None, default: int = 0) -> int:
    return int(round(as_float(value, float(default))))


def read_index(index_path: Path) -> list[dict[str, str]]:
    with index_path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_examples(records: list[dict[str, str]], actions: tuple[str, ...]) -> list[dict[str, str]]:
    best: dict[str, dict[str, str]] = {}
    for record in records:
        if record.get("status") != "ok":
            continue
        family = action_family(record["full_id"])
        if family not in actions:
            continue
        pair_rows = as_int(record.get("pair_rows"))
        previous = best.get(family)
        if previous is None or pair_rows > as_int(previous.get("pair_rows")):
            best[family] = record
    missing = [action for action in actions if action not in best]
    if missing:
        raise ValueError(f"Could not find HOI v2 records for action families: {missing}")
    return [best[action] for action in actions]


def load_pairs(pair_path: Path) -> dict[int, list[dict[str, str]]]:
    by_frame: dict[int, list[dict[str, str]]] = defaultdict(list)
    with pair_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frame_id = as_int(row.get("frame_id"), -1)
            if frame_id >= 0:
                by_frame[frame_id].append(row)
    return dict(by_frame)


def choose_clip(by_frame: dict[int, list[dict[str, str]]], fps: float, frame_count: int, clip_seconds: float) -> tuple[int, int]:
    if frame_count <= 0:
        frame_count = max(by_frame.keys(), default=0) + 1
    window = max(1, min(int(round(clip_seconds * fps)), frame_count))
    scores = [0.0] * frame_count
    for frame_id, rows in by_frame.items():
        if 0 <= frame_id < frame_count:
            top_scores = sorted((as_float(row.get("simple_interaction_score")) for row in rows), reverse=True)[:4]
            contact_bonus = sum(0.08 for row in rows if as_int(row.get("contact_like")) == 1)
            scores[frame_id] = sum(top_scores) + contact_bonus
    current = sum(scores[:window])
    best_score = current
    best_start = 0
    for start in range(1, max(1, frame_count - window + 1)):
        current += scores[start + window - 1] - scores[start - 1]
        if current > best_score:
            best_score = current
            best_start = start
    return best_start, best_start + window


def put_text_box(frame, text: str, x: int, y: int, color=(255, 255, 255), scale=0.52) -> None:
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    cv2.rectangle(frame, (x - 4, y - height - baseline - 4), (x + width + 4, y + baseline + 4), TEXT_BG, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_box(frame, row: dict[str, str], prefix: str, color, label: str) -> None:
    x1 = as_int(row.get(f"{prefix}_x1"))
    y1 = as_int(row.get(f"{prefix}_y1"))
    x2 = as_int(row.get(f"{prefix}_x2"))
    y2 = as_int(row.get(f"{prefix}_y2"))
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    put_text_box(frame, label, x1, max(18, y1 - 7), color=color, scale=0.45)


def top_rows(rows: list[dict[str, str]], side: str, top_k: int) -> list[dict[str, str]]:
    side_rows = [row for row in rows if row.get("hand_side") == side]
    return sorted(
        side_rows,
        key=lambda row: (
            as_float(row.get("simple_interaction_score")),
            as_float(row.get("bbox_iou")),
            as_float(row.get("object_conf")),
        ),
        reverse=True,
    )[:top_k]


def draw_pairs(frame, rows: list[dict[str, str]], top_k: int) -> list[str]:
    panel_lines: list[str] = []
    for side in ("left", "right", "unknown"):
        selected = top_rows(rows, side, top_k)
        if not selected:
            continue
        for rank, row in enumerate(selected, start=1):
            score = as_float(row.get("simple_interaction_score"))
            alpha = max(0.4, 1.0 - 0.22 * (rank - 1))
            hand_color = tuple(int(channel * alpha) for channel in SIDE_COLORS[side])
            object_color = tuple(int(channel * alpha) for channel in OBJECT_COLORS[side])
            obj = row.get("object_class") or f"class_{row.get('object_class_id', '')}"
            draw_box(frame, row, "object", object_color, f"{side[0].upper()} top{rank} {obj} {score:.2f}")
            if rank == 1:
                draw_box(frame, row, "hand", hand_color, f"{side} hand")
            if rank <= 3:
                gap = as_float(row.get("bbox_gap_object_norm"), math.nan)
                iou = as_float(row.get("bbox_iou"), math.nan)
                panel_lines.append(f"{side[0].upper()} {rank}: {obj} score={score:.2f} gap={gap:.2f} iou={iou:.2f}")
    return panel_lines


def draw_panel(frame, full_id: str, frame_id: int, start_frame: int, fps: float, lines: list[str]) -> None:
    height, width = frame.shape[:2]
    title = f"{task_from_full_id(full_id)}  frame {frame_id}  clip t={(frame_id - start_frame) / max(fps, 1e-6):.1f}s"
    lines = [title, *lines[:8]]
    panel_w = min(width, max(560, int(width * 0.42)))
    panel_h = min(height, 28 + 24 * len(lines))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
    y = 24
    for index, line in enumerate(lines):
        color = (255, 255, 255) if index == 0 else (225, 238, 240)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
        y += 24


def render_demo(
    full_id: str,
    pair_path: Path,
    video_path: Path,
    output_path: Path,
    clip_seconds: float,
    top_k: int,
    overwrite: bool,
) -> dict:
    if output_path.exists() and not overwrite:
        return {"full_id": full_id, "status": "skipped_exists", "output_path": str(output_path)}
    if not video_path.exists():
        return {"full_id": full_id, "status": "missing_video", "video_path": str(video_path)}
    if not pair_path.exists():
        return {"full_id": full_id, "status": "missing_pair", "pair_path": str(pair_path)}

    by_frame = load_pairs(pair_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"full_id": full_id, "status": "cannot_open_video", "video_path": str(video_path)}
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame, end_frame = choose_clip(by_frame, fps, frame_count, clip_seconds)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        return {"full_id": full_id, "status": "cannot_open_writer", "output_path": str(output_path)}

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_id = start_frame
    written = 0
    while frame_id < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        rows = by_frame.get(frame_id, [])
        panel_lines = draw_pairs(frame, rows, top_k)
        draw_panel(frame, full_id, frame_id, start_frame, fps, panel_lines or ["no HOI pair in this frame"])
        writer.write(frame)
        frame_id += 1
        written += 1

    cap.release()
    writer.release()
    return {
        "full_id": full_id,
        "action_family": action_family(full_id),
        "status": "ok",
        "video_path": str(video_path),
        "pair_path": str(pair_path),
        "output_path": str(output_path),
        "fps": fps,
        "start_frame": start_frame,
        "end_frame": frame_id,
        "written_frame_count": written,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hoi-root", type=Path, default=DEFAULT_HOI_ROOT)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--actions", nargs="+", default=list(TARGET_ACTIONS))
    parser.add_argument("--full-id", nargs="+", help="Render explicit recordings instead of auto-selecting by action.")
    parser.add_argument("--clip-seconds", type=float, default=45.0)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_path = args.hoi_root / "hand_object_interaction_v2_index.csv"
    if args.full_id:
        selected = [{"full_id": full_id, "pair_rows": "", "status": "ok"} for full_id in args.full_id[:3]]
    else:
        selected = select_examples(read_index(index_path), tuple(args.actions))

    records = []
    for record in selected:
        full_id = record["full_id"]
        participant = participant_from_full_id(full_id)
        output_path = args.output_root / participant / f"{full_id}_hoi_v2_simple_demo.mp4"
        records.append(
            render_demo(
                full_id=full_id,
                pair_path=pair_path_for(full_id, args.hoi_root),
                video_path=video_path_for(full_id, args.video_root),
                output_path=output_path,
                clip_seconds=max(1.0, args.clip_seconds),
                top_k=max(1, args.top_k),
                overwrite=args.overwrite,
            )
        )

    args.output_root.mkdir(parents=True, exist_ok=True)
    index_out = args.output_root / "hoi_v2_simple_demo_video_index.csv"
    summary_out = args.output_root / "hoi_v2_simple_demo_video_summary.json"
    with index_out.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = sorted({key for record in records for key in record})
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "selected_full_ids": [record["full_id"] for record in records],
        "ok_count": sum(1 for record in records if record.get("status") == "ok"),
        "output_root": str(args.output_root),
        "index_path": str(index_out),
        "rendered_videos": [record.get("output_path") for record in records if record.get("status") == "ok"],
    }
    summary_out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
