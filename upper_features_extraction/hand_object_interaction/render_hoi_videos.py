#!/usr/bin/env python3
"""Render full hand-object interaction inspection videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from upper_features_extraction import config

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


DATASET_ROOT = Path("/home/julius/datasets/seminar_object_detection")
DEFAULT_VIDEO_ROOT = DATASET_ROOT / "iAssistADL_dataset_export_2026-05-18" / "world_camera_videos"
DEFAULT_OUTPUT_ROOT = DATASET_ROOT / "video_demonstration" / "hand_object_interaction"
FRAME_SUFFIX = "_hand_object_frame_features"
PAIR_SUFFIX = "_hand_object_pair_features"
WORLD_CAMERA_SUFFIX = "_world_camera"
SIDES = ("left", "right")
SIDE_PREFIX = {"left": "L", "right": "R"}
SIDE_COLOR = {"left": (60, 220, 255), "right": (80, 180, 255)}
OBJECT_COLOR = {"left": (80, 255, 120), "right": (255, 160, 80)}
TEXT_BG = (20, 20, 20)


def parse_full_id(path: Path, suffix: str) -> str:
    stem = path.stem
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def participant_from_full_id(full_id: str) -> str:
    return "_".join(full_id.split("_")[:4])


def iter_frame_paths(frame_root: Path, participant: str | None, full_ids: set[str] | None, limit: int | None):
    paths = sorted(frame_root.glob("*/*_hand_object_frame_features.csv"))
    if participant:
        paths = [path for path in paths if path.parent.name == participant]
    if full_ids:
        paths = [path for path in paths if parse_full_id(path, FRAME_SUFFIX) in full_ids]
    if limit is not None:
        paths = paths[:limit]
    return paths


def video_path_for(full_id: str, video_root: Path) -> Path:
    participant = participant_from_full_id(full_id)
    return video_root / participant / f"{full_id}{WORLD_CAMERA_SUFFIX}.mp4"


def pair_path_for(full_id: str, pair_root: Path) -> Path:
    participant = participant_from_full_id(full_id)
    return pair_root / participant / f"{full_id}{PAIR_SUFFIX}.csv"


def as_int(value, default=0):
    if pd.isna(value):
        return default
    return int(round(float(value)))


def as_float(value, default=np.nan):
    if pd.isna(value):
        return default
    return float(value)


def put_text_box(frame, text, xy, scale=0.55, color=(255, 255, 255), bg=TEXT_BG, thickness=1):
    x, y = xy
    (w, h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(frame, (x - 3, y - h - baseline - 4), (x + w + 3, y + baseline + 3), bg, -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_box(frame, row, prefix, color, label):
    x1 = as_int(row.get(f"{prefix}_x1", np.nan))
    y1 = as_int(row.get(f"{prefix}_y1", np.nan))
    x2 = as_int(row.get(f"{prefix}_x2", np.nan))
    y2 = as_int(row.get(f"{prefix}_y2", np.nan))
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    put_text_box(frame, label, (x1, max(18, y1 - 6)), scale=0.45, color=color, thickness=1)


def load_by_frame(path: Path, wanted_columns: list[str] | None = None) -> dict[int, pd.DataFrame]:
    if not path.exists():
        return {}
    if wanted_columns is None:
        df = pd.read_csv(path)
    else:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        usecols = [col for col in wanted_columns if col in header]
        df = pd.read_csv(path, usecols=usecols)
    if "frame_id" not in df.columns or df.empty:
        return {}
    df["frame_id"] = pd.to_numeric(df["frame_id"], errors="coerce")
    df = df.dropna(subset=["frame_id"]).copy()
    df["frame_id"] = df["frame_id"].astype(int)
    return {int(frame_id): group.copy() for frame_id, group in df.groupby("frame_id", sort=False)}


def top_pair_rows(pair_frame: pd.DataFrame, side: str, top_k: int) -> pd.DataFrame:
    if pair_frame.empty:
        return pair_frame
    side_df = pair_frame[pair_frame["hand_side"].eq(side)]
    if side_df.empty:
        return side_df
    score_col = "selection_score" if "selection_score" in side_df.columns else "interaction_score"
    sort_cols = [score_col]
    for col in ("visual_object_score", "interaction_score", "object_conf"):
        if col in side_df.columns and col not in sort_cols:
            sort_cols.append(col)
    return side_df.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(top_k)


def draw_pairs(frame, pair_frame: pd.DataFrame, top_k: int):
    for side in SIDES:
        rows = top_pair_rows(pair_frame, side, top_k)
        for rank, (_, row) in enumerate(rows.iterrows(), start=1):
            alpha = max(0.35, 1.0 - 0.22 * (rank - 1))
            hand_color = tuple(int(channel * alpha) for channel in SIDE_COLOR[side])
            object_color = tuple(int(channel * alpha) for channel in OBJECT_COLOR[side])
            score = as_float(row.get("selection_score", row.get("interaction_score", np.nan)))
            object_label = f"{SIDE_PREFIX[side]} top{rank} {row.get('object_class', '')} {score:.2f}"
            draw_box(frame, row, "object", object_color, object_label)
            if rank == 1:
                draw_box(frame, row, "hand", hand_color, f"{SIDE_PREFIX[side]} hand")


def frame_summary_lines(row: pd.Series | None, top_k: int) -> list[str]:
    if row is None:
        return ["no HOI frame summary"]
    lines = []
    frame_id = as_int(row.get("frame_id", 0))
    timestamp = as_float(row.get("timestamp", 0.0), 0.0)
    lines.append(f"frame {frame_id}  t={timestamp:.2f}s")
    for side in ("L", "R"):
        active = row.get(f"{side}_active_object", np.nan)
        sel = as_float(row.get(f"{side}_selection_score", np.nan))
        step = row.get(f"{side}_step_object", np.nan)
        step_score = as_float(row.get(f"{side}_step_score", np.nan))
        temporal = row.get(f"{side}_temporal_0p5_object", np.nan)
        temporal_score = as_float(row.get(f"{side}_temporal_0p5_score", np.nan))
        lines.append(f"{side} active: {active if pd.notna(active) else '-'} sel={sel:.2f}")
        lines.append(f"{side} 0.5s:   {temporal if pd.notna(temporal) else '-'} score={temporal_score:.2f}")
        lines.append(f"{side} step:   {step if pd.notna(step) else '-'} score={step_score:.2f}")
        top_parts = []
        for rank in range(1, top_k + 1):
            obj = row.get(f"{side}_top{rank}_object", np.nan)
            score = as_float(row.get(f"{side}_top{rank}_selection_score", np.nan))
            if pd.notna(obj):
                top_parts.append(f"{rank}:{obj} {score:.2f}")
        if top_parts:
            lines.append(f"{side} top{top_k}:  " + " | ".join(top_parts))
    return lines


def draw_panel(frame, lines: list[str]):
    h, w = frame.shape[:2]
    panel_w = min(680, max(440, int(w * 0.36)))
    panel_h = min(h, 28 + 23 * len(lines))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
    y = 24
    for i, line in enumerate(lines):
        color = (255, 255, 255) if i == 0 else (220, 235, 240)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)
        y += 23


def render_video(
    full_id: str,
    video_path: Path,
    frame_path: Path,
    pair_path: Path,
    output_path: Path,
    fps_override: float | None,
    top_k: int,
    overwrite: bool,
):
    if output_path.exists() and not overwrite:
        return {"full_id": full_id, "status": "skipped_exists", "output_path": str(output_path)}
    if not video_path.exists():
        return {"full_id": full_id, "status": "missing_video", "video_path": str(video_path)}
    if not pair_path.exists():
        return {"full_id": full_id, "status": "missing_pair", "pair_path": str(pair_path)}

    frame_columns = ["frame_id", "timestamp"]
    for side in ("L", "R"):
        frame_columns.extend(
            [
                f"{side}_active_object",
                f"{side}_selection_score",
                f"{side}_temporal_0p5_object",
                f"{side}_temporal_0p5_score",
                f"{side}_step_object",
                f"{side}_step_score",
            ]
        )
        for rank in range(1, top_k + 1):
            frame_columns.extend([f"{side}_top{rank}_object", f"{side}_top{rank}_selection_score"])
    pair_columns = [
        "frame_id",
        "hand_side",
        "object_class",
        "selection_score",
        "interaction_score",
        "visual_object_score",
        "object_conf",
        "hand_x1",
        "hand_y1",
        "hand_x2",
        "hand_y2",
        "object_x1",
        "object_y1",
        "object_x2",
        "object_y2",
    ]
    frame_by_id = load_by_frame(frame_path, frame_columns)
    pair_by_id = load_by_frame(pair_path, pair_columns)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"full_id": full_id, "status": "cannot_open_video", "video_path": str(video_path)}
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(fps_override or cap.get(cv2.CAP_PROP_FPS) or config.HAND_OBJECT_VIDEO_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        return {"full_id": full_id, "status": "cannot_open_writer", "output_path": str(output_path)}

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        summary_df = frame_by_id.get(frame_idx)
        summary_row = summary_df.iloc[0] if summary_df is not None and not summary_df.empty else None
        pair_frame = pair_by_id.get(frame_idx, pd.DataFrame())
        if not pair_frame.empty:
            draw_pairs(frame, pair_frame, top_k=top_k)
        draw_panel(frame, frame_summary_lines(summary_row, top_k=top_k))
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
    return {
        "full_id": full_id,
        "status": "ok",
        "video_path": str(video_path),
        "frame_path": str(frame_path),
        "pair_path": str(pair_path),
        "output_path": str(output_path),
        "fps": fps,
        "source_frame_count": frame_count,
        "written_frame_count": frame_idx,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Render full HOI overlay videos for visual inspection.")
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--frame-root", type=Path, default=config.HAND_OBJECT_INTERACTION_FRAME_OUTPUT_ROOT)
    parser.add_argument("--pair-root", type=Path, default=config.HAND_OBJECT_INTERACTION_PAIR_OUTPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--participant")
    parser.add_argument("--full-id", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    full_ids = set(args.full_id) if args.full_id else None
    frame_paths = iter_frame_paths(args.frame_root, args.participant, full_ids, args.limit)
    iterator = tqdm(frame_paths, desc="render HOI videos") if tqdm is not None else frame_paths
    records = []
    for frame_path in iterator:
        full_id = parse_full_id(frame_path, FRAME_SUFFIX)
        participant = participant_from_full_id(full_id)
        output_path = args.output_root / participant / f"{full_id}_hoi_overlay.mp4"
        records.append(
            render_video(
                full_id=full_id,
                video_path=video_path_for(full_id, args.video_root),
                frame_path=frame_path,
                pair_path=pair_path_for(full_id, args.pair_root),
                output_path=output_path,
                fps_override=args.fps,
                top_k=max(1, args.top_k),
                overwrite=args.overwrite,
            )
        )
    args.output_root.mkdir(parents=True, exist_ok=True)
    index_path = args.output_root / "hand_object_interaction_video_index.csv"
    summary_path = args.output_root / "hand_object_interaction_video_summary.json"
    pd.DataFrame(records).to_csv(index_path, index=False)
    summary = {
        "recording_count": len(records),
        "ok_count": int(sum(record.get("status") == "ok" for record in records)),
        "skipped_count": int(sum(record.get("status") == "skipped_exists" for record in records)),
        "output_root": str(args.output_root),
        "index_path": str(index_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
