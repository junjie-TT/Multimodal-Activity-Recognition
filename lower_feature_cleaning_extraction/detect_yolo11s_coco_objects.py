#!/usr/bin/env python3
"""Run original YOLO11s COCO object detection on iAssistADL world-camera videos."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from lower_feature_cleaning_extraction.config import (  # noqa: E402
    DATASET_ROOT,
    SKIP_TASKS,
    WORLD_CAMERA_VIDEO_ROOT,
)
from lower_feature_cleaning_extraction.detect_hands_yolo import (  # noqa: E402
    iter_video_paths,
    normalize_class_filter,
    parse_world_camera_video_name,
    read_detection_csv_summary,
    resolve_model_path,
    should_keep_detection,
    write_rows_csv,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional.
    tqdm = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "train_YOLO_hands" / "yolo11s.pt"
DEFAULT_OUTPUT_ROOT = DATASET_ROOT / "lower_feature_outcome" / "yolo11s_coco_object_outcome_csv"
DEFAULT_DEMO_ROOT = DATASET_ROOT / "video_demonstration" / "yolo11s_coco_object"
DEFAULT_DEMO_SUFFIX = "yolo11s_coco_demo_full_video"

CSV_COLUMNS = [
    "video",
    "frame_id",
    "class_id",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
    "cx",
    "cy",
    "width",
    "height",
    "area",
]

INDEX_COLUMNS = [
    "participant_id",
    "task_name",
    "action_name",
    "full_id",
    "video_path",
    "csv_path",
    "frames_processed",
    "frames_with_detections",
    "detections",
    "skipped_existing",
]

DEMO_INDEX_COLUMNS = [
    "action_name",
    "participant_id",
    "task_name",
    "full_id",
    "video_path",
    "demo_video_path",
    "frames_written",
    "skipped_existing",
]


def normalize_action_name(task_name: str) -> str:
    """Collapse numbered repeats, e.g. Tasche_packen1/2 -> Tasche_packen."""
    action = re.sub(r"\d+$", "", task_name)
    action = action.replace("several_Tasks", "several_Task")
    return action.rstrip("_")


def csv_output_path(output_root: Path, full_id: str) -> Path:
    return output_root / f"{full_id}_world_camera.csv"


def detect_objects_in_video(
    model,
    video_path: Path,
    output_path: Path,
    class_filter=None,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 640,
    device: str | None = None,
    vid_stride: int = 1,
    batch: int = 1,
    overwrite: bool = False,
) -> dict:
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        existing = read_detection_csv_summary(output_path)
        return {
            "csv_path": str(output_path),
            "frames_processed": "",
            "frames_with_detections": int(existing["frames_with_detections"]),
            "detections": int(existing["detections"]),
            "skipped_existing": True,
        }

    rows = []
    frame_id = -1
    video_name = video_path.stem

    stream = model.predict(
        source=str(video_path),
        stream=True,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        vid_stride=vid_stride,
        batch=batch,
        verbose=False,
    )

    for frame_id, result in enumerate(stream):
        names = result.names or {}
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            continue

        xyxy = boxes.xyxy.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        confidence = boxes.conf.cpu().numpy()

        for detection_idx in range(len(boxes)):
            class_id = int(cls[detection_idx])
            class_name = str(names.get(class_id, class_id))
            if not should_keep_detection(class_id, class_name, class_filter):
                continue

            x1, y1, x2, y2 = (float(v) for v in xyxy[detection_idx])
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            rows.append(
                {
                    "video": video_name,
                    "frame_id": int(frame_id * vid_stride),
                    "class_id": class_id,
                    "confidence": float(confidence[detection_idx]),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cx": x1 + width / 2.0,
                    "cy": y1 + height / 2.0,
                    "width": width,
                    "height": height,
                    "area": width * height,
                }
            )

    write_rows_csv(output_path, rows, CSV_COLUMNS)
    return {
        "csv_path": str(output_path),
        "frames_processed": int(frame_id + 1),
        "frames_with_detections": int(len({row["frame_id"] for row in rows})),
        "detections": int(len(rows)),
        "skipped_existing": False,
    }


def pick_one_video_per_action(rows: list[dict], use_normalized_action: bool = True) -> list[dict]:
    selected = {}
    for row in sorted(rows, key=lambda item: (item["action_name"], item["full_id"])):
        key = row["action_name"] if use_normalized_action else row["task_name"]
        if key not in selected:
            selected[key] = row
    return list(selected.values())


def render_detection_demo_video(
    model,
    video_path: Path,
    output_path: Path,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 640,
    device: str | None = None,
    overwrite: bool = False,
) -> dict:
    if output_path.exists() and not overwrite:
        return {
            "demo_video_path": str(output_path),
            "frames_written": "",
            "skipped_existing": True,
        }

    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: opencv-python. Install cv2 before rendering demo videos."
        ) from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise OSError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise OSError(f"Could not open output video writer: {output_path}")

    frames_written = 0
    iterator = range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
    if tqdm is not None:
        iterator = tqdm(iterator, desc=f"Rendering {video_path.stem}", unit="frame")

    for _ in iterator:
        ok, frame = cap.read()
        if not ok:
            break

        result = model.predict(
            source=frame,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            verbose=False,
        )[0]
        writer.write(result.plot())
        frames_written += 1

    writer.release()
    cap.release()
    return {
        "demo_video_path": str(output_path),
        "frames_written": int(frames_written),
        "skipped_existing": False,
    }


def run_yolo11s_coco_objects(
    model_path=DEFAULT_MODEL,
    video_root=WORLD_CAMERA_VIDEO_ROOT,
    output_root=DEFAULT_OUTPUT_ROOT,
    demo_root=DEFAULT_DEMO_ROOT,
    class_names=None,
    participant=None,
    full_id=None,
    conf: float = 0.25,
    iou: float = 0.7,
    imgsz: int = 640,
    device: str | None = None,
    vid_stride: int = 1,
    batch: int = 1,
    limit: int | None = None,
    include_calibration: bool = False,
    overwrite: bool = False,
    render_demos: bool = True,
    overwrite_demos: bool = False,
    normalized_action_demos: bool = True,
) -> tuple[list[dict], dict]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: ultralytics. Install it in the active environment "
            "before running YOLO11s COCO object detection."
        ) from exc

    model_path = resolve_model_path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO11s model not found: {model_path}")

    video_root = Path(video_root)
    output_root = Path(output_root)
    demo_root = Path(demo_root)
    class_filter = normalize_class_filter(class_names)
    skip_tasks = None if include_calibration else SKIP_TASKS

    video_paths = iter_video_paths(
        video_root,
        participant=participant,
        full_id=full_id,
        skip_tasks=skip_tasks,
        limit=limit,
    )
    if not video_paths:
        raise FileNotFoundError(f"No world camera videos found under: {video_root}")

    model = YOLO(str(model_path))
    rows = []
    iterator = video_paths
    if tqdm is not None:
        iterator = tqdm(video_paths, desc="Detecting COCO objects with YOLO11s", unit="video")

    for video_path in iterator:
        participant_id, task_name, video_full_id = parse_world_camera_video_name(video_path)
        action_name = normalize_action_name(task_name)
        result = detect_objects_in_video(
            model=model,
            video_path=video_path,
            output_path=csv_output_path(output_root, video_full_id),
            class_filter=class_filter,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            device=device,
            vid_stride=vid_stride,
            batch=batch,
            overwrite=overwrite,
        )
        rows.append(
            {
                "participant_id": participant_id,
                "task_name": task_name,
                "action_name": action_name,
                "full_id": video_full_id,
                "video_path": str(video_path),
                **result,
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / "yolo11s_coco_object_outcome_index.csv"
    write_rows_csv(index_path, rows, INDEX_COLUMNS)

    demos = []
    if render_demos:
        selected = pick_one_video_per_action(rows, use_normalized_action=normalized_action_demos)
        demo_iterator = selected
        if tqdm is not None:
            demo_iterator = tqdm(selected, desc="Rendering one full demo per action", unit="video")

        for row in demo_iterator:
            video_path = Path(row["video_path"])
            output_path = demo_root / f"{row['full_id']}_world_camera_{DEFAULT_DEMO_SUFFIX}.mp4"
            demo_result = render_detection_demo_video(
                model=model,
                video_path=video_path,
                output_path=output_path,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=device,
                overwrite=overwrite_demos,
            )
            demos.append(
                {
                    "action_name": row["action_name"],
                    "participant_id": row["participant_id"],
                    "task_name": row["task_name"],
                    "full_id": row["full_id"],
                    "video_path": row["video_path"],
                    **demo_result,
                }
            )

        demo_root.mkdir(parents=True, exist_ok=True)
        write_rows_csv(demo_root / "yolo11s_coco_demo_index.csv", demos, DEMO_INDEX_COLUMNS)

    detections_by_action = defaultdict(int)
    for row in rows:
        detections_by_action[row["action_name"]] += int(row["detections"])

    summary = {
        "model_path": str(model_path),
        "video_root": str(video_root),
        "output_root": str(output_root),
        "index_csv": str(index_path),
        "demo_root": str(demo_root) if render_demos else None,
        "video_count": int(len(rows)),
        "action_count": int(len({row["action_name"] for row in rows})),
        "demo_video_count": int(len(demos)),
        "total_detections": int(sum(int(row["detections"]) for row in rows)),
        "detections_by_action": dict(sorted(detections_by_action.items())),
        "class_filter": sorted(class_filter) if class_filter else None,
        "conf": float(conf),
        "iou": float(iou),
        "imgsz": int(imgsz),
        "device": device,
        "vid_stride": int(vid_stride),
        "skipped_calibration_tasks": [] if include_calibration else sorted(SKIP_TASKS),
    }

    summary_path = output_root / "yolo11s_coco_object_outcome_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    summary["summary_json"] = str(summary_path)

    return rows, summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run original YOLO11s COCO object detection on iAssistADL world-camera "
            "videos, write per-video CSV files, and render one full demo video per action."
        )
    )
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help=f"Default: {DEFAULT_MODEL}")
    parser.add_argument("--video-root", default=str(WORLD_CAMERA_VIDEO_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--demo-root", default=str(DEFAULT_DEMO_ROOT))
    parser.add_argument(
        "--class-name",
        dest="class_names",
        action="append",
        default=None,
        help="Class name or id to keep. Repeat or comma-separate. Default writes all COCO classes.",
    )
    parser.add_argument("--participant", default=None, help="Optional participant id, e.g. IA_4_H_02.")
    parser.add_argument(
        "--full-id",
        action="append",
        default=None,
        help="Optional video full id, e.g. IA_4_H_02_Brief_schreiben1. Can be repeated.",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="Ultralytics device string, e.g. 0, cpu, cuda:0.")
    parser.add_argument("--vid-stride", type=int, default=1)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for quick tests.")
    parser.add_argument("--include-calibration", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing CSV files.")
    parser.add_argument("--no-demos", action="store_true", help="Only write CSV files.")
    parser.add_argument("--overwrite-demos", action="store_true", help="Overwrite existing demo videos.")
    parser.add_argument(
        "--demo-by-task",
        action="store_true",
        help="Render one demo per exact task repeat instead of one per normalized action.",
    )
    return parser


def main(argv=None) -> None:
    args = build_arg_parser().parse_args(argv)
    rows, summary = run_yolo11s_coco_objects(
        model_path=args.model,
        video_root=args.video_root,
        output_root=args.output_root,
        demo_root=args.demo_root,
        class_names=args.class_names,
        participant=args.participant,
        full_id=args.full_id,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        device=args.device,
        vid_stride=args.vid_stride,
        batch=args.batch,
        limit=args.limit,
        include_calibration=args.include_calibration,
        overwrite=args.overwrite,
        render_demos=not args.no_demos,
        overwrite_demos=args.overwrite_demos,
        normalized_action_demos=not args.demo_by_task,
    )

    print("YOLO11s COCO object detection completed.")
    print(json.dumps(summary, indent=2))
    print("\nDetections by participant:")
    detections_by_participant = defaultdict(int)
    for row in rows:
        detections_by_participant[row["participant_id"]] += int(row["detections"])

    for participant_id in sorted(detections_by_participant):
        print(f"{participant_id}    {detections_by_participant[participant_id]}")


if __name__ == "__main__":
    main()
