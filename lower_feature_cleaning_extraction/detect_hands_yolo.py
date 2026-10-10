import argparse
import csv
import json
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from lower_feature_cleaning_extraction.config import (
    HAND_YOLO_MODEL_PATH,
    SKIP_TASKS,
    WORLD_CAMERA_VIDEO_ROOT,
    YOLO_HAND_OUTCOME_ROOT,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


CSV_COLUMNS = [
    "frame_id",
    "class_id",
    "class_name",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2",
]


def parse_world_camera_video_name(path):
    participant_id = path.parent.name
    stem = path.stem
    if stem.endswith("_world_camera"):
        full_id = stem[: -len("_world_camera")]
    else:
        full_id = stem

    prefix = f"{participant_id}_"
    task_name = full_id[len(prefix):] if full_id.startswith(prefix) else full_id
    return participant_id, task_name, full_id


def normalize_class_filter(values):
    if not values:
        return None

    normalized = set()
    for value in values:
        for item in value.split(","):
            item = item.strip()
            if item:
                normalized.add(item.lower())
    return normalized or None


def should_keep_detection(class_id, class_name, class_filter):
    if class_filter is None:
        return True

    return str(class_id).lower() in class_filter or str(class_name).lower() in class_filter


def write_rows_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_detection_csv_summary(path):
    detections = 0
    frame_ids = set()
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            detections += 1
            frame_id = row.get("frame_id")
            if frame_id not in (None, ""):
                frame_ids.add(frame_id)

    return {
        "frames_with_detections": len(frame_ids),
        "detections": detections,
    }


def resolve_model_path(model_path):
    model_path = Path(model_path).expanduser()
    if model_path.is_dir():
        candidates = sorted(model_path.glob("*.pt"))
        if not candidates:
            raise FileNotFoundError(f"No .pt model files found in: {model_path}")
        if len(candidates) > 1:
            candidate_list = "\n".join(str(path) for path in candidates)
            raise ValueError(
                "Multiple .pt model files found. Pass one explicitly with --model:\n"
                f"{candidate_list}"
            )
        return candidates[0]
    return model_path


def iter_video_paths(video_root, participant=None, full_id=None, skip_tasks=SKIP_TASKS, limit=None):
    video_root = Path(video_root)
    pattern = "*/*_world_camera.mp4"
    paths = sorted(video_root.glob(pattern))

    if participant:
        paths = [path for path in paths if path.parent.name == participant]

    if full_id:
        wanted = set(full_id)
        paths = [
            path
            for path in paths
            if parse_world_camera_video_name(path)[2] in wanted
        ]

    if skip_tasks:
        paths = [
            path
            for path in paths
            if parse_world_camera_video_name(path)[1] not in skip_tasks
        ]

    if limit is not None:
        paths = paths[:limit]

    return paths


def detect_hands_in_video(
    model,
    video_path,
    output_path,
    class_filter=None,
    conf=0.25,
    iou=0.7,
    imgsz=640,
    device="0",
    vid_stride=1,
    batch=1,
    overwrite=False,
):
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        existing = read_detection_csv_summary(output_path)
        return {
            "csv_path": str(output_path),
            "frames_with_detections": int(existing["frames_with_detections"]),
            "detections": int(existing["detections"]),
            "skipped_existing": True,
        }

    rows = []
    frame_id = -1

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

            x1, y1, x2, y2 = xyxy[detection_idx]
            rows.append(
                {
                    "frame_id": int(frame_id * vid_stride),
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": float(confidence[detection_idx]),
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                }
            )

    write_rows_csv(output_path, rows, CSV_COLUMNS)
    frames_with_detections = len({row["frame_id"] for row in rows})

    return {
        "csv_path": str(output_path),
        "frames_processed": int(frame_id + 1),
        "frames_with_detections": int(frames_with_detections),
        "detections": int(len(rows)),
        "skipped_existing": False,
    }


def run_hand_yolo_detection(
    model_path,
    video_root=WORLD_CAMERA_VIDEO_ROOT,
    output_root=YOLO_HAND_OUTCOME_ROOT,
    class_names=None,
    participant=None,
    full_id=None,
    conf=0.25,
    iou=0.7,
    imgsz=640,
    device=None,
    vid_stride=1,
    batch=1,
    limit=None,
    include_calibration=False,
    overwrite=False,
):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: ultralytics. Install it in the active environment "
            "before running hand YOLO detection."
        ) from exc

    model_path = resolve_model_path(model_path)
    video_root = Path(video_root)
    output_root = Path(output_root)
    class_filter = normalize_class_filter(class_names)

    if not model_path.exists():
        raise FileNotFoundError(f"YOLO hand model not found: {model_path}")

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
        iterator = tqdm(video_paths, desc="Detecting hands with YOLO", unit="video")

    for video_path in iterator:
        participant_id, task_name, video_full_id = parse_world_camera_video_name(video_path)
        output_path = output_root / f"{video_full_id}_world_camera.csv"
        result = detect_hands_in_video(
            model=model,
            video_path=video_path,
            output_path=output_path,
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
                "full_id": video_full_id,
                "video_path": str(video_path),
                **result,
            }
        )

    output_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / "yolo_hand_outcome_index.csv"
    index_columns = [
        "participant_id",
        "task_name",
        "full_id",
        "video_path",
        "csv_path",
        "frames_processed",
        "frames_with_detections",
        "detections",
        "skipped_existing",
    ]
    write_rows_csv(index_path, rows, index_columns)

    summary = {
        "model_path": str(model_path),
        "video_root": str(video_root),
        "output_root": str(output_root),
        "index_csv": str(index_path),
        "video_count": int(len(rows)),
        "total_detections": int(sum(row["detections"] for row in rows)),
        "class_filter": sorted(class_filter) if class_filter else None,
        "conf": float(conf),
        "iou": float(iou),
        "imgsz": int(imgsz),
        "device": device,
        "vid_stride": int(vid_stride),
        "skipped_calibration_tasks": [] if include_calibration else sorted(SKIP_TASKS),
    }

    summary_path = output_root / "yolo_hand_outcome_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return rows, summary


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Run a hand-specific Ultralytics YOLO model on world camera videos."
    )
    parser.add_argument(
        "--model",
        default=str(HAND_YOLO_MODEL_PATH),
        help=(
            "Path to the hand YOLO weights or a directory containing one .pt file. "
            f"Default: {HAND_YOLO_MODEL_PATH}"
        ),
    )
    parser.add_argument("--video-root", default=str(WORLD_CAMERA_VIDEO_ROOT))
    parser.add_argument("--output-root", default=str(YOLO_HAND_OUTCOME_ROOT))
    parser.add_argument(
        "--class-name",
        dest="class_names",
        action="append",
        default=None,
        help=(
            "Class name or id to keep. Can be repeated or comma-separated. "
            "By default all model detections are written."
        ),
    )
    parser.add_argument("--participant", default=None, help="Optional participant id filter, e.g. IA_4_H_02.")
    parser.add_argument(
        "--full-id",
        action="append",
        default=None,
        help="Optional full recording id filter. Can be repeated.",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument(
        "--device",
        default="0",
        help="Ultralytics device string, e.g. 0, cpu, cuda:0. Default: 0.",
    )
    parser.add_argument("--vid-stride", type=int, default=1, help="Process every Nth frame.")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of videos.")
    parser.add_argument(
        "--include-calibration",
        action="store_true",
        help="Include Eye1, Eye2, and Recalibration1 videos.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output CSV files.")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    rows, summary = run_hand_yolo_detection(
        model_path=args.model,
        video_root=args.video_root,
        output_root=args.output_root,
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
    )

    print("Hand YOLO detection completed.")
    print(json.dumps(summary, indent=2))
    print("\nDetections by participant:")
    detections_by_participant = {}
    for row in rows:
        participant_id = row["participant_id"]
        detections_by_participant.setdefault(participant_id, 0)
        detections_by_participant[participant_id] += row["detections"]

    for participant_id in sorted(detections_by_participant):
        print(f"{participant_id}    {detections_by_participant[participant_id]}")


if __name__ == "__main__":
    main()
