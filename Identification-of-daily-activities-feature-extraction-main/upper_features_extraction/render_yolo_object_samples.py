import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

import cv2


DATASET_ROOT = Path("/home/julius/datasets/seminar_object_detection")
DEFAULT_CSV_ROOT = DATASET_ROOT / "lower_feature_outcome" / "yolo_object_outcome_csv"
DEFAULT_VIDEO_ROOT = DATASET_ROOT / "iAssistADL_dataset_export_2026-05-18" / "world_camera_videos"
DEFAULT_OUTPUT_ROOT = DATASET_ROOT / "video_demonstration" / "yolo_object_samples"

COCO_CLASS_NAMES = {
    0: "person",
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


def parse_full_id(path):
    stem = Path(path).stem
    suffix = "_world_camera"
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def participant_from_full_id(full_id):
    return "_".join(full_id.split("_")[:4])


def task_from_full_id(full_id):
    participant = participant_from_full_id(full_id)
    prefix = f"{participant}_"
    return full_id[len(prefix) :] if full_id.startswith(prefix) else full_id


def normalize_action(task_name):
    base = re.sub(r"\d+$", "", task_name)
    base = base.replace("Teekochen", "Tee_kochen")
    base = base.replace("several_Task_", "several_Tasks_")
    return base


def find_video(video_root, full_id):
    participant = participant_from_full_id(full_id)
    path = Path(video_root) / participant / f"{full_id}_world_camera.mp4"
    return path if path.exists() else None


def choose_samples(csv_root, video_root, requested_actions=None):
    grouped = defaultdict(list)
    for csv_path in sorted(Path(csv_root).glob("*_world_camera.csv")):
        full_id = parse_full_id(csv_path)
        action = normalize_action(task_from_full_id(full_id))
        if requested_actions and action not in requested_actions:
            continue
        video_path = find_video(video_root, full_id)
        if video_path is not None:
            grouped[action].append((full_id, csv_path, video_path))

    samples = []
    for action in sorted(grouped):
        full_id, csv_path, video_path = grouped[action][0]
        samples.append(
            {
                "action": action,
                "full_id": full_id,
                "csv_path": csv_path,
                "video_path": video_path,
            }
        )
    return samples


def class_color(class_id):
    palette = [
        (50, 210, 255),
        (80, 190, 90),
        (255, 170, 60),
        (235, 90, 95),
        (180, 130, 255),
        (255, 220, 80),
        (60, 220, 180),
        (220, 150, 220),
    ]
    return palette[int(class_id) % len(palette)]


def clamp_box(row, width, height):
    x1 = max(0, min(width - 1, int(round(float(row["x1"])))))
    y1 = max(0, min(height - 1, int(round(float(row["y1"])))))
    x2 = max(0, min(width - 1, int(round(float(row["x2"])))))
    y2 = max(0, min(height - 1, int(round(float(row["y2"])))))
    return x1, y1, x2, y2


def load_detections(csv_path, start_frame, end_frame, min_confidence):
    detections = defaultdict(list)
    with Path(csv_path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                frame_id = int(round(float(row["frame_id"])))
                confidence = float(row["confidence"])
            except (KeyError, TypeError, ValueError):
                continue
            if frame_id < start_frame or frame_id >= end_frame or confidence < min_confidence:
                continue
            detections[frame_id].append(row)
    return detections


def draw_label(frame, text, x, y, color, font_scale=0.55):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    tw, th = text_size
    label_y = max(th + baseline + 6, y)
    cv2.rectangle(frame, (x, label_y - th - baseline - 6), (min(frame.shape[1] - 1, x + tw + 8), label_y), color, -1)
    cv2.putText(frame, text, (x + 4, label_y - baseline - 3), font, font_scale, (20, 20, 20), thickness, cv2.LINE_AA)


def render_sample(sample, output_root, duration_s, start_s, min_confidence, detector_label):
    video_path = sample["video_path"]
    csv_path = sample["csv_path"]
    full_id = sample["full_id"]
    action = sample["action"]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = max(0, int(round(start_s * fps)))
    end_frame = min(total_frames, start_frame + int(round(duration_s * fps)))
    if end_frame <= start_frame:
        raise RuntimeError(f"Empty render range for {video_path}")

    detections = load_detections(csv_path, start_frame, end_frame, min_confidence)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    duration_label = f"{int(duration_s)}s" if float(duration_s).is_integer() else f"{duration_s:g}s"
    output_path = output_root / f"{full_id}_world_camera_yolo_object_demo_{duration_label}.mp4"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_id = start_frame
    written = 0
    while frame_id < end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        frame_detections = detections.get(frame_id, [])
        for det in frame_detections:
            try:
                class_id = int(round(float(det["class_id"])))
                confidence = float(det["confidence"])
                x1, y1, x2, y2 = clamp_box(det, width, height)
            except (KeyError, TypeError, ValueError):
                continue
            color = class_color(class_id)
            class_name = det.get("class_name") or COCO_CLASS_NAMES.get(class_id, f"class_{class_id}")
            label = f"{class_name} {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            draw_label(frame, label, x1, y1, color)

        title = f"{detector_label} | {action} | {full_id} | frame {frame_id} | boxes {len(frame_detections)}"
        draw_label(frame, title, 16, 28, (245, 245, 245), font_scale=0.6)
        writer.write(frame)
        written += 1
        frame_id += 1

    cap.release()
    writer.release()

    if written == 0:
        raise RuntimeError(f"No frames written for {video_path}")

    return {
        "action": action,
        "full_id": full_id,
        "video_path": str(video_path),
        "csv_path": str(csv_path),
        "output_path": str(output_path),
        "fps": f"{fps:.3f}",
        "start_frame": start_frame,
        "end_frame": start_frame + written,
        "written_frames": written,
        "min_confidence": min_confidence,
    }


def write_manifest(records, output_root):
    path = Path(output_root) / "sample_manifest.csv"
    fieldnames = [
        "action",
        "full_id",
        "video_path",
        "csv_path",
        "output_path",
        "fps",
        "start_frame",
        "end_frame",
        "written_frames",
        "min_confidence",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return path


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Render one YOLO object detection demo video per action.")
    parser.add_argument("--csv-root", type=Path, default=DEFAULT_CSV_ROOT)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--duration-s", type=float, default=30.0)
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--min-confidence", type=float, default=0.25)
    parser.add_argument("--detector-label", default="YOLO object CSV")
    parser.add_argument("--action", action="append", help="Normalized action to render. Can be repeated.")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    requested_actions = set(args.action) if args.action else None
    samples = choose_samples(args.csv_root, args.video_root, requested_actions=requested_actions)
    if not samples:
        raise SystemExit("No samples found with both CSV and video.")

    records = []
    for sample in samples:
        record = render_sample(
            sample,
            output_root=args.output_root,
            duration_s=args.duration_s,
            start_s=args.start_s,
            min_confidence=args.min_confidence,
            detector_label=args.detector_label,
        )
        records.append(record)
        print(f"rendered {record['action']}: {record['output_path']}")

    manifest_path = write_manifest(records, args.output_root)
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
