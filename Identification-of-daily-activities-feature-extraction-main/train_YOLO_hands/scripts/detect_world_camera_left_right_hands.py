#!/usr/bin/env python3
"""Run the trained YOLO11s left/right hand detector on world camera videos."""

from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lower_feature_cleaning_extraction.detect_hands_yolo import (  # noqa: E402
    build_arg_parser,
    run_hand_yolo_detection,
)


DEFAULT_MODEL = (
    "/home/julius/models/hand_detection/"
    "yolo11s_egohands_real_adl_mixed/weights/best.pt"
)
DEFAULT_VIDEO_ROOT = (
    "/home/julius/datasets/seminar_object_detection/"
    "iAssistADL_dataset_export_2026-05-18/world_camera_videos"
)
DEFAULT_OUTPUT_ROOT = (
    "/home/julius/datasets/seminar_object_detection/lower_feature_outcome/"
    "yolo11s_left_right_hand_outcome_csv"
)


def main(argv=None) -> None:
    parser = build_arg_parser()
    parser.description = (
        "Run the trained YOLO11s left/right hand detector on iAssistADL world camera videos."
    )
    parser.set_defaults(
        model=DEFAULT_MODEL,
        video_root=DEFAULT_VIDEO_ROOT,
        output_root=DEFAULT_OUTPUT_ROOT,
        class_names=None,
        conf=0.25,
        iou=0.7,
        imgsz=640,
        device="0",
        vid_stride=1,
        batch=1,
    )

    for action in parser._actions:
        if action.dest == "model":
            action.help = f"Path to YOLO11s left/right hand weights. Default: {DEFAULT_MODEL}"
        elif action.dest == "video_root":
            action.help = f"Root containing participant world camera videos. Default: {DEFAULT_VIDEO_ROOT}"
        elif action.dest == "output_root":
            action.help = f"Directory for per-video detection CSV outputs. Default: {DEFAULT_OUTPUT_ROOT}"

    args = parser.parse_args(argv)
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

    print("YOLO11s left/right hand detection completed.")
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
