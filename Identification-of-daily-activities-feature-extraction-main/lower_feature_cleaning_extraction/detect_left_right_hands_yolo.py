import json

from lower_feature_cleaning_extraction.config import (
    LEFT_RIGHT_HAND_YOLO_MODEL_PATH,
    YOLO_LEFT_RIGHT_HAND_OUTCOME_ROOT,
)
from lower_feature_cleaning_extraction.detect_hands_yolo import (
    build_arg_parser,
    run_hand_yolo_detection,
)


def main(argv=None):
    parser = build_arg_parser()
    parser.description = "Run the trained left/right hand YOLO model on world camera videos."
    parser.set_defaults(
        model=str(LEFT_RIGHT_HAND_YOLO_MODEL_PATH),
        output_root=str(YOLO_LEFT_RIGHT_HAND_OUTCOME_ROOT),
    )
    for action in parser._actions:
        if action.dest == "model":
            action.help = (
                "Path to the trained left/right hand YOLO weights. "
                f"Default: {LEFT_RIGHT_HAND_YOLO_MODEL_PATH}"
            )
        elif action.dest == "output_root":
            action.help = (
                "Directory for left/right hand detection CSV outputs. "
                f"Default: {YOLO_LEFT_RIGHT_HAND_OUTCOME_ROOT}"
            )
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

    print("Left/right hand YOLO detection completed.")
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
