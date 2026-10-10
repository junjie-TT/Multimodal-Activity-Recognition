import argparse
import json
from pathlib import Path

from lower_feature_cleaning_extraction.config import LEFT_RIGHT_HAND_YOLO_MODEL_ROOT


def train_hand_yolo(
    data_yaml,
    model="yolov8s.pt",
    epochs=100,
    imgsz=640,
    batch=16,
    device="0",
    project=LEFT_RIGHT_HAND_YOLO_MODEL_ROOT,
    name="egohands_left_right",
    patience=25,
):
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: ultralytics. Run this in the YOLO/object_detection environment."
        ) from exc

    data_yaml = Path(data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(data_yaml)

    model = YOLO(str(model))
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project),
        name=name,
        patience=patience,
        exist_ok=True,
    )

    save_dir = Path(results.save_dir)
    summary = {
        "data_yaml": str(data_yaml),
        "model": str(model.model_name if hasattr(model, "model_name") else model),
        "save_dir": str(save_dir),
        "best_weights": str(save_dir / "weights" / "best.pt"),
        "last_weights": str(save_dir / "weights" / "last.pt"),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
    }
    summary_path = save_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train a left/right hand YOLO model.")
    parser.add_argument("--data-yaml", required=True, help="Prepared YOLO data.yaml path.")
    parser.add_argument("--model", default="yolov8s.pt", help="Base model, e.g. yolov8n.pt or yolov8s.pt.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--project", default=str(LEFT_RIGHT_HAND_YOLO_MODEL_ROOT))
    parser.add_argument("--name", default="egohands_left_right")
    parser.add_argument("--patience", type=int, default=25)
    return parser


def main():
    args = build_arg_parser().parse_args()
    summary = train_hand_yolo(
        data_yaml=args.data_yaml,
        model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        patience=args.patience,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
