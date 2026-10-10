import argparse
import csv
import os
import time
from pathlib import Path

import cv2
from ultralytics import YOLO

from yolo_features_extraction import config
from yolo_features_extraction.archive.adl14_experiment.prepare_real_adl_hands_yolo_dataset import CLASS_NAMES


DEFAULT_MODEL_PATH = Path("runs/real_adl_hands/yolo11s_real_adl_14cls_gpu/weights/best.pt")
DEFAULT_OUTPUT_ROOT = (
    config.DATASET_ROOT
    / "lower_feature_outcome"
    / "yolo_real_adl_14_object_outcome_csv"
)
CUSTOM_CLASS_ID_OFFSET = 1000


class ADL14ObjectExtractor:
    def __init__(self, model_path, conf_thres=0.25, class_id_offset=CUSTOM_CLASS_ID_OFFSET):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        self.conf_thres = conf_thres
        self.class_id_offset = int(class_id_offset)

    def extract(self, frame, frame_id=None):
        results = self.model(frame, verbose=False)[0]
        if results.boxes is None:
            return []

        detections = []
        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.conf_thres:
                continue

            local_class_id = int(box.cls[0])
            class_name = CLASS_NAMES[local_class_id] if 0 <= local_class_id < len(CLASS_NAMES) else f"adl_{local_class_id}"
            class_id = self.class_id_offset + local_class_id
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            width = x2 - x1
            height = y2 - y1

            detections.append(
                {
                    "video": None,
                    "frame_id": frame_id,
                    "class_id": class_id,
                    "local_class_id": local_class_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cx": (x1 + x2) / 2.0,
                    "cy": (y1 + y2) / 2.0,
                    "width": width,
                    "height": height,
                    "area": width * height,
                    "source_model": "adl14",
                }
            )
        return detections

    def process_video(self, video_path):
        print(f"\n[VIDEO] Opening: {video_path}")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_id = 0
        detections = []
        start_time = time.time()

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_detections = self.extract(frame, frame_id=frame_id)
            for det in frame_detections:
                det["video"] = Path(video_path).stem
            detections.extend(frame_detections)

            if frame_id % 100 == 0:
                elapsed = max(time.time() - start_time, 1e-9)
                print(f"[PROCESS] {frame_id}/{total_frames} | fps={frame_id / elapsed:.2f}")
            frame_id += 1

        cap.release()
        print(f"[DONE] Video finished in {time.time() - start_time:.2f}s")
        return detections

    @staticmethod
    def find_videos(input_folder):
        input_folder = Path(input_folder)
        video_files = []
        for root, _, files in os.walk(input_folder):
            for file_name in files:
                path = Path(root) / file_name
                if path.suffix.lower() in config.VIDEO_EXTENSIONS:
                    video_files.append(path)
        return sorted(video_files)

    @staticmethod
    def should_skip_video(video_path):
        return any(keyword in video_path.stem for keyword in config.SKIP_VIDEO_KEYWORDS)

    @staticmethod
    def save_csv(rows, output_path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        columns = [
            "video",
            "frame_id",
            "class_id",
            "local_class_id",
            "class_name",
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
            "source_model",
        ]
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def process_folder(self, input_folder, output_folder, skip_existing=True, limit=None):
        output_folder = Path(output_folder)
        video_files = self.find_videos(input_folder)
        if limit is not None:
            video_files = video_files[:limit]
        print(f"[FOLDER] Found {len(video_files)} videos")

        for index, video_path in enumerate(video_files, start=1):
            if self.should_skip_video(video_path):
                print(f"[SKIP CALIBRATION] {video_path}")
                continue

            output_path = output_folder / f"{video_path.stem}.csv"
            if skip_existing and output_path.exists() and output_path.stat().st_size > 0:
                print(f"[SKIP EXISTS] {output_path}")
                continue

            print("\n==============================")
            print(f"[FILE] {index}/{len(video_files)}")
            print(f"[PATH] {video_path}")
            print("==============================")
            rows = self.process_video(video_path)
            self.save_csv(rows, output_path)
            print(f"[DONE] Saved CSV: {output_path}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run the ADL 14-class YOLO model on world-camera videos.")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--input-root", type=Path, default=config.VIDEO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--conf", type=float, default=config.CONFIDENCE_THRESHOLD)
    parser.add_argument("--class-id-offset", type=int, default=CUSTOM_CLASS_ID_OFFSET)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    extractor = ADL14ObjectExtractor(
        model_path=args.model,
        conf_thres=args.conf,
        class_id_offset=args.class_id_offset,
    )
    extractor.process_folder(
        input_folder=args.input_root,
        output_folder=args.output_root,
        skip_existing=not args.overwrite,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
