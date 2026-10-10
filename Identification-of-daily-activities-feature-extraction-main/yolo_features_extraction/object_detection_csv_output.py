import argparse
import cv2
import os
import time
import csv
from pathlib import Path

from ultralytics import YOLO

try:
    from yolo_features_extraction import config
except ModuleNotFoundError:
    import config


BASE_DIR = Path(__file__).resolve().parent


class YOLOFeatureExtractor:
    def __init__(self, model_path=None, conf_thres=0.25):
        if model_path is None:
            model_path = config.MODEL_PATH

        model_path = self.resolve_model_path(model_path)

        print("[INIT] Loading YOLO model...")
        self.model = YOLO(str(model_path))
        self.conf_thres = conf_thres
        print(f"[INIT] Model loaded: {model_path}")

    @staticmethod
    def resolve_model_path(model_path):
        path = Path(model_path)
        if path.is_absolute():
            return path

        local_path = BASE_DIR / path
        if local_path.exists():
            return local_path

        return Path.cwd() / path

    def extract(self, frame, frame_id=None):
        results = self.model(frame, verbose=False)[0]

        features = []

        if results.boxes is None:
            return features

        for box in results.boxes:
            conf = float(box.conf[0])
            if conf < self.conf_thres:
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_id = int(box.cls[0])

            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            w = x2 - x1
            h = y2 - y1
            area = w * h

            features.append({
                "frame_id": frame_id,
                "bbox": [x1, y1, x2, y2],
                "class_id": cls_id,
                "confidence": conf,
                "center": [cx, cy],
                "size": [w, h],
                "area": area
            })

        return features

    def process_video(self, video_path):
        print(f"\n[VIDEO] Opening: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("[ERROR] Cannot open video!")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[VIDEO] Total frames: {total_frames}")

        all_frames = []
        frame_id = 0

        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.time()

            detections = self.extract(frame, frame_id=frame_id)

            all_frames.append({
                "frame_id": frame_id,
                "detections": detections
            })

            dt = time.time() - t0
            fps = 1.0 / dt if dt > 0 else 0

            if frame_id % 10 == 0:
                print(f"[PROCESS] {frame_id}/{total_frames} | fps={fps:.2f}")

            frame_id += 1

        cap.release()

        print(f"[DONE] Video finished in {time.time() - start_time:.2f}s")

        return all_frames

    @staticmethod
    def should_skip_video(video_path):
        return any(keyword in video_path.stem for keyword in config.SKIP_VIDEO_KEYWORDS)

    @staticmethod
    def find_videos(input_folder):
        input_folder = Path(input_folder)
        video_files = []

        for root, dirs, files in os.walk(input_folder):
            for file_name in files:
                path = Path(root) / file_name
                if path.suffix.lower() in config.VIDEO_EXTENSIONS:
                    video_files.append(path)

        return sorted(video_files)

    @staticmethod
    def save_csv(data, output_path, video_name):
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            writer.writerow([
                "video",
                "frame_id",
                "class_id",
                "confidence",
                "x1", "y1", "x2", "y2",
                "cx", "cy",
                "width", "height",
                "area"
            ])

            for frame in data:
                frame_id = frame["frame_id"]

                for det in frame["detections"]:
                    x1, y1, x2, y2 = det["bbox"]
                    cx, cy = det["center"]
                    w, h = det["size"]
                    area = det["area"]

                    writer.writerow([
                        video_name,
                        frame_id,
                        det["class_id"],
                        det["confidence"],
                        x1, y1, x2, y2,
                        cx, cy,
                        w, h,
                        area
                    ])

    def process_folder(self, input_folder, output_folder, skip_existing=True, limit=None):
        input_folder = Path(input_folder)
        output_folder = Path(output_folder)
        output_folder.mkdir(parents=True, exist_ok=True)

        video_files = self.find_videos(input_folder)
        if limit is not None:
            video_files = video_files[:limit]

        print(f"[FOLDER] Found {len(video_files)} videos")

        for idx, video_path in enumerate(video_files):
            if self.should_skip_video(video_path):
                print(f"[SKIP CALIBRATION] {video_path}")
                continue

            base_name = video_path.stem
            output_path = output_folder / f"{base_name}.csv"

            if skip_existing and output_path.exists() and output_path.stat().st_size > 0:
                print(f"[SKIP EXISTS] {output_path}")
                continue

            print("\n==============================")
            print(f"[FILE] {idx+1}/{len(video_files)}")
            print(f"[PATH] {video_path}")
            print("==============================")

            data = self.process_video(video_path)

            self.save_csv(data, output_path, base_name)

            print(f"[DONE] Saved CSV: {output_path}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run YOLO object detection on world-camera videos and export CSVs.")
    parser.add_argument("--model", type=Path, default=config.MODEL_PATH)
    parser.add_argument("--input-root", type=Path, default=config.VIDEO_ROOT)
    parser.add_argument("--output-root", type=Path, default=config.OUTPUT_ROOT)
    parser.add_argument("--conf", type=float, default=config.CONFIDENCE_THRESHOLD)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    yolo = YOLOFeatureExtractor(
        model_path=args.model,
        conf_thres=args.conf,
    )

    yolo.process_folder(
        input_folder=args.input_root,
        output_folder=args.output_root,
        skip_existing=not args.overwrite,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
