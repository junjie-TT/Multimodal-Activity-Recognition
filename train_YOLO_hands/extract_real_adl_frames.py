#!/usr/bin/env python3
"""Extract frames from Real ADL first-person world camera videos for manual YOLO labeling."""

import argparse
import os
from pathlib import Path

import cv2
from tqdm import tqdm


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract frames from Real ADL world camera videos for manual hand labeling."
    )
    parser.add_argument(
        "--video_dir",
        required=True,
        help="Input root directory containing .mp4 world camera videos.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory to save extracted JPEG frames.",
    )
    parser.add_argument(
        "--interval_sec",
        type=float,
        default=1.0,
        help="Frame extraction interval in seconds. Default is 1.0.",
    )
    parser.add_argument(
        "--max_frames_per_video",
        type=int,
        default=0,
        help="Maximum frames to extract per video. 0 means no limit.",
    )
    parser.add_argument(
        "--image_quality",
        type=int,
        default=95,
        help="JPEG quality for saved images. Default is 95.",
    )
    return parser.parse_args()


def find_video_paths(video_dir: Path):
    video_paths = []
    for root, _, files in os.walk(video_dir):
        for filename in files:
            if filename.lower().endswith(".mp4"):
                video_paths.append(Path(root) / filename)
    video_paths.sort()
    return video_paths


def make_output_dir(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)


def extract_frames_from_video(
    video_path: Path,
    output_dir: Path,
    interval_sec: float,
    max_frames_per_video: int,
    image_quality: int,
) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"Warning: Unable to open video: {video_path}")
        return 0

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps != fps:  # check NaN as well
        fps = 30.0
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = total_frames / fps if total_frames > 0 else 0.0

    if duration_sec <= 0.0:
        print(f"Warning: Video has invalid duration or frame count: {video_path}")
        capture.release()
        return 0

    if interval_sec <= 0.0:
        interval_sec = 1.0

    participant = video_path.parent.name or "unknown"
    video_stem = video_path.stem

    extracted_count = 0
    frame_index = 0
    timestamp = 0.0

    while True:
        if max_frames_per_video > 0 and extracted_count >= max_frames_per_video:
            break

        frame_index = int(round(timestamp * fps))
        if frame_index >= total_frames:
            break

        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = capture.read()
        if not success or frame is None:
            print(
                f"Warning: Failed to read frame {frame_index} from {video_path}."
            )
            break

        timestamp_ms = int(round((frame_index / fps) * 1000.0))
        output_name = (
            f"{participant}*{video_stem}_frame*{frame_index:06d}_time*{timestamp_ms:08d}ms.jpg"
        )
        output_path = output_dir / output_name

        save_params = [int(cv2.IMWRITE_JPEG_QUALITY), max(0, min(image_quality, 100))]
        success = cv2.imwrite(str(output_path), frame, save_params)
        if not success:
            print(f"Warning: Failed to write image: {output_path}")
            break

        extracted_count += 1
        timestamp += interval_sec

    capture.release()
    return extracted_count


def main():
    args = parse_arguments()
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    make_output_dir(output_dir)

    video_paths = find_video_paths(video_dir)
    if not video_paths:
        print(f"No .mp4 videos found in directory: {video_dir}")
        return

    summary = {}
    total_extracted = 0

    print(f"Found {len(video_paths)} video(s) under {video_dir}")
    for video_path in tqdm(video_paths, desc="Processing videos", unit="video"):
        extracted = extract_frames_from_video(
            video_path=video_path,
            output_dir=output_dir,
            interval_sec=args.interval_sec,
            max_frames_per_video=args.max_frames_per_video,
            image_quality=args.image_quality,
        )
        summary[str(video_path)] = extracted
        total_extracted += extracted

    print("\nExtraction summary")
    print("------------------")
    print(f"Video root directory: {video_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found videos: {len(video_paths)}")
    for video_path, count in summary.items():
        print(f"  {Path(video_path).name}: {count} frame(s)")
    print(f"Total extracted frames: {total_extracted}")


if __name__ == "__main__":
    main()


# Usage example:
# python train_YOLO_hands/extract_real_adl_frames.py \
#   --video_dir /home/julius/datasets/seminar_object_detection/iAssistADL_dataset_export_2026-05-18/world_camera_videos \
#   --output_dir /home/julius/datasets/real_adl_hands/frames_pilot_500 \
#   --interval_sec 1.0 \
#   --max_frames_per_video 20
