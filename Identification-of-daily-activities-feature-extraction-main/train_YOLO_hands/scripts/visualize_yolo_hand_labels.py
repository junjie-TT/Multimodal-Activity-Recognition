#!/usr/bin/env python3
"""Visualize YOLO left/right hand labels by drawing boxes on sampled images."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASS_NAMES = {0: "left_hand", 1: "right_hand"}
COLORS = {0: (40, 180, 255), 1: (80, 220, 80)}


def find_images(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def read_boxes(label_path: Path) -> list[tuple[int, float, float, float, float]]:
    boxes = []
    if not label_path.exists():
        print(f"Warning: missing label for visualization: {label_path}")
        return boxes

    with label_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                print(f"Warning: {label_path.name}:{line_number} has {len(parts)} columns, expected 5")
                continue
            try:
                class_id = int(parts[0])
                x_center, y_center, width, height = [float(value) for value in parts[1:]]
            except ValueError:
                print(f"Warning: {label_path.name}:{line_number} contains non-numeric values")
                continue
            if class_id not in CLASS_NAMES:
                print(f"Warning: {label_path.name}:{line_number} class_id must be 0 or 1")
                continue
            boxes.append((class_id, x_center, y_center, width, height))
    return boxes


def draw_boxes(image, boxes: list[tuple[int, float, float, float, float]]):
    height, width = image.shape[:2]
    for class_id, x_center, y_center, box_width, box_height in boxes:
        x1 = int(round((x_center - box_width / 2.0) * width))
        y1 = int(round((y_center - box_height / 2.0) * height))
        x2 = int(round((x_center + box_width / 2.0) * width))
        y2 = int(round((y_center + box_height / 2.0) * height))
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))

        color = COLORS[class_id]
        label = CLASS_NAMES[class_id]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        text_size, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        text_w, text_h = text_size
        label_y1 = max(0, y1 - text_h - baseline - 4)
        cv2.rectangle(image, (x1, label_y1), (x1 + text_w + 6, label_y1 + text_h + baseline + 4), color, -1)
        cv2.putText(image, label, (x1 + 3, label_y1 + text_h + 1), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    return image


def visualize(image_dir: Path, label_dir: Path, out_dir: Path, num_samples: int, seed: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    images = find_images(image_dir)
    rng = random.Random(seed)
    if num_samples < len(images):
        images = rng.sample(images, num_samples)
    images = sorted(images)

    written = 0
    for image_path in tqdm(images, desc="Visualizing labels", unit="image"):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: failed to read image: {image_path}")
            continue
        boxes = read_boxes(label_dir / f"{image_path.stem}.txt")
        image = draw_boxes(image, boxes)
        if cv2.imwrite(str(out_dir / image_path.name), image):
            written += 1
        else:
            print(f"Warning: failed to write visualization: {out_dir / image_path.name}")

    print("\nSummary")
    print("-------")
    print(f"Input images available: {len(find_images(image_dir))}")
    print(f"Requested samples: {num_samples}")
    print(f"Visualizations written: {written}")
    print(f"Output path: {out_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize YOLO left/right hand labels.")
    parser.add_argument("--image_dir", required=True, help="Directory containing images.")
    parser.add_argument("--label_dir", required=True, help="Directory containing YOLO .txt labels.")
    parser.add_argument("--out_dir", required=True, help="Directory to write visualized images.")
    parser.add_argument("--num_samples", type=int, default=100, help="Number of random images to visualize.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)
    out_dir = Path(args.out_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    visualize(image_dir=image_dir, label_dir=label_dir, out_dir=out_dir, num_samples=args.num_samples, seed=args.seed)


if __name__ == "__main__":
    main()
