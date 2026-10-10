#!/usr/bin/env python3
"""Prepare raw Real ADL hand labels in YOLO format, skipping unlabeled images."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASS_NAMES = {0: "left_hand", 1: "right_hand"}


def find_images(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def validate_label_file(label_path: Path) -> tuple[int, int, int]:
    counts = {0: 0, 1: 0}
    warning_count = 0

    if not label_path.exists():
        return 0, 0, 0

    with label_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                print(f"Warning: {label_path.name}:{line_number} has {len(parts)} columns, expected 5")
                warning_count += 1
                continue

            try:
                class_id = int(parts[0])
                x_center, y_center, width, height = [float(value) for value in parts[1:]]
            except ValueError:
                print(f"Warning: {label_path.name}:{line_number} contains non-numeric values")
                warning_count += 1
                continue

            errors = []
            if class_id not in CLASS_NAMES:
                errors.append("class_id must be 0 or 1")
            if not 0.0 <= x_center <= 1.0:
                errors.append("x_center must be in [0, 1]")
            if not 0.0 <= y_center <= 1.0:
                errors.append("y_center must be in [0, 1]")
            if not 0.0 < width <= 1.0:
                errors.append("width must be in (0, 1]")
            if not 0.0 < height <= 1.0:
                errors.append("height must be in (0, 1]")

            if errors:
                print(f"Warning: {label_path.name}:{line_number} invalid label: {'; '.join(errors)}")
                warning_count += 1
                continue

            counts[class_id] += 1

    return counts[0], counts[1], warning_count


def prepare_dataset(image_dir: Path, label_dir: Path, out_root: Path) -> None:
    out_images = out_root / "images"
    out_labels = out_root / "labels"
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    images = find_images(image_dir)
    existing_labels = 0
    skipped_missing_labels = 0
    left_count = 0
    right_count = 0
    warnings = 0

    for image_path in tqdm(images, desc="Preparing Real ADL raw labels", unit="image"):
        src_label = label_dir / f"{image_path.stem}.txt"
        if not src_label.exists():
            print(f"Warning: missing label for {image_path.name}; skipping image")
            skipped_missing_labels += 1
            continue

        dst_image = out_images / image_path.name
        shutil.copy2(image_path, dst_image)

        dst_label = out_labels / f"{image_path.stem}.txt"
        shutil.copy2(src_label, dst_label)
        existing_labels += 1

        label_left, label_right, label_warnings = validate_label_file(dst_label)
        left_count += label_left
        right_count += label_right
        warnings += label_warnings

    print("\nSummary")
    print("-------")
    print(f"Images found: {len(images)}")
    print(f"Images copied with labels: {existing_labels}")
    print(f"Images skipped because label file was missing: {skipped_missing_labels}")
    print(f"left_hand boxes: {left_count}")
    print(f"right_hand boxes: {right_count}")
    print(f"Total boxes: {left_count + right_count}")
    print(f"Warnings: {warnings}")
    print(f"Output path: {out_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare raw Real ADL YOLO left/right hand labels.")
    parser.add_argument("--image_dir", required=True, help="Directory containing Real ADL images.")
    parser.add_argument("--label_dir", required=True, help="Directory containing LabelImg YOLO .txt labels.")
    parser.add_argument("--out_root", required=True, help="Output root for raw YOLO dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)
    out_root = Path(args.out_root)

    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")

    prepare_dataset(image_dir=image_dir, label_dir=label_dir, out_root=out_root)


if __name__ == "__main__":
    main()
