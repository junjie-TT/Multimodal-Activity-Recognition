#!/usr/bin/env python3
"""Check a YOLO left/right hand dataset for file pairing and label validity."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
VALID_CLASSES = {0, 1}


def find_images(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def validate_label(label_path: Path) -> tuple[Counter, list[str], bool]:
    counts = Counter()
    errors = []
    has_non_empty_line = False

    with label_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            has_non_empty_line = True
            parts = line.split()
            if len(parts) != 5:
                errors.append(f"{label_path.name}:{line_number} has {len(parts)} columns, expected 5")
                continue
            try:
                class_id = int(parts[0])
                x_center, y_center, width, height = [float(value) for value in parts[1:]]
            except ValueError:
                errors.append(f"{label_path.name}:{line_number} contains non-numeric values")
                continue

            line_errors = []
            if class_id not in VALID_CLASSES:
                line_errors.append("class_id must be 0 or 1")
            if not 0.0 <= x_center <= 1.0:
                line_errors.append("x_center must be in [0, 1]")
            if not 0.0 <= y_center <= 1.0:
                line_errors.append("y_center must be in [0, 1]")
            if not 0.0 < width <= 1.0:
                line_errors.append("width must be in (0, 1]")
            if not 0.0 < height <= 1.0:
                line_errors.append("height must be in (0, 1]")

            if line_errors:
                errors.append(f"{label_path.name}:{line_number} invalid label: {'; '.join(line_errors)}")
            else:
                counts[class_id] += 1

    return counts, errors, not has_non_empty_line


def check_dataset(image_dir: Path, label_dir: Path) -> None:
    images = find_images(image_dir)
    labels = sorted(label_dir.glob("*.txt"))
    image_stems = {path.stem for path in images}
    label_stems = {path.stem for path in labels}

    missing_labels = sorted(image_stems - label_stems)
    extra_labels = sorted(label_stems - image_stems)
    class_counts = Counter()
    empty_labels = 0
    errors = []

    for label_path in labels:
        counts, label_errors, is_empty = validate_label(label_path)
        class_counts.update(counts)
        errors.extend(label_errors)
        if is_empty:
            empty_labels += 1

    print("Dataset check summary")
    print("---------------------")
    print(f"Images: {len(images)}")
    print(f"Label files: {len(labels)}")
    print(f"Missing labels: {len(missing_labels)}")
    print(f"Extra labels: {len(extra_labels)}")
    print(f"left_hand boxes: {class_counts[0]}")
    print(f"right_hand boxes: {class_counts[1]}")
    print(f"Empty label files: {empty_labels}")
    print(f"Total boxes: {class_counts[0] + class_counts[1]}")
    print(f"Format errors: {len(errors)}")

    samples = []
    samples.extend(f"Missing label for image stem: {stem}" for stem in missing_labels[:50])
    samples.extend(f"Extra label without image stem: {stem}" for stem in extra_labels[:50])
    samples.extend(errors[:50])

    if samples:
        print("\nFirst error samples")
        print("-------------------")
        for sample in samples[:50]:
            print(f"- {sample}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check a YOLO left/right hand dataset.")
    parser.add_argument("--image_dir", required=True, help="Directory containing images.")
    parser.add_argument("--label_dir", required=True, help="Directory containing YOLO .txt labels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not label_dir.is_dir():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    check_dataset(image_dir=image_dir, label_dir=label_dir)


if __name__ == "__main__":
    main()
