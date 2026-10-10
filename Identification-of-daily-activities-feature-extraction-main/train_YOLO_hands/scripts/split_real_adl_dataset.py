#!/usr/bin/env python3
"""Split Real ADL YOLO hand data by filename-derived groups to reduce frame leakage."""

from __future__ import annotations

import argparse
import random
import re
import shutil
from pathlib import Path

from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def find_images(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def group_key(image_path: Path) -> str:
    stem = image_path.stem
    participant_match = re.match(r"^(IA_\d+_[A-Za-z]+_\d+)", stem)
    if participant_match:
        return participant_match.group(1)

    frame_split = re.split(r"[_*.-]frame[_*.-]?\d+", stem, maxsplit=1)
    if frame_split and frame_split[0] and frame_split[0] != stem:
        return frame_split[0]

    time_split = re.split(r"[_*.-]time[_*.-]?\d+", stem, maxsplit=1)
    if time_split and time_split[0] and time_split[0] != stem:
        return time_split[0]

    parts = stem.split("_")
    if len(parts) >= 4:
        return "_".join(parts[:4])
    return stem


def assign_groups(images: list[Path], train_count: int, val_count: int, seed: int) -> dict[str, list[Path]]:
    groups = {}
    for image_path in images:
        groups.setdefault(group_key(image_path), []).append(image_path)

    group_items = list(groups.items())
    rng = random.Random(seed)
    rng.shuffle(group_items)

    splits = {"train": [], "val": [], "test": []}
    for _, group_images in group_items:
        if len(splits["train"]) < train_count:
            splits["train"].extend(group_images)
        elif len(splits["val"]) < val_count:
            splits["val"].extend(group_images)
        else:
            splits["test"].extend(group_images)

    for split_name in splits:
        splits[split_name] = sorted(splits[split_name])
    return splits


def copy_split(split_name: str, images: list[Path], label_dir: Path, out_root: Path) -> tuple[int, int]:
    image_out = out_root / "images" / split_name
    label_out = out_root / "labels" / split_name
    image_out.mkdir(parents=True, exist_ok=True)
    label_out.mkdir(parents=True, exist_ok=True)

    copied = 0
    skipped = 0
    for image_path in tqdm(images, desc=f"Copying {split_name}", unit="image"):
        src_label = label_dir / f"{image_path.stem}.txt"
        if not src_label.exists():
            print(f"Warning: missing label for {image_path.name}; skipping image")
            skipped += 1
            continue

        dst_label = label_out / f"{image_path.stem}.txt"
        shutil.copy2(image_path, image_out / image_path.name)
        shutil.copy2(src_label, dst_label)
        copied += 1
    return copied, skipped


def write_data_yaml(out_root: Path) -> None:
    content = f"""path: {out_root}

train: images/train
val: images/val
test: images/test

names:
  0: left_hand
  1: right_hand
"""
    (out_root / "data.yaml").write_text(content, encoding="utf-8")


def split_dataset(image_dir: Path, label_dir: Path, out_root: Path, train_count: int, val_count: int, seed: int) -> None:
    images = []
    skipped_before_split = 0
    for image_path in find_images(image_dir):
        if (label_dir / f"{image_path.stem}.txt").exists():
            images.append(image_path)
        else:
            print(f"Warning: missing label for {image_path.name}; excluding from split")
            skipped_before_split += 1

    splits = assign_groups(images, train_count=train_count, val_count=val_count, seed=seed)

    copied_counts = {}
    skipped_during_copy = {}
    for split_name, split_images in splits.items():
        copied, skipped = copy_split(split_name=split_name, images=split_images, label_dir=label_dir, out_root=out_root)
        copied_counts[split_name] = copied
        skipped_during_copy[split_name] = skipped

    write_data_yaml(out_root)

    print("\nSummary")
    print("-------")
    print(f"Images skipped before split because label file was missing: {skipped_before_split}")
    print(f"Train images: {copied_counts['train']}")
    print(f"Val images: {copied_counts['val']}")
    print(f"Test images: {copied_counts['test']}")
    print(f"Images skipped during copy: {sum(skipped_during_copy.values())}")
    print(f"Output path: {out_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split Real ADL YOLO hand dataset by participant/video groups.")
    parser.add_argument("--image_dir", required=True, help="Raw image directory.")
    parser.add_argument("--label_dir", required=True, help="Raw label directory.")
    parser.add_argument("--out_root", required=True, help="Output root for split YOLO dataset.")
    parser.add_argument("--train_count", type=int, default=1000, help="Target train image count.")
    parser.add_argument("--val_count", type=int, default=200, help="Target val image count.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for group order.")
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
    split_dataset(
        image_dir=image_dir,
        label_dir=label_dir,
        out_root=out_root,
        train_count=args.train_count,
        val_count=args.val_count,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
