#!/usr/bin/env python3
"""Build a mixed source-hand + Real ADL YOLO dataset for left/right hand detection."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def find_images(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def copy_pair(image_path: Path, label_path: Path, image_out: Path, label_out: Path, prefix: str) -> bool:
    if not label_path.exists():
        print(f"Warning: missing label for {image_path}; skipping image")
        return False

    dst_image_name = f"{prefix}{image_path.name}"
    dst_label_name = f"{Path(dst_image_name).stem}.txt"
    shutil.copy2(image_path, image_out / dst_image_name)
    shutil.copy2(label_path, label_out / dst_label_name)
    return True


def copy_real_split(real_root: Path, out_root: Path, split_name: str) -> int:
    src_images = real_root / "images" / split_name
    src_labels = real_root / "labels" / split_name
    dst_images = out_root / "images" / split_name
    dst_labels = out_root / "labels" / split_name
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    images = find_images(src_images)
    copied = 0
    for image_path in tqdm(images, desc=f"Copying Real ADL {split_name}", unit="image"):
        if copy_pair(
            image_path=image_path,
            label_path=src_labels / f"{image_path.stem}.txt",
            image_out=dst_images,
            label_out=dst_labels,
            prefix="real_",
        ):
            copied += 1
    return copied


def copy_ego_train(ego_images: Path, ego_labels: Path, out_root: Path, ego_sample: int, seed: int) -> int:
    all_images = find_images(ego_images)
    images = []
    skipped_missing_labels = 0
    for image_path in all_images:
        if (ego_labels / f"{image_path.stem}.txt").exists():
            images.append(image_path)
        else:
            skipped_missing_labels += 1
    if skipped_missing_labels:
        print(f"Warning: skipping {skipped_missing_labels} source image(s) without label files")

    rng = random.Random(seed)
    if ego_sample < len(images):
        images = rng.sample(images, ego_sample)
    else:
        print(f"Warning: ego_sample={ego_sample} >= available source images={len(images)}; using all images")
    images = sorted(images)

    dst_images = out_root / "images" / "train"
    dst_labels = out_root / "labels" / "train"
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    copied = 0
    for image_path in tqdm(images, desc="Copying sampled source train", unit="image"):
        if copy_pair(
            image_path=image_path,
            label_path=ego_labels / f"{image_path.stem}.txt",
            image_out=dst_images,
            label_out=dst_labels,
            prefix="ego_",
        ):
            copied += 1
    return copied


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


def build_mixed_dataset(
    ego_images: Path,
    ego_labels: Path,
    real_root: Path,
    out_root: Path,
    ego_sample: int,
    seed: int,
) -> None:
    real_train_count = copy_real_split(real_root=real_root, out_root=out_root, split_name="train")
    val_count = copy_real_split(real_root=real_root, out_root=out_root, split_name="val")
    test_count = copy_real_split(real_root=real_root, out_root=out_root, split_name="test")
    ego_count = copy_ego_train(ego_images=ego_images, ego_labels=ego_labels, out_root=out_root, ego_sample=ego_sample, seed=seed)
    write_data_yaml(out_root)

    print("\nSummary")
    print("-------")
    print(f"Real ADL train images: {real_train_count}")
    print(f"Sampled source images: {ego_count}")
    print(f"Mixed train images: {real_train_count + ego_count}")
    print(f"Val images: {val_count}")
    print(f"Test images: {test_count}")
    print(f"Output path: {out_root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build mixed source-hand + Real ADL YOLO hand dataset.")
    parser.add_argument("--ego_images", required=True, help="Source train image directory, e.g. EgoHands images/train.")
    parser.add_argument("--ego_labels", required=True, help="Source train label directory, e.g. EgoHands labels/train.")
    parser.add_argument("--real_root", required=True, help="Split Real ADL YOLO dataset root.")
    parser.add_argument("--out_root", required=True, help="Output root for mixed YOLO dataset.")
    parser.add_argument("--ego_sample", type=int, default=2500, help="Number of source train images to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for source sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ego_images = Path(args.ego_images)
    ego_labels = Path(args.ego_labels)
    real_root = Path(args.real_root)
    out_root = Path(args.out_root)
    for directory in [ego_images, ego_labels, real_root / "images" / "train", real_root / "labels" / "train"]:
        if not directory.is_dir():
            raise FileNotFoundError(f"Required directory not found: {directory}")
    build_mixed_dataset(
        ego_images=ego_images,
        ego_labels=ego_labels,
        real_root=real_root,
        out_root=out_root,
        ego_sample=args.ego_sample,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
