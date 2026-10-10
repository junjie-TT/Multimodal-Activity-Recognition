import argparse
import json
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat

try:
    import yaml
except ImportError:
    yaml = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
CLASS_NAMES = ["left_hand", "right_hand"]


def split_video_dirs(video_dirs, train_ratio=0.8, val_ratio=0.1, seed=42):
    video_dirs = list(video_dirs)
    rng = random.Random(seed)
    rng.shuffle(video_dirs)

    n_total = len(video_dirs)
    n_train = int(round(n_total * train_ratio))
    n_val = int(round(n_total * val_ratio))

    return {
        "train": video_dirs[:n_train],
        "val": video_dirs[n_train:n_train + n_val],
        "test": video_dirs[n_train + n_val:],
    }


def polygon_to_yolo_bbox(polygon, width, height):
    polygon = np.asarray(polygon, dtype=float)
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] < 2:
        return None

    x1 = float(np.clip(np.nanmin(polygon[:, 0]), 0, width - 1))
    y1 = float(np.clip(np.nanmin(polygon[:, 1]), 0, height - 1))
    x2 = float(np.clip(np.nanmax(polygon[:, 0]), 0, width - 1))
    y2 = float(np.clip(np.nanmax(polygon[:, 1]), 0, height - 1))

    if x2 <= x1 or y2 <= y1:
        return None

    x_center = ((x1 + x2) / 2.0) / width
    y_center = ((y1 + y2) / 2.0) / height
    box_width = (x2 - x1) / width
    box_height = (y2 - y1) / height

    return x_center, y_center, box_width, box_height


def field_to_class_id(field_name, other_mode):
    if field_name == "myleft":
        return 0
    if field_name == "myright":
        return 1
    if field_name == "yourleft":
        return 0 if other_mode == "merge_by_side" else None
    if field_name == "yourright":
        return 1 if other_mode == "merge_by_side" else None
    return None


def copy_or_symlink(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)


def write_data_yaml(output_root):
    data = {
        "path": str(Path(output_root).resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {idx: name for idx, name in enumerate(CLASS_NAMES)},
    }
    data_yaml = Path(output_root) / "data.yaml"
    if yaml is None:
        with open(data_yaml, "w") as f:
            f.write(f"path: {data['path']}\n")
            f.write("train: train/images\n")
            f.write("val: val/images\n")
            f.write("test: test/images\n")
            f.write("names:\n")
            for idx, name in enumerate(CLASS_NAMES):
                f.write(f"  {idx}: {name}\n")
    else:
        with open(data_yaml, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
    return data_yaml


def convert_video_dir(video_dir, split, output_root, other_mode, link_mode):
    polygons_path = video_dir / "polygons.mat"
    if not polygons_path.exists():
        raise FileNotFoundError(polygons_path)

    polygons = loadmat(polygons_path, squeeze_me=True, struct_as_record=False)["polygons"]
    image_paths = sorted(path for path in video_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if len(image_paths) != len(polygons):
        raise ValueError(
            f"Image count and polygon count differ in {video_dir}: "
            f"{len(image_paths)} images, {len(polygons)} polygons"
        )

    rows = []
    for image_path, polygon_record in zip(image_paths, polygons):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Could not read image: {image_path}")
        height, width = image.shape[:2]

        relative_name = f"{video_dir.name}_{image_path.name}"
        out_image = output_root / split / "images" / relative_name
        out_label = output_root / split / "labels" / Path(relative_name).with_suffix(".txt")

        label_lines = []
        dropped = 0
        for field_name in ("myleft", "myright", "yourleft", "yourright"):
            class_id = field_to_class_id(field_name, other_mode)
            polygon = getattr(polygon_record, field_name)
            if class_id is None:
                if np.asarray(polygon).size:
                    dropped += 1
                continue

            bbox = polygon_to_yolo_bbox(polygon, width, height)
            if bbox is None:
                continue

            label_lines.append(
                f"{class_id} " + " ".join(f"{value:.6f}" for value in bbox)
            )

        copy_or_symlink(image_path, out_image, link_mode)
        out_label.parent.mkdir(parents=True, exist_ok=True)
        with open(out_label, "w") as f:
            if label_lines:
                f.write("\n".join(label_lines) + "\n")

        rows.append(
            {
                "split": split,
                "video": video_dir.name,
                "image": str(image_path),
                "labels": len(label_lines),
                "dropped_other_hands": dropped,
            }
        )

    return rows


def convert_egohands_mat_to_yolo(
    source_root,
    output_root,
    other_mode="ignore",
    link_mode="symlink",
    train_ratio=0.8,
    val_ratio=0.1,
    seed=42,
):
    source_root = Path(source_root)
    output_root = Path(output_root)
    labelled_root = source_root / "_LABELLED_SAMPLES"
    if not labelled_root.exists():
        raise FileNotFoundError(labelled_root)

    video_dirs = sorted(path for path in labelled_root.iterdir() if path.is_dir())
    split_dirs = split_video_dirs(video_dirs, train_ratio=train_ratio, val_ratio=val_ratio, seed=seed)

    all_rows = []
    for split, dirs in split_dirs.items():
        for video_dir in dirs:
            all_rows.extend(
                convert_video_dir(
                    video_dir=video_dir,
                    split=split,
                    output_root=output_root,
                    other_mode=other_mode,
                    link_mode=link_mode,
                )
            )

    output_root.mkdir(parents=True, exist_ok=True)
    data_yaml = write_data_yaml(output_root)
    summary = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "data_yaml": str(data_yaml),
        "other_mode": other_mode,
        "link_mode": link_mode,
        "train_ratio": train_ratio,
        "val_ratio": val_ratio,
        "seed": seed,
        "videos": {split: len(dirs) for split, dirs in split_dirs.items()},
        "images": len(all_rows),
        "boxes": int(sum(row["labels"] for row in all_rows)),
        "dropped_other_hands": int(sum(row["dropped_other_hands"] for row in all_rows)),
        "class_names": CLASS_NAMES,
    }

    summary_path = output_root / "convert_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Convert official EgoHands polygons.mat annotations to YOLO left/right hand boxes."
    )
    parser.add_argument("--source-root", required=True, help="Official EgoHands extracted root.")
    parser.add_argument("--output-root", required=True, help="YOLO dataset output root.")
    parser.add_argument(
        "--other-mode",
        choices=("ignore", "merge_by_side"),
        default="ignore",
        help="Ignore yourleft/yourright, or merge them into left/right by side.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="Use symlinks to avoid duplicating images, or copy files.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main():
    args = build_arg_parser().parse_args()
    summary = convert_egohands_mat_to_yolo(
        source_root=args.source_root,
        output_root=args.output_root,
        other_mode=args.other_mode,
        link_mode=args.link_mode,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
