import argparse
import json
import shutil
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


DEFAULT_LEFT_NAMES = {
    "left_hand",
    "left hand",
    "own left",
    "own_left",
    "own left hand",
    "own_left_hand",
}
DEFAULT_RIGHT_NAMES = {
    "right_hand",
    "right hand",
    "own right",
    "own_right",
    "own right hand",
    "own_right_hand",
}
DEFAULT_OTHER_LEFT_NAMES = {
    "other left",
    "other_left",
    "other left hand",
    "other_left_hand",
}
DEFAULT_OTHER_RIGHT_NAMES = {
    "other right",
    "other_right",
    "other right hand",
    "other_right_hand",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "val", "test")


def normalize_name(name):
    return str(name).strip().lower().replace("-", " ").replace("_", " ")


def load_names(source_root, data_yaml=None):
    if yaml is None:
        raise ImportError("Missing dependency: pyyaml. Install it with `pip install pyyaml`.")

    data_yaml = Path(data_yaml) if data_yaml else Path(source_root) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing data.yaml: {data_yaml}")

    with open(data_yaml) as f:
        data = yaml.safe_load(f)

    names = data.get("names")
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names, key=lambda item: int(item))]
    if not isinstance(names, list):
        raise ValueError(f"Could not parse class names from: {data_yaml}")

    return [str(name) for name in names], data_yaml


def build_class_mapping(names, other_mode):
    mapping = {}
    ignored = {}

    for class_id, name in enumerate(names):
        normalized = normalize_name(name)
        compact = normalized.replace(" ", "_")

        if normalized in DEFAULT_LEFT_NAMES or compact in DEFAULT_LEFT_NAMES:
            mapping[class_id] = 0
        elif normalized in DEFAULT_RIGHT_NAMES or compact in DEFAULT_RIGHT_NAMES:
            mapping[class_id] = 1
        elif normalized in DEFAULT_OTHER_LEFT_NAMES or compact in DEFAULT_OTHER_LEFT_NAMES:
            if other_mode == "merge_by_side":
                mapping[class_id] = 0
            elif other_mode == "other_hand":
                mapping[class_id] = 2
            else:
                ignored[class_id] = name
        elif normalized in DEFAULT_OTHER_RIGHT_NAMES or compact in DEFAULT_OTHER_RIGHT_NAMES:
            if other_mode == "merge_by_side":
                mapping[class_id] = 1
            elif other_mode == "other_hand":
                mapping[class_id] = 2
            else:
                ignored[class_id] = name
        else:
            ignored[class_id] = name

    output_names = ["left_hand", "right_hand"]
    if other_mode == "other_hand":
        output_names.append("other_hand")

    return mapping, ignored, output_names


def find_split_dirs(source_root):
    source_root = Path(source_root)
    split_dirs = {}
    for split in SPLITS:
        split_dir = source_root / split
        if split_dir.exists():
            canonical = "val" if split == "valid" else split
            split_dirs[canonical] = split_dir
    return split_dirs


def find_images_dir(split_dir):
    images_dir = split_dir / "images"
    return images_dir if images_dir.exists() else split_dir


def find_labels_dir(split_dir):
    labels_dir = split_dir / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError(f"Missing labels directory: {labels_dir}")
    return labels_dir


def iter_images(images_dir):
    return sorted(path for path in Path(images_dir).rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)


def convert_label_file(input_path, output_path, class_mapping):
    kept = 0
    dropped = 0
    output_lines = []

    if input_path.exists():
        with open(input_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                old_class = int(float(parts[0]))
                new_class = class_mapping.get(old_class)
                if new_class is None:
                    dropped += 1
                    continue
                output_lines.append(" ".join([str(new_class), *parts[1:5]]))
                kept += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        if output_lines:
            f.write("\n".join(output_lines) + "\n")

    return kept, dropped


def copy_or_symlink(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src)


def write_data_yaml(output_root, output_names):
    data = {
        "path": str(Path(output_root).resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "names": {idx: name for idx, name in enumerate(output_names)},
    }
    data_yaml = Path(output_root) / "data.yaml"
    if yaml is not None:
        with open(data_yaml, "w") as f:
            yaml.safe_dump(data, f, sort_keys=False)
    else:
        with open(data_yaml, "w") as f:
            f.write(f"path: {data['path']}\n")
            f.write("train: train/images\nval: val/images\ntest: test/images\nnames:\n")
            for idx, name in enumerate(output_names):
                f.write(f"  {idx}: {name}\n")
    return data_yaml


def prepare_dataset(source_root, output_root, data_yaml=None, other_mode="ignore", link_mode="symlink"):
    source_root = Path(source_root)
    output_root = Path(output_root)
    names, source_data_yaml = load_names(source_root, data_yaml)
    class_mapping, ignored, output_names = build_class_mapping(names, other_mode)

    if 0 not in class_mapping.values() or 1 not in class_mapping.values():
        raise ValueError(
            "Class mapping did not find both left_hand and right_hand. "
            f"Source names were: {names}"
        )

    split_dirs = find_split_dirs(source_root)
    if "train" not in split_dirs or "val" not in split_dirs:
        raise FileNotFoundError(
            "Expected YOLO dataset splits under train/ and valid/ or val/."
        )

    rows = []
    totals = {"images": 0, "kept_boxes": 0, "dropped_boxes": 0}
    for split, split_dir in split_dirs.items():
        images_dir = find_images_dir(split_dir)
        labels_dir = find_labels_dir(split_dir)
        out_images = output_root / split / "images"
        out_labels = output_root / split / "labels"

        for image_path in iter_images(images_dir):
            relative = image_path.relative_to(images_dir)
            label_path = labels_dir / relative.with_suffix(".txt")
            out_image_path = out_images / relative
            out_label_path = out_labels / relative.with_suffix(".txt")

            copy_or_symlink(image_path, out_image_path, link_mode)
            kept, dropped = convert_label_file(label_path, out_label_path, class_mapping)

            rows.append(
                {
                    "split": split,
                    "image": str(image_path),
                    "label": str(label_path),
                    "kept_boxes": kept,
                    "dropped_boxes": dropped,
                }
            )
            totals["images"] += 1
            totals["kept_boxes"] += kept
            totals["dropped_boxes"] += dropped

    output_root.mkdir(parents=True, exist_ok=True)
    out_data_yaml = write_data_yaml(output_root, output_names)
    summary = {
        "source_root": str(source_root),
        "source_data_yaml": str(source_data_yaml),
        "output_root": str(output_root),
        "output_data_yaml": str(out_data_yaml),
        "source_names": names,
        "output_names": output_names,
        "class_mapping": {str(key): value for key, value in class_mapping.items()},
        "ignored_classes": {str(key): value for key, value in ignored.items()},
        "other_mode": other_mode,
        "link_mode": link_mode,
        **totals,
    }
    summary_path = output_root / "prepare_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Convert EgoHands-style YOLO labels to left_hand/right_hand YOLO training data."
    )
    parser.add_argument("--source-root", required=True, help="Downloaded EgoHands YOLO dataset root.")
    parser.add_argument("--output-root", required=True, help="Prepared YOLO dataset output root.")
    parser.add_argument("--data-yaml", default=None, help="Optional source data.yaml path.")
    parser.add_argument(
        "--other-mode",
        choices=("ignore", "merge_by_side", "other_hand"),
        default="ignore",
        help="How to handle other-left/other-right classes.",
    )
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="Use symlinks to avoid duplicating images, or copy files.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    summary = prepare_dataset(
        source_root=args.source_root,
        output_root=args.output_root,
        data_yaml=args.data_yaml,
        other_mode=args.other_mode,
        link_mode=args.link_mode,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
