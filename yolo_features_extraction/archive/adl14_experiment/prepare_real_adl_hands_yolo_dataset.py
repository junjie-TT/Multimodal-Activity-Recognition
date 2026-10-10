from pathlib import Path
import random
import shutil


IMAGE_ROOT = Path("/home/julius/datasets/real_adl_hands/frames_pilot_500")
LABEL_ROOT = Path("/home/julius/datasets/real_adl_hands/frames_pilot_500_agumentation")
OUTPUT_ROOT = Path("datasets/real_adl_hands_yolo_labeled")
SEED = 42
VAL_FRACTION = 0.2

CLASS_NAMES = [
    "pen",
    "paper",
    "bowl",
    "plate",
    "spoon",
    "cup",
    "knife",
    "fork",
    "container",
    "bottle",
    "bar",
    "glass",
    "key",
    "wallet",
]


def reset_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def symlink_force(source, target):
    if target.exists() or target.is_symlink():
        target.unlink()
    target.symlink_to(source)


def label_has_boxes(path):
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) == 5:
            return True
    return False


def validate_label(path):
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        parts = line.strip().split()
        if not parts:
            continue
        if len(parts) != 5:
            raise ValueError(f"{path}:{line_number} is not YOLO bbox format: {line!r}")
        class_id = int(float(parts[0]))
        if class_id < 0 or class_id >= len(CLASS_NAMES):
            raise ValueError(f"{path}:{line_number} class id {class_id} outside 0..{len(CLASS_NAMES)-1}")
        for value in parts[1:]:
            numeric = float(value)
            if numeric < 0.0 or numeric > 1.0:
                raise ValueError(f"{path}:{line_number} normalized bbox value outside [0, 1]: {line!r}")


def main():
    labels = sorted(path for path in LABEL_ROOT.glob("*.txt") if path.name != "classes.txt")
    samples = []
    missing_images = []
    empty_labels = []
    for label_path in labels:
        image_path = IMAGE_ROOT / f"{label_path.stem}.jpg"
        if not image_path.exists():
            missing_images.append(label_path.name)
            continue
        if not label_has_boxes(label_path):
            empty_labels.append(label_path.name)
            continue
        validate_label(label_path)
        samples.append((image_path, label_path))

    if not samples:
        raise RuntimeError("No labeled image/label pairs found.")

    random.Random(SEED).shuffle(samples)
    val_count = max(1, round(len(samples) * VAL_FRACTION))
    split_samples = {
        "val": samples[:val_count],
        "train": samples[val_count:],
    }

    for split in split_samples:
        reset_dir(OUTPUT_ROOT / "images" / split)
        reset_dir(OUTPUT_ROOT / "labels" / split)

    for split, split_items in split_samples.items():
        for image_path, label_path in split_items:
            symlink_force(image_path, OUTPUT_ROOT / "images" / split / image_path.name)
            symlink_force(label_path, OUTPUT_ROOT / "labels" / split / label_path.name)

    yaml_path = OUTPUT_ROOT / "real_adl_hands_14cls.yaml"
    names = "\n".join(f"  {idx}: {name}" for idx, name in enumerate(CLASS_NAMES))
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {OUTPUT_ROOT.resolve()}",
                "train: images/train",
                "val: images/val",
                f"nc: {len(CLASS_NAMES)}",
                "names:",
                names,
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(f"dataset: {OUTPUT_ROOT.resolve()}")
    print(f"yaml: {yaml_path.resolve()}")
    print(f"train images: {len(split_samples['train'])}")
    print(f"val images: {len(split_samples['val'])}")
    print(f"missing images ignored: {len(missing_images)}")
    print(f"empty labels ignored: {len(empty_labels)}")


if __name__ == "__main__":
    main()
