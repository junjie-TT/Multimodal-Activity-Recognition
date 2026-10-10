import argparse
from pathlib import Path

import pandas as pd

from yolo_features_extraction import config


COCO_CLASS_NAMES = {
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    73: "book",
}

SEMANTIC_EQUIVALENTS = {
    "bottle": {"bottle"},
    "cup": {"cup", "wine glass"},
    "glass": {"cup", "wine glass"},
    "bowl": {"bowl"},
    "knife": {"knife"},
    "fork": {"fork"},
    "spoon": {"spoon"},
    "paper": {"book"},
}

DEFAULT_COCO_ROOT = config.OUTPUT_ROOT
DEFAULT_ADL14_ROOT = (
    config.DATASET_ROOT
    / "lower_feature_outcome"
    / "yolo_real_adl_14_object_outcome_csv"
)
DEFAULT_OUTPUT_ROOT = (
    config.DATASET_ROOT
    / "lower_feature_outcome"
    / "yolo_object_outcome_csv_coco_plus_adl14"
)
DEFAULT_DUPLICATE_IOU_THRESHOLD = 0.60


def normalize_coco(coco_df):
    coco_df = coco_df.copy()
    coco_df["local_class_id"] = pd.NA
    coco_df["class_name"] = pd.to_numeric(coco_df["class_id"], errors="coerce").map(COCO_CLASS_NAMES)
    coco_df["source_model"] = "coco"
    return coco_df


def normalize_adl14(adl_df):
    adl_df = adl_df.copy()
    if "source_model" not in adl_df.columns and "model_source" in adl_df.columns:
        adl_df = adl_df.rename(columns={"model_source": "source_model"})
    adl_df["source_model"] = adl_df.get("source_model", "adl14").fillna("adl14")
    return adl_df


def bbox_iou(row_a, row_b):
    x1 = max(float(row_a["x1"]), float(row_b["x1"]))
    y1 = max(float(row_a["y1"]), float(row_b["y1"]))
    x2 = min(float(row_a["x2"]), float(row_b["x2"]))
    y2 = min(float(row_a["y2"]), float(row_b["y2"]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    intersection = inter_w * inter_h
    if intersection <= 0.0:
        return 0.0

    area_a = max(0.0, float(row_a["x2"]) - float(row_a["x1"])) * max(0.0, float(row_a["y2"]) - float(row_a["y1"]))
    area_b = max(0.0, float(row_b["x2"]) - float(row_b["x1"])) * max(0.0, float(row_b["y2"]) - float(row_b["y1"]))
    union = area_a + area_b - intersection
    return intersection / union if union > 0.0 else 0.0


def semantically_equivalent(coco_row, adl_row):
    adl_name = str(adl_row.get("class_name", "")).strip().lower()
    coco_name = str(coco_row.get("class_name", "")).strip().lower()
    return coco_name in SEMANTIC_EQUIVALENTS.get(adl_name, set())


def drop_coco_duplicates(merged, iou_threshold=DEFAULT_DUPLICATE_IOU_THRESHOLD):
    if merged.empty or "source_model" not in merged.columns:
        return merged

    drop_indices = set()
    for _, frame_df in merged.groupby("frame_id", sort=False):
        coco_rows = frame_df[frame_df["source_model"].eq("coco")]
        adl_rows = frame_df[frame_df["source_model"].eq("adl14")]
        if coco_rows.empty or adl_rows.empty:
            continue

        for coco_index, coco_row in coco_rows.iterrows():
            for _, adl_row in adl_rows.iterrows():
                if semantically_equivalent(coco_row, adl_row) and bbox_iou(coco_row, adl_row) >= iou_threshold:
                    drop_indices.add(coco_index)
                    break

    if not drop_indices:
        return merged
    return merged.drop(index=sorted(drop_indices)).reset_index(drop=True)


def merge_one(coco_path, adl_path, output_path, iou_threshold=DEFAULT_DUPLICATE_IOU_THRESHOLD):
    frames = []
    if coco_path.exists():
        frames.append(normalize_coco(pd.read_csv(coco_path)))
    if adl_path.exists():
        frames.append(normalize_adl14(pd.read_csv(adl_path)))
    if not frames:
        return False

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = drop_coco_duplicates(merged, iou_threshold=iou_threshold)
    sort_cols = [col for col in ["frame_id", "source_model", "class_id", "confidence"] if col in merged.columns]
    ascending = [True, True, True, False][: len(sort_cols)]
    merged = merged.sort_values(sort_cols, ascending=ascending) if sort_cols else merged

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    return True


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Merge COCO YOLO CSVs with ADL 14-class YOLO CSVs.")
    parser.add_argument("--coco-root", type=Path, default=DEFAULT_COCO_ROOT)
    parser.add_argument("--adl14-root", type=Path, default=DEFAULT_ADL14_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--duplicate-iou-threshold", type=float, default=DEFAULT_DUPLICATE_IOU_THRESHOLD)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    csv_names = sorted({p.name for p in args.coco_root.glob("*.csv")} | {p.name for p in args.adl14_root.glob("*.csv")})
    written = 0
    skipped = 0
    for name in csv_names:
        output_path = args.output_root / name
        if output_path.exists() and output_path.stat().st_size > 0 and not args.overwrite:
            skipped += 1
            continue
        if merge_one(
            args.coco_root / name,
            args.adl14_root / name,
            output_path,
            iou_threshold=args.duplicate_iou_threshold,
        ):
            written += 1
    print(
        {
            "coco_root": str(args.coco_root),
            "adl14_root": str(args.adl14_root),
            "output_root": str(args.output_root),
            "duplicate_iou_threshold": args.duplicate_iou_threshold,
            "written": written,
            "skipped": skipped,
        }
    )


if __name__ == "__main__":
    main()
