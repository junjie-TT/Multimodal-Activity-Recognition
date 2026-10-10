import json
from pathlib import Path

import numpy as np
import pandas as pd

from upper_features_extraction import config
from upper_features_extraction.utils.geometry import point_in_bbox

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


GAZE_COLUMNS = ["video_frame_idx", "x", "y"]
YOLO_COLUMNS = ["frame_id", "class_id", "confidence", "x1", "y1", "x2", "y2"]


def parse_full_id_from_yolo_path(path):
    return Path(path).name.replace("_world_camera.csv", "")


def build_yolo_lookup(yolo_root=config.YOLO_OBJECT_OUTCOME_ROOT):
    return {
        parse_full_id_from_yolo_path(path): path
        for path in Path(yolo_root).glob("*_world_camera.csv")
    }


def read_gaze_index(index_path=config.GAZE_XY_FILLED_INDEX):
    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(
            f"Missing gaze nearest-fill index: {index_path}. "
            "Run lower_feature_cleaning_extraction.fill_gaze_xy_nearest first."
        )

    index_df = pd.read_csv(index_path)
    gaze_df = index_df[index_df["modality"] == config.GAZE_MODALITY].copy()
    if gaze_df.empty:
        raise ValueError(f"No {config.GAZE_MODALITY} rows found in {index_path}.")

    return gaze_df


def normalize_hit_class(hit_classes, has_detection, valid_gaze_sample_count):
    if valid_gaze_sample_count == 0:
        return "NO_VALID_GAZE"
    if not has_detection:
        return "NO_DETECTION"
    if not hit_classes:
        return "NONE"
    return json.dumps(sorted(hit_classes))


def compute_frame_level_point_in_bbox(gaze_df, yolo_df):
    gaze = gaze_df.copy()
    yolo = yolo_df.copy()

    for col in GAZE_COLUMNS:
        if col not in gaze.columns:
            raise ValueError(f"gaze CSV is missing required column: {col}")
    for col in YOLO_COLUMNS:
        if col not in yolo.columns:
            raise ValueError(f"YOLO CSV is missing required column: {col}")

    gaze["video_frame_idx"] = pd.to_numeric(gaze["video_frame_idx"], errors="coerce")
    gaze["x"] = pd.to_numeric(gaze["x"], errors="coerce")
    gaze["y"] = pd.to_numeric(gaze["y"], errors="coerce")
    gaze = gaze[np.isfinite(gaze["video_frame_idx"])].copy()
    gaze["video_frame_idx"] = gaze["video_frame_idx"].astype(int)

    yolo["frame_id"] = pd.to_numeric(yolo["frame_id"], errors="coerce")
    yolo = yolo[np.isfinite(yolo["frame_id"])].copy()
    yolo["frame_id"] = yolo["frame_id"].astype(int)
    yolo_by_frame = {
        int(frame_id): frame_df
        for frame_id, frame_df in yolo[YOLO_COLUMNS].groupby("frame_id", sort=False)
    }

    frame_rows = []
    box_hit_rows = []
    for frame_id, frame_gaze in gaze.groupby("video_frame_idx", sort=True):
        detections = yolo_by_frame.get(int(frame_id))
        has_detection = detections is not None and len(detections) > 0

        valid_gaze = frame_gaze[np.isfinite(frame_gaze["x"]) & np.isfinite(frame_gaze["y"])]
        hit_classes = set()
        hit_sample_count = 0
        hit_detection_count = 0
        hit_box_count = 0

        if has_detection and not valid_gaze.empty:
            sample_hit_mask = np.zeros(len(valid_gaze), dtype=bool)

            for detection_idx, (_, box) in enumerate(detections.iterrows()):
                box_sample_mask = valid_gaze.apply(
                    lambda row: point_in_bbox(row["x"], row["y"], box),
                    axis=1,
                ).to_numpy()
                box_hit_sample_count = int(box_sample_mask.sum())

                if box_hit_sample_count == 0:
                    continue

                hit_box_count += 1
                hit_detection_count += box_hit_sample_count
                hit_classes.add(int(box["class_id"]))
                sample_hit_mask |= box_sample_mask

                box_hit_rows.append(
                    {
                        "video_frame_idx": int(frame_id),
                        "detection_idx_in_frame": int(detection_idx),
                        "class_id": int(box["class_id"]),
                        "confidence": float(box["confidence"]),
                        "x1": float(box["x1"]),
                        "y1": float(box["y1"]),
                        "x2": float(box["x2"]),
                        "y2": float(box["y2"]),
                        "hit_gaze_sample_count": box_hit_sample_count,
                        "valid_gaze_sample_count": int(len(valid_gaze)),
                        "hit_ratio_in_box": box_hit_sample_count / len(valid_gaze),
                    }
                )

            hit_sample_count = int(sample_hit_mask.sum())

        gaze_sample_count = int(len(frame_gaze))
        valid_gaze_sample_count = int(len(valid_gaze))
        hit_ratio = (
            hit_sample_count / valid_gaze_sample_count
            if valid_gaze_sample_count > 0
            else np.nan
        )

        frame_rows.append(
            {
                "video_frame_idx": int(frame_id),
                "gaze_sample_count": gaze_sample_count,
                "valid_gaze_sample_count": valid_gaze_sample_count,
                "yolo_detection_count": int(len(detections)) if has_detection else 0,
                "hit_box_count": int(hit_box_count),
                "hit_sample_count": int(hit_sample_count),
                "hit_detection_count": int(hit_detection_count),
                "hit_ratio": hit_ratio,
                "hit_class": normalize_hit_class(
                    hit_classes,
                    has_detection=has_detection,
                    valid_gaze_sample_count=valid_gaze_sample_count,
                ),
            }
        )

    return pd.DataFrame(frame_rows), pd.DataFrame(box_hit_rows)


def frame_output_path_for_record(participant_id, full_id, output_root):
    return Path(output_root) / participant_id / f"{full_id}_gaze_yolo_point_in_bbox_frame.csv"


def box_hit_output_path_for_record(participant_id, full_id, output_root):
    return Path(output_root) / participant_id / f"{full_id}_gaze_yolo_point_in_bbox_frame_boxes.csv"


def run_point_in_bbox_frame_aggregation(
    gaze_index_path=config.GAZE_XY_FILLED_INDEX,
    yolo_root=config.YOLO_OBJECT_OUTCOME_ROOT,
    output_root=config.POINT_IN_BBOX_FRAME_OUTPUT_ROOT,
):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    gaze_index_df = read_gaze_index(gaze_index_path)
    yolo_lookup = build_yolo_lookup(yolo_root)

    records = list(gaze_index_df.to_dict("records"))
    iterator = records
    if tqdm is not None:
        iterator = tqdm(records, desc="Point-in-BBox frame aggregation", unit="recording")

    index_rows = []
    missing_yolo_rows = []

    for record in iterator:
        participant_id = record["participant_id"]
        task_name = record["task_name"]
        full_id = record["full_id"]
        gaze_path = Path(record["output_csv_path"])
        yolo_path = yolo_lookup.get(full_id)

        if yolo_path is None:
            missing_yolo_rows.append(
                {
                    "participant_id": participant_id,
                    "task_name": task_name,
                    "full_id": full_id,
                    "gaze_csv_path": str(gaze_path),
                    "expected_yolo_csv": str(Path(yolo_root) / f"{full_id}_world_camera.csv"),
                }
            )
            continue

        gaze_df = pd.read_csv(gaze_path)
        yolo_df = pd.read_csv(yolo_path)
        out_df, box_hit_df = compute_frame_level_point_in_bbox(gaze_df, yolo_df)

        out_path = frame_output_path_for_record(participant_id, full_id, output_root)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)

        box_hit_path = box_hit_output_path_for_record(participant_id, full_id, output_root)
        box_hit_df.to_csv(box_hit_path, index=False)

        valid_frames = out_df[out_df["valid_gaze_sample_count"] > 0]
        index_rows.append(
            {
                "participant_id": participant_id,
                "task_name": task_name,
                "full_id": full_id,
                "gaze_csv_path": str(gaze_path),
                "yolo_csv_path": str(yolo_path),
                "output_csv_path": str(out_path),
                "box_hit_csv_path": str(box_hit_path),
                "frame_count": int(len(out_df)),
                "box_hit_row_count": int(len(box_hit_df)),
                "valid_gaze_frame_count": int(len(valid_frames)),
                "hit_frame_count": int((valid_frames["hit_sample_count"] > 0).sum()),
                "no_detection_frame_count": int((out_df["hit_class"] == "NO_DETECTION").sum()),
                "no_valid_gaze_frame_count": int((out_df["hit_class"] == "NO_VALID_GAZE").sum()),
                "mean_hit_ratio_on_valid_gaze_frames": float(valid_frames["hit_ratio"].mean())
                if not valid_frames.empty
                else np.nan,
            }
        )

    index_df = pd.DataFrame(index_rows)
    missing_yolo_df = pd.DataFrame(missing_yolo_rows)
    index_df.to_csv(config.POINT_IN_BBOX_FRAME_INDEX, index=False)
    missing_yolo_df.to_csv(config.POINT_IN_BBOX_FRAME_MISSING_YOLO, index=False)

    summary = {
        "gaze_index_path": str(gaze_index_path),
        "yolo_root": str(yolo_root),
        "output_root": str(output_root),
        "index_csv": str(config.POINT_IN_BBOX_FRAME_INDEX),
        "missing_yolo_csv": str(config.POINT_IN_BBOX_FRAME_MISSING_YOLO),
        "processed_recording_count": int(len(index_df)),
        "missing_yolo_count": int(len(missing_yolo_df)),
        "policy": (
            "No gaze downsampling is performed. Every gaze sample is tested "
            "against YOLO boxes from its video_frame_idx, then results are "
            "aggregated to one row per video frame."
        ),
    }
    with open(config.POINT_IN_BBOX_FRAME_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)

    return index_df, missing_yolo_df, summary


def main():
    index_df, missing_yolo_df, summary = run_point_in_bbox_frame_aggregation()
    print("Point-in-BBox frame aggregation completed.")
    print(json.dumps(summary, indent=2))

    if not index_df.empty:
        print("\nProcessed recordings:")
        print(
            index_df[
                [
                    "full_id",
                    "frame_count",
                    "valid_gaze_frame_count",
                    "hit_frame_count",
                    "mean_hit_ratio_on_valid_gaze_frames",
                ]
            ].to_string(index=False)
        )

    if not missing_yolo_df.empty:
        print("\nMissing YOLO recordings:")
        print(missing_yolo_df[["full_id", "expected_yolo_csv"]].to_string(index=False))


if __name__ == "__main__":
    main()
