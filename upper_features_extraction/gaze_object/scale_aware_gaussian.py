import json
from pathlib import Path

import numpy as np
import pandas as pd

from upper_features_extraction import config
from upper_features_extraction.gaze_object.point_in_bbox import (
    build_yolo_lookup,
    read_gaze_index,
)
from upper_features_extraction.utils.geometry import (
    bbox_diagonal,
    gaussian_score,
    point_to_bbox_center_distance,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


GAZE_COLUMNS = ["video_frame_idx", "x", "y"]
YOLO_COLUMNS = ["frame_id", "class_id", "confidence", "x1", "y1", "x2", "y2"]


def class_penalty_factors(class_id):
    if int(class_id) in config.GAZE_OBJECT_PENALIZED_CLASS_IDS:
        return config.PENALIZED_CLASS_SIGMA_FACTOR, config.PENALIZED_CLASS_SCORE_FACTOR
    return 1.0, 1.0


def normalize_best_class(best_class, has_detection, valid_gaze_sample_count, best_score, threshold):
    if valid_gaze_sample_count == 0:
        return "NO_VALID_GAZE"
    if not has_detection:
        return "NO_DETECTION"
    if best_class is None or best_score < threshold:
        return "NONE"
    return int(best_class)


def compute_frame_level_scale_aware_gaussian(
    gaze_df,
    yolo_df,
    alpha=config.GAUSSIAN_ALPHA,
    threshold=config.GAUSSIAN_SCORE_THRESHOLD,
):
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
    box_score_rows = []
    for frame_id, frame_gaze in gaze.groupby("video_frame_idx", sort=True):
        detections = yolo_by_frame.get(int(frame_id))
        has_detection = detections is not None and len(detections) > 0

        valid_gaze = frame_gaze[np.isfinite(frame_gaze["x"]) & np.isfinite(frame_gaze["y"])]
        best_score = 0.0
        best_class = None
        best_detection_idx = None

        if has_detection and not valid_gaze.empty:
            for detection_idx, (_, box) in enumerate(detections.iterrows()):
                sigma_factor, score_factor = class_penalty_factors(box["class_id"])
                sigma = alpha * bbox_diagonal(box) * sigma_factor
                sample_scores = []
                for _, gaze_row in valid_gaze.iterrows():
                    distance = point_to_bbox_center_distance(gaze_row["x"], gaze_row["y"], box)
                    score = (
                        gaussian_score(distance, sigma)
                        * float(box["confidence"])
                        * score_factor
                    )
                    sample_scores.append(score)

                if not sample_scores:
                    continue

                mean_score = float(np.mean(sample_scores))
                max_score = float(np.max(sample_scores))
                if mean_score > best_score:
                    best_score = mean_score
                    best_class = int(box["class_id"])
                    best_detection_idx = int(detection_idx)

                box_score_rows.append(
                    {
                        "video_frame_idx": int(frame_id),
                        "detection_idx_in_frame": int(detection_idx),
                        "class_id": int(box["class_id"]),
                        "confidence": float(box["confidence"]),
                        "x1": float(box["x1"]),
                        "y1": float(box["y1"]),
                        "x2": float(box["x2"]),
                        "y2": float(box["y2"]),
                        "sigma": float(sigma),
                        "class_sigma_factor": float(sigma_factor),
                        "class_score_factor": float(score_factor),
                        "valid_gaze_sample_count": int(len(valid_gaze)),
                        "mean_score": mean_score,
                        "max_score": max_score,
                    }
                )

        valid_gaze_sample_count = int(len(valid_gaze))
        frame_rows.append(
            {
                "video_frame_idx": int(frame_id),
                "gaze_sample_count": int(len(frame_gaze)),
                "valid_gaze_sample_count": valid_gaze_sample_count,
                "yolo_detection_count": int(len(detections)) if has_detection else 0,
                "best_detection_idx_in_frame": best_detection_idx,
                "best_class": normalize_best_class(
                    best_class,
                    has_detection=has_detection,
                    valid_gaze_sample_count=valid_gaze_sample_count,
                    best_score=best_score,
                    threshold=threshold,
                ),
                "best_score": float(best_score),
                "score_threshold": float(threshold),
            }
        )

    return pd.DataFrame(frame_rows), pd.DataFrame(box_score_rows)


def frame_output_path_for_record(participant_id, full_id, output_root):
    return Path(output_root) / participant_id / f"{full_id}_gaze_yolo_scale_aware_gaussian_frame.csv"


def box_score_output_path_for_record(participant_id, full_id, output_root):
    return Path(output_root) / participant_id / f"{full_id}_gaze_yolo_scale_aware_gaussian_frame_boxes.csv"


def run_scale_aware_gaussian_frame_aggregation(
    gaze_index_path=config.GAZE_XY_FILLED_INDEX,
    yolo_root=config.YOLO_OBJECT_OUTCOME_ROOT,
    output_root=config.SCALE_AWARE_GAUSSIAN_FRAME_OUTPUT_ROOT,
    alpha=config.GAUSSIAN_ALPHA,
    threshold=config.GAUSSIAN_SCORE_THRESHOLD,
):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    gaze_index_df = read_gaze_index(gaze_index_path)
    yolo_lookup = build_yolo_lookup(yolo_root)

    records = list(gaze_index_df.to_dict("records"))
    iterator = records
    if tqdm is not None:
        iterator = tqdm(records, desc="Scale-aware Gaussian frame aggregation", unit="recording")

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
        out_df, box_score_df = compute_frame_level_scale_aware_gaussian(
            gaze_df,
            yolo_df,
            alpha=alpha,
            threshold=threshold,
        )

        out_path = frame_output_path_for_record(participant_id, full_id, output_root)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)

        box_score_path = box_score_output_path_for_record(participant_id, full_id, output_root)
        box_score_df.to_csv(box_score_path, index=False)

        valid_frames = out_df[out_df["valid_gaze_sample_count"] > 0]
        scored_frames = valid_frames[pd.to_numeric(valid_frames["best_score"], errors="coerce") >= threshold]
        index_rows.append(
            {
                "participant_id": participant_id,
                "task_name": task_name,
                "full_id": full_id,
                "gaze_csv_path": str(gaze_path),
                "yolo_csv_path": str(yolo_path),
                "output_csv_path": str(out_path),
                "box_score_csv_path": str(box_score_path),
                "frame_count": int(len(out_df)),
                "box_score_row_count": int(len(box_score_df)),
                "valid_gaze_frame_count": int(len(valid_frames)),
                "scored_frame_count": int(len(scored_frames)),
                "no_detection_frame_count": int((out_df["best_class"] == "NO_DETECTION").sum()),
                "no_valid_gaze_frame_count": int((out_df["best_class"] == "NO_VALID_GAZE").sum()),
                "mean_best_score_on_valid_gaze_frames": float(valid_frames["best_score"].mean())
                if not valid_frames.empty
                else np.nan,
            }
        )

    index_df = pd.DataFrame(index_rows)
    missing_yolo_df = pd.DataFrame(missing_yolo_rows)
    index_df.to_csv(config.SCALE_AWARE_GAUSSIAN_FRAME_INDEX, index=False)
    missing_yolo_df.to_csv(config.SCALE_AWARE_GAUSSIAN_FRAME_MISSING_YOLO, index=False)

    summary = {
        "gaze_index_path": str(gaze_index_path),
        "yolo_root": str(yolo_root),
        "output_root": str(output_root),
        "index_csv": str(config.SCALE_AWARE_GAUSSIAN_FRAME_INDEX),
        "missing_yolo_csv": str(config.SCALE_AWARE_GAUSSIAN_FRAME_MISSING_YOLO),
        "processed_recording_count": int(len(index_df)),
        "missing_yolo_count": int(len(missing_yolo_df)),
        "alpha": float(alpha),
        "score_threshold": float(threshold),
        "penalized_class_ids": list(config.GAZE_OBJECT_PENALIZED_CLASS_IDS),
        "penalized_class_sigma_factor": float(config.PENALIZED_CLASS_SIGMA_FACTOR),
        "penalized_class_score_factor": float(config.PENALIZED_CLASS_SCORE_FACTOR),
        "policy": (
            "No gaze downsampling is performed. Each gaze sample is scored "
            "against each YOLO box using a Gaussian centered at the bbox center, "
            "with sigma = alpha * bbox_diagonal. Scores are multiplied by YOLO "
            "confidence and averaged per frame-box. Penalized classes use a "
            "smaller sigma and lower score prior."
        ),
    }
    with open(config.SCALE_AWARE_GAUSSIAN_FRAME_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)

    return index_df, missing_yolo_df, summary


def main():
    index_df, missing_yolo_df, summary = run_scale_aware_gaussian_frame_aggregation()
    print("Scale-aware Gaussian frame aggregation completed.")
    print(json.dumps(summary, indent=2))

    if not index_df.empty:
        print("\nProcessed recordings:")
        print(
            index_df[
                [
                    "full_id",
                    "frame_count",
                    "valid_gaze_frame_count",
                    "scored_frame_count",
                    "mean_best_score_on_valid_gaze_frames",
                ]
            ].to_string(index=False)
        )

    if not missing_yolo_df.empty:
        print("\nMissing YOLO recordings:")
        print(missing_yolo_df[["full_id", "expected_yolo_csv"]].to_string(index=False))


if __name__ == "__main__":
    main()
