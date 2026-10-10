import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from upper_features_extraction import config
from upper_features_extraction.gaze_object.point_in_bbox import (
    build_yolo_lookup,
    read_gaze_index,
)
from upper_features_extraction.utils.geometry import (
    bbox_center,
    bbox_diagonal,
    gaussian_overlap_density,
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


def entropy(probabilities, eps=config.POSTERIOR_EPS):
    probs = np.asarray(probabilities, dtype=float)
    probs = probs[np.isfinite(probs) & (probs > eps)]
    if probs.size == 0:
        return np.nan
    return float(-(probs * np.log(probs)).sum())


def normalize_scores(scores, eps=config.POSTERIOR_EPS):
    scores = np.asarray(scores, dtype=float)
    scores = np.where(np.isfinite(scores) & (scores > 0), scores, 0.0)
    total = float(scores.sum())
    if total <= eps:
        return np.zeros_like(scores)
    return scores / total


def gaze_distribution_summary(valid_gaze):
    if valid_gaze.empty:
        return {
            "gaze_mean_x": np.nan,
            "gaze_mean_y": np.nan,
            "gaze_std_x": np.nan,
            "gaze_std_y": np.nan,
        }

    return {
        "gaze_mean_x": float(valid_gaze["x"].mean()),
        "gaze_mean_y": float(valid_gaze["y"].mean()),
        "gaze_std_x": float(valid_gaze["x"].std(ddof=0)),
        "gaze_std_y": float(valid_gaze["y"].std(ddof=0)),
    }


def compute_frame_level_probabilistic_heatmap(
    gaze_df,
    yolo_df,
    gaze_sigma=config.GAZE_HEATMAP_SIGMA_PX,
    object_alpha=config.OBJECT_DISTRIBUTION_ALPHA,
    eps=config.POSTERIOR_EPS,
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
    object_rows = []

    for frame_id, frame_gaze in gaze.groupby("video_frame_idx", sort=True):
        detections = yolo_by_frame.get(int(frame_id))
        has_detection = detections is not None and len(detections) > 0
        valid_gaze = frame_gaze[np.isfinite(frame_gaze["x"]) & np.isfinite(frame_gaze["y"])]

        object_frame_rows = []
        if has_detection and not valid_gaze.empty:
            for detection_idx, (_, box) in enumerate(detections.iterrows()):
                sigma_factor, score_factor = class_penalty_factors(box["class_id"])
                object_sigma = object_alpha * bbox_diagonal(box) * sigma_factor
                object_sigma = np.clip(
                    object_sigma,
                    config.MIN_OBJECT_SIGMA_PX,
                    config.MAX_OBJECT_SIGMA_PX,
                )
                object_center_x, object_center_y = bbox_center(box)
                sample_overlap_scores = []

                for _, gaze_row in valid_gaze.iterrows():
                    distance = point_to_bbox_center_distance(gaze_row["x"], gaze_row["y"], box)
                    overlap = gaussian_overlap_density(
                        distance,
                        sigma_a=gaze_sigma,
                        sigma_b=object_sigma,
                    )
                    sample_overlap_scores.append(overlap)

                raw_overlap_score = float(np.mean(sample_overlap_scores))
                confidence_weighted_score = (
                    raw_overlap_score
                    * float(box["confidence"])
                    * score_factor
                )
                object_frame_rows.append(
                    {
                        "video_frame_idx": int(frame_id),
                        "detection_idx_in_frame": int(detection_idx),
                        "class_id": int(box["class_id"]),
                        "confidence": float(box["confidence"]),
                        "x1": float(box["x1"]),
                        "y1": float(box["y1"]),
                        "x2": float(box["x2"]),
                        "y2": float(box["y2"]),
                        "object_center_x": float(object_center_x),
                        "object_center_y": float(object_center_y),
                        "gaze_sigma": float(gaze_sigma),
                        "object_sigma": float(object_sigma),
                        "class_sigma_factor": float(sigma_factor),
                        "class_score_factor": float(score_factor),
                        "valid_gaze_sample_count": int(len(valid_gaze)),
                        "raw_overlap_score": raw_overlap_score,
                        "confidence_weighted_score": confidence_weighted_score,
                    }
                )

        if object_frame_rows:
            posterior = normalize_scores(
                [row["confidence_weighted_score"] for row in object_frame_rows],
                eps=eps,
            )
            for row, prob in zip(object_frame_rows, posterior):
                row["object_posterior_probability"] = float(prob)
            object_rows.extend(object_frame_rows)

            best_idx = int(np.argmax(posterior))
            best_row = object_frame_rows[best_idx]
            uncertainty_entropy = entropy(posterior, eps=eps)
            normalized_entropy = (
                uncertainty_entropy / math.log(len(posterior))
                if len(posterior) > 1 and np.isfinite(uncertainty_entropy)
                else 0.0
            )
            best_class = int(best_row["class_id"])
            best_detection_idx = int(best_row["detection_idx_in_frame"])
            best_probability = float(best_row["object_posterior_probability"])
            best_score = float(best_row["confidence_weighted_score"])
            bbox_width = float(best_row["x2"] - best_row["x1"])
            bbox_height = float(best_row["y2"] - best_row["y1"])
            if (
                np.isfinite(bbox_width)
                and np.isfinite(bbox_height)
                and bbox_width > 0
                and bbox_height > 0
            ):
                best_gaze_object_dx_norm = (
                    float(valid_gaze["x"].mean()) - float(best_row["object_center_x"])
                ) / bbox_width
                best_gaze_object_dy_norm = (
                    float(valid_gaze["y"].mean()) - float(best_row["object_center_y"])
                ) / bbox_height
                best_gaze_object_dist_norm = math.sqrt(
                    best_gaze_object_dx_norm ** 2 + best_gaze_object_dy_norm ** 2
                )
            else:
                best_gaze_object_dx_norm = np.nan
                best_gaze_object_dy_norm = np.nan
                best_gaze_object_dist_norm = np.nan
        else:
            uncertainty_entropy = np.nan
            normalized_entropy = np.nan
            best_class = "NO_VALID_GAZE" if valid_gaze.empty else "NO_DETECTION"
            best_detection_idx = np.nan
            best_probability = np.nan
            best_score = 0.0
            best_gaze_object_dx_norm = np.nan
            best_gaze_object_dy_norm = np.nan
            best_gaze_object_dist_norm = np.nan

        frame_rows.append(
            {
                "video_frame_idx": int(frame_id),
                "gaze_sample_count": int(len(frame_gaze)),
                "valid_gaze_sample_count": int(len(valid_gaze)),
                "yolo_detection_count": int(len(detections)) if has_detection else 0,
                "best_detection_idx_in_frame": best_detection_idx,
                "best_class": best_class,
                "best_object_probability": best_probability,
                "best_confidence_weighted_score": best_score,
                "best_gaze_object_dx_norm": best_gaze_object_dx_norm,
                "best_gaze_object_dy_norm": best_gaze_object_dy_norm,
                "best_gaze_object_dist_norm": best_gaze_object_dist_norm,
                "object_probability_entropy": uncertainty_entropy,
                "object_probability_normalized_entropy": normalized_entropy,
                **gaze_distribution_summary(valid_gaze),
            }
        )

    return pd.DataFrame(frame_rows), pd.DataFrame(object_rows)


def frame_output_path_for_record(participant_id, full_id, output_root):
    return Path(output_root) / participant_id / f"{full_id}_gaze_yolo_probabilistic_heatmap_frame.csv"


def object_output_path_for_record(participant_id, full_id, output_root):
    return Path(output_root) / participant_id / f"{full_id}_gaze_yolo_probabilistic_heatmap_frame_objects.csv"


def run_probabilistic_heatmap_frame_aggregation(
    gaze_index_path=config.GAZE_XY_FILLED_INDEX,
    yolo_root=config.YOLO_OBJECT_OUTCOME_ROOT,
    output_root=config.PROBABILISTIC_HEATMAP_FRAME_OUTPUT_ROOT,
    gaze_sigma=config.GAZE_HEATMAP_SIGMA_PX,
    object_alpha=config.OBJECT_DISTRIBUTION_ALPHA,
    eps=config.POSTERIOR_EPS,
):
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    gaze_index_df = read_gaze_index(gaze_index_path)
    yolo_lookup = build_yolo_lookup(yolo_root)

    records = list(gaze_index_df.to_dict("records"))
    iterator = records
    if tqdm is not None:
        iterator = tqdm(records, desc="Probabilistic heatmap frame aggregation", unit="recording")

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
        out_df, object_df = compute_frame_level_probabilistic_heatmap(
            gaze_df,
            yolo_df,
            gaze_sigma=gaze_sigma,
            object_alpha=object_alpha,
            eps=eps,
        )

        out_path = frame_output_path_for_record(participant_id, full_id, output_root)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(out_path, index=False)

        object_path = object_output_path_for_record(participant_id, full_id, output_root)
        object_df.to_csv(object_path, index=False)

        valid_frames = out_df[out_df["valid_gaze_sample_count"] > 0]
        probability_frames = valid_frames[pd.to_numeric(valid_frames["best_object_probability"], errors="coerce") > 0]
        index_rows.append(
            {
                "participant_id": participant_id,
                "task_name": task_name,
                "full_id": full_id,
                "gaze_csv_path": str(gaze_path),
                "yolo_csv_path": str(yolo_path),
                "output_csv_path": str(out_path),
                "object_probability_csv_path": str(object_path),
                "frame_count": int(len(out_df)),
                "object_probability_row_count": int(len(object_df)),
                "valid_gaze_frame_count": int(len(valid_frames)),
                "probability_frame_count": int(len(probability_frames)),
                "no_detection_frame_count": int((out_df["best_class"] == "NO_DETECTION").sum()),
                "no_valid_gaze_frame_count": int((out_df["best_class"] == "NO_VALID_GAZE").sum()),
                "mean_best_object_probability": float(valid_frames["best_object_probability"].mean())
                if not valid_frames.empty
                else np.nan,
                "mean_normalized_entropy": float(valid_frames["object_probability_normalized_entropy"].mean())
                if not valid_frames.empty
                else np.nan,
            }
        )

    index_df = pd.DataFrame(index_rows)
    missing_yolo_df = pd.DataFrame(missing_yolo_rows)
    index_df.to_csv(config.PROBABILISTIC_HEATMAP_FRAME_INDEX, index=False)
    missing_yolo_df.to_csv(config.PROBABILISTIC_HEATMAP_FRAME_MISSING_YOLO, index=False)

    summary = {
        "gaze_index_path": str(gaze_index_path),
        "yolo_root": str(yolo_root),
        "output_root": str(output_root),
        "index_csv": str(config.PROBABILISTIC_HEATMAP_FRAME_INDEX),
        "missing_yolo_csv": str(config.PROBABILISTIC_HEATMAP_FRAME_MISSING_YOLO),
        "processed_recording_count": int(len(index_df)),
        "missing_yolo_count": int(len(missing_yolo_df)),
        "gaze_sigma": float(gaze_sigma),
        "object_alpha": float(object_alpha),
        "min_object_sigma_px": float(config.MIN_OBJECT_SIGMA_PX),
        "max_object_sigma_px": float(config.MAX_OBJECT_SIGMA_PX),
        "penalized_class_ids": list(config.GAZE_OBJECT_PENALIZED_CLASS_IDS),
        "penalized_class_sigma_factor": float(config.PENALIZED_CLASS_SIGMA_FACTOR),
        "penalized_class_score_factor": float(config.PENALIZED_CLASS_SCORE_FACTOR),
        "policy": (
            "No gaze downsampling is performed. Gaze is modeled as a Gaussian "
            "mixture over samples in each video frame. Each YOLO object is "
            "modeled as a Gaussian centered at its bbox center with sigma = "
            "object_alpha * bbox_diagonal, clipped to configured min/max bounds, "
            "and prior weight = YOLO confidence. "
            "Penalized classes use a smaller sigma and lower prior. Frame-object "
            "posteriors are normalized over all detections in the frame."
        ),
    }
    with open(config.PROBABILISTIC_HEATMAP_FRAME_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)

    return index_df, missing_yolo_df, summary


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Build gaze-object probabilistic heatmap frame features.")
    parser.add_argument("--gaze-index-path", type=Path, default=config.GAZE_XY_FILLED_INDEX)
    parser.add_argument("--yolo-root", type=Path, default=config.YOLO_OBJECT_OUTCOME_ROOT)
    parser.add_argument("--output-root", type=Path, default=config.PROBABILISTIC_HEATMAP_FRAME_OUTPUT_ROOT)
    parser.add_argument("--gaze-sigma", type=float, default=config.GAZE_HEATMAP_SIGMA_PX)
    parser.add_argument("--object-alpha", type=float, default=config.OBJECT_DISTRIBUTION_ALPHA)
    parser.add_argument("--eps", type=float, default=config.POSTERIOR_EPS)
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    index_df, missing_yolo_df, summary = run_probabilistic_heatmap_frame_aggregation(
        gaze_index_path=args.gaze_index_path,
        yolo_root=args.yolo_root,
        output_root=args.output_root,
        gaze_sigma=args.gaze_sigma,
        object_alpha=args.object_alpha,
        eps=args.eps,
    )
    print("Probabilistic heatmap frame aggregation completed.")
    print(json.dumps(summary, indent=2))

    if not index_df.empty:
        print("\nProcessed recordings:")
        print(
            index_df[
                [
                    "full_id",
                    "frame_count",
                    "valid_gaze_frame_count",
                    "probability_frame_count",
                    "mean_best_object_probability",
                    "mean_normalized_entropy",
                ]
            ].to_string(index=False)
        )

    if not missing_yolo_df.empty:
        print("\nMissing YOLO recordings:")
        print(missing_yolo_df[["full_id", "expected_yolo_csv"]].to_string(index=False))


if __name__ == "__main__":
    main()
