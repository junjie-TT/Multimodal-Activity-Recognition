#!/usr/bin/env python3
"""Build compact hand-object interaction training features from detections."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from upper_features_extraction import config

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


OUTPUT_CSV = (
    Path("/home/julius/datasets/seminar_object_detection/input_train")
    / "hand_object_compact_train_features.csv"
)
WORLD_CAMERA_SUFFIX = "_world_camera"
REQUIRED_DETECTION_COLUMNS = ("frame_id", "confidence", "x1", "y1", "x2", "y2")
REQUIRED_OBJECT_COLUMNS = ("class_id", *REQUIRED_DETECTION_COLUMNS)
MERGE_KEY_COLUMNS = ["participant_id", "task_name", "full_id", "video_frame_idx"]
NO_VALID_HAND_OBJECT_CLASS = "NO_VALID_HAND_OBJECT"
SIDES = ("left", "right")
OBJECT_SLOTS = (1, 2)


def side_object_slot_columns(side: str, slot: int) -> list[str]:
    return [
        f"{side}_object_{slot}_class",
        f"{side}_object_{slot}_probability",
        f"{side}_object_{slot}_dx_norm",
        f"{side}_object_{slot}_dy_norm",
        f"{side}_object_{slot}_dist_norm",
    ]


def object_slot_class_dummy_columns(side: str, slot: int) -> list[str]:
    return [
        *(f"{side}_object_{slot}_class_{class_id}" for class_id in range(80)),
        f"{side}_object_{slot}_class_{NO_VALID_HAND_OBJECT_CLASS}",
    ]


CLASS_FEATURE_COLUMNS = [
    *(f"{side}_object_{slot}_class" for side in SIDES for slot in OBJECT_SLOTS),
]
CLASS_DUMMY_COLUMNS = [
    *(
        column
        for side in SIDES
        for slot in OBJECT_SLOTS
        for column in object_slot_class_dummy_columns(side, slot)
    ),
]
NUMERIC_FEATURE_COLUMNS = [
    *(
        column
        for side in SIDES
        for slot in OBJECT_SLOTS
        for column in side_object_slot_columns(side, slot)
        if not column.endswith("_class")
    ),
    *(f"{side}_object_margin" for side in SIDES),
    *(f"has_valid_{side}_hand_object" for side in SIDES),
    "has_valid_hand_object",
]
OUTPUT_FEATURE_COLUMNS = [
    *(column for side in SIDES for slot in OBJECT_SLOTS for column in side_object_slot_columns(side, slot)),
    *(f"{side}_object_margin" for side in SIDES),
    *(f"has_valid_{side}_hand_object" for side in SIDES),
    "has_valid_hand_object",
]
FRAME_FEATURE_COLUMNS = [
    "video_frame_idx",
    *OUTPUT_FEATURE_COLUMNS,
]
MAX_MATCH_DISTANCE_PX = 220.0
OBJECT_DIAG_MIN_PX = 30.0
DEFAULT_IMAGE_WIDTH = 1600.0
DEFAULT_IMAGE_HEIGHT = 1200.0
IGNORED_OBJECT_CLASS_IDS = {0, 37, 59, 60, 71, 72}
LARGE_OBJECT_AREA_RATIO_THRESHOLD = 0.5
LARGE_OBJECT_SCORE_FACTOR = 0.005
EPS = 1e-12


def parse_full_id_from_yolo_path(path: Path) -> str:
    stem = Path(path).stem
    if stem.endswith(WORLD_CAMERA_SUFFIX):
        return stem[: -len(WORLD_CAMERA_SUFFIX)]
    return stem


def participant_from_full_id(full_id: str) -> str:
    return "_".join(full_id.split("_")[:4])


def task_from_full_id(full_id: str) -> str:
    participant_id = participant_from_full_id(full_id)
    prefix = f"{participant_id}_"
    if full_id.startswith(prefix):
        return full_id[len(prefix) :]
    return full_id


def normalize_class_value(value: object) -> str:
    if pd.isna(value):
        return NO_VALID_HAND_OBJECT_CLASS
    text = str(value).strip()
    if not text:
        return NO_VALID_HAND_OBJECT_CLASS
    try:
        numeric = float(text)
    except ValueError:
        return text
    if np.isfinite(numeric) and numeric.is_integer():
        return str(int(numeric))
    return text


def normalize_hand_side(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip().lower()
    if "left" in text:
        return "left"
    if "right" in text:
        return "right"
    return "unknown"


def infer_hand_side(row: pd.Series) -> str:
    if "hand_side" in row.index:
        side = normalize_hand_side(row["hand_side"])
        if side != "unknown":
            return side

    if "class_name" in row.index:
        side = normalize_hand_side(row["class_name"])
        if side != "unknown":
            return side

    class_id = row.get("class_id")
    if pd.isna(class_id):
        return "unknown"
    try:
        class_id = int(float(class_id))
    except (TypeError, ValueError):
        return "unknown"
    if class_id == 0:
        return "left"
    if class_id == 1:
        return "right"
    return "unknown"


def require_columns(path: Path, columns: tuple[str, ...], label: str) -> None:
    header = pd.read_csv(path, nrows=0).columns
    missing = sorted(set(columns) - set(header))
    if missing:
        raise ValueError(f"{path} is missing required {label} columns: {missing}")


def numeric_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def read_detection_frame_ids(path: Path | None) -> set[int]:
    if path is None or not Path(path).exists():
        return set()
    header = pd.read_csv(path, nrows=0).columns
    if "frame_id" not in header:
        return set()
    frame_ids = pd.to_numeric(pd.read_csv(path, usecols=["frame_id"])["frame_id"], errors="coerce")
    frame_ids = frame_ids[np.isfinite(frame_ids)]
    if frame_ids.empty:
        return set()
    return set(frame_ids.round().astype(int))


def infer_image_area_from_detections(df: pd.DataFrame, image_width: float | None, image_height: float | None) -> float:
    if image_width is not None and image_height is not None and image_width > 0 and image_height > 0:
        return float(image_width) * float(image_height)
    max_x = pd.to_numeric(df.get("x2", pd.Series(dtype=float)), errors="coerce").max()
    max_y = pd.to_numeric(df.get("y2", pd.Series(dtype=float)), errors="coerce").max()
    if pd.isna(max_x) or pd.isna(max_y) or max_x <= 0 or max_y <= 0:
        return 0.0
    return float(max_x) * float(max_y)


def read_object_detections(
    path: Path,
    image_width: float | None = None,
    image_height: float | None = None,
) -> pd.DataFrame:
    require_columns(path, REQUIRED_OBJECT_COLUMNS, "object detection")
    df = pd.read_csv(path)
    df = numeric_columns(df, REQUIRED_OBJECT_COLUMNS)
    df = df.dropna(subset=list(REQUIRED_OBJECT_COLUMNS)).copy()
    if df.empty:
        return pd.DataFrame(columns=["frame_id", "object_class_id"])

    image_area = infer_image_area_from_detections(df, image_width, image_height)
    df["frame_id"] = df["frame_id"].round().astype(int)
    df["object_class_id"] = df["class_id"].round().astype(int)
    df = df[~df["object_class_id"].isin(IGNORED_OBJECT_CLASS_IDS)].copy()
    df["object_confidence"] = df["confidence"].astype(float)
    df["object_x1"] = df["x1"].astype(float)
    df["object_y1"] = df["y1"].astype(float)
    df["object_x2"] = df["x2"].astype(float)
    df["object_y2"] = df["y2"].astype(float)
    df["object_width"] = df["object_x2"] - df["object_x1"]
    df["object_height"] = df["object_y2"] - df["object_y1"]
    df = df[(df["object_width"] > 0.0) & (df["object_height"] > 0.0)].copy()
    df["object_area"] = df["object_width"] * df["object_height"]
    if image_area > 0:
        df["large_object_score_factor"] = np.where(
            df["object_area"] > image_area * LARGE_OBJECT_AREA_RATIO_THRESHOLD,
            LARGE_OBJECT_SCORE_FACTOR,
            1.0,
        )
    else:
        df["large_object_score_factor"] = 1.0
    df["object_cx"] = (df["object_x1"] + df["object_x2"]) / 2.0
    df["object_cy"] = (df["object_y1"] + df["object_y2"]) / 2.0
    return df[
        [
            "frame_id",
            "object_class_id",
            "object_confidence",
            "object_x1",
            "object_y1",
            "object_x2",
            "object_y2",
            "object_width",
            "object_height",
            "large_object_score_factor",
            "object_cx",
            "object_cy",
        ]
    ]


def read_hand_detections(path: Path) -> pd.DataFrame:
    require_columns(path, REQUIRED_DETECTION_COLUMNS, "hand detection")
    df = pd.read_csv(path)
    numeric_source_columns = tuple(
        column for column in ("class_id", *REQUIRED_DETECTION_COLUMNS) if column in df.columns
    )
    df = numeric_columns(df, numeric_source_columns)
    df = df.dropna(subset=list(REQUIRED_DETECTION_COLUMNS)).copy()
    if df.empty:
        return pd.DataFrame(columns=["frame_id", "hand_side"])

    df["frame_id"] = df["frame_id"].round().astype(int)
    df["hand_confidence"] = df["confidence"].astype(float)
    df["hand_side"] = df.apply(infer_hand_side, axis=1)
    df["hand_x1"] = df["x1"].astype(float)
    df["hand_y1"] = df["y1"].astype(float)
    df["hand_x2"] = df["x2"].astype(float)
    df["hand_y2"] = df["y2"].astype(float)
    df["hand_width"] = df["hand_x2"] - df["hand_x1"]
    df["hand_height"] = df["hand_y2"] - df["hand_y1"]
    df = df[(df["hand_width"] > 0.0) & (df["hand_height"] > 0.0)].copy()
    df["hand_cx"] = (df["hand_x1"] + df["hand_x2"]) / 2.0
    df["hand_cy"] = (df["hand_y1"] + df["hand_y2"]) / 2.0
    return df[
        [
            "frame_id",
            "hand_side",
            "hand_confidence",
            "hand_x1",
            "hand_y1",
            "hand_x2",
            "hand_y2",
            "hand_cx",
            "hand_cy",
        ]
    ]


def empty_frame_row(frame_id: int) -> dict:
    row = {"video_frame_idx": int(frame_id), "has_valid_hand_object": 0}
    for side in SIDES:
        for slot in OBJECT_SLOTS:
            row[f"{side}_object_{slot}_class"] = NO_VALID_HAND_OBJECT_CLASS
            row[f"{side}_object_{slot}_probability"] = 0.0
            row[f"{side}_object_{slot}_dx_norm"] = 0.0
            row[f"{side}_object_{slot}_dy_norm"] = 0.0
            row[f"{side}_object_{slot}_dist_norm"] = 0.0
        row[f"{side}_object_margin"] = 0.0
        row[f"has_valid_{side}_hand_object"] = 0
    return row


def empty_frame_features() -> pd.DataFrame:
    return pd.DataFrame(columns=FRAME_FEATURE_COLUMNS)


def empty_output_features() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            *MERGE_KEY_COLUMNS,
            *CLASS_DUMMY_COLUMNS,
            *NUMERIC_FEATURE_COLUMNS,
        ]
    )


def side_object_slot_features(ranked: pd.DataFrame, side: str) -> pd.DataFrame:
    side_ranked = ranked[ranked["hand_side"].eq(side)].copy()
    if side_ranked.empty:
        return pd.DataFrame(columns=["video_frame_idx"])

    out = pd.DataFrame({"video_frame_idx": sorted(side_ranked["frame_id"].astype(int).unique())})
    for slot in OBJECT_SLOTS:
        slot_df = side_ranked[side_ranked["object_rank"].eq(slot)].copy()
        if slot_df.empty:
            continue
        slot_df["video_frame_idx"] = slot_df["frame_id"].astype(int)
        slot_df[f"{side}_object_{slot}_class"] = slot_df["object_class_id"].map(normalize_class_value)
        slot_df[f"{side}_object_{slot}_probability"] = slot_df["posterior"].astype(float)
        slot_df[f"{side}_object_{slot}_dx_norm"] = slot_df["hand_object_dx_px"] / MAX_MATCH_DISTANCE_PX
        slot_df[f"{side}_object_{slot}_dy_norm"] = slot_df["hand_object_dy_px"] / MAX_MATCH_DISTANCE_PX
        slot_df[f"{side}_object_{slot}_dist_norm"] = slot_df["distance_norm"]
        out = out.merge(
            slot_df[
                [
                    "video_frame_idx",
                    f"{side}_object_{slot}_class",
                    f"{side}_object_{slot}_probability",
                    f"{side}_object_{slot}_dx_norm",
                    f"{side}_object_{slot}_dy_norm",
                    f"{side}_object_{slot}_dist_norm",
                ]
            ],
            on="video_frame_idx",
            how="left",
            sort=True,
        )

    out[f"has_valid_{side}_hand_object"] = 1
    return out


def fill_missing_side_features(features: pd.DataFrame) -> pd.DataFrame:
    for side in SIDES:
        valid_col = f"has_valid_{side}_hand_object"

        for slot in OBJECT_SLOTS:
            class_col = f"{side}_object_{slot}_class"
            numeric_cols = [
                f"{side}_object_{slot}_probability",
                f"{side}_object_{slot}_dx_norm",
                f"{side}_object_{slot}_dy_norm",
                f"{side}_object_{slot}_dist_norm",
            ]

            if class_col not in features.columns:
                features[class_col] = NO_VALID_HAND_OBJECT_CLASS
            features[class_col] = features[class_col].fillna(NO_VALID_HAND_OBJECT_CLASS)
            for col in numeric_cols:
                if col not in features.columns:
                    features[col] = 0.0
                features[col] = pd.to_numeric(features[col], errors="coerce").fillna(0.0)

        margin_col = f"{side}_object_margin"
        features[margin_col] = (
            pd.to_numeric(features[f"{side}_object_1_probability"], errors="coerce").fillna(0.0)
            - pd.to_numeric(features[f"{side}_object_2_probability"], errors="coerce").fillna(0.0)
        )
        if valid_col not in features.columns:
            features[valid_col] = 0
        features[valid_col] = pd.to_numeric(features[valid_col], errors="coerce").fillna(0).astype("int8")

    features["has_valid_hand_object"] = (
        pd.to_numeric(features["has_valid_left_hand_object"], errors="coerce").fillna(0).gt(0)
        | pd.to_numeric(features["has_valid_right_hand_object"], errors="coerce").fillna(0).gt(0)
    ).astype("int8")
    return features


def compact_features_for_record(
    object_path: Path,
    hand_path: Path | None,
    full_id: str,
    image_width: float | None = None,
    image_height: float | None = None,
) -> pd.DataFrame:
    object_detected_frame_ids = read_detection_frame_ids(object_path)
    hand_detected_frame_ids = read_detection_frame_ids(hand_path)
    objects = read_object_detections(object_path, image_width=image_width, image_height=image_height)
    hands = read_hand_detections(hand_path) if hand_path is not None and hand_path.exists() else pd.DataFrame()

    detected_frame_ids = object_detected_frame_ids | hand_detected_frame_ids
    max_frame_id = max(detected_frame_ids) if detected_frame_ids else None
    frame_ids = list(range(max_frame_id + 1)) if max_frame_id is not None and max_frame_id >= 0 else []
    if not frame_ids:
        frame_df = pd.DataFrame(columns=["video_frame_idx"])
    else:
        frame_df = pd.DataFrame({"video_frame_idx": frame_ids})

    if objects.empty or hands.empty:
        rows = [empty_frame_row(frame_id) for frame_id in frame_ids]
        features = pd.DataFrame(rows, columns=FRAME_FEATURE_COLUMNS)
    else:
        hands_for_sides = hands[hands["hand_side"].isin(SIDES)].copy()
        pairs = hands_for_sides.merge(objects, on="frame_id", how="inner", sort=False)
        if pairs.empty:
            rows = [empty_frame_row(frame_id) for frame_id in frame_ids]
            features = pd.DataFrame(rows, columns=FRAME_FEATURE_COLUMNS)
        else:
            closest_x = np.minimum(
                np.maximum(pairs["hand_cx"].to_numpy(dtype=float), pairs["object_x1"].to_numpy(dtype=float)),
                pairs["object_x2"].to_numpy(dtype=float),
            )
            closest_y = np.minimum(
                np.maximum(pairs["hand_cy"].to_numpy(dtype=float), pairs["object_y1"].to_numpy(dtype=float)),
                pairs["object_y2"].to_numpy(dtype=float),
            )
            pairs["hand_object_dx_px"] = pairs["hand_cx"].to_numpy(dtype=float) - closest_x
            pairs["hand_object_dy_px"] = pairs["hand_cy"].to_numpy(dtype=float) - closest_y
            pairs["hand_object_distance_px"] = np.hypot(
                pairs["hand_object_dx_px"],
                pairs["hand_object_dy_px"],
            )
            pairs = pairs[pairs["hand_object_distance_px"] <= MAX_MATCH_DISTANCE_PX].copy()
            object_diag_safe = np.maximum(
                np.hypot(
                    pairs["object_width"].to_numpy(dtype=float),
                    pairs["object_height"].to_numpy(dtype=float),
                ),
                OBJECT_DIAG_MIN_PX,
            )
            pairs["distance_norm"] = pairs["hand_object_distance_px"] / object_diag_safe
            pairs["interaction_score"] = (
                pairs["large_object_score_factor"] * np.exp(-pairs["distance_norm"])
            )
            pairs["score_sum"] = pairs.groupby(["frame_id", "hand_side"])["interaction_score"].transform("sum")
            pairs = pairs[pairs["score_sum"] > EPS].copy()

            if pairs.empty:
                rows = [empty_frame_row(frame_id) for frame_id in frame_ids]
                features = pd.DataFrame(rows, columns=FRAME_FEATURE_COLUMNS)
            else:
                pairs["posterior"] = pairs["interaction_score"] / pairs["score_sum"]
                ranked = pairs.sort_values(
                    ["frame_id", "hand_side", "posterior", "hand_object_distance_px"],
                    ascending=[True, True, False, True],
                ).copy()
                ranked["object_rank"] = ranked.groupby(["frame_id", "hand_side"]).cumcount() + 1
                ranked = ranked[ranked["object_rank"].isin(OBJECT_SLOTS)].copy()
                features = frame_df.copy()
                for side in SIDES:
                    features = features.merge(
                        side_object_slot_features(ranked, side),
                        on="video_frame_idx",
                        how="left",
                        sort=True,
                    )
                features = fill_missing_side_features(features)

    if features.empty and frame_df.empty:
        features = empty_frame_features()

    features.insert(0, "full_id", full_id)
    features.insert(0, "task_name", task_from_full_id(full_id))
    features.insert(0, "participant_id", participant_from_full_id(full_id))
    return features


def iter_object_paths(
    object_root: Path,
    participant: str | None = None,
    full_id: list[str] | None = None,
    limit: int | None = None,
) -> list[Path]:
    paths = sorted(Path(object_root).rglob(f"*{WORLD_CAMERA_SUFFIX}.csv"))
    if participant:
        paths = [
            path
            for path in paths
            if parse_full_id_from_yolo_path(path).startswith(f"{participant}_")
        ]
    if full_id:
        wanted = set(full_id)
        paths = [path for path in paths if parse_full_id_from_yolo_path(path) in wanted]
    if limit is not None:
        paths = paths[: max(0, limit)]
    return paths


def build_hand_path_lookup(hand_root: Path) -> dict[str, Path]:
    return {
        parse_full_id_from_yolo_path(path): path
        for path in sorted(Path(hand_root).rglob(f"*{WORLD_CAMERA_SUFFIX}.csv"))
    }


def build_compact_train_features(
    object_root: Path = config.YOLO_OBJECT_OUTCOME_ROOT,
    hand_root: Path = config.YOLO_LEFT_RIGHT_HAND_OUTCOME_ROOT,
    participant: str | None = None,
    full_id: list[str] | None = None,
    limit: int | None = None,
    image_width: float | None = DEFAULT_IMAGE_WIDTH,
    image_height: float | None = DEFAULT_IMAGE_HEIGHT,
) -> pd.DataFrame:
    object_paths = iter_object_paths(object_root, participant=participant, full_id=full_id, limit=limit)
    hand_lookup = build_hand_path_lookup(hand_root)
    iterator = object_paths
    if tqdm is not None:
        iterator = tqdm(object_paths, desc="Compact hand-object train features", unit="recording")

    frames = []
    for object_path in iterator:
        current_full_id = parse_full_id_from_yolo_path(object_path)
        hand_path = hand_lookup.get(current_full_id)
        frames.append(
            compact_features_for_record(
                object_path,
                hand_path,
                current_full_id,
                image_width=image_width,
                image_height=image_height,
            )
        )

    if not frames:
        return empty_output_features()

    features = pd.concat(frames, ignore_index=True, sort=False)
    class_dummy_frames = []
    for side in SIDES:
        for slot in OBJECT_SLOTS:
            class_col = f"{side}_object_{slot}_class"
            probability_col = f"{side}_object_{slot}_probability"

            features[class_col] = features[class_col].map(normalize_class_value)
            probabilities = pd.to_numeric(features[probability_col], errors="coerce").fillna(0.0)
            features.loc[probabilities.le(0.0), class_col] = NO_VALID_HAND_OBJECT_CLASS
            dummies = pd.get_dummies(features[class_col], prefix=class_col)
            dummies = dummies.reindex(
                columns=object_slot_class_dummy_columns(side, slot),
                fill_value=0,
            ).astype("int8")
            class_dummy_frames.append(dummies)

    class_dummies = pd.concat(class_dummy_frames, axis=1)
    for column in NUMERIC_FEATURE_COLUMNS:
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(0.0)
    features["has_valid_hand_object"] = features["has_valid_hand_object"].astype("int8")

    return pd.concat(
        [
            features[MERGE_KEY_COLUMNS],
            class_dummies,
            features[NUMERIC_FEATURE_COLUMNS],
        ],
        axis=1,
    )


def write_compact_train_features(
    object_root: Path = config.YOLO_OBJECT_OUTCOME_ROOT,
    hand_root: Path = config.YOLO_LEFT_RIGHT_HAND_OUTCOME_ROOT,
    output_csv: Path = OUTPUT_CSV,
    participant: str | None = None,
    full_id: list[str] | None = None,
    limit: int | None = None,
    image_width: float | None = DEFAULT_IMAGE_WIDTH,
    image_height: float | None = DEFAULT_IMAGE_HEIGHT,
) -> tuple[pd.DataFrame, dict]:
    features = build_compact_train_features(
        object_root=object_root,
        hand_root=hand_root,
        participant=participant,
        full_id=full_id,
        limit=limit,
        image_width=image_width,
        image_height=image_height,
    )
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(output_csv, index=False)

    summary = {
        "object_root": str(object_root),
        "hand_root": str(hand_root),
        "output_csv": str(output_csv),
        "row_count": int(len(features)),
        "column_count": int(len(features.columns)),
        "class_column_count": len(CLASS_FEATURE_COLUMNS),
        "one_hot_class_column_count": len(CLASS_DUMMY_COLUMNS),
        "ignored_object_class_ids": sorted(IGNORED_OBJECT_CLASS_IDS),
        "large_object_area_ratio_threshold": LARGE_OBJECT_AREA_RATIO_THRESHOLD,
        "large_object_score_factor": LARGE_OBJECT_SCORE_FACTOR,
        "max_match_distance_px": MAX_MATCH_DISTANCE_PX,
        "object_diag_min_px": OBJECT_DIAG_MIN_PX,
        "image_width": image_width,
        "image_height": image_height,
    }
    return features, summary


def build_processed_recordings_summary(features: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "full_id",
        "frame_count",
        "valid_hand_object_frame_count",
        "valid_left_hand_object_frame_count",
        "valid_right_hand_object_frame_count",
        "mean_left_object_1_probability",
        "mean_right_object_1_probability",
        "mean_left_object_margin",
        "mean_right_object_margin",
    ]
    if features.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for full_id, frame_df in features.groupby("full_id", sort=True):
        valid = pd.to_numeric(
            frame_df["has_valid_hand_object"],
            errors="coerce",
        ).fillna(0)
        left_valid = pd.to_numeric(
            frame_df["has_valid_left_hand_object"],
            errors="coerce",
        ).fillna(0)
        right_valid = pd.to_numeric(
            frame_df["has_valid_right_hand_object"],
            errors="coerce",
        ).fillna(0)
        left_probability = pd.to_numeric(
            frame_df["left_object_1_probability"],
            errors="coerce",
        ).fillna(0.0)
        right_probability = pd.to_numeric(
            frame_df["right_object_1_probability"],
            errors="coerce",
        ).fillna(0.0)
        left_margin = pd.to_numeric(
            frame_df["left_object_margin"],
            errors="coerce",
        ).fillna(0.0)
        right_margin = pd.to_numeric(
            frame_df["right_object_margin"],
            errors="coerce",
        ).fillna(0.0)
        valid_left_probability = left_probability[left_valid.gt(0)]
        valid_right_probability = right_probability[right_valid.gt(0)]
        valid_left_margin = left_margin[left_valid.gt(0)]
        valid_right_margin = right_margin[right_valid.gt(0)]
        rows.append(
            {
                "full_id": full_id,
                "frame_count": int(len(frame_df)),
                "valid_hand_object_frame_count": int(valid.gt(0).sum()),
                "valid_left_hand_object_frame_count": int(left_valid.gt(0).sum()),
                "valid_right_hand_object_frame_count": int(right_valid.gt(0).sum()),
                "mean_left_object_1_probability": (
                    float(valid_left_probability.mean()) if not valid_left_probability.empty else 0.0
                ),
                "mean_right_object_1_probability": (
                    float(valid_right_probability.mean()) if not valid_right_probability.empty else 0.0
                ),
                "mean_left_object_margin": (
                    float(valid_left_margin.mean()) if not valid_left_margin.empty else 0.0
                ),
                "mean_right_object_margin": (
                    float(valid_right_margin.mean()) if not valid_right_margin.empty else 0.0
                ),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build compact hand-object interaction training features from detections."
    )
    parser.add_argument("--object-root", type=Path, default=config.YOLO_OBJECT_OUTCOME_ROOT)
    parser.add_argument("--hand-root", type=Path, default=config.YOLO_LEFT_RIGHT_HAND_OUTCOME_ROOT)
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--participant")
    parser.add_argument("--full-id", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--image-width", type=float, default=DEFAULT_IMAGE_WIDTH)
    parser.add_argument("--image-height", type=float, default=DEFAULT_IMAGE_HEIGHT)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_arg_parser().parse_args(argv)
    features, summary = write_compact_train_features(
        object_root=args.object_root,
        hand_root=args.hand_root,
        output_csv=args.output_csv,
        participant=args.participant,
        full_id=args.full_id,
        limit=args.limit,
        image_width=args.image_width,
        image_height=args.image_height,
    )

    print("Compact hand-object training features completed.")
    print(json.dumps(summary, indent=2))
    processed_recordings = build_processed_recordings_summary(features)
    if not processed_recordings.empty:
        print("\nProcessed recordings:")
        print(processed_recordings.to_string(index=False))
    if not features.empty:
        print("\nOutput columns:")
        print(", ".join(features.columns))


if __name__ == "__main__":
    main()
