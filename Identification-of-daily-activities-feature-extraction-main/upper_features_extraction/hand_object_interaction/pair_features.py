import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from upper_features_extraction import config

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


EPS = 1e-9
SIDES = ("left", "right")
PHASE_NONE = 0
PHASE_APPROACH = 1
PHASE_CONTACT = 2
PHASE_MANIPULATE = 3
PHASE_RELEASE = 4

DEFAULT_OBJECT_EXCLUDE_CLASS_IDS = {0, 60}
CELL_PHONE_CLASS_ID = 67
DEFAULT_LARGE_OBJECT_SOFT_PENALTY_START_RATIO = 0.20
DEFAULT_LARGE_OBJECT_SOFT_PENALTY_FULL_RATIO = 0.70
DEFAULT_LARGE_OBJECT_SCORE_WEIGHT = 0.05
LARGE_OBJECT_SOFT_PENALTY_START_RATIO = DEFAULT_LARGE_OBJECT_SOFT_PENALTY_START_RATIO
LARGE_OBJECT_SOFT_PENALTY_FULL_RATIO = DEFAULT_LARGE_OBJECT_SOFT_PENALTY_FULL_RATIO
LARGE_OBJECT_SCORE_WEIGHT = DEFAULT_LARGE_OBJECT_SCORE_WEIGHT
CELL_PHONE_SCORE_PENALTY = 0.15
DEFAULT_TOP_K_OBJECTS = 3
DEFAULT_HOI_SCORE_CLASSES = (
    "cup",
    "bottle",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "book",
    "cell phone",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "scissors",
    "pen",
    "paper",
    "envelope",
    "sponge",
    "cloth",
)
SMALL_OBJECT_PRIORITY_CLASSES = {
    "pen",
    "paper",
    "fork",
    "knife",
    "spoon",
    "bar",
    "key",
    "wallet",
    "mouse",
    "remote",
    "scissors",
}
SMALL_OBJECT_PRIORITY_BONUS = 0.14
SMALL_OBJECT_PRIORITY_GAP_NORM_THRESHOLD = 0.45
SMALL_OBJECT_PRIORITY_DISTANCE_SIZE_THRESHOLD = 1.65
SMALL_OBJECT_PRIORITY_MAX_AREA_RATIO = 0.08
COCO_CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    8: "boat",
    9: "traffic light",
    10: "fire hydrant",
    11: "stop sign",
    12: "parking meter",
    13: "bench",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
    24: "backpack",
    25: "umbrella",
    26: "handbag",
    27: "tie",
    28: "suitcase",
    29: "frisbee",
    30: "skis",
    31: "snowboard",
    32: "sports ball",
    33: "kite",
    34: "baseball bat",
    35: "baseball glove",
    36: "skateboard",
    37: "surfboard",
    38: "tennis racket",
    39: "bottle",
    40: "wine glass",
    41: "cup",
    42: "fork",
    43: "knife",
    44: "spoon",
    45: "bowl",
    46: "banana",
    47: "apple",
    48: "sandwich",
    49: "orange",
    50: "broccoli",
    51: "carrot",
    52: "hot dog",
    53: "pizza",
    54: "donut",
    55: "cake",
    56: "chair",
    57: "couch",
    58: "potted plant",
    59: "bed",
    60: "dining table",
    61: "toilet",
    62: "tv",
    63: "laptop",
    64: "mouse",
    65: "remote",
    66: "keyboard",
    67: "cell phone",
    68: "microwave",
    69: "oven",
    70: "toaster",
    71: "sink",
    72: "refrigerator",
    73: "book",
    74: "clock",
    75: "vase",
    76: "scissors",
    77: "teddy bear",
    78: "hair drier",
    79: "toothbrush",
}

SKELETON_COLUMNS = [
    "frame_idx",
    "time_s",
    "left_wrist_speed",
    "left_hand_center_speed",
    "left_elbow_angle",
    "left_wrist_angle_velocity",
    "left_hand_motion_energy_0p35s",
    "right_wrist_speed",
    "right_hand_center_speed",
    "right_elbow_angle",
    "right_wrist_angle_velocity",
    "right_hand_motion_energy_0p35s",
    "left_right_hand_center_dist",
    "both_hands_active",
]

IMU_COLUMNS = [
    "frame_idx",
    "time_s",
    "left_hand_abs_gyro_mag",
    "left_wrist_angle_velocity_deg_s",
    "left_wrist_relative_gyro_mag",
    "right_hand_abs_gyro_mag",
    "right_wrist_angle_velocity_deg_s",
    "right_wrist_relative_gyro_mag",
]


def parse_full_id_from_yolo_path(path):
    stem = Path(path).stem
    suffix = "_world_camera"
    return stem[: -len(suffix)] if stem.endswith(suffix) else stem


def participant_from_full_id(full_id):
    return "_".join(full_id.split("_")[:4])


def task_from_full_id(full_id):
    participant = participant_from_full_id(full_id)
    prefix = f"{participant}_"
    return full_id[len(prefix) :] if full_id.startswith(prefix) else full_id


def numeric_columns(df, columns):
    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def object_class_name(class_id):
    class_id = int(class_id)
    return COCO_CLASS_NAMES.get(class_id, f"class_{class_id}")


def normalize_object_class_name(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower().replace("_", " ")


def read_object_detections(path, exclude_class_ids=DEFAULT_OBJECT_EXCLUDE_CLASS_IDS):
    df = pd.read_csv(path)
    df = numeric_columns(
        df,
        ["frame_id", "class_id", "confidence", "x1", "y1", "x2", "y2", "cx", "cy", "width", "height", "area"],
    )
    required = {"frame_id", "class_id", "confidence", "x1", "y1", "x2", "y2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing object YOLO columns: {sorted(missing)}")

    df = df.dropna(subset=["frame_id", "class_id", "confidence", "x1", "y1", "x2", "y2"]).copy()
    if exclude_class_ids:
        df = df[~df["class_id"].astype(int).isin(exclude_class_ids)].copy()

    df["frame_id"] = df["frame_id"].round().astype(int)
    df["class_id"] = df["class_id"].round().astype(int)
    df["object_class_id"] = df["class_id"]
    if "class_name" in df.columns:
        class_names = df["class_name"].map(normalize_object_class_name)
        df["object_class"] = class_names.where(class_names.ne(""), df["class_id"].map(object_class_name))
    else:
        df["object_class"] = df["class_id"].map(object_class_name)
    if "cx" not in df.columns:
        df["cx"] = (df["x1"] + df["x2"]) / 2.0
    if "cy" not in df.columns:
        df["cy"] = (df["y1"] + df["y2"]) / 2.0
    df["width"] = (df["x2"] - df["x1"]).clip(lower=0.0)
    df["height"] = (df["y2"] - df["y1"]).clip(lower=0.0)
    df["area"] = df["width"] * df["height"]
    df["detection_idx_in_frame"] = df.groupby("frame_id").cumcount()
    return assign_object_tracks(df)


def hand_side_from_row(row):
    name = str(row.get("class_name", "")).lower()
    if "left" in name:
        return "left"
    if "right" in name:
        return "right"
    class_id = row.get("class_id")
    if pd.notna(class_id):
        return "left" if int(class_id) == 0 else "right"
    return np.nan


def read_hand_detections(path):
    df = pd.read_csv(path)
    df = numeric_columns(df, ["frame_id", "class_id", "confidence", "x1", "y1", "x2", "y2"])
    required = {"frame_id", "confidence", "x1", "y1", "x2", "y2"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing left/right hand YOLO columns: {sorted(missing)}")

    df = df.dropna(subset=["frame_id", "confidence", "x1", "y1", "x2", "y2"]).copy()
    df["frame_id"] = df["frame_id"].round().astype(int)
    df["hand_side"] = df.apply(hand_side_from_row, axis=1)
    df = df[df["hand_side"].isin(SIDES)].copy()
    df["hand_cx"] = (df["x1"] + df["x2"]) / 2.0
    df["hand_cy"] = (df["y1"] + df["y2"]) / 2.0
    df["hand_width"] = (df["x2"] - df["x1"]).clip(lower=0.0)
    df["hand_height"] = (df["y2"] - df["y1"]).clip(lower=0.0)
    df["hand_area"] = df["hand_width"] * df["hand_height"]
    df = df.sort_values(["frame_id", "hand_side", "confidence"], ascending=[True, True, False])
    df = df.drop_duplicates(["frame_id", "hand_side"], keep="first").copy()
    return df.rename(
        columns={
            "confidence": "hand_conf",
            "x1": "hand_x1",
            "y1": "hand_y1",
            "x2": "hand_x2",
            "y2": "hand_y2",
        }
    )


def bbox_iou(a, b):
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return inter / denom if denom > EPS else 0.0


def bbox_gap(a, b):
    dx = max(b[0] - a[2], a[0] - b[2], 0.0)
    dy = max(b[1] - a[3], a[1] - b[3], 0.0)
    return float(np.hypot(dx, dy))


def point_in_bbox(x, y, box):
    return int(box[0] <= x <= box[2] and box[1] <= y <= box[3])


def expand_bbox(box, ratio):
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    return (
        x1 - width * ratio,
        y1 - height * ratio,
        x2 + width * ratio,
        y2 + height * ratio,
    )


def assign_object_tracks(df, iou_threshold=0.2, max_center_shift_factor=1.5):
    rows = []
    next_track_id = 0
    active = {}

    for frame_id, frame_df in df.sort_values(["frame_id", "class_id", "confidence"]).groupby("frame_id", sort=True):
        current = {}
        for _, row in frame_df.iterrows():
            cls = int(row["class_id"])
            box = (row["x1"], row["y1"], row["x2"], row["y2"])
            center = np.array([row["cx"], row["cy"]], dtype=float)
            diag = float(np.hypot(row["width"], row["height"]))
            best_key = None
            best_score = -np.inf

            for key, prev in active.items():
                if key[0] != cls or key in current:
                    continue
                prev_box = prev["box"]
                prev_center = prev["center"]
                iou = bbox_iou(box, prev_box)
                center_shift = float(np.linalg.norm(center - prev_center))
                max_shift = max(diag, prev["diag"], 1.0) * max_center_shift_factor
                if iou < iou_threshold and center_shift > max_shift:
                    continue
                score = iou - 0.01 * center_shift / max(max_shift, EPS)
                if score > best_score:
                    best_key = key
                    best_score = score

            if best_key is None:
                track_id = next_track_id
                next_track_id += 1
                best_key = (cls, track_id)
            else:
                track_id = best_key[1]

            current[best_key] = {"box": box, "center": center, "diag": diag, "last_frame": frame_id}
            out = row.copy()
            out["object_id"] = int(track_id)
            rows.append(out)

        active = {
            key: value
            for key, value in {**active, **current}.items()
            if frame_id - value["last_frame"] <= 2
        }

    return pd.DataFrame(rows).reset_index(drop=True)


def infer_image_size(objects, hands, image_width=None, image_height=None):
    if image_width and image_height:
        return float(image_width), float(image_height)
    max_x = max(objects["x2"].max() if len(objects) else 0, hands["hand_x2"].max() if len(hands) else 0)
    max_y = max(objects["y2"].max() if len(objects) else 0, hands["hand_y2"].max() if len(hands) else 0)
    return float(image_width or max_x), float(image_height or max_y)


def build_pair_rows(objects, hands, fps, image_width, image_height, expand_ratio):
    if objects.empty or hands.empty:
        return pd.DataFrame()

    rows = []
    image_diagonal = float(np.hypot(image_width, image_height))
    image_area = max(float(image_width * image_height), EPS)
    object_groups = {frame_id: frame for frame_id, frame in objects.groupby("frame_id")}

    for _, hand in hands.sort_values(["frame_id", "hand_side"]).iterrows():
        frame_objects = object_groups.get(int(hand["frame_id"]))
        if frame_objects is None or frame_objects.empty:
            continue

        hand_box = (hand["hand_x1"], hand["hand_y1"], hand["hand_x2"], hand["hand_y2"])
        expanded_hand_box = expand_bbox(hand_box, expand_ratio)
        hand_center = np.array([hand["hand_cx"], hand["hand_cy"]], dtype=float)

        for _, obj in frame_objects.iterrows():
            object_box = (obj["x1"], obj["y1"], obj["x2"], obj["y2"])
            object_center = np.array([obj["cx"], obj["cy"]], dtype=float)
            center_distance = float(np.linalg.norm(hand_center - object_center))
            object_size = float(np.sqrt(max(obj["area"], EPS)))
            gap = bbox_gap(hand_box, object_box)
            iou = bbox_iou(hand_box, object_box)
            expanded_iou = bbox_iou(expanded_hand_box, object_box)
            object_area_ratio = float(obj["area"] / image_area)

            rows.append(
                {
                    "frame_id": int(hand["frame_id"]),
                    "timestamp": int(hand["frame_id"]) / fps,
                    "hand_side": hand["hand_side"],
                    "object_class_id": int(obj["object_class_id"]),
                    "object_class": obj["object_class"],
                    "object_id": int(obj["object_id"]),
                    "object_detection_idx_in_frame": int(obj["detection_idx_in_frame"]),
                    "object_conf": float(obj["confidence"]),
                    "hand_conf": float(hand["hand_conf"]),
                    "hand_x1": hand["hand_x1"],
                    "hand_y1": hand["hand_y1"],
                    "hand_x2": hand["hand_x2"],
                    "hand_y2": hand["hand_y2"],
                    "object_x1": obj["x1"],
                    "object_y1": obj["y1"],
                    "object_x2": obj["x2"],
                    "object_y2": obj["y2"],
                    "hand_cx": hand["hand_cx"],
                    "hand_cy": hand["hand_cy"],
                    "object_cx": obj["cx"],
                    "object_cy": obj["cy"],
                    "object_area": obj["area"],
                    "object_area_ratio": object_area_ratio,
                    "center_distance": center_distance,
                    "center_distance_norm": center_distance / max(image_diagonal, EPS),
                    "distance_to_object_size": center_distance / object_size,
                    "bbox_iou": iou,
                    "bbox_gap": gap,
                    "bbox_gap_norm": gap / object_size,
                    "expanded_iou": expanded_iou,
                    "hand_center_in_object_bbox": point_in_bbox(hand["hand_cx"], hand["hand_cy"], object_box),
                    "object_center_in_hand_bbox": point_in_bbox(obj["cx"], obj["cy"], hand_box),
                }
            )

    return pd.DataFrame(rows)


def cosine_similarity_2d(a, b):
    a_norm = np.linalg.norm(a, axis=1)
    b_norm = np.linalg.norm(b, axis=1)
    denom = a_norm * b_norm
    out = np.full(len(a), np.nan, dtype=float)
    valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1) & (denom > EPS)
    out[valid] = np.sum(a[valid] * b[valid], axis=1) / denom[valid]
    return out


def add_temporal_features(pairs, fps, contact_gap_threshold, contact_iou_threshold):
    if pairs.empty:
        return pairs

    pairs = pairs.sort_values(["hand_side", "object_class", "object_id", "frame_id"]).copy()
    group_cols = ["hand_side", "object_class", "object_id"]
    grouped = pairs.groupby(group_cols, sort=False)

    for col in ["center_distance", "hand_cx", "hand_cy", "object_cx", "object_cy"]:
        pairs[f"prev_{col}"] = grouped[col].shift(1)

    pairs["prev_frame_id"] = grouped["frame_id"].shift(1)
    dt = (pairs["frame_id"] - pairs["prev_frame_id"]) / fps
    valid_dt = dt.where(dt > EPS)

    pairs["delta_distance"] = pairs["center_distance"] - pairs["prev_center_distance"]
    pairs["approach_speed"] = -pairs["delta_distance"] / valid_dt
    pairs["separation_speed"] = pairs["delta_distance"] / valid_dt

    hand_velocity = np.column_stack(
        [
            (pairs["hand_cx"] - pairs["prev_hand_cx"]) / valid_dt,
            (pairs["hand_cy"] - pairs["prev_hand_cy"]) / valid_dt,
        ]
    )
    object_velocity = np.column_stack(
        [
            (pairs["object_cx"] - pairs["prev_object_cx"]) / valid_dt,
            (pairs["object_cy"] - pairs["prev_object_cy"]) / valid_dt,
        ]
    )
    pairs["hand_velocity_2d"] = np.linalg.norm(hand_velocity, axis=1)
    pairs["object_velocity_2d"] = np.linalg.norm(object_velocity, axis=1)
    pairs["velocity_similarity"] = cosine_similarity_2d(hand_velocity, object_velocity)
    pairs["relative_velocity"] = np.linalg.norm(hand_velocity - object_velocity, axis=1)
    pairs["object_still_score"] = 1.0 / (1.0 + pairs["object_velocity_2d"].fillna(0.0) / 100.0)

    pairs["contact_candidate"] = (
        (pairs["bbox_gap_norm"] <= contact_gap_threshold) | (pairs["expanded_iou"] >= contact_iou_threshold)
    ).astype(int)
    pairs["contact_duration_frames"] = (
        pairs.groupby(group_cols, sort=False)["contact_candidate"]
        .transform(lambda s: s.groupby((s != s.shift()).cumsum()).cumcount() + 1)
        .where(pairs["contact_candidate"].eq(1), 0)
    )
    pairs["contact_duration_s"] = pairs["contact_duration_frames"] / fps
    return pairs.drop(columns=[col for col in pairs.columns if col.startswith("prev_")])


def robust_norm(series, scale=None):
    values = pd.to_numeric(series, errors="coerce").astype(float)
    if scale is None:
        finite = values[np.isfinite(values)]
        scale = float(finite.quantile(0.9)) if len(finite) else 1.0
    scale = max(scale, EPS)
    return (values / scale).clip(lower=0.0, upper=1.0).fillna(0.0)


def numeric_series(pairs, column, default=0.0):
    if column in pairs.columns:
        return pd.to_numeric(pairs[column], errors="coerce").astype(float)
    return pd.Series(default, index=pairs.index, dtype=float)


def compute_visual_object_score(pairs):
    gap_score = np.exp(-pairs["bbox_gap_norm"].clip(lower=0.0, upper=10.0)).fillna(0.0)
    distance_score = np.exp(-pairs["distance_to_object_size"].clip(lower=0.0, upper=10.0)).fillna(0.0)
    object_conf_score = pairs["object_conf"].fillna(0.0).clip(0.0, 1.0)
    return (
        0.25 * gap_score
        + 0.20 * pairs["expanded_iou"].fillna(0.0).clip(0.0, 1.0)
        + 0.15 * pairs["object_center_in_hand_bbox"].fillna(0.0)
        + 0.10 * pairs["hand_center_in_object_bbox"].fillna(0.0)
        + 0.15 * distance_score
        + 0.15 * object_conf_score
    ).clip(0.0, 1.0)


def compute_motion_manipulation_score(pairs):
    skeleton_score = (
        0.45 * robust_norm(numeric_series(pairs, "wrist_velocity_3d"))
        + 0.35 * robust_norm(numeric_series(pairs, "hand_local_motion_energy"))
        + 0.20 * robust_norm(numeric_series(pairs, "wrist_angle_velocity"))
    )
    imu_score = (
        0.45 * robust_norm(numeric_series(pairs, "hand_gyro_mag"))
        + 0.35 * robust_norm(numeric_series(pairs, "imu_wrist_gyro_mag"))
        + 0.20 * robust_norm(numeric_series(pairs, "imu_wrist_angle_velocity"))
    )
    hand_velocity_score = robust_norm(numeric_series(pairs, "hand_velocity_2d"), scale=250.0)
    velocity_similarity = numeric_series(pairs, "velocity_similarity")
    velocity_similarity_score = ((velocity_similarity + 1.0) / 2.0).clip(0.0, 1.0).fillna(0.0)
    relative_velocity = numeric_series(pairs, "relative_velocity")
    relative_velocity_score = (1.0 / (1.0 + relative_velocity / 150.0)).where(relative_velocity.notna(), 0.0)
    contact_duration_score = (numeric_series(pairs, "contact_duration_s").fillna(0.0) / 0.5).clip(0.0, 1.0)
    motion_score = (
        0.30 * skeleton_score
        + 0.25 * imu_score
        + 0.15 * hand_velocity_score
        + 0.15 * velocity_similarity_score
        + 0.05 * relative_velocity_score
        + 0.10 * contact_duration_score
    ).clip(0.0, 1.0)
    return motion_score, skeleton_score, imu_score


def add_penalties_and_bonuses(
    pairs,
    cell_phone_penalty=CELL_PHONE_SCORE_PENALTY,
    large_object_score_weight=LARGE_OBJECT_SCORE_WEIGHT,
    large_object_soft_penalty_start_ratio=LARGE_OBJECT_SOFT_PENALTY_START_RATIO,
    large_object_soft_penalty_full_ratio=LARGE_OBJECT_SOFT_PENALTY_FULL_RATIO,
):
    object_area_ratio = pairs["object_area_ratio"].fillna(0.0).clip(0.0, 1.0)
    penalty_span = max(large_object_soft_penalty_full_ratio - large_object_soft_penalty_start_ratio, EPS)
    large_object_excess = ((object_area_ratio - large_object_soft_penalty_start_ratio) / penalty_span).clip(0.0, 1.0)
    large_object_penalty = large_object_excess.pow(2) * (3.0 - 2.0 * large_object_excess)
    cell_phone_penalty_indicator = pairs["object_class_id"].eq(CELL_PHONE_CLASS_ID).astype(float)
    object_class_norm = pairs["object_class"].map(normalize_object_class_name)
    small_object_priority = object_class_norm.isin(SMALL_OBJECT_PRIORITY_CLASSES)
    close_to_hand = (
        pairs["bbox_gap_norm"].fillna(np.inf).le(SMALL_OBJECT_PRIORITY_GAP_NORM_THRESHOLD)
        | pairs["distance_to_object_size"].fillna(np.inf).le(SMALL_OBJECT_PRIORITY_DISTANCE_SIZE_THRESHOLD)
        | pairs["expanded_iou"].fillna(0.0).gt(0.0)
        | pairs["hand_center_in_object_bbox"].fillna(0.0).eq(1)
        | pairs["object_center_in_hand_bbox"].fillna(0.0).eq(1)
    )
    small_enough = object_area_ratio.le(SMALL_OBJECT_PRIORITY_MAX_AREA_RATIO)
    small_object_priority_bonus = (
        small_object_priority & close_to_hand & small_enough
    ).astype(float) * SMALL_OBJECT_PRIORITY_BONUS

    pairs["large_object_penalty"] = large_object_penalty
    pairs["large_object_score_penalty"] = large_object_score_weight * large_object_penalty
    pairs["cell_phone_penalty"] = cell_phone_penalty_indicator
    pairs["cell_phone_score_penalty"] = cell_phone_penalty * cell_phone_penalty_indicator
    pairs["small_object_priority_bonus"] = small_object_priority_bonus.to_numpy(dtype=float)
    return pairs


def find_skeleton_path(participant_id, full_id):
    root = config.SKELETON_UPPER_LIMB_KINEMATICS_FRAME_OUTPUT_ROOT / participant_id
    paths = sorted(root.glob(f"{full_id}_Body_*_skeleton_upper_limb_kinematics_frame_features.csv"))
    if not paths:
        return None
    best_path = None
    best_score = -1
    for path in paths:
        try:
            cols = pd.read_csv(path, nrows=0).columns
            usecols = [col for col in ["left_wrist_speed", "right_wrist_speed"] if col in cols]
            if not usecols:
                score = 0
            else:
                score = int(pd.read_csv(path, usecols=usecols)[usecols].notna().sum().sum())
        except Exception:
            score = 0
        if score > best_score:
            best_path = path
            best_score = score
    return best_path


def find_imu_path(participant_id, full_id):
    path = (
        config.IMU_JOINT_KINEMATICS_FRAME_OUTPUT_ROOT
        / participant_id
        / f"{full_id}_imu_joint_kinematics_frame_features.csv"
    )
    return path if path.exists() else None


def read_feature_file(path, wanted_columns):
    if path is None:
        return None
    header = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [col for col in wanted_columns if col in header]
    if "time_s" not in usecols:
        return None
    df = pd.read_csv(path, usecols=usecols)
    return numeric_columns(df, usecols).dropna(subset=["time_s"]).sort_values("time_s")


def merge_asof_features(pairs, features, suffix):
    if pairs.empty or features is None or features.empty:
        return pairs
    left = pairs.sort_values("timestamp").copy()
    right = features.rename(columns={"time_s": f"{suffix}_time_s"}).sort_values(f"{suffix}_time_s")
    merged = pd.merge_asof(
        left,
        right,
        left_on="timestamp",
        right_on=f"{suffix}_time_s",
        direction="nearest",
        tolerance=0.05,
    )
    if "frame_idx" in merged.columns:
        merged = merged.rename(columns={"frame_idx": f"{suffix}_frame_idx"})
    return merged.sort_values(["frame_id", "hand_side", "object_class", "object_id"]).reset_index(drop=True)


def add_side_specific_motion_columns(pairs):
    for base, left_col, right_col in [
        ("wrist_velocity_3d", "left_wrist_speed", "right_wrist_speed"),
        ("hand_velocity_3d", "left_hand_center_speed", "right_hand_center_speed"),
        ("elbow_angle", "left_elbow_angle", "right_elbow_angle"),
        ("wrist_angle_velocity", "left_wrist_angle_velocity", "right_wrist_angle_velocity"),
        ("hand_local_motion_energy", "left_hand_motion_energy_0p35s", "right_hand_motion_energy_0p35s"),
        ("hand_gyro_mag", "left_hand_abs_gyro_mag", "right_hand_abs_gyro_mag"),
        ("imu_wrist_angle_velocity", "left_wrist_angle_velocity_deg_s", "right_wrist_angle_velocity_deg_s"),
        ("imu_wrist_gyro_mag", "left_wrist_relative_gyro_mag", "right_wrist_relative_gyro_mag"),
    ]:
        left = pairs[left_col] if left_col in pairs.columns else np.nan
        right = pairs[right_col] if right_col in pairs.columns else np.nan
        pairs[base] = np.where(pairs["hand_side"].eq("left"), left, right)
    return pairs


def add_scores_and_phases(
    pairs,
    manipulate_threshold,
    cell_phone_penalty=CELL_PHONE_SCORE_PENALTY,
    large_object_score_weight=LARGE_OBJECT_SCORE_WEIGHT,
    large_object_soft_penalty_start_ratio=LARGE_OBJECT_SOFT_PENALTY_START_RATIO,
    large_object_soft_penalty_full_ratio=LARGE_OBJECT_SOFT_PENALTY_FULL_RATIO,
):
    if pairs.empty:
        return pairs

    approach_score = robust_norm(pairs["approach_speed"], scale=400.0)
    velocity_similarity_raw = numeric_series(pairs, "velocity_similarity")
    velocity_similarity = ((velocity_similarity_raw + 1.0) / 2.0).clip(0.0, 1.0).fillna(0.0)
    relative_velocity_raw = numeric_series(pairs, "relative_velocity")
    relative_velocity_score = (1.0 / (1.0 + relative_velocity_raw / 150.0)).where(relative_velocity_raw.notna(), 0.0)
    object_motion_score = robust_norm(pairs["object_velocity_2d"], scale=250.0)
    temporal_score = 0.35 * approach_score + 0.35 * velocity_similarity + 0.20 * relative_velocity_score + 0.10 * object_motion_score

    visual_object_score = compute_visual_object_score(pairs)
    motion_manipulation_score, skeleton_score, imu_score = compute_motion_manipulation_score(pairs)

    pairs["visual_object_score"] = visual_object_score
    pairs["visual_score"] = visual_object_score
    pairs["temporal_score"] = temporal_score
    pairs["skeleton_motion_score"] = skeleton_score
    pairs["imu_motion_score"] = imu_score
    pairs["motion_manipulation_score"] = motion_manipulation_score
    pairs = add_penalties_and_bonuses(
        pairs,
        cell_phone_penalty=cell_phone_penalty,
        large_object_score_weight=large_object_score_weight,
        large_object_soft_penalty_start_ratio=large_object_soft_penalty_start_ratio,
        large_object_soft_penalty_full_ratio=large_object_soft_penalty_full_ratio,
    )
    pairs["interaction_score"] = (
        0.50 * visual_object_score
        + 0.20 * temporal_score
        + 0.30 * motion_manipulation_score
        - pairs["large_object_score_penalty"]
        - pairs["cell_phone_score_penalty"]
    ).clip(0.0, 1.0)
    pairs["selection_score"] = np.clip(
        (
            0.85 * visual_object_score
            + 0.15 * temporal_score
            - pairs["large_object_score_penalty"]
            - pairs["cell_phone_score_penalty"]
            + pairs["small_object_priority_bonus"]
        ).to_numpy(dtype=float),
        0.0,
        1.0,
    )

    contact = pairs["contact_candidate"].eq(1)
    manipulating = contact & (
        (pairs["interaction_score"] >= manipulate_threshold)
        & (
            (pairs["contact_duration_s"] >= 0.2)
            | (velocity_similarity >= 0.65)
            | (pairs["imu_motion_score"] >= 0.35)
            | (pairs["skeleton_motion_score"] >= 0.35)
        )
    )
    release = (
        pairs["contact_candidate"].eq(0)
        & (pairs.groupby(["hand_side", "object_class", "object_id"], sort=False)["contact_candidate"].shift(1).eq(1))
        & (pairs["separation_speed"].fillna(0.0) > 0.0)
    )
    approach = (
        pairs["approach_speed"].fillna(0.0).gt(0.0)
        & (pairs["hand_velocity_2d"].fillna(0.0).gt(10.0))
        & (pairs["object_still_score"].fillna(0.0).gt(0.5))
    )

    pairs["interaction_phase"] = PHASE_NONE
    pairs.loc[approach, "interaction_phase"] = PHASE_APPROACH
    pairs.loc[contact, "interaction_phase"] = PHASE_CONTACT
    pairs.loc[manipulating, "interaction_phase"] = PHASE_MANIPULATE
    pairs.loc[release, "interaction_phase"] = PHASE_RELEASE
    pairs["interaction_phase_name"] = pairs["interaction_phase"].map(
        {
            PHASE_NONE: "none",
            PHASE_APPROACH: "approach",
            PHASE_CONTACT: "contact_candidate",
            PHASE_MANIPULATE: "manipulate_hold",
            PHASE_RELEASE: "release",
        }
    )
    return pairs


PHASE_PRIORITY = {
    "none": 0,
    "approach": 1,
    "release": 2,
    "contact_candidate": 3,
    "manipulate_hold": 4,
}
PRIORITY_PHASE = {value: key for key, value in PHASE_PRIORITY.items()}


def choose_proxy_phase(object_window):
    if object_window.empty:
        return np.nan
    recent_timestamp = object_window["timestamp"].max()
    recent = object_window[object_window["timestamp"].ge(recent_timestamp - 0.25)]
    candidates = recent if not recent.empty else object_window
    phase_scores = candidates["interaction_phase_name"].map(PHASE_PRIORITY).fillna(0)
    return candidates.iloc[int(phase_scores.to_numpy().argmax())]["interaction_phase_name"]


def choose_temporal_proxy(window_df, side, window_frame_count):
    side_window = window_df[window_df["hand_side"].eq(side)]
    if side_window.empty:
        return None

    candidates = []
    group_cols = ["object_class", "object_id"]
    if "object_class_id" in side_window.columns:
        group_cols.insert(0, "object_class_id")

    for key, object_window in side_window.groupby(group_cols, sort=False):
        score_col = "selection_score" if "selection_score" in object_window.columns else "interaction_score"
        scores = object_window[score_col].astype(float)
        support_frames = int(object_window["frame_id"].nunique())
        support_ratio = support_frames / max(float(window_frame_count), 1.0)
        proxy_score = 0.65 * float(scores.mean()) + 0.25 * float(scores.max()) + 0.10 * support_ratio
        latest = object_window.sort_values(["frame_id", "interaction_score"], ascending=[False, False]).iloc[0]
        candidates.append(
            {
                "object_class_id": latest.get("object_class_id", np.nan),
                "object_class": latest["object_class"],
                "object_id": int(latest["object_id"]),
                "proxy_score": proxy_score,
                "support_frames": support_frames,
                "support_ratio": support_ratio,
                "proxy_phase": choose_proxy_phase(object_window),
            }
        )

    if not candidates:
        return None
    return max(candidates, key=lambda item: item["proxy_score"])


def add_temporal_proxy_features(pairs, proxy_window_s=config.HAND_OBJECT_PROXY_WINDOW_S, prefix="temporal_proxy"):
    if pairs.empty:
        return pairs

    pairs = pairs.copy()
    unique_timestamps = pairs[["frame_id", "timestamp"]].drop_duplicates("frame_id").sort_values("frame_id")["timestamp"]
    fps_estimate = 1.0 / max(float(unique_timestamps.diff().dropna().median()), EPS)
    expected_window_frames = max(1, int(round(proxy_window_s * fps_estimate)))
    group_cols = ["hand_side", "object_class", "object_id"]
    if "object_class_id" in pairs.columns:
        group_cols.insert(1, "object_class_id")

    pairs["phase_priority"] = pairs["interaction_phase_name"].map(PHASE_PRIORITY).fillna(0).astype(float)
    score_col = f"{prefix}_score"
    support_frames_col = f"{prefix}_support_frames"
    support_ratio_col = f"{prefix}_support_ratio"
    phase_col = f"{prefix}_phase"
    pairs[score_col] = np.nan
    pairs[support_frames_col] = 0
    pairs[support_ratio_col] = 0.0
    pairs[phase_col] = "none"

    for _, index in pairs.groupby(group_cols, sort=False).groups.items():
        track = pairs.loc[index].sort_values("timestamp")
        timed = track.set_index(pd.to_timedelta(track["timestamp"], unit="s"))
        base_score_col = "selection_score" if "selection_score" in timed.columns else "interaction_score"
        rolling_score = timed[base_score_col].rolling(f"{proxy_window_s}s", min_periods=1)
        score_mean = rolling_score.mean().to_numpy(dtype=float)
        score_max = rolling_score.max().to_numpy(dtype=float)
        support = rolling_score.count().to_numpy(dtype=float)
        support_ratio = np.clip(support / expected_window_frames, 0.0, 1.0)
        proxy_score = 0.65 * score_mean + 0.25 * score_max + 0.10 * support_ratio

        phase_priority = (
            timed["phase_priority"]
            .rolling(f"{proxy_window_s}s", min_periods=1)
            .max()
            .fillna(0)
            .astype(int)
            .to_numpy()
        )
        phases = [PRIORITY_PHASE.get(int(value), "none") for value in phase_priority]

        pairs.loc[track.index, score_col] = proxy_score
        pairs.loc[track.index, support_frames_col] = support.astype(int)
        pairs.loc[track.index, support_ratio_col] = support_ratio
        pairs.loc[track.index, phase_col] = phases

    return pairs.drop(columns=["phase_priority"])


def score_column_or_fallback(df, score_col):
    if score_col in df.columns:
        return score_col
    if "selection_score" in df.columns:
        return "selection_score"
    return "interaction_score"


def sort_candidates(df, score_col):
    if df.empty:
        return df
    score_col = score_column_or_fallback(df, score_col)
    sort_cols = [score_col]
    ascending = [False]
    for col in ["visual_object_score", "interaction_score", "object_conf"]:
        if col in df.columns and col not in sort_cols:
            sort_cols.append(col)
            ascending.append(False)
    return df.sort_values(sort_cols, ascending=ascending)


def class_score_name(class_name):
    return normalize_object_class_name(class_name).replace(" ", "_")


def add_topk_object_features(row, side_df, side, top_k, score_col):
    side_prefix = side[0].upper()
    candidates = sort_candidates(side_df, score_col)
    for rank in range(1, top_k + 1):
        prefix = f"{side_prefix}_top{rank}"
        if len(candidates) < rank:
            row[f"{prefix}_object_class_id"] = np.nan
            row[f"{prefix}_object"] = np.nan
            row[f"{prefix}_object_id"] = np.nan
            row[f"{prefix}_visual_object_score"] = np.nan
            row[f"{prefix}_interaction_score"] = np.nan
            row[f"{prefix}_selection_score"] = np.nan
            row[f"{prefix}_phase"] = np.nan
            continue
        candidate = candidates.iloc[rank - 1]
        row[f"{prefix}_object_class_id"] = candidate.get("object_class_id", np.nan)
        row[f"{prefix}_object"] = candidate["object_class"]
        row[f"{prefix}_object_id"] = candidate["object_id"]
        row[f"{prefix}_visual_object_score"] = candidate.get("visual_object_score", candidate.get("visual_score", np.nan))
        row[f"{prefix}_interaction_score"] = candidate.get("interaction_score", np.nan)
        row[f"{prefix}_selection_score"] = candidate.get("selection_score", candidate.get("interaction_score", np.nan))
        row[f"{prefix}_phase"] = candidate.get("interaction_phase_name", np.nan)
    return row


def add_temporal_topk_object_features(row, side_df, side, top_k, proxy_prefix):
    side_prefix = side[0].upper()
    score_col = f"{proxy_prefix}_score"
    phase_col = f"{proxy_prefix}_phase"
    group_cols = ["object_class", "object_id"]
    if "object_class_id" in side_df.columns:
        group_cols.insert(0, "object_class_id")
    candidates = []
    if not side_df.empty:
        for _, object_window in side_df.groupby(group_cols, sort=False):
            best = sort_candidates(object_window, score_col).iloc[0]
            candidates.append(best)
    candidates = pd.DataFrame(candidates) if candidates else pd.DataFrame()
    candidates = sort_candidates(candidates, score_col)
    output_prefix = proxy_prefix.replace("temporal_proxy", "temporal")
    for rank in range(1, top_k + 1):
        prefix = f"{side_prefix}_{output_prefix}_top{rank}"
        if len(candidates) < rank:
            row[f"{prefix}_object"] = np.nan
            row[f"{prefix}_score"] = np.nan
            row[f"{prefix}_phase"] = np.nan
            continue
        candidate = candidates.iloc[rank - 1]
        row[f"{prefix}_object"] = candidate["object_class"]
        row[f"{prefix}_score"] = candidate.get(score_col, np.nan)
        row[f"{prefix}_phase"] = candidate.get(phase_col, np.nan)
    return row


def temporal_window_for_frame(pairs, frame_timestamp, side, proxy_window_s):
    if pairs.empty:
        return pairs
    start_time = frame_timestamp - proxy_window_s
    return pairs[
        pairs["hand_side"].eq(side)
        & pairs["timestamp"].le(frame_timestamp)
        & pairs["timestamp"].ge(start_time)
    ]


def add_class_score_vector(row, side_df, side, class_names, score_col):
    side_prefix = side[0].upper()
    score_col = score_column_or_fallback(side_df, score_col) if not side_df.empty else score_col
    normalized_classes = [normalize_object_class_name(value) for value in class_names]
    object_classes = side_df["object_class"].map(normalize_object_class_name) if not side_df.empty else pd.Series(dtype=str)
    for class_name in normalized_classes:
        column = f"{side_prefix}_score_{class_score_name(class_name)}"
        if side_df.empty or score_col not in side_df.columns:
            row[column] = 0.0
            continue
        values = pd.to_numeric(side_df.loc[object_classes.eq(class_name), score_col], errors="coerce")
        row[column] = float(values.max()) if len(values.dropna()) else 0.0
    return row


def add_step_proxy_features(row, side, temporal_0p5):
    side_prefix = side[0].upper()
    prefix = f"{side_prefix}_step"
    if temporal_0p5 is None:
        row[f"{prefix}_object_class_id"] = np.nan
        row[f"{prefix}_object"] = np.nan
        row[f"{prefix}_object_id"] = np.nan
        row[f"{prefix}_score"] = np.nan
        row[f"{prefix}_phase"] = np.nan
        row[f"{prefix}_support_frames"] = 0
        row[f"{prefix}_support_ratio"] = 0.0
        return row
    row[f"{prefix}_object_class_id"] = temporal_0p5.get("object_class_id", np.nan)
    row[f"{prefix}_object"] = temporal_0p5["object_class"]
    row[f"{prefix}_object_id"] = temporal_0p5["object_id"]
    row[f"{prefix}_score"] = temporal_0p5["temporal_proxy_0p5_score"]
    row[f"{prefix}_phase"] = temporal_0p5["temporal_proxy_0p5_phase"]
    row[f"{prefix}_support_frames"] = temporal_0p5["temporal_proxy_0p5_support_frames"]
    row[f"{prefix}_support_ratio"] = temporal_0p5["temporal_proxy_0p5_support_ratio"]
    return row


def build_frame_summary(
    pairs,
    proxy_window_s=config.HAND_OBJECT_PROXY_WINDOW_S,
    top_k_objects=DEFAULT_TOP_K_OBJECTS,
    hoi_score_classes=DEFAULT_HOI_SCORE_CLASSES,
    class_score_col="selection_score",
):
    if pairs.empty:
        return pd.DataFrame()

    if "temporal_proxy_score" not in pairs.columns:
        pairs = add_temporal_proxy_features(pairs, proxy_window_s=proxy_window_s)
    if "temporal_proxy_0p5_score" not in pairs.columns:
        pairs = add_temporal_proxy_features(
            pairs,
            proxy_window_s=config.HAND_OBJECT_PROXY_SHORT_WINDOW_S,
            prefix="temporal_proxy_0p5",
        )

    rows = []
    for frame_id, frame_df in pairs.groupby("frame_id", sort=True):
        frame_timestamp = float(frame_df["timestamp"].iloc[0])
        row = {"frame_id": int(frame_id), "timestamp": frame_timestamp}
        side_best = {}
        temporal_best = {}
        temporal_0p5_best = {}
        for side in SIDES:
            side_df = frame_df[frame_df["hand_side"].eq(side)]
            temporal_side_df = temporal_window_for_frame(pairs, frame_timestamp, side, proxy_window_s)
            temporal_0p5_side_df = temporal_window_for_frame(
                pairs,
                frame_timestamp,
                side,
                config.HAND_OBJECT_PROXY_SHORT_WINDOW_S,
            )
            row = add_topk_object_features(row, side_df, side, top_k_objects, "selection_score")
            row = add_temporal_topk_object_features(
                row,
                temporal_0p5_side_df,
                side,
                top_k_objects,
                "temporal_proxy_0p5",
            )
            row = add_class_score_vector(row, side_df, side, hoi_score_classes, class_score_col)
            if side_df.empty:
                row[f"{side[0].upper()}_active_object_class_id"] = np.nan
                row[f"{side[0].upper()}_active_object"] = np.nan
                row[f"{side[0].upper()}_active_object_id"] = np.nan
                row[f"{side[0].upper()}_visual_object_score"] = np.nan
                row[f"{side[0].upper()}_interaction_score"] = np.nan
                row[f"{side[0].upper()}_selection_score"] = np.nan
                row[f"{side[0].upper()}_interaction_phase"] = np.nan
            else:
                active_score_col = "selection_score" if "selection_score" in side_df.columns else "interaction_score"
                best = sort_candidates(side_df, active_score_col).iloc[0]
                side_best[side] = best
                row[f"{side[0].upper()}_active_object_class_id"] = best.get("object_class_id", np.nan)
                row[f"{side[0].upper()}_active_object"] = best["object_class"]
                row[f"{side[0].upper()}_active_object_id"] = best["object_id"]
                row[f"{side[0].upper()}_visual_object_score"] = best.get("visual_object_score", best.get("visual_score", np.nan))
                row[f"{side[0].upper()}_interaction_score"] = best["interaction_score"]
                row[f"{side[0].upper()}_selection_score"] = best.get("selection_score", best["interaction_score"])
                row[f"{side[0].upper()}_interaction_phase"] = best["interaction_phase_name"]

            if temporal_side_df.empty:
                row[f"{side[0].upper()}_temporal_object_class_id"] = np.nan
                row[f"{side[0].upper()}_temporal_object"] = np.nan
                row[f"{side[0].upper()}_temporal_object_id"] = np.nan
                row[f"{side[0].upper()}_temporal_score"] = np.nan
                row[f"{side[0].upper()}_temporal_phase"] = np.nan
                row[f"{side[0].upper()}_temporal_support_frames"] = 0
                row[f"{side[0].upper()}_temporal_support_ratio"] = 0.0
            else:
                temporal = sort_candidates(temporal_side_df, "temporal_proxy_score").iloc[0]
                temporal_best[side] = temporal
                row[f"{side[0].upper()}_temporal_object_class_id"] = temporal.get("object_class_id", np.nan)
                row[f"{side[0].upper()}_temporal_object"] = temporal["object_class"]
                row[f"{side[0].upper()}_temporal_object_id"] = temporal["object_id"]
                row[f"{side[0].upper()}_temporal_score"] = temporal["temporal_proxy_score"]
                row[f"{side[0].upper()}_temporal_phase"] = temporal["temporal_proxy_phase"]
                row[f"{side[0].upper()}_temporal_support_frames"] = temporal["temporal_proxy_support_frames"]
                row[f"{side[0].upper()}_temporal_support_ratio"] = temporal["temporal_proxy_support_ratio"]

            if temporal_0p5_side_df.empty:
                row[f"{side[0].upper()}_temporal_0p5_object_class_id"] = np.nan
                row[f"{side[0].upper()}_temporal_0p5_object"] = np.nan
                row[f"{side[0].upper()}_temporal_0p5_object_id"] = np.nan
                row[f"{side[0].upper()}_temporal_0p5_score"] = np.nan
                row[f"{side[0].upper()}_temporal_0p5_phase"] = np.nan
                row[f"{side[0].upper()}_temporal_0p5_support_frames"] = 0
                row[f"{side[0].upper()}_temporal_0p5_support_ratio"] = 0.0
                row = add_step_proxy_features(row, side, None)
                continue

            temporal_0p5 = sort_candidates(temporal_0p5_side_df, "temporal_proxy_0p5_score").iloc[0]
            temporal_0p5_best[side] = temporal_0p5
            row[f"{side[0].upper()}_temporal_0p5_object_class_id"] = temporal_0p5.get("object_class_id", np.nan)
            row[f"{side[0].upper()}_temporal_0p5_object"] = temporal_0p5["object_class"]
            row[f"{side[0].upper()}_temporal_0p5_object_id"] = temporal_0p5["object_id"]
            row[f"{side[0].upper()}_temporal_0p5_score"] = temporal_0p5["temporal_proxy_0p5_score"]
            row[f"{side[0].upper()}_temporal_0p5_phase"] = temporal_0p5["temporal_proxy_0p5_phase"]
            row[f"{side[0].upper()}_temporal_0p5_support_frames"] = temporal_0p5["temporal_proxy_0p5_support_frames"]
            row[f"{side[0].upper()}_temporal_0p5_support_ratio"] = temporal_0p5["temporal_proxy_0p5_support_ratio"]
            row = add_step_proxy_features(row, side, temporal_0p5)

        left = side_best.get("left")
        right = side_best.get("right")
        same_object = (
            left is not None
            and right is not None
            and left["object_class"] == right["object_class"]
            and int(left["object_id"]) == int(right["object_id"])
        )
        row["two_hand_object"] = left["object_class"] if same_object else np.nan
        row["two_hand_object_id"] = int(left["object_id"]) if same_object else np.nan
        row["two_hand_interaction_score"] = (
            float((left["interaction_score"] + right["interaction_score"]) / 2.0) if same_object else 0.0
        )
        temporal_left = temporal_best.get("left")
        temporal_right = temporal_best.get("right")
        temporal_same_object = (
            temporal_left is not None
            and temporal_right is not None
            and temporal_left["object_class"] == temporal_right["object_class"]
            and int(temporal_left["object_id"]) == int(temporal_right["object_id"])
        )
        row["two_hand_temporal_object"] = temporal_left["object_class"] if temporal_same_object else np.nan
        row["two_hand_temporal_object_id"] = int(temporal_left["object_id"]) if temporal_same_object else np.nan
        row["two_hand_temporal_score"] = (
            float((temporal_left["temporal_proxy_score"] + temporal_right["temporal_proxy_score"]) / 2.0) if temporal_same_object else 0.0
        )
        temporal_0p5_left = temporal_0p5_best.get("left")
        temporal_0p5_right = temporal_0p5_best.get("right")
        temporal_0p5_same_object = (
            temporal_0p5_left is not None
            and temporal_0p5_right is not None
            and temporal_0p5_left["object_class"] == temporal_0p5_right["object_class"]
            and int(temporal_0p5_left["object_id"]) == int(temporal_0p5_right["object_id"])
        )
        row["two_hand_temporal_0p5_object"] = temporal_0p5_left["object_class"] if temporal_0p5_same_object else np.nan
        row["two_hand_temporal_0p5_object_id"] = int(temporal_0p5_left["object_id"]) if temporal_0p5_same_object else np.nan
        row["two_hand_temporal_0p5_score"] = (
            float((temporal_0p5_left["temporal_proxy_0p5_score"] + temporal_0p5_right["temporal_proxy_0p5_score"]) / 2.0)
            if temporal_0p5_same_object
            else 0.0
        )
        row["left_right_hand_center_dist"] = frame_df["left_right_hand_center_dist"].dropna().iloc[0] if "left_right_hand_center_dist" in frame_df and frame_df["left_right_hand_center_dist"].notna().any() else np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def safe_numeric_stat(df, column, stat):
    if df.empty or column not in df.columns:
        return np.nan
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return np.nan
    if stat == "mean":
        return float(values.mean())
    if stat == "median":
        return float(values.median())
    raise ValueError(f"Unsupported stat: {stat}")


def top_class_counts_json(series, top_n=10):
    values = series.map(normalize_object_class_name).replace("", np.nan).dropna()
    return json.dumps(values.value_counts().head(top_n).to_dict(), ensure_ascii=False)


def selected_active_rows(pairs):
    if pairs.empty:
        return pd.DataFrame()
    rows = []
    score_col = "selection_score" if "selection_score" in pairs.columns else "interaction_score"
    for _, side_df in pairs.groupby(["frame_id", "hand_side"], sort=False):
        rows.append(sort_candidates(side_df, score_col).iloc[0])
    return pd.DataFrame(rows)


def selected_class_frame_ratio(active_rows, class_name, frame_count):
    if active_rows.empty or frame_count <= 0:
        return 0.0
    classes = active_rows["object_class"].map(normalize_object_class_name)
    frame_count_for_class = active_rows.loc[classes.eq(normalize_object_class_name(class_name)), "frame_id"].nunique()
    return float(frame_count_for_class / frame_count)


def build_diagnostics(
    recording_info,
    objects,
    hands,
    pairs,
    frame_summary,
    large_object_soft_penalty_start_ratio=LARGE_OBJECT_SOFT_PENALTY_START_RATIO,
):
    frame_count = int(len(frame_summary))
    active_rows = selected_active_rows(pairs)
    large_selected_ratio = 0.0
    if not active_rows.empty and frame_count > 0 and "object_area_ratio" in active_rows.columns:
        large_selected_ratio = float(
            active_rows.loc[
                pd.to_numeric(active_rows["object_area_ratio"], errors="coerce").ge(
                    large_object_soft_penalty_start_ratio
                ),
                "frame_id",
            ].nunique()
            / frame_count
        )

    diagnostics = {
        "participant_id": recording_info["participant_id"],
        "task_name": recording_info["task_name"],
        "full_id": recording_info["full_id"],
        "object_rows": int(len(objects)),
        "hand_rows": int(len(hands)),
        "pair_rows": int(len(pairs)),
        "frame_rows": frame_count,
        "left_hand_frame_count": int(hands.loc[hands["hand_side"].eq("left"), "frame_id"].nunique()) if not hands.empty else 0,
        "right_hand_frame_count": int(hands.loc[hands["hand_side"].eq("right"), "frame_id"].nunique()) if not hands.empty else 0,
        "object_frame_count": int(objects["frame_id"].nunique()) if not objects.empty else 0,
        "L_active_object_frame_ratio": float(frame_summary["L_active_object"].notna().mean()) if frame_count and "L_active_object" in frame_summary.columns else 0.0,
        "R_active_object_frame_ratio": float(frame_summary["R_active_object"].notna().mean()) if frame_count and "R_active_object" in frame_summary.columns else 0.0,
        "L_active_object_top_classes": top_class_counts_json(frame_summary["L_active_object"]) if "L_active_object" in frame_summary.columns else "{}",
        "R_active_object_top_classes": top_class_counts_json(frame_summary["R_active_object"]) if "R_active_object" in frame_summary.columns else "{}",
        "cell_phone_selected_frame_ratio": selected_class_frame_ratio(active_rows, "cell phone", frame_count),
        "book_selected_frame_ratio": selected_class_frame_ratio(active_rows, "book", frame_count),
        "large_object_selected_frame_ratio": large_selected_ratio,
        "selection_score_mean": safe_numeric_stat(pairs, "selection_score", "mean"),
        "selection_score_median": safe_numeric_stat(pairs, "selection_score", "median"),
        "visual_object_score_mean": safe_numeric_stat(pairs, "visual_object_score", "mean"),
        "visual_object_score_median": safe_numeric_stat(pairs, "visual_object_score", "median"),
        "motion_manipulation_score_mean": safe_numeric_stat(pairs, "motion_manipulation_score", "mean"),
        "motion_manipulation_score_median": safe_numeric_stat(pairs, "motion_manipulation_score", "median"),
    }
    return diagnostics


def process_recording(
    object_path,
    hand_path,
    pair_output_path,
    frame_output_path,
    fps,
    image_width=None,
    image_height=None,
    expand_ratio=config.HAND_OBJECT_EXPAND_RATIO,
    contact_gap_threshold=config.HAND_OBJECT_CONTACT_GAP_NORM_THRESHOLD,
    contact_iou_threshold=config.HAND_OBJECT_CONTACT_IOU_THRESHOLD,
    manipulate_threshold=config.HAND_OBJECT_MANIPULATE_SCORE_THRESHOLD,
    exclude_class_ids=DEFAULT_OBJECT_EXCLUDE_CLASS_IDS,
    top_k_objects=DEFAULT_TOP_K_OBJECTS,
    hoi_score_classes=DEFAULT_HOI_SCORE_CLASSES,
    cell_phone_penalty=CELL_PHONE_SCORE_PENALTY,
    large_object_score_weight=LARGE_OBJECT_SCORE_WEIGHT,
    large_object_soft_penalty_start_ratio=LARGE_OBJECT_SOFT_PENALTY_START_RATIO,
    large_object_soft_penalty_full_ratio=LARGE_OBJECT_SOFT_PENALTY_FULL_RATIO,
):
    full_id = parse_full_id_from_yolo_path(object_path)
    participant_id = participant_from_full_id(full_id)
    task_name = task_from_full_id(full_id)
    recording_info = {"participant_id": participant_id, "task_name": task_name, "full_id": full_id}

    objects = read_object_detections(object_path, exclude_class_ids=exclude_class_ids)
    hands = read_hand_detections(hand_path)
    width, height = infer_image_size(objects, hands, image_width=image_width, image_height=image_height)
    pairs = build_pair_rows(objects, hands, fps=fps, image_width=width, image_height=height, expand_ratio=expand_ratio)
    pairs = add_temporal_features(pairs, fps, contact_gap_threshold, contact_iou_threshold)

    skeleton = read_feature_file(find_skeleton_path(participant_id, full_id), SKELETON_COLUMNS)
    imu = read_feature_file(find_imu_path(participant_id, full_id), IMU_COLUMNS)
    pairs = merge_asof_features(pairs, skeleton, "skeleton")
    pairs = merge_asof_features(pairs, imu, "imu")

    if not pairs.empty:
        pairs.insert(0, "full_id", full_id)
        pairs.insert(0, "task_name", task_name)
        pairs.insert(0, "participant_id", participant_id)
        pairs = add_side_specific_motion_columns(pairs)
        pairs = add_scores_and_phases(
            pairs,
            manipulate_threshold,
            cell_phone_penalty=cell_phone_penalty,
            large_object_score_weight=large_object_score_weight,
            large_object_soft_penalty_start_ratio=large_object_soft_penalty_start_ratio,
            large_object_soft_penalty_full_ratio=large_object_soft_penalty_full_ratio,
        )
        pairs = add_temporal_proxy_features(pairs, proxy_window_s=config.HAND_OBJECT_PROXY_WINDOW_S)
        pairs = add_temporal_proxy_features(
            pairs,
            proxy_window_s=config.HAND_OBJECT_PROXY_SHORT_WINDOW_S,
            prefix="temporal_proxy_0p5",
        )

    frame_summary = build_frame_summary(
        pairs,
        top_k_objects=top_k_objects,
        hoi_score_classes=hoi_score_classes,
    )
    if not frame_summary.empty:
        frame_summary.insert(0, "full_id", full_id)
        frame_summary.insert(0, "task_name", task_name)
        frame_summary.insert(0, "participant_id", participant_id)

    pair_output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_output_path.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(pair_output_path, index=False)
    frame_summary.to_csv(frame_output_path, index=False)

    diagnostics = build_diagnostics(
        recording_info,
        objects,
        hands,
        pairs,
        frame_summary,
        large_object_soft_penalty_start_ratio=large_object_soft_penalty_start_ratio,
    )

    return {
        "participant_id": participant_id,
        "task_name": task_name,
        "full_id": full_id,
        "object_path": str(object_path),
        "hand_path": str(hand_path),
        "pair_output_path": str(pair_output_path),
        "frame_output_path": str(frame_output_path),
        "pair_rows": int(len(pairs)),
        "frame_rows": int(len(frame_summary)),
        "object_rows": int(len(objects)),
        "hand_rows": int(len(hands)),
        "has_skeleton": skeleton is not None,
        "has_imu": imu is not None,
        "diagnostics": diagnostics,
    }


def iter_object_paths(input_root, participant=None, full_id=None, limit=None):
    paths = sorted(Path(input_root).glob("*_world_camera.csv"))
    if participant:
        paths = [path for path in paths if parse_full_id_from_yolo_path(path).startswith(f"{participant}_")]
    if full_id:
        wanted = set(full_id)
        paths = [path for path in paths if parse_full_id_from_yolo_path(path) in wanted]
    if limit is not None:
        paths = paths[:limit]
    return paths


def parse_class_id_set(value):
    if not value:
        return set()
    return {int(part.strip()) for part in value.split(",") if part.strip()}


def parse_hoi_score_classes(value):
    if not value:
        return list(DEFAULT_HOI_SCORE_CLASSES)
    return [normalize_object_class_name(part) for part in value.split(",") if part.strip()]


def merge_existing_records(path, records, key="full_id"):
    new_df = pd.DataFrame(records)
    old_df = None
    if path.exists():
        try:
            old_df = pd.read_csv(path)
        except Exception:
            old_df = None
    if new_df.empty:
        return old_df if old_df is not None else new_df
    if old_df is None or key not in new_df.columns:
        return new_df
    if old_df.empty or key not in old_df.columns:
        return new_df
    combined = pd.concat([old_df, new_df], ignore_index=True, sort=False)
    return combined.drop_duplicates(subset=[key], keep="last")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Build hand-object interaction pair and frame features.")
    parser.add_argument("--object-root", type=Path, default=config.YOLO_OBJECT_OUTCOME_ROOT)
    parser.add_argument("--hand-root", type=Path, default=config.YOLO_LEFT_RIGHT_HAND_OUTCOME_ROOT)
    parser.add_argument("--output-root", type=Path, default=config.HAND_OBJECT_INTERACTION_OUTPUT_ROOT)
    parser.add_argument("--participant")
    parser.add_argument("--full-id", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fps", type=float, default=config.HAND_OBJECT_VIDEO_FPS)
    parser.add_argument("--image-width", type=float)
    parser.add_argument("--image-height", type=float)
    parser.add_argument("--expand-ratio", type=float, default=config.HAND_OBJECT_EXPAND_RATIO)
    parser.add_argument("--contact-gap-threshold", type=float, default=config.HAND_OBJECT_CONTACT_GAP_NORM_THRESHOLD)
    parser.add_argument("--contact-iou-threshold", type=float, default=config.HAND_OBJECT_CONTACT_IOU_THRESHOLD)
    parser.add_argument("--manipulate-threshold", type=float, default=config.HAND_OBJECT_MANIPULATE_SCORE_THRESHOLD)
    parser.add_argument("--top-k-objects", type=int, default=DEFAULT_TOP_K_OBJECTS)
    parser.add_argument("--cell-phone-penalty", type=float, default=CELL_PHONE_SCORE_PENALTY)
    parser.add_argument("--large-object-score-weight", type=float, default=DEFAULT_LARGE_OBJECT_SCORE_WEIGHT)
    parser.add_argument(
        "--large-object-soft-penalty-start-ratio",
        type=float,
        default=DEFAULT_LARGE_OBJECT_SOFT_PENALTY_START_RATIO,
    )
    parser.add_argument(
        "--large-object-soft-penalty-full-ratio",
        type=float,
        default=DEFAULT_LARGE_OBJECT_SOFT_PENALTY_FULL_RATIO,
    )
    parser.add_argument(
        "--hoi-score-classes",
        default=",".join(DEFAULT_HOI_SCORE_CLASSES),
        help="Comma-separated object class names for per-class HOI score vector columns.",
    )
    parser.add_argument(
        "--exclude-class-ids",
        default=",".join(str(value) for value in sorted(DEFAULT_OBJECT_EXCLUDE_CLASS_IDS)),
        help="Comma-separated object YOLO class ids to drop before pairing.",
    )
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    pair_root = args.output_root / "pair_features"
    frame_root = args.output_root / "frame_features"
    object_paths = iter_object_paths(args.object_root, participant=args.participant, full_id=args.full_id, limit=args.limit)
    iterator = tqdm(object_paths, desc="hand-object interaction") if tqdm is not None else object_paths

    records = []
    diagnostics_records = []
    missing_hands = []
    exclude_class_ids = parse_class_id_set(args.exclude_class_ids)
    hoi_score_classes = parse_hoi_score_classes(args.hoi_score_classes)

    for object_path in iterator:
        full_id = parse_full_id_from_yolo_path(object_path)
        participant_id = participant_from_full_id(full_id)
        hand_path = args.hand_root / f"{full_id}_world_camera.csv"
        if not hand_path.exists():
            missing_hands.append({"full_id": full_id, "object_path": str(object_path), "hand_path": str(hand_path)})
            continue

        pair_output_path = pair_root / participant_id / f"{full_id}_hand_object_pair_features.csv"
        frame_output_path = frame_root / participant_id / f"{full_id}_hand_object_frame_features.csv"
        if not args.overwrite and pair_output_path.exists() and frame_output_path.exists():
            continue

        record = process_recording(
            object_path=object_path,
            hand_path=hand_path,
            pair_output_path=pair_output_path,
            frame_output_path=frame_output_path,
            fps=args.fps,
            image_width=args.image_width,
            image_height=args.image_height,
            expand_ratio=args.expand_ratio,
            contact_gap_threshold=args.contact_gap_threshold,
            contact_iou_threshold=args.contact_iou_threshold,
            manipulate_threshold=args.manipulate_threshold,
            exclude_class_ids=exclude_class_ids,
            top_k_objects=max(1, args.top_k_objects),
            hoi_score_classes=hoi_score_classes,
            cell_phone_penalty=args.cell_phone_penalty,
            large_object_score_weight=args.large_object_score_weight,
            large_object_soft_penalty_start_ratio=args.large_object_soft_penalty_start_ratio,
            large_object_soft_penalty_full_ratio=args.large_object_soft_penalty_full_ratio,
        )
        diagnostics_records.append(record.pop("diagnostics"))
        records.append(record)

    args.output_root.mkdir(parents=True, exist_ok=True)
    index_path = args.output_root / "hand_object_interaction_index.csv"
    summary_path = args.output_root / "hand_object_interaction_summary.json"
    missing_path = args.output_root / "missing_left_right_hand_yolo.csv"
    diagnostics_path = args.output_root / "hand_object_interaction_diagnostics.csv"

    index_df = pd.DataFrame(records) if args.overwrite else merge_existing_records(index_path, records)
    diagnostics_df = (
        pd.DataFrame(diagnostics_records)
        if args.overwrite
        else merge_existing_records(diagnostics_path, diagnostics_records)
    )
    index_df.to_csv(index_path, index=False)
    diagnostics_df.to_csv(diagnostics_path, index=False)
    pd.DataFrame(missing_hands).to_csv(missing_path, index=False)
    summary = {
        "recording_count": int(len(index_df)),
        "processed_recording_count": len(records),
        "missing_hand_yolo_count": len(missing_hands),
        "pair_rows": int(pd.to_numeric(index_df.get("pair_rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
        "frame_rows": int(pd.to_numeric(index_df.get("frame_rows", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()),
        "object_root": str(args.object_root),
        "hand_root": str(args.hand_root),
        "output_root": str(args.output_root),
        "fps": args.fps,
        "exclude_class_ids": sorted(exclude_class_ids),
        "top_k_objects": max(1, args.top_k_objects),
        "hoi_score_classes": hoi_score_classes,
        "cell_phone_penalty": args.cell_phone_penalty,
        "large_object_score_weight": args.large_object_score_weight,
        "large_object_soft_penalty_start_ratio": args.large_object_soft_penalty_start_ratio,
        "large_object_soft_penalty_full_ratio": args.large_object_soft_penalty_full_ratio,
        "diagnostics_path": str(diagnostics_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
