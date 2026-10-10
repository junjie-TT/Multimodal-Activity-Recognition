import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from upper_features_extraction import config

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


ALIGNED_SENSOR_ROOT = config.LOWER_FEATURE_OUTCOME_ROOT / "aligned_sensor_with_YOLO"
EPS = 1e-9

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

CORE_JOINTS = [
    "neck",
    "spine_1",
    "spine_2",
    "spine_3",
]
SIDE_BASE_JOINTS = [
    "shoulder",
    "elbow",
    "wrist",
    "hip",
    "hand_index_1",
    "hand_middle_4",
    "hand_pinky_1",
    "hand_thumb_4",
]
HAND_JOINTS = [
    "hand_index_1",
    "hand_middle_4",
    "hand_pinky_1",
    "hand_thumb_4",
]
SIDES = ("left", "right")
BODY_JOINTS = CORE_JOINTS + [
    f"{side}_{joint}" for side in SIDES for joint in SIDE_BASE_JOINTS
]


def participant_from_full_id(full_id):
    return "_".join(full_id.split("_")[:4])


def parse_recording_dir(recording_dir):
    recording_dir = Path(recording_dir)
    participant_id = recording_dir.parent.name
    full_id = recording_dir.name
    prefix = f"{participant_id}_"
    task_name = full_id[len(prefix):] if full_id.startswith(prefix) else full_id
    return participant_id, task_name, full_id


def body_id_from_path(path):
    return Path(path).stem.replace("zed__skeleton__Body_", "")


def skeleton_usecols(path):
    header = pd.read_csv(path, nrows=0).columns.tolist()
    wanted = ["frame_idx", "time_s"]
    for joint in BODY_JOINTS:
        wanted.extend([f"{joint}_x", f"{joint}_y", f"{joint}_z"])
    return [col for col in wanted if col in header]


def read_skeleton_sensor_axis(path):
    usecols = skeleton_usecols(path)
    df = pd.read_csv(path, usecols=usecols)
    for column in usecols:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if "time_s" not in df.columns:
        raise ValueError(f"{path} has no original time_s column")
    if "frame_idx" not in df.columns:
        df["frame_idx"] = np.round(df["time_s"].to_numpy(dtype=float) * 120.0)

    for joint in BODY_JOINTS:
        cols = [f"{joint}_{axis}" for axis in ("x", "y", "z")]
        if not set(cols).issubset(df.columns):
            continue
        values = df[cols].to_numpy(dtype=float)
        invalid = ~np.isfinite(values).all(axis=1) | np.all(
            np.isclose(values, -1.0, equal_nan=False), axis=1
        )
        df.loc[invalid, cols] = np.nan

    return df.dropna(subset=["time_s"]).sort_values("time_s").reset_index(drop=True)


def joint_xyz(df, joint):
    cols = [f"{joint}_{axis}" for axis in ("x", "y", "z")]
    if not set(cols).issubset(df.columns):
        return np.full((len(df), 3), np.nan, dtype=float)
    return df[cols].to_numpy(dtype=float)


def valid_xyz(values):
    return np.isfinite(values).all(axis=1)


def nanmean_points(points):
    stack = np.stack(points, axis=2)
    valid_counts = np.isfinite(stack).all(axis=1).sum(axis=1)
    clean = np.where(np.isfinite(stack), stack, 0.0)
    counts = np.isfinite(stack).sum(axis=2)
    out = np.divide(
        clean.sum(axis=2),
        counts,
        out=np.full(stack.shape[:2], np.nan, dtype=float),
        where=counts > 0,
    )
    out[valid_counts == 0] = np.nan
    return out


def norm_rows(values):
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    valid = valid_xyz(values)
    out[valid] = np.linalg.norm(values[valid], axis=1)
    return out


def normalize_rows(values):
    values = np.asarray(values, dtype=float)
    out = np.full_like(values, np.nan, dtype=float)
    norms = norm_rows(values)
    valid = np.isfinite(norms) & (norms > EPS)
    out[valid] = values[valid] / norms[valid, None]
    return out


def point_velocity(values, time_s):
    values = np.asarray(values, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    velocity = np.full_like(values, np.nan, dtype=float)
    if len(values) < 2:
        return velocity
    dt = np.diff(time_s)
    delta = values[1:] - values[:-1]
    valid = (
        np.isfinite(delta).all(axis=1)
        & np.isfinite(dt)
        & (dt > EPS)
        & valid_xyz(values[1:])
        & valid_xyz(values[:-1])
    )
    velocity[1:][valid] = delta[valid] / dt[valid, None]
    return velocity


def scalar_velocity(values, time_s):
    values = np.asarray(values, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    if len(values) < 2:
        return out
    dt = np.diff(time_s)
    delta = values[1:] - values[:-1]
    valid = np.isfinite(delta) & np.isfinite(dt) & (dt > EPS)
    out[1:][valid] = delta[valid] / dt[valid]
    return out


def scalar_acceleration(values, time_s):
    return scalar_velocity(scalar_velocity(values, time_s), time_s)


def point_acceleration_mag(values, time_s):
    return norm_rows(point_velocity(point_velocity(values, time_s), time_s))


def point_jerk_mag(values, time_s):
    velocity = point_velocity(values, time_s)
    acceleration = point_velocity(velocity, time_s)
    jerk = point_velocity(acceleration, time_s)
    return norm_rows(jerk)


def angle_between(v1, v2):
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    out = np.full(len(v1), np.nan, dtype=float)
    valid = valid_xyz(v1) & valid_xyz(v2)
    if not valid.any():
        return out
    n1 = np.linalg.norm(v1[valid], axis=1)
    n2 = np.linalg.norm(v2[valid], axis=1)
    denom = n1 * n2
    good = denom > EPS
    valid_idx = np.flatnonzero(valid)
    dots = np.sum(v1[valid][good] * v2[valid][good], axis=1) / denom[good]
    out[valid_idx[good]] = np.degrees(np.arccos(np.clip(dots, -1.0, 1.0)))
    return out


def distance(a, b):
    out = np.full(len(a), np.nan, dtype=float)
    valid = valid_xyz(a) & valid_xyz(b)
    out[valid] = np.linalg.norm(a[valid] - b[valid], axis=1)
    return out


def point_line_distance(point, line_a, line_b):
    out = np.full(len(point), np.nan, dtype=float)
    ab = line_b - line_a
    ap = point - line_a
    valid = valid_xyz(point) & valid_xyz(line_a) & valid_xyz(line_b)
    denom = norm_rows(ab)
    valid &= np.isfinite(denom) & (denom > EPS)
    if not valid.any():
        return out
    cross = np.cross(ap[valid], ab[valid])
    out[valid] = np.linalg.norm(cross, axis=1) / denom[valid]
    return out


def rolling_stats(frame_df, column, window_s, prefix=None):
    prefix = prefix or column
    indexed = frame_df.set_index(pd.to_timedelta(frame_df["time_s"], unit="s"))[column]
    rolling = indexed.rolling(f"{window_s}s", min_periods=1)
    frame_df[f"{prefix}_mean_0p35s"] = rolling.mean().to_numpy()
    frame_df[f"{prefix}_std_0p35s"] = rolling.std(ddof=0).to_numpy()
    frame_df[f"{prefix}_min_0p35s"] = rolling.min().to_numpy()
    frame_df[f"{prefix}_max_0p35s"] = rolling.max().to_numpy()
    frame_df[f"{prefix}_range_0p35s"] = (
        frame_df[f"{prefix}_max_0p35s"] - frame_df[f"{prefix}_min_0p35s"]
    )


def rolling_abs_mean(frame_df, column, window_s, out_column):
    indexed = frame_df.set_index(pd.to_timedelta(frame_df["time_s"], unit="s"))[column].abs()
    frame_df[out_column] = indexed.rolling(f"{window_s}s", min_periods=1).mean().to_numpy()


def rolling_corr(frame_df, left_column, right_column, window_s, out_column):
    time_index = pd.to_timedelta(frame_df["time_s"], unit="s")
    left = frame_df.set_index(time_index)[left_column]
    right = frame_df.set_index(time_index)[right_column]
    frame_df[out_column] = left.rolling(f"{window_s}s", min_periods=3).corr(right).to_numpy()


def pairwise_max_distance(points):
    out = np.full(len(points[0]), np.nan, dtype=float)
    distances = []
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            distances.append(distance(points[i], points[j]))
    for values in distances:
        valid = np.isfinite(values)
        out[valid] = np.where(np.isfinite(out[valid]), np.maximum(out[valid], values[valid]), values[valid])
    return out


def hand_shape_area(points):
    out = np.full(len(points[0]), np.nan, dtype=float)
    stack = np.stack(points, axis=1)
    valid = np.isfinite(stack).all(axis=2).all(axis=1)
    if not valid.any():
        return out
    center = np.mean(stack[valid], axis=1)
    rel = stack[valid] - center[:, None, :]
    area = np.zeros(valid.sum(), dtype=float)
    for idx in range(stack.shape[1]):
        nxt = (idx + 1) % stack.shape[1]
        area += 0.5 * np.linalg.norm(np.cross(rel[:, idx], rel[:, nxt]), axis=1)
    out[valid] = area
    return out


def build_trunk_basis(neck, spine_3, left_shoulder, right_shoulder):
    vertical = normalize_rows(neck - spine_3)
    lateral = normalize_rows(right_shoulder - left_shoulder)
    forward = normalize_rows(np.cross(lateral, vertical))
    return vertical, lateral, forward


def project_on_axis(displacement, axis):
    out = np.full(len(displacement), np.nan, dtype=float)
    valid = valid_xyz(displacement) & valid_xyz(axis)
    out[valid] = np.sum(displacement[valid] * axis[valid], axis=1)
    return out


def skeleton_quality(path):
    df = read_skeleton_sensor_axis(path)
    if df.empty:
        return 0
    valid_counts = []
    for joint in BODY_JOINTS:
        valid_counts.append(int(valid_xyz(joint_xyz(df, joint)).sum()))
    return int(np.nanmean(valid_counts)) if valid_counts else 0


def select_main_skeleton_path(recording_dir):
    paths = sorted(Path(recording_dir).glob("zed__skeleton__Body_*.csv"))
    if not paths:
        return None
    scored = [(skeleton_quality(path), path) for path in paths]
    return sorted(scored, key=lambda item: item[0], reverse=True)[0][1]


def iter_recording_dirs(input_root=ALIGNED_SENSOR_ROOT, participant=None, full_id=None, limit=None):
    input_root = Path(input_root)
    dirs = sorted(path for path in input_root.glob("*/*") if path.is_dir())
    if participant:
        dirs = [path for path in dirs if path.parent.name == participant]
    if full_id:
        wanted = set(full_id)
        dirs = [path for path in dirs if path.name in wanted]
    dirs = [path for path in dirs if any(path.glob("zed__skeleton__Body_*.csv"))]
    if limit is not None:
        dirs = dirs[:limit]
    return dirs


def extract_frame_features(
    skeleton_df,
    participant_id,
    task_name,
    full_id,
    body_id,
    window_s=config.SKELETON_UPPER_LIMB_KINEMATICS_WINDOW_S,
    both_hands_active_threshold=config.SKELETON_BOTH_HANDS_ACTIVE_SPEED_THRESHOLD,
):
    time_s = skeleton_df["time_s"].to_numpy(dtype=float)
    frame_idx = skeleton_df["frame_idx"].to_numpy(dtype=float)

    neck = joint_xyz(skeleton_df, "neck")
    spine_1 = joint_xyz(skeleton_df, "spine_1")
    spine_2 = joint_xyz(skeleton_df, "spine_2")
    spine_3 = joint_xyz(skeleton_df, "spine_3")
    left_hip = joint_xyz(skeleton_df, "left_hip")
    right_hip = joint_xyz(skeleton_df, "right_hip")
    left_shoulder = joint_xyz(skeleton_df, "left_shoulder")
    right_shoulder = joint_xyz(skeleton_df, "right_shoulder")

    hip_center = nanmean_points([left_hip, right_hip])
    shoulder_center = nanmean_points([left_shoulder, right_shoulder])
    trunk_center = nanmean_points([neck, spine_1, spine_2, spine_3, hip_center])
    vertical, lateral, forward = build_trunk_basis(neck, spine_3, left_shoulder, right_shoulder)
    trunk_vec = neck - spine_3
    global_vertical = np.tile(np.array([0.0, 1.0, 0.0]), (len(skeleton_df), 1))

    out = pd.DataFrame(
        {
            "participant_id": participant_id,
            "task_name": task_name,
            "full_id": full_id,
            "body_id": body_id,
            "frame_idx": frame_idx,
            "time_s": time_s,
            "neck_x": neck[:, 0],
            "neck_y": neck[:, 1],
            "neck_z": neck[:, 2],
            "trunk_center_x": trunk_center[:, 0],
            "trunk_center_y": trunk_center[:, 1],
            "trunk_center_z": trunk_center[:, 2],
        }
    )

    neck_acc = point_acceleration_mag(neck, time_s)
    trunk_angle = angle_between(trunk_vec, global_vertical)
    out["neck_speed"] = norm_rows(point_velocity(neck, time_s))
    out["neck_acceleration_mag"] = neck_acc
    out["neck_jerk_mag"] = point_jerk_mag(neck, time_s)
    out["neck_to_hip_dist"] = distance(neck, hip_center)
    out["shoulder_center_to_hip_dist"] = distance(shoulder_center, hip_center)
    out["trunk_angle"] = trunk_angle
    out["trunk_angle_velocity"] = scalar_velocity(trunk_angle, time_s)
    out["trunk_angle_acceleration"] = scalar_acceleration(trunk_angle, time_s)
    out["neck_spine_angle"] = angle_between(neck - spine_2, hip_center - spine_2)

    hand_centers = {}
    hand_speeds = {}

    for side in SIDES:
        shoulder = joint_xyz(skeleton_df, f"{side}_shoulder")
        elbow = joint_xyz(skeleton_df, f"{side}_elbow")
        wrist = joint_xyz(skeleton_df, f"{side}_wrist")
        hip = joint_xyz(skeleton_df, f"{side}_hip")
        fingers = [joint_xyz(skeleton_df, f"{side}_{joint}") for joint in HAND_JOINTS]
        hand_center = nanmean_points(fingers)
        hand_centers[side] = hand_center

        upper_arm_vec = elbow - shoulder
        forearm_vec = wrist - elbow
        hand_vec = hand_center - wrist
        elbow_angle = angle_between(shoulder - elbow, wrist - elbow)
        wrist_angle = angle_between(forearm_vec, hand_vec)
        shoulder_angle = angle_between(upper_arm_vec, trunk_vec)

        wrist_speed = norm_rows(point_velocity(wrist, time_s))
        hand_center_speed = norm_rows(point_velocity(hand_center, time_s))
        hand_speeds[side] = hand_center_speed

        displacement_from_trunk = hand_center - trunk_center
        out[f"{side}_wrist_x"] = wrist[:, 0]
        out[f"{side}_wrist_y"] = wrist[:, 1]
        out[f"{side}_wrist_z"] = wrist[:, 2]
        out[f"{side}_hand_center_x"] = hand_center[:, 0]
        out[f"{side}_hand_center_y"] = hand_center[:, 1]
        out[f"{side}_hand_center_z"] = hand_center[:, 2]

        out[f"{side}_wrist_speed"] = wrist_speed
        out[f"{side}_wrist_acceleration_mag"] = point_acceleration_mag(wrist, time_s)
        out[f"{side}_wrist_jerk_mag"] = point_jerk_mag(wrist, time_s)
        out[f"{side}_hand_center_speed"] = hand_center_speed
        out[f"{side}_hand_center_acceleration_mag"] = point_acceleration_mag(hand_center, time_s)
        out[f"{side}_hand_center_jerk_mag"] = point_jerk_mag(hand_center, time_s)
        out[f"{side}_elbow_speed"] = norm_rows(point_velocity(elbow, time_s))
        out[f"{side}_shoulder_speed"] = norm_rows(point_velocity(shoulder, time_s))

        out[f"{side}_elbow_angle"] = elbow_angle
        out[f"{side}_elbow_angle_velocity"] = scalar_velocity(elbow_angle, time_s)
        out[f"{side}_elbow_angle_acceleration"] = scalar_acceleration(elbow_angle, time_s)
        out[f"{side}_wrist_angle"] = wrist_angle
        out[f"{side}_wrist_angle_velocity"] = scalar_velocity(wrist_angle, time_s)
        out[f"{side}_wrist_angle_acceleration"] = scalar_acceleration(wrist_angle, time_s)
        out[f"{side}_shoulder_angle"] = shoulder_angle
        out[f"{side}_shoulder_angle_velocity"] = scalar_velocity(shoulder_angle, time_s)
        out[f"{side}_shoulder_angle_acceleration"] = scalar_acceleration(shoulder_angle, time_s)

        out[f"{side}_upper_arm_length"] = distance(shoulder, elbow)
        out[f"{side}_forearm_length"] = distance(elbow, wrist)
        out[f"{side}_wrist_to_shoulder_dist"] = distance(wrist, shoulder)
        out[f"{side}_wrist_to_neck_dist"] = distance(wrist, neck)
        out[f"{side}_wrist_to_hip_dist"] = distance(wrist, hip)
        out[f"{side}_wrist_to_trunk_dist"] = point_line_distance(wrist, spine_3, neck)
        out[f"{side}_hand_to_trunk_dist"] = point_line_distance(hand_center, spine_3, neck)
        out[f"{side}_hand_center_to_spine_dist"] = point_line_distance(hand_center, spine_3, neck)
        out[f"{side}_hand_height_relative_to_shoulder"] = project_on_axis(
            hand_center - shoulder, vertical
        )
        out[f"{side}_hand_forward_relative_to_trunk"] = project_on_axis(
            displacement_from_trunk, forward
        )
        out[f"{side}_hand_lateral_relative_to_trunk"] = project_on_axis(
            displacement_from_trunk, lateral
        )

        out[f"{side}_thumb_index_dist"] = distance(fingers[3], fingers[0])
        out[f"{side}_finger_spread"] = pairwise_max_distance(fingers)
        out[f"{side}_hand_shape_area"] = hand_shape_area(fingers)

        for metric in [
            f"{side}_wrist_speed",
            f"{side}_hand_center_speed",
            f"{side}_elbow_speed",
            f"{side}_shoulder_speed",
            f"{side}_elbow_angle",
            f"{side}_wrist_angle",
            f"{side}_shoulder_angle",
        ]:
            rolling_stats(out, metric, window_s)
        for metric in [
            f"{side}_elbow_angle_velocity",
            f"{side}_wrist_angle_velocity",
            f"{side}_shoulder_angle_velocity",
        ]:
            rolling_abs_mean(out, metric, window_s, f"{metric}_abs_mean_0p35s")
        out[f"{side}_hand_motion_energy_0p35s"] = (
            out.set_index(pd.to_timedelta(out["time_s"], unit="s"))[
                f"{side}_hand_center_speed"
            ]
            .pow(2)
            .rolling(f"{window_s}s", min_periods=1)
            .mean()
            .to_numpy()
        )

    out["left_right_wrist_dist"] = distance(
        joint_xyz(skeleton_df, "left_wrist"), joint_xyz(skeleton_df, "right_wrist")
    )
    out["left_right_hand_center_dist"] = distance(hand_centers["left"], hand_centers["right"])
    out["relative_hand_speed_diff"] = hand_speeds["left"] - hand_speeds["right"]
    out["relative_hand_speed_abs_diff"] = np.abs(out["relative_hand_speed_diff"])
    out["both_hands_active"] = (
        (hand_speeds["left"] > both_hands_active_threshold)
        & (hand_speeds["right"] > both_hands_active_threshold)
    ).astype(int)
    speed_sum = hand_speeds["left"] + hand_speeds["right"]
    out["dominant_hand_speed_ratio"] = np.where(
        np.isfinite(speed_sum) & (speed_sum > EPS),
        np.maximum(hand_speeds["left"], hand_speeds["right"]) / speed_sum,
        np.nan,
    )
    rolling_corr(
        out,
        "left_hand_center_speed",
        "right_hand_center_speed",
        window_s,
        "hands_moving_together_corr",
    )

    for metric in [
        "neck_speed",
        "neck_acceleration_mag",
        "trunk_angle",
        "neck_to_hip_dist",
        "left_right_wrist_dist",
        "left_right_hand_center_dist",
    ]:
        rolling_stats(out, metric, window_s)
    rolling_abs_mean(out, "trunk_angle_velocity", window_s, "trunk_angle_velocity_abs_mean_0p35s")
    out["trunk_stability_0p35s"] = 1.0 / (out["trunk_angle_std_0p35s"] + 1e-6)
    out["neck_stability_0p35s"] = 1.0 / (out["neck_speed_std_0p35s"] + 1e-6)

    return out


def output_path(participant_id, full_id, body_id, output_root=None):
    output_root = Path(output_root or config.SKELETON_UPPER_LIMB_KINEMATICS_FRAME_OUTPUT_ROOT)
    return output_root / participant_id / f"{full_id}_Body_{body_id}_skeleton_upper_limb_kinematics_frame_features.csv"


def process_recording(recording_dir, output_root=None, window_s=None, overwrite=False):
    window_s = float(window_s or config.SKELETON_UPPER_LIMB_KINEMATICS_WINDOW_S)
    participant_id, task_name, full_id = parse_recording_dir(recording_dir)
    skeleton_path = select_main_skeleton_path(recording_dir)
    if skeleton_path is None:
        return None

    body_id = body_id_from_path(skeleton_path)
    frame_path = output_path(participant_id, full_id, body_id, output_root=output_root)
    if frame_path.exists() and not overwrite:
        existing = pd.read_csv(frame_path, usecols=["time_s"])
        return {
            "participant_id": participant_id,
            "task_name": task_name,
            "full_id": full_id,
            "body_id": body_id,
            "skeleton_csv_path": str(skeleton_path),
            "frame_features_csv": str(frame_path),
            "row_count": int(len(existing)),
            "sensor_frame_count": int(len(existing)),
            "skipped_existing": True,
        }

    skeleton_df = read_skeleton_sensor_axis(skeleton_path)
    frame_df = extract_frame_features(
        skeleton_df,
        participant_id=participant_id,
        task_name=task_name,
        full_id=full_id,
        body_id=body_id,
        window_s=window_s,
    )

    frame_path.parent.mkdir(parents=True, exist_ok=True)
    frame_df.to_csv(frame_path, index=False)

    median_dt = frame_df["time_s"].diff().dropna().median() if len(frame_df) > 1 else np.nan
    return {
        "participant_id": participant_id,
        "task_name": task_name,
        "full_id": full_id,
        "body_id": body_id,
        "skeleton_csv_path": str(skeleton_path),
        "frame_features_csv": str(frame_path),
        "row_count": int(len(frame_df)),
        "sensor_frame_count": int(len(skeleton_df)),
        "median_dt_s": float(median_dt) if np.isfinite(median_dt) else np.nan,
        "has_video_columns": bool(any("video" in col for col in frame_df.columns)),
        "skipped_existing": False,
    }


def extract_skeleton_upper_limb_kinematics(
    input_root=ALIGNED_SENSOR_ROOT,
    output_root=None,
    participant=None,
    full_id=None,
    window_s=None,
    limit=None,
    overwrite=False,
):
    output_root = Path(output_root or config.SKELETON_UPPER_LIMB_KINEMATICS_FRAME_OUTPUT_ROOT)
    window_s = float(window_s or config.SKELETON_UPPER_LIMB_KINEMATICS_WINDOW_S)
    recording_dirs = iter_recording_dirs(input_root, participant=participant, full_id=full_id, limit=limit)
    if not recording_dirs:
        raise FileNotFoundError(f"No skeleton recording dirs found under {input_root}")

    records = []
    iterator = recording_dirs
    if tqdm is not None:
        iterator = tqdm(recording_dirs, desc="Extracting skeleton upper-limb kinematics", unit="recording")
    for recording_dir in iterator:
        row = process_recording(
            recording_dir,
            output_root=output_root,
            window_s=window_s,
            overwrite=overwrite,
        )
        if row is not None:
            records.append(row)

    index_df = pd.DataFrame(records)
    output_base = output_root.parent
    output_base.mkdir(parents=True, exist_ok=True)
    index_path = output_base / config.SKELETON_UPPER_LIMB_KINEMATICS_INDEX.name
    summary_path = output_base / config.SKELETON_UPPER_LIMB_KINEMATICS_SUMMARY.name
    index_df.to_csv(index_path, index=False)
    summary = {
        "input_root": str(input_root),
        "output_root": str(output_base),
        "frame_output_root": str(output_root),
        "index_csv": str(index_path),
        "recording_count": int(len(index_df)),
        "row_count": int(index_df["row_count"].sum()) if not index_df.empty else 0,
        "sensor_frame_count": int(index_df["sensor_frame_count"].sum()) if not index_df.empty else 0,
        "window_s": float(window_s),
        "time_axis": "original skeleton sensor time_s/frame_idx",
        "uses_video_time_or_frame": False,
        "both_hands_active_speed_threshold": float(
            config.SKELETON_BOTH_HANDS_ACTIVE_SPEED_THRESHOLD
        ),
        "joints": BODY_JOINTS,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return index_df, summary


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Extract 120Hz upper-limb kinematics from ZED 3D skeleton CSVs."
    )
    parser.add_argument("--input-root", type=Path, default=ALIGNED_SENSOR_ROOT)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=config.SKELETON_UPPER_LIMB_KINEMATICS_FRAME_OUTPUT_ROOT,
        help="Frame feature output root. Index and summary are written to its parent.",
    )
    parser.add_argument("--participant")
    parser.add_argument("--full-id", nargs="*")
    parser.add_argument("--window-s", type=float, default=config.SKELETON_UPPER_LIMB_KINEMATICS_WINDOW_S)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main():
    args = build_arg_parser().parse_args()
    index_df, summary = extract_skeleton_upper_limb_kinematics(
        input_root=args.input_root,
        output_root=args.output_root,
        participant=args.participant,
        full_id=args.full_id,
        window_s=args.window_s,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2))
    if not index_df.empty:
        print(index_df[["participant_id", "full_id", "body_id", "row_count"]].head().to_string(index=False))


if __name__ == "__main__":
    main()
