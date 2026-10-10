import argparse
import json
import os
from pathlib import Path
import random
import sys

import cv2
import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from upper_features_extraction import config

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


ALIGNED_SENSOR_ROOT = config.LOWER_FEATURE_OUTCOME_ROOT / "aligned_sensor_with_YOLO"

CORE_JOINTS = [
    "neck",
    "spine_1",
    "spine_2",
    "spine_3",
]
SIDE_JOINTS = [
    "shoulder",
    "elbow",
    "wrist",
    "hip",
    "hand_index_1",
    "hand_middle_4",
    "hand_pinky_1",
    "hand_thumb_4",
]
BODY_JOINTS = CORE_JOINTS + [f"{side}_{joint}" for side in ("left", "right") for joint in SIDE_JOINTS]

HAND_JOINTS = [
    "hand_index_1",
    "hand_middle_4",
    "hand_pinky_1",
    "hand_thumb_4",
]

BONES = [
    ("spine_3", "spine_2"),
    ("spine_2", "spine_1"),
    ("spine_1", "neck"),
    ("neck", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("left_wrist", "left_hand_index_1"),
    ("left_wrist", "left_hand_middle_4"),
    ("left_wrist", "left_hand_pinky_1"),
    ("left_wrist", "left_hand_thumb_4"),
    ("neck", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("right_wrist", "right_hand_index_1"),
    ("right_wrist", "right_hand_middle_4"),
    ("right_wrist", "right_hand_pinky_1"),
    ("right_wrist", "right_hand_thumb_4"),
    ("spine_3", "left_hip"),
    ("spine_3", "right_hip"),
    ("left_hip", "right_hip"),
]

BODY_MODEL_LIMBS = [
    ("neck", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("neck", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("spine_3", "left_hip"),
    ("spine_3", "right_hip"),
    ("left_hip", "right_hip"),
]

BODY_MODEL_TORSO = ("left_shoulder", "right_shoulder", "right_hip", "left_hip")

LABEL_JOINTS = {
    "neck": "neck",
    "spine_1": "spine 1",
    "spine_2": "spine 2",
    "spine_3": "spine 3",
    "left_shoulder": "L shoulder",
    "left_elbow": "L elbow",
    "left_wrist": "L wrist",
    "left_hip": "L hip",
    "left_hand_index_1": "L index",
    "left_hand_middle_4": "L middle",
    "left_hand_pinky_1": "L pinky",
    "left_hand_thumb_4": "L thumb",
    "right_shoulder": "R shoulder",
    "right_elbow": "R elbow",
    "right_wrist": "R wrist",
    "right_hip": "R hip",
    "right_hand_index_1": "R index",
    "right_hand_middle_4": "R middle",
    "right_hand_pinky_1": "R pinky",
    "right_hand_thumb_4": "R thumb",
}

WINDOW_COLUMNS = [
    "participant_id",
    "task_name",
    "full_id",
    "body_id",
    "start_time_s",
    "end_time_s",
    "start_frame",
    "end_frame",
    "samples",
    "body_center_speed_mean",
    "body_center_speed_max",
    "body_motion_energy_mean",
    "body_motion_energy_max",
    "upper_limb_motion_mean",
    "upper_limb_motion_max",
    "trunk_motion_mean",
    "left_hand_speed_mean",
    "left_hand_speed_max",
    "right_hand_speed_mean",
    "right_hand_speed_max",
    "hand_activity_balance",
    "max_joint_speed",
    "mean_joint_acceleration",
    "body_center_displacement_3d",
    "neck_displacement_3d",
    "left_wrist_displacement_3d",
    "right_wrist_displacement_3d",
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
    wanted = ["time_s", "video_time_s", "video_frame_idx"]
    for joint in BODY_JOINTS:
        wanted.extend([f"{joint}_x", f"{joint}_y", f"{joint}_z"])
    return [col for col in wanted if col in header]


def read_skeleton(path):
    usecols = skeleton_usecols(path)
    df = pd.read_csv(path, usecols=usecols)
    for column in usecols:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    if "video_time_s" in df.columns:
        df["feature_time_s"] = df["video_time_s"]
    elif "time_s" in df.columns:
        df["feature_time_s"] = df["time_s"]
    else:
        df["feature_time_s"] = np.arange(len(df), dtype=float) / 30.0

    if "video_frame_idx" in df.columns:
        df["feature_frame_idx"] = df["video_frame_idx"].round()
    else:
        df["feature_frame_idx"] = np.round(df["feature_time_s"] * 30.0)

    for joint in BODY_JOINTS:
        cols = [f"{joint}_{axis}" for axis in ("x", "y", "z")]
        if not set(cols).issubset(df.columns):
            continue
        values = df[cols].to_numpy(dtype=float)
        invalid = ~np.isfinite(values).all(axis=1) | np.all(np.isclose(values, -1.0, equal_nan=False), axis=1)
        for col in cols:
            df.loc[invalid, col] = np.nan

    df = df.dropna(subset=["feature_time_s"]).sort_values("feature_time_s").reset_index(drop=True)
    return df


def joint_xyz(df, joint):
    cols = [f"{joint}_{axis}" for axis in ("x", "y", "z")]
    if not set(cols).issubset(df.columns):
        return np.full((len(df), 3), np.nan, dtype=float)
    return df[cols].to_numpy(dtype=float)


def valid_xyz(values):
    return np.isfinite(values).all(axis=1)


def point_speed(values, time_s):
    values = np.asarray(values, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    dt = np.diff(time_s, prepend=np.nan)
    dv = np.full_like(values, np.nan, dtype=float)
    if len(values) > 1:
        dv[1:] = values[1:] - values[:-1]
    speed = np.full(len(values), np.nan, dtype=float)
    valid = valid_xyz(values) & np.isfinite(dv).all(axis=1) & np.isfinite(dt) & (dt > 0)
    speed[valid] = np.linalg.norm(dv[valid], axis=1) / dt[valid]
    return speed


def point_acceleration(speed, time_s):
    time_s = np.asarray(time_s, dtype=float)
    ds = np.full(len(speed), np.nan, dtype=float)
    if len(speed) > 1:
        ds[1:] = speed[1:] - speed[:-1]
    dt = np.diff(time_s, prepend=np.nan)
    acceleration = np.full(len(speed), np.nan, dtype=float)
    valid = np.isfinite(ds) & np.isfinite(dt) & (dt > 0)
    acceleration[valid] = np.abs(ds[valid]) / dt[valid]
    return acceleration


def median_points(points):
    stack = np.stack(points, axis=2)
    return np.nanmedian(stack, axis=2)


def distance_between(df, joint_a, joint_b):
    a = joint_xyz(df, joint_a)
    b = joint_xyz(df, joint_b)
    out = np.full(len(df), np.nan, dtype=float)
    valid = valid_xyz(a) & valid_xyz(b)
    out[valid] = np.linalg.norm(a[valid] - b[valid], axis=1)
    return out


def extract_frame_features(skeleton_df, participant_id, task_name, full_id, body_id):
    time_s = skeleton_df["feature_time_s"].to_numpy(dtype=float)
    frame_idx = skeleton_df["feature_frame_idx"].to_numpy(dtype=float)

    joint_speeds = {}
    joint_accels = {}
    for joint in BODY_JOINTS:
        speed = point_speed(joint_xyz(skeleton_df, joint), time_s)
        joint_speeds[joint] = speed
        joint_accels[joint] = point_acceleration(speed, time_s)

    torso_points = [joint_xyz(skeleton_df, joint) for joint in ["neck", "spine_1", "spine_2", "spine_3", "left_hip", "right_hip"]]
    body_center = median_points(torso_points)
    body_center_speed = point_speed(body_center, time_s)

    left_hand_center = median_points([joint_xyz(skeleton_df, f"left_{joint}") for joint in HAND_JOINTS])
    right_hand_center = median_points([joint_xyz(skeleton_df, f"right_{joint}") for joint in HAND_JOINTS])
    left_hand_speed = point_speed(left_hand_center, time_s)
    right_hand_speed = point_speed(right_hand_center, time_s)

    speed_matrix = np.column_stack([joint_speeds[joint] for joint in BODY_JOINTS])
    accel_matrix = np.column_stack([joint_accels[joint] for joint in BODY_JOINTS])
    arm_joints = [
        joint
        for joint in BODY_JOINTS
        if any(joint.startswith(prefix) for prefix in ("left_shoulder", "left_elbow", "left_wrist", "left_hand", "right_shoulder", "right_elbow", "right_wrist", "right_hand"))
    ]
    trunk_joints = ["neck", "spine_1", "spine_2", "spine_3", "left_hip", "right_hip"]

    out = pd.DataFrame(
        {
            "participant_id": participant_id,
            "task_name": task_name,
            "full_id": full_id,
            "body_id": body_id,
            "time_s": time_s,
            "video_frame_idx": frame_idx,
            "body_center_x": body_center[:, 0],
            "body_center_y": body_center[:, 1],
            "body_center_z": body_center[:, 2],
            "neck_x": joint_xyz(skeleton_df, "neck")[:, 0],
            "neck_y": joint_xyz(skeleton_df, "neck")[:, 1],
            "neck_z": joint_xyz(skeleton_df, "neck")[:, 2],
            "left_wrist_x": joint_xyz(skeleton_df, "left_wrist")[:, 0],
            "left_wrist_y": joint_xyz(skeleton_df, "left_wrist")[:, 1],
            "left_wrist_z": joint_xyz(skeleton_df, "left_wrist")[:, 2],
            "right_wrist_x": joint_xyz(skeleton_df, "right_wrist")[:, 0],
            "right_wrist_y": joint_xyz(skeleton_df, "right_wrist")[:, 1],
            "right_wrist_z": joint_xyz(skeleton_df, "right_wrist")[:, 2],
            "body_center_speed_3d": body_center_speed,
            "body_motion_energy_3d": np.nanmean(speed_matrix, axis=1),
            "upper_limb_motion_3d": np.nanmean(np.column_stack([joint_speeds[joint] for joint in arm_joints]), axis=1),
            "trunk_motion_3d": np.nanmean(np.column_stack([joint_speeds[joint] for joint in trunk_joints]), axis=1),
            "left_hand_center_x": left_hand_center[:, 0],
            "left_hand_center_y": left_hand_center[:, 1],
            "left_hand_center_z": left_hand_center[:, 2],
            "left_hand_speed_3d": left_hand_speed,
            "right_hand_center_x": right_hand_center[:, 0],
            "right_hand_center_y": right_hand_center[:, 1],
            "right_hand_center_z": right_hand_center[:, 2],
            "right_hand_speed_3d": right_hand_speed,
            "max_joint_speed_3d": np.nanmax(speed_matrix, axis=1),
            "mean_joint_acceleration_3d": np.nanmean(accel_matrix, axis=1),
            "shoulder_width_3d": distance_between(skeleton_df, "left_shoulder", "right_shoulder"),
            "hip_width_3d": distance_between(skeleton_df, "left_hip", "right_hip"),
            "left_forearm_length_3d": distance_between(skeleton_df, "left_elbow", "left_wrist"),
            "right_forearm_length_3d": distance_between(skeleton_df, "right_elbow", "right_wrist"),
        }
    )
    return out


def displacement(series_x, series_y, series_z):
    values = np.column_stack([series_x, series_y, series_z]).astype(float)
    valid = valid_xyz(values)
    if valid.sum() < 2:
        return np.nan
    first = values[np.flatnonzero(valid)[0]]
    last = values[np.flatnonzero(valid)[-1]]
    return float(np.linalg.norm(last - first))


def extract_window_features(frame_df, window_s=1.0, stride_s=0.5):
    if frame_df.empty:
        return pd.DataFrame(columns=WINDOW_COLUMNS)

    rows = []
    start = float(frame_df["time_s"].min())
    end = float(frame_df["time_s"].max())
    current = start
    while current + window_s <= end + 1e-9:
        window_end = current + window_s
        win = frame_df[(frame_df["time_s"] >= current) & (frame_df["time_s"] < window_end)]
        current += stride_s
        if len(win) < 2:
            continue

        left_mean = float(np.nanmean(win["left_hand_speed_3d"]))
        right_mean = float(np.nanmean(win["right_hand_speed_3d"]))
        denom = left_mean + right_mean
        balance = (left_mean - right_mean) / denom if np.isfinite(denom) and denom > 1e-9 else np.nan

        rows.append(
            {
                "participant_id": win["participant_id"].iloc[0],
                "task_name": win["task_name"].iloc[0],
                "full_id": win["full_id"].iloc[0],
                "body_id": win["body_id"].iloc[0],
                "start_time_s": float(win["time_s"].iloc[0]),
                "end_time_s": float(win["time_s"].iloc[-1]),
                "start_frame": int(round(win["video_frame_idx"].iloc[0])),
                "end_frame": int(round(win["video_frame_idx"].iloc[-1])),
                "samples": int(len(win)),
                "body_center_speed_mean": float(np.nanmean(win["body_center_speed_3d"])),
                "body_center_speed_max": float(np.nanmax(win["body_center_speed_3d"])),
                "body_motion_energy_mean": float(np.nanmean(win["body_motion_energy_3d"])),
                "body_motion_energy_max": float(np.nanmax(win["body_motion_energy_3d"])),
                "upper_limb_motion_mean": float(np.nanmean(win["upper_limb_motion_3d"])),
                "upper_limb_motion_max": float(np.nanmax(win["upper_limb_motion_3d"])),
                "trunk_motion_mean": float(np.nanmean(win["trunk_motion_3d"])),
                "left_hand_speed_mean": left_mean,
                "left_hand_speed_max": float(np.nanmax(win["left_hand_speed_3d"])),
                "right_hand_speed_mean": right_mean,
                "right_hand_speed_max": float(np.nanmax(win["right_hand_speed_3d"])),
                "hand_activity_balance": float(balance) if np.isfinite(balance) else np.nan,
                "max_joint_speed": float(np.nanmax(win["max_joint_speed_3d"])),
                "mean_joint_acceleration": float(np.nanmean(win["mean_joint_acceleration_3d"])),
                "body_center_displacement_3d": displacement(win["body_center_x"], win["body_center_y"], win["body_center_z"]),
                "neck_displacement_3d": displacement(win["neck_x"], win["neck_y"], win["neck_z"]),
                "left_wrist_displacement_3d": displacement(win["left_wrist_x"], win["left_wrist_y"], win["left_wrist_z"]),
                "right_wrist_displacement_3d": displacement(win["right_wrist_x"], win["right_wrist_y"], win["right_wrist_z"]),
            }
        )
    return pd.DataFrame(rows, columns=WINDOW_COLUMNS)


def skeleton_quality(path):
    df = read_skeleton(path)
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
    scored = sorted(scored, key=lambda item: item[0], reverse=True)
    return scored[0][1]


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


def output_paths(participant_id, full_id, body_id, output_root=config.SKELETON_3D_MOTION_OUTPUT_ROOT):
    frame_path = config.SKELETON_3D_FRAME_OUTPUT_ROOT / participant_id / f"{full_id}_Body_{body_id}_skeleton_3d_frame_features.csv"
    window_path = config.SKELETON_3D_WINDOW_OUTPUT_ROOT / participant_id / f"{full_id}_Body_{body_id}_skeleton_3d_window_features.csv"
    return frame_path, window_path


def process_recording(recording_dir, window_s=1.0, stride_s=0.5, overwrite=False):
    participant_id, task_name, full_id = parse_recording_dir(recording_dir)
    skeleton_path = select_main_skeleton_path(recording_dir)
    if skeleton_path is None:
        return None

    body_id = body_id_from_path(skeleton_path)
    frame_path, window_path = output_paths(participant_id, full_id, body_id)
    if frame_path.exists() and window_path.exists() and not overwrite:
        return {
            "participant_id": participant_id,
            "task_name": task_name,
            "full_id": full_id,
            "body_id": body_id,
            "skeleton_csv_path": str(skeleton_path),
            "frame_features_csv": str(frame_path),
            "window_features_csv": str(window_path),
            "frame_count": len(pd.read_csv(frame_path, usecols=["time_s"])),
            "window_count": len(pd.read_csv(window_path, usecols=["start_time_s"])),
            "skipped_existing": True,
        }

    skeleton_df = read_skeleton(skeleton_path)
    frame_df = extract_frame_features(skeleton_df, participant_id, task_name, full_id, body_id)
    window_df = extract_window_features(frame_df, window_s=window_s, stride_s=stride_s)

    frame_path.parent.mkdir(parents=True, exist_ok=True)
    window_path.parent.mkdir(parents=True, exist_ok=True)
    frame_df.to_csv(frame_path, index=False)
    window_df.to_csv(window_path, index=False)

    return {
        "participant_id": participant_id,
        "task_name": task_name,
        "full_id": full_id,
        "body_id": body_id,
        "skeleton_csv_path": str(skeleton_path),
        "frame_features_csv": str(frame_path),
        "window_features_csv": str(window_path),
        "frame_count": int(len(frame_df)),
        "window_count": int(len(window_df)),
        "mean_body_motion_energy": float(window_df["body_motion_energy_mean"].mean()) if not window_df.empty else np.nan,
        "mean_upper_limb_motion": float(window_df["upper_limb_motion_mean"].mean()) if not window_df.empty else np.nan,
        "skipped_existing": False,
    }


def extract_skeleton_3d_motion_features(
    input_root=ALIGNED_SENSOR_ROOT,
    participant=None,
    full_id=None,
    window_s=1.0,
    stride_s=0.5,
    limit=None,
    overwrite=False,
):
    recording_dirs = iter_recording_dirs(input_root, participant=participant, full_id=full_id, limit=limit)
    if not recording_dirs:
        raise FileNotFoundError(f"No skeleton recording dirs found under {input_root}")

    config.SKELETON_3D_MOTION_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    iterator = recording_dirs
    if tqdm is not None:
        iterator = tqdm(recording_dirs, desc="Extracting skeleton 3D motion", unit="recording")
    for recording_dir in iterator:
        row = process_recording(recording_dir, window_s=window_s, stride_s=stride_s, overwrite=overwrite)
        if row is not None:
            records.append(row)

    index_df = pd.DataFrame(records)
    index_df.to_csv(config.SKELETON_3D_MOTION_INDEX, index=False)
    summary = {
        "input_root": str(input_root),
        "output_root": str(config.SKELETON_3D_MOTION_OUTPUT_ROOT),
        "index_csv": str(config.SKELETON_3D_MOTION_INDEX),
        "recording_count": int(len(index_df)),
        "frame_count": int(index_df["frame_count"].sum()) if not index_df.empty else 0,
        "window_count": int(index_df["window_count"].sum()) if not index_df.empty else 0,
        "body_joints": BODY_JOINTS,
        "window_s": float(window_s),
        "stride_s": float(stride_s),
    }
    with open(config.SKELETON_3D_MOTION_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)
    return index_df, summary


def prepare_render_points(skeleton_df, start_s=0.0, duration_s=30.0, render_stride=3):
    end_s = start_s + duration_s if duration_s is not None else np.inf
    part = skeleton_df[(skeleton_df["feature_time_s"] >= start_s) & (skeleton_df["feature_time_s"] <= end_s)].copy()
    part = part.iloc[:: max(1, int(render_stride))].reset_index(drop=True)
    return part


def set_equal_3d_limits(ax, coords, padding=0.12, zoom=1.0, min_radius=0.28):
    valid = coords[np.isfinite(coords).all(axis=1)]
    if len(valid) == 0:
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        return
    mins = valid.min(axis=0)
    maxs = valid.max(axis=0)
    center = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0 + padding)
    radius = max(radius / max(float(zoom), 1e-6), min_radius)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def style_compact_3d_axes(ax):
    ax.set_xlabel("x", fontsize=14, labelpad=-2)
    ax.set_ylabel("y", fontsize=14, labelpad=-2)
    ax.set_zlabel("z", fontsize=14, labelpad=-2)
    ax.tick_params(axis="both", which="major", labelsize=11, pad=-1)
    ax.xaxis.pane.set_alpha(0.04)
    ax.yaxis.pane.set_alpha(0.04)
    ax.zaxis.pane.set_alpha(0.04)


def frame_coords(row):
    coords = []
    for joint in BODY_JOINTS:
        values = np.array([row.get(f"{joint}_{axis}", np.nan) for axis in ("x", "y", "z")], dtype=float)
        if np.isfinite(values).all():
            coords.append(values)
    if not coords:
        return np.empty((0, 3), dtype=float)
    return np.vstack(coords)


def segment_color(joint_a, joint_b):
    if joint_a.startswith("left_") or joint_b.startswith("left_"):
        return "#2ca25f"
    if joint_a.startswith("right_") or joint_b.startswith("right_"):
        return "#de2d26"
    return "#3182bd"


def estimate_body_scale(points):
    left_shoulder = points.get("left_shoulder")
    right_shoulder = points.get("right_shoulder")
    if left_shoulder is not None and right_shoulder is not None:
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        if np.isfinite(shoulder_width) and shoulder_width > 1e-6:
            return float(shoulder_width)
    coords = np.vstack(list(points.values())) if points else np.empty((0, 3))
    if len(coords) >= 2:
        span = np.nanmax(coords, axis=0) - np.nanmin(coords, axis=0)
        scale = float(np.nanmax(span))
        if np.isfinite(scale) and scale > 1e-6:
            return scale * 0.22
    return 0.25


def draw_head_model(ax, points):
    neck = points.get("neck")
    spine_1 = points.get("spine_1")
    if neck is None:
        return

    scale = estimate_body_scale(points)
    radius = max(scale * 0.22, 0.035)
    if spine_1 is not None:
        up = neck - spine_1
        norm = np.linalg.norm(up)
        if np.isfinite(norm) and norm > 1e-6:
            up = up / norm
        else:
            up = np.array([0.0, 1.0, 0.0])
    else:
        up = np.array([0.0, 1.0, 0.0])
    center = neck + up * radius * 1.25

    u = np.linspace(0, 2 * np.pi, 12)
    v = np.linspace(0, np.pi, 8)
    xs = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    ys = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    zs = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, color="#9ecae1", alpha=0.35, linewidth=0, shade=True)


def draw_torso_model(ax, points):
    torso = [points.get(joint) for joint in BODY_MODEL_TORSO]
    if any(point is None for point in torso):
        return

    poly = Poly3DCollection(
        [torso],
        facecolors="#9ecae1",
        edgecolors="#3182bd",
        linewidths=1.5,
        alpha=0.22,
    )
    ax.add_collection3d(poly)


def draw_limb_model(ax, points):
    for joint_a, joint_b in BODY_MODEL_LIMBS:
        pa = points.get(joint_a)
        pb = points.get(joint_b)
        if pa is None or pb is None:
            continue
        color = segment_color(joint_a, joint_b)
        ax.plot(
            [pa[0], pb[0]],
            [pa[1], pb[1]],
            [pa[2], pb[2]],
            color=color,
            linewidth=8.0,
            alpha=0.28,
            solid_capstyle="round",
        )


def draw_body_model(ax, points):
    draw_torso_model(ax, points)
    draw_limb_model(ax, points)
    draw_head_model(ax, points)


def local_render_coords(render_df, row_idx, context=8):
    start = max(0, row_idx - context)
    end = min(len(render_df), row_idx + context + 1)
    coords = [frame_coords(render_df.iloc[i]) for i in range(start, end)]
    coords = [item for item in coords if len(item)]
    if not coords:
        return np.empty((0, 3), dtype=float)
    return np.vstack(coords)


def render_skeleton_3d_video(
    skeleton_path,
    output_path,
    full_id=None,
    duration_s=30.0,
    start_s=0.0,
    render_stride=3,
    output_fps=15,
    width=1280,
    height=960,
    zoom=2.2,
    show_labels=True,
    show_body_model=True,
):
    skeleton_df = read_skeleton(skeleton_path)
    render_df = prepare_render_points(skeleton_df, start_s=start_s, duration_s=duration_s, render_stride=render_stride)
    if render_df.empty:
        raise ValueError(f"No renderable skeleton rows in {skeleton_path}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.9)
    ax.set_position([0.0, 0.0, 1.0, 0.9])

    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not create output video: {output_path}")

    title = full_id or Path(skeleton_path).parent.name
    for row_idx, row in render_df.iterrows():
        ax.clear()
        ax.view_init(elev=18, azim=-70)
        set_equal_3d_limits(ax, local_render_coords(render_df, row_idx), padding=0.06, zoom=zoom, min_radius=0.22)
        style_compact_3d_axes(ax)
        ax.set_title(
            f"{title} | frame {int(row['feature_frame_idx'])} | t={row['feature_time_s']:.2f}s",
            fontsize=34,
            fontweight="bold",
            pad=8,
        )
        ax.grid(True, alpha=0.25)

        points = {}
        for joint in BODY_JOINTS:
            values = np.array([row.get(f"{joint}_{axis}", np.nan) for axis in ("x", "y", "z")], dtype=float)
            if np.isfinite(values).all():
                points[joint] = values
                color = "#2ca25f" if joint.startswith("left_") else "#de2d26" if joint.startswith("right_") else "#3182bd"
                size = 62 if joint in LABEL_JOINTS else 36
                ax.scatter(values[0], values[1], values[2], s=size, color=color, depthshade=True)
                if show_labels and joint in LABEL_JOINTS:
                    ax.text(
                        values[0],
                        values[1],
                        values[2],
                        f" {LABEL_JOINTS[joint]}",
                        fontsize=11,
                        color=color,
                        zorder=10,
                    )

        if show_body_model:
            draw_body_model(ax, points)

        for a, b in BONES:
            if a not in points or b not in points:
                continue
            pa = points[a]
            pb = points[b]
            color = segment_color(a, b)
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], [pa[2], pb[2]], color=color, linewidth=3.2)

        fig.canvas.draw()
        image = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if image.shape[1] != width or image.shape[0] != height:
            image = cv2.resize(image, (width, height))
        writer.write(image)

    writer.release()
    plt.close(fig)
    return output_path


def choose_demo_recordings(input_root=ALIGNED_SENSOR_ROOT, count=3, seed=42, full_id=None):
    if full_id:
        return iter_recording_dirs(input_root, full_id=full_id)
    dirs = iter_recording_dirs(input_root)
    if len(dirs) <= count:
        return dirs
    rng = random.Random(seed)
    return rng.sample(dirs, count)


def render_demo_videos(
    input_root=ALIGNED_SENSOR_ROOT,
    output_root=config.SKELETON_3D_DEMO_OUTPUT_ROOT,
    count=3,
    seed=42,
    full_id=None,
    duration_s=30.0,
    start_s=0.0,
    render_stride=3,
    output_fps=15,
    zoom=2.2,
    show_labels=True,
    show_body_model=True,
):
    recording_dirs = choose_demo_recordings(input_root, count=count, seed=seed, full_id=full_id)
    rows = []
    iterator = recording_dirs
    if tqdm is not None:
        iterator = tqdm(recording_dirs, desc="Rendering 3D skeleton demos", unit="video")
    for recording_dir in iterator:
        participant_id, task_name, current_full_id = parse_recording_dir(recording_dir)
        skeleton_path = select_main_skeleton_path(recording_dir)
        if skeleton_path is None:
            continue
        body_id = body_id_from_path(skeleton_path)
        label_tag = "labels" if show_labels else "nolabels"
        model_tag = "bodymodel" if show_body_model else "bones"
        output_path = Path(output_root) / (
            f"{current_full_id}_Body_{body_id}_skeleton_3d_demo_{int(duration_s)}s_zoom{zoom:g}_{label_tag}_{model_tag}.mp4"
        )
        render_skeleton_3d_video(
            skeleton_path,
            output_path,
            full_id=current_full_id,
            duration_s=duration_s,
            start_s=start_s,
            render_stride=render_stride,
            output_fps=output_fps,
            zoom=zoom,
            show_labels=show_labels,
            show_body_model=show_body_model,
        )
        rows.append(
            {
                "participant_id": participant_id,
                "task_name": task_name,
                "full_id": current_full_id,
                "body_id": body_id,
                "skeleton_csv_path": str(skeleton_path),
                "demo_video_path": str(output_path),
            }
        )
    return pd.DataFrame(rows)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Extract 3D skeleton body motion features and render 3D skeleton demos.")
    parser.add_argument("--input-root", default=str(ALIGNED_SENSOR_ROOT))
    parser.add_argument("--participant", default=None)
    parser.add_argument("--full-id", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--window-s", type=float, default=1.0)
    parser.add_argument("--stride-s", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--demo-only", action="store_true")
    parser.add_argument("--render-demo", action="store_true")
    parser.add_argument("--demo-count", type=int, default=3)
    parser.add_argument("--demo-seed", type=int, default=42)
    parser.add_argument("--demo-duration-s", type=float, default=30.0)
    parser.add_argument("--demo-start-s", type=float, default=0.0)
    parser.add_argument("--demo-stride", type=int, default=3)
    parser.add_argument("--demo-fps", type=int, default=15)
    parser.add_argument("--demo-output-root", default=str(config.SKELETON_3D_DEMO_OUTPUT_ROOT))
    parser.add_argument("--demo-zoom", type=float, default=2.2)
    parser.add_argument("--demo-labels", dest="demo_labels", action="store_true", default=True)
    parser.add_argument("--no-demo-labels", dest="demo_labels", action="store_false")
    parser.add_argument("--body-model", dest="body_model", action="store_true", default=True)
    parser.add_argument("--no-body-model", dest="body_model", action="store_false")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    input_root = Path(args.input_root)

    if not args.demo_only:
        index_df, summary = extract_skeleton_3d_motion_features(
            input_root=input_root,
            participant=args.participant,
            full_id=args.full_id,
            window_s=args.window_s,
            stride_s=args.stride_s,
            limit=args.limit,
            overwrite=args.overwrite,
        )
        print("Skeleton 3D motion feature extraction completed.")
        print(json.dumps(summary, indent=2))
        if not index_df.empty:
            print("\nTop recordings by upper-limb motion:")
            cols = ["full_id", "body_id", "frame_count", "window_count", "mean_upper_limb_motion"]
            available_cols = [col for col in cols if col in index_df.columns]
            print(index_df.sort_values("mean_upper_limb_motion", ascending=False)[available_cols].head(20).to_string(index=False))

    if args.render_demo or args.demo_only:
        demo_df = render_demo_videos(
            input_root=input_root,
            output_root=Path(args.demo_output_root),
            count=args.demo_count,
            seed=args.demo_seed,
            full_id=args.full_id,
            duration_s=args.demo_duration_s,
            start_s=args.demo_start_s,
            render_stride=args.demo_stride,
            output_fps=args.demo_fps,
            zoom=args.demo_zoom,
            show_labels=args.demo_labels,
            show_body_model=args.body_model,
        )
        print("\nRendered 3D skeleton demo videos:")
        print(demo_df.to_string(index=False))


if __name__ == "__main__":
    main()
