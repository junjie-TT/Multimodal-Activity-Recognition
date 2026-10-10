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


SKIP_TASKS = {"Eye1", "Eye2", "Recalibration1"}

# Mapping inferred for the aligned Movella DOT CSV exports. The raw export
# filenames use D422... while some older notes include a DOTD422... prefix.
DOT_TO_BODY = {
    "D422CD008603": "left_forearm",
    "DOTD422CD008603": "left_forearm",
    "D422CD008CC7": "right_hand",
    "DOTD422CD008CC7": "right_hand",
    "D422CD009F54": "left_hand",
    "DOTD422CD009F54": "left_hand",
    "D422CD009F5B": "right_forearm",
    "DOTD422CD009F5B": "right_forearm",
    "D422CD009F60": "right_upper_arm",
    "DOTD422CD009F60": "right_upper_arm",
    "D422CD00A220": "left_upper_arm",
    "DOTD422CD00A220": "left_upper_arm",
    "D422CD00A795": "trunk",
    "DOTD422CD00A795": "trunk",
}

BODY_ORDER = [
    "trunk",
    "left_upper_arm",
    "left_forearm",
    "left_hand",
    "right_upper_arm",
    "right_forearm",
    "right_hand",
]

JOINT_DEFS = [
    ("left_shoulder", "trunk", "left_upper_arm"),
    ("left_elbow", "left_upper_arm", "left_forearm"),
    ("left_wrist", "left_forearm", "left_hand"),
    ("right_shoulder", "trunk", "right_upper_arm"),
    ("right_elbow", "right_upper_arm", "right_forearm"),
    ("right_wrist", "right_forearm", "right_hand"),
]

QUAT_COLS = ("quat_w", "quat_x", "quat_y", "quat_z")
GYRO_COLS = ("gyro_x", "gyro_y", "gyro_z")


def parse_recording_dir(recording_dir):
    recording_dir = Path(recording_dir)
    participant_id = recording_dir.parent.name
    full_id = recording_dir.name
    prefix = f"{participant_id}_"
    task_name = full_id[len(prefix):] if full_id.startswith(prefix) else full_id
    return participant_id, task_name, full_id


def dot_id_from_path(path):
    stem = Path(path).stem
    return stem.replace("movella__DOT_", "").replace("movella__", "")


def body_from_path(path):
    return DOT_TO_BODY.get(dot_id_from_path(path))


def iter_recording_dirs(input_root=config.VIDEO_ALIGNED_SENSOR_ROOT, participant=None, full_id=None, include_calibration=False, limit=None):
    input_root = Path(input_root)
    dirs = sorted(path for path in input_root.glob("*/*") if path.is_dir())

    if participant:
        dirs = [path for path in dirs if path.parent.name == participant]

    if full_id:
        wanted = set(full_id)
        dirs = [path for path in dirs if path.name in wanted]

    if not include_calibration:
        dirs = [
            path
            for path in dirs
            if parse_recording_dir(path)[1] not in SKIP_TASKS
        ]

    dirs = [path for path in dirs if any(path.glob("movella__DOT*.csv"))]

    if limit is not None:
        dirs = dirs[:limit]
    return dirs


def numeric_column(df, column):
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def read_imu_csv(path):
    df = pd.read_csv(path)
    for column in ["frame_idx", "time_s", *QUAT_COLS, *GYRO_COLS]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "time_s" not in df.columns:
        raise ValueError(f"Missing time_s in {path}")

    df = df.dropna(subset=["time_s"]).copy()
    df = df.sort_values("time_s").reset_index(drop=True)
    df["feature_time_s"] = df["time_s"].astype(float)
    if "frame_idx" in df.columns:
        df["feature_frame_idx"] = numeric_column(df, "frame_idx").round()
    else:
        df["feature_frame_idx"] = np.round(df["feature_time_s"] * 120.0)
    return df


def read_recording_imus(recording_dir):
    sensors = {}
    for path in sorted(Path(recording_dir).glob("movella__DOT*.csv")):
        body = body_from_path(path)
        if body is None:
            continue
        df = read_imu_csv(path)
        if not all(col in df.columns for col in QUAT_COLS):
            continue
        sensors[body] = {
            "path": path,
            "df": df,
        }
    return sensors


def reference_body(sensors):
    for body in ["trunk", "left_upper_arm", "right_upper_arm", "left_forearm", "right_forearm", "left_hand", "right_hand"]:
        if body in sensors:
            return body
    return next(iter(sensors.keys()))


def normalize_quat_wxyz(q):
    q = np.asarray(q, dtype=float)
    norms = np.linalg.norm(q, axis=1)
    valid = np.isfinite(q).all(axis=1) & (norms > 1e-8)
    out = np.full_like(q, np.nan, dtype=float)
    out[valid] = q[valid] / norms[valid, None]
    return out, valid


def enforce_quat_continuity(q):
    q = np.asarray(q, dtype=float).copy()
    for idx in range(1, len(q)):
        if not np.isfinite(q[idx]).all() or not np.isfinite(q[idx - 1]).all():
            continue
        if float(np.dot(q[idx - 1], q[idx])) < 0:
            q[idx] *= -1.0
    return q


def interp_values(source_time, target_time, values):
    source_time = np.asarray(source_time, dtype=float)
    target_time = np.asarray(target_time, dtype=float)
    values = np.asarray(values, dtype=float)
    out = np.full((len(target_time), values.shape[1]), np.nan, dtype=float)
    valid_time = np.isfinite(source_time)

    for col_idx in range(values.shape[1]):
        col = values[:, col_idx]
        valid = valid_time & np.isfinite(col)
        if int(valid.sum()) < 2:
            continue
        out[:, col_idx] = np.interp(target_time, source_time[valid], col[valid], left=np.nan, right=np.nan)
    return out


def align_sensor_to_time(sensor_df, target_time):
    source_time = sensor_df["feature_time_s"].to_numpy(dtype=float)
    quat = sensor_df[list(QUAT_COLS)].to_numpy(dtype=float)
    quat, quat_valid = normalize_quat_wxyz(quat)
    quat[~quat_valid] = np.nan
    quat = enforce_quat_continuity(quat)
    quat_interp = interp_values(source_time, target_time, quat)
    quat_interp, quat_interp_valid = normalize_quat_wxyz(quat_interp)

    gyro = sensor_df.reindex(columns=list(GYRO_COLS)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    gyro_interp = interp_values(source_time, target_time, gyro)

    return {
        "quat": quat_interp,
        "quat_valid": quat_interp_valid,
        "gyro": gyro_interp,
    }


def quat_conjugate(q):
    out = np.asarray(q, dtype=float).copy()
    out[:, 1:] *= -1.0
    return out


def quat_multiply(q1, q2):
    q1 = np.asarray(q1, dtype=float)
    q2 = np.asarray(q2, dtype=float)
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

    return np.column_stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_relative(parent_q, child_q):
    rel = quat_multiply(quat_conjugate(parent_q), child_q)
    rel, valid = normalize_quat_wxyz(rel)
    rel[~valid] = np.nan
    return rel, valid


def quat_angle_deg(q):
    q, valid = normalize_quat_wxyz(q)
    out = np.full(len(q), np.nan, dtype=float)
    w = np.clip(np.abs(q[:, 0]), -1.0, 1.0)
    out[valid] = np.degrees(2.0 * np.arccos(w[valid]))
    return out


def quat_to_euler_deg(q):
    q, valid = normalize_quat_wxyz(q)
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]
    roll = np.full(len(q), np.nan, dtype=float)
    pitch = np.full(len(q), np.nan, dtype=float)
    yaw = np.full(len(q), np.nan, dtype=float)
    if valid.any():
        t0 = 2.0 * (w * x + y * z)
        t1 = 1.0 - 2.0 * (x * x + y * y)
        t2 = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
        t3 = 2.0 * (w * z + x * y)
        t4 = 1.0 - 2.0 * (y * y + z * z)
        roll[valid] = np.degrees(np.arctan2(t0[valid], t1[valid]))
        pitch[valid] = np.degrees(np.arcsin(t2[valid]))
        yaw[valid] = np.degrees(np.arctan2(t3[valid], t4[valid]))
    return roll, pitch, yaw


def vector_norm(values):
    values = np.asarray(values, dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    valid = np.isfinite(values).all(axis=1)
    out[valid] = np.linalg.norm(values[valid], axis=1)
    return out


def finite_difference(values, times):
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    dt = np.diff(times, prepend=np.nan)
    diff = np.diff(values, prepend=np.nan)
    valid = np.isfinite(diff) & np.isfinite(dt) & (dt > 0)
    out[valid] = diff[valid] / dt[valid]
    return out


def rolling_series(values, times):
    return pd.Series(values, index=pd.to_timedelta(times, unit="s"))


def rolling_mean(values, times, window_s):
    return rolling_series(values, times).rolling(f"{int(round(window_s * 1000))}ms", min_periods=1).mean().to_numpy(dtype=float)


def rolling_std(values, times, window_s):
    return rolling_series(values, times).rolling(f"{int(round(window_s * 1000))}ms", min_periods=1).std().fillna(0.0).to_numpy(dtype=float)


def rolling_range(values, times, window_s):
    def finite_range(window_values):
        window_values = window_values[np.isfinite(window_values)]
        if len(window_values) == 0:
            return np.nan
        return float(np.nanmax(window_values) - np.nanmin(window_values))

    return (
        rolling_series(values, times)
        .rolling(f"{int(round(window_s * 1000))}ms", min_periods=1)
        .apply(finite_range, raw=True)
        .to_numpy(dtype=float)
    )


def build_frame_features(recording_dir, window_s=config.IMU_JOINT_KINEMATICS_WINDOW_S):
    participant_id, task_name, full_id = parse_recording_dir(recording_dir)
    sensors = read_recording_imus(recording_dir)
    if not sensors:
        raise ValueError(f"No mapped Movella IMU CSV files found in {recording_dir}")

    ref_body = reference_body(sensors)
    ref_df = sensors[ref_body]["df"]
    times = ref_df["feature_time_s"].to_numpy(dtype=float)
    frame_idx = ref_df["feature_frame_idx"].to_numpy(dtype=float)

    columns = {
        "participant_id": participant_id,
        "task_name": task_name,
        "full_id": full_id,
        "frame_idx": frame_idx,
        "time_s": numeric_column(ref_df, "time_s"),
        "reference_body": ref_body,
    }

    aligned = {
        body: align_sensor_to_time(sensor["df"], times)
        for body, sensor in sensors.items()
    }

    for body in BODY_ORDER:
        if body not in aligned:
            continue
        quat = aligned[body]["quat"]
        roll, pitch, yaw = quat_to_euler_deg(quat)
        gyro = aligned[body]["gyro"]
        columns[f"{body}_quat_valid"] = aligned[body]["quat_valid"].astype(int)
        columns[f"{body}_abs_roll"] = roll
        columns[f"{body}_abs_pitch"] = pitch
        columns[f"{body}_abs_yaw"] = yaw
        columns[f"{body}_abs_gyro_mag"] = vector_norm(gyro)

    if "trunk" in aligned:
        columns["trunk_posture_roll"] = columns.get("trunk_abs_roll")
        columns["trunk_posture_pitch"] = columns.get("trunk_abs_pitch")
        columns["trunk_posture_yaw"] = columns.get("trunk_abs_yaw")
        columns["trunk_posture_gyro_mag"] = columns.get("trunk_abs_gyro_mag")

    for joint_name, parent_body, child_body in JOINT_DEFS:
        if parent_body not in aligned or child_body not in aligned:
            continue
        rel_q, rel_valid = quat_relative(aligned[parent_body]["quat"], aligned[child_body]["quat"])
        rel_roll, rel_pitch, rel_yaw = quat_to_euler_deg(rel_q)
        angle = quat_angle_deg(rel_q)
        rel_gyro = aligned[child_body]["gyro"] - aligned[parent_body]["gyro"]
        rel_gyro_mag = vector_norm(rel_gyro)

        columns[f"{joint_name}_relative_valid"] = rel_valid.astype(int)
        columns[f"{joint_name}_angle_deg"] = angle
        columns[f"{joint_name}_angle_velocity_deg_s"] = finite_difference(angle, times)
        columns[f"{joint_name}_relative_roll"] = rel_roll
        columns[f"{joint_name}_relative_pitch"] = rel_pitch
        columns[f"{joint_name}_relative_yaw"] = rel_yaw
        columns[f"{joint_name}_relative_gyro_mag"] = rel_gyro_mag
        columns[f"{joint_name}_angle_mean_0p35s"] = rolling_mean(angle, times, window_s)
        columns[f"{joint_name}_angle_std_0p35s"] = rolling_std(angle, times, window_s)
        columns[f"{joint_name}_angle_range_0p35s"] = rolling_range(angle, times, window_s)

    for column in ["trunk_abs_roll", "trunk_abs_pitch", "trunk_abs_yaw"]:
        if column in columns:
            values = np.asarray(columns[column], dtype=float)
            columns[f"{column}_std_0p35s"] = rolling_std(values, times, window_s)
            columns[f"{column}_range_0p35s"] = rolling_range(values, times, window_s)

    out = pd.DataFrame(columns)
    out.attrs["mapped_body_count"] = len(aligned)
    out.attrs["reference_body"] = ref_body
    return out


def output_path_for(recording_dir, output_root):
    participant_id, _, full_id = parse_recording_dir(recording_dir)
    return Path(output_root) / participant_id / f"{full_id}_imu_joint_kinematics_frame_features.csv"


def process_recording(recording_dir, output_root, window_s=config.IMU_JOINT_KINEMATICS_WINDOW_S, overwrite=False):
    participant_id, task_name, full_id = parse_recording_dir(recording_dir)
    output_path = output_path_for(recording_dir, output_root)
    if output_path.exists() and not overwrite:
        existing = pd.read_csv(output_path, usecols=["frame_idx"])
        return {
            "participant_id": participant_id,
            "task_name": task_name,
            "full_id": full_id,
            "frame_feature_csv": str(output_path),
            "row_count": int(len(existing)),
            "sensor_frame_count": int(existing["frame_idx"].nunique()),
            "mapped_body_count": np.nan,
            "reference_body": "existing",
            "skipped_existing": True,
        }

    frame_df = build_frame_features(recording_dir, window_s=window_s)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame_df.to_csv(output_path, index=False)
    return {
        "participant_id": participant_id,
        "task_name": task_name,
        "full_id": full_id,
        "frame_feature_csv": str(output_path),
        "row_count": int(len(frame_df)),
        "sensor_frame_count": int(frame_df["frame_idx"].nunique()),
        "mapped_body_count": int(frame_df.attrs.get("mapped_body_count", 0)),
        "reference_body": frame_df.attrs.get("reference_body", ""),
        "skipped_existing": False,
    }


def extract_imu_joint_kinematics(
    input_root=config.VIDEO_ALIGNED_SENSOR_ROOT,
    output_root=config.IMU_JOINT_KINEMATICS_FRAME_OUTPUT_ROOT,
    index_path=None,
    summary_path=None,
    participant=None,
    full_id=None,
    window_s=config.IMU_JOINT_KINEMATICS_WINDOW_S,
    include_calibration=False,
    limit=None,
    overwrite=False,
):
    input_root = Path(input_root)
    output_root = Path(output_root)
    index_path = Path(index_path) if index_path is not None else output_root.parent / config.IMU_JOINT_KINEMATICS_INDEX.name
    summary_path = Path(summary_path) if summary_path is not None else output_root.parent / config.IMU_JOINT_KINEMATICS_SUMMARY.name

    recording_dirs = iter_recording_dirs(
        input_root=input_root,
        participant=participant,
        full_id=full_id,
        include_calibration=include_calibration,
        limit=limit,
    )
    if not recording_dirs:
        raise FileNotFoundError(f"No Movella IMU recording dirs found under {input_root}")

    rows = []
    iterator = recording_dirs
    if tqdm is not None:
        iterator = tqdm(recording_dirs, desc="Extracting IMU joint kinematics", unit="recording")
    for recording_dir in iterator:
        rows.append(process_recording(recording_dir, output_root=output_root, window_s=window_s, overwrite=overwrite))

    index_df = pd.DataFrame(rows)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(index_path, index=False)

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "index_csv": str(index_path),
        "recording_count": int(len(index_df)),
        "row_count": int(index_df["row_count"].sum()),
        "sensor_frame_count": int(index_df["sensor_frame_count"].sum()),
        "window_s": float(window_s),
        "dot_to_body": DOT_TO_BODY,
        "body_order": BODY_ORDER,
        "joint_defs": [
            {"joint": joint, "parent": parent, "child": child}
            for joint, parent, child in JOINT_DEFS
        ],
        "skipped_calibration_tasks": [] if include_calibration else sorted(SKIP_TASKS),
        "feature_policy": {
            "time_axis": "Uses original sensor time_s/frame_idx at the Movella sampling rate, not video_time_s/video_frame_idx.",
            "implementation": "Independent upper_features_extraction implementation; does not call feature_extraction.modalities.imu.",
            "absolute_orientation": "Per body segment absolute roll/pitch/yaw.",
            "relative_joint_angle": "q_relative = conjugate(parent_segment_quat) * child_segment_quat; angle_deg = 2*acos(abs(w)).",
            "window": "Joint angle mean/std/range use a causal window over current sample and previous window_s seconds.",
        },
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    return index_df, summary


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Extract upper-level IMU joint kinematics from aligned Movella DOT CSV files.")
    parser.add_argument("--input-root", default=str(config.VIDEO_ALIGNED_SENSOR_ROOT))
    parser.add_argument("--output-root", default=str(config.IMU_JOINT_KINEMATICS_FRAME_OUTPUT_ROOT))
    parser.add_argument("--index-path", default=None)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--participant", default=None)
    parser.add_argument("--full-id", action="append", default=None)
    parser.add_argument("--window-s", type=float, default=config.IMU_JOINT_KINEMATICS_WINDOW_S)
    parser.add_argument("--include-calibration", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    index_df, summary = extract_imu_joint_kinematics(
        input_root=args.input_root,
        output_root=args.output_root,
        index_path=args.index_path,
        summary_path=args.summary_path,
        participant=args.participant,
        full_id=args.full_id,
        window_s=args.window_s,
        include_calibration=args.include_calibration,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    print("IMU joint kinematics extraction completed.")
    print(json.dumps(summary, indent=2))
    print("\nProcessed recordings:")
    print(index_df[["participant_id", "full_id", "row_count", "sensor_frame_count", "mapped_body_count", "reference_body"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
