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


ACC_COLS = ("accel_x", "accel_y", "accel_z")
GYRO_COLS = ("gyro_x", "gyro_y", "gyro_z")
QUAT_COLS = ("quat_w", "quat_x", "quat_y", "quat_z")
OUTPUT_COLUMNS = [
    "participant_id",
    "task_name",
    "full_id",
    "frame_idx",
    "time_s",
    "video_time_s",
    "video_frame_idx",
    "window_start_time_s",
    "window_duration_s",
    "window_sample_count",
    "acc_mag",
    "gyro_mag",
    "linear_accel_x",
    "linear_accel_y",
    "linear_accel_z",
    "acc_dyn_mag",
    "motion_intensity",
    "gyro_energy",
    "acc_energy",
    "roll",
    "pitch",
    "yaw",
    "roll_range",
    "pitch_range",
    "yaw_range",
    "orientation_change",
    "orientation_velocity",
    "gyro_std",
    "roll_std",
    "pitch_std",
    "yaw_std",
    "low_motion_ratio",
    "head_stability_score",
    "jerk_x",
    "jerk_y",
    "jerk_z",
    "jerk_mag",
    "angular_acc_x",
    "angular_acc_y",
    "angular_acc_z",
    "angular_acc_mag",
    "gravity_compensation_method",
    "gravity_compensation_error_median",
    "gravity_magnitude",
]
SKIP_TASKS = {"Eye1", "Eye2", "Recalibration1"}


def parse_recording_path(csv_path, input_root):
    rel = Path(csv_path).relative_to(input_root)
    participant_id = rel.parts[0]
    full_id = rel.parts[1]
    prefix = f"{participant_id}_"
    task_name = full_id[len(prefix):] if full_id.startswith(prefix) else full_id
    return participant_id, task_name, full_id


def iter_eye_imu_csvs(input_root, participant=None, full_id=None, include_calibration=False, limit=None):
    input_root = Path(input_root)
    paths = sorted(input_root.glob(f"*/*/{config.HEAD_IMU_MODALITY}.csv"))

    if participant:
        paths = [path for path in paths if path.relative_to(input_root).parts[0] == participant]

    if full_id:
        wanted = set(full_id)
        paths = [path for path in paths if path.relative_to(input_root).parts[1] in wanted]

    if not include_calibration:
        paths = [
            path
            for path in paths
            if parse_recording_path(path, input_root)[1] not in SKIP_TASKS
        ]

    if limit is not None:
        paths = paths[:limit]

    return paths


def numeric_column(df, column):
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def estimate_sample_rate(times):
    times = np.asarray(times, dtype=float)
    diffs = np.diff(times)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return 120.0
    return float(1.0 / np.median(diffs))


def read_eye_imu_csv(path):
    df = pd.read_csv(path)
    required = {*ACC_COLS, *GYRO_COLS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing eye-tracker IMU columns in {path}: {missing}")

    out = df.copy()
    for column in [
        "frame_idx",
        "time_s",
        "video_time_s",
        "video_frame_idx",
        *ACC_COLS,
        *GYRO_COLS,
        *QUAT_COLS,
    ]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    time_col = "video_time_s" if "video_time_s" in out.columns else "time_s"
    if time_col not in out.columns:
        raise ValueError(f"Missing time_s/video_time_s in {path}")

    out = out.dropna(subset=[time_col]).copy()
    out = out.sort_values(time_col).reset_index(drop=True)
    out["feature_time_s"] = out[time_col].astype(float)
    if "video_frame_idx" in out.columns:
        out["feature_frame_idx"] = numeric_column(out, "video_frame_idx").round()
    else:
        out["feature_frame_idx"] = np.round(out["feature_time_s"] * 30.0)
    return out


def normalize_quat_wxyz(q):
    q = np.asarray(q, dtype=float)
    norms = np.linalg.norm(q, axis=1)
    valid = np.isfinite(q).all(axis=1) & (norms > 1e-8)
    out = np.full_like(q, np.nan, dtype=float)
    out[valid] = q[valid] / norms[valid, None]
    return out, valid


def quat_conjugate_wxyz(q):
    out = q.copy()
    out[:, 1:] *= -1.0
    return out


def quat_rotate_wxyz(q, vectors):
    q = np.asarray(q, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    q_vec = q[:, 1:]
    q_w = q[:, :1]
    t = 2.0 * np.cross(q_vec, vectors)
    return vectors + q_w * t + np.cross(q_vec, t)


def quat_to_euler_deg_wxyz(q):
    q, valid = normalize_quat_wxyz(q)
    w = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    roll = np.full(len(q), np.nan, dtype=float)
    pitch = np.full(len(q), np.nan, dtype=float)
    yaw = np.full(len(q), np.nan, dtype=float)
    if not valid.any():
        return roll, pitch, yaw

    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    t2 = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)

    roll[valid] = np.degrees(np.arctan2(t0[valid], t1[valid]))
    pitch[valid] = np.degrees(np.arcsin(t2[valid]))
    yaw[valid] = np.degrees(np.arctan2(t3[valid], t4[valid]))
    return roll, pitch, yaw


def rolling_median(values, window_samples):
    return (
        pd.Series(values)
        .rolling(window_samples, min_periods=1)
        .median()
        .to_numpy(dtype=float)
    )


def low_motion_mask(gyro_mag, valid_acc, valid_quat):
    valid = valid_acc & valid_quat & np.isfinite(gyro_mag)
    if int(valid.sum()) < 10:
        return valid
    cutoff = np.nanquantile(gyro_mag[valid], 0.25)
    low = valid & (gyro_mag <= cutoff)
    return low if int(low.sum()) >= 10 else valid


def gravity_compensated_acceleration(acc, quat, gyro_mag, gravity_magnitude):
    gravity_magnitude = float(gravity_magnitude)
    quat, valid_quat = normalize_quat_wxyz(quat)
    valid_acc = np.isfinite(acc).all(axis=1)
    score_mask = low_motion_mask(gyro_mag, valid_acc, valid_quat)
    if int(score_mask.sum()) < 10:
        return None, {
            "method": "unavailable",
            "error_median": np.nan,
        }

    candidates = []
    for z_sign in (1.0, -1.0):
        gravity_world = np.tile(np.array([0.0, 0.0, z_sign * gravity_magnitude]), (len(acc), 1))
        for rotation_name, q_for_rotation in (
            ("q_rotates_world_to_sensor", quat),
            ("q_rotates_sensor_to_world", quat_conjugate_wxyz(quat)),
        ):
            gravity_sensor = quat_rotate_wxyz(q_for_rotation, gravity_world)
            linear_acc = acc - gravity_sensor
            linear_mag = np.linalg.norm(linear_acc, axis=1)
            score = float(np.nanmedian(linear_mag[score_mask]))
            candidates.append(
                {
                    "method": f"{rotation_name}; gravity_world_z={z_sign:+.0f}",
                    "linear_acc": linear_acc,
                    "score": score,
                }
            )

    best = min(candidates, key=lambda item: item["score"])
    linear_acc = np.full_like(acc, np.nan, dtype=float)
    linear_acc[valid_acc & valid_quat] = best["linear_acc"][valid_acc & valid_quat]
    return linear_acc, {
        "method": best["method"],
        "error_median": float(best["score"]),
        "gravity_magnitude": gravity_magnitude,
    }


def infer_gravity_magnitude(acc, gyro_mag):
    acc_mag = np.linalg.norm(np.asarray(acc, dtype=float), axis=1)
    valid = np.isfinite(acc_mag)
    finite_gyro = np.isfinite(gyro_mag)
    if finite_gyro.any():
        cutoff = np.nanquantile(gyro_mag[finite_gyro], 0.5)
        low_gyro = finite_gyro & (gyro_mag <= cutoff)
        if int((valid & low_gyro).sum()) >= 10:
            valid = valid & low_gyro

    if int(valid.sum()) == 0:
        return 9.80665

    gravity = float(np.nanmedian(acc_mag[valid]))
    if not np.isfinite(gravity) or gravity <= 0:
        return 9.80665
    return gravity


def resolve_gravity_magnitude(gravity_magnitude, acc, gyro_mag):
    if isinstance(gravity_magnitude, str):
        if gravity_magnitude.lower() != "auto":
            return float(gravity_magnitude)
        return infer_gravity_magnitude(acc, gyro_mag)
    return float(gravity_magnitude)


def resolve_dynamic_acceleration(df, acc, gyro_mag, sample_rate, acc_mode, gravity_magnitude):
    if acc_mode not in {"auto", "quaternion", "magnitude_baseline"}:
        raise ValueError(f"Unsupported acc_mode: {acc_mode}")

    resolved_gravity_magnitude = resolve_gravity_magnitude(gravity_magnitude, acc, gyro_mag)
    has_quat = all(col in df.columns for col in QUAT_COLS)
    if acc_mode in {"auto", "quaternion"} and has_quat:
        quat = df[list(QUAT_COLS)].to_numpy(dtype=float)
        linear_acc, info = gravity_compensated_acceleration(acc, quat, gyro_mag, resolved_gravity_magnitude)
        if linear_acc is not None:
            return linear_acc, "quaternion", info
        if acc_mode == "quaternion":
            raise ValueError("Quaternion gravity compensation requested, but valid quaternion data was not available.")

    acc_mag = np.linalg.norm(acc, axis=1)
    baseline_samples = max(3, int(round(sample_rate * 1.0)))
    baseline = rolling_median(acc_mag, baseline_samples)
    acc_dyn_mag = np.abs(acc_mag - baseline)
    linear_acc = np.full_like(acc, np.nan, dtype=float)
    valid = np.isfinite(acc).all(axis=1) & np.isfinite(acc_mag) & (acc_mag > 1e-9)
    linear_acc[valid] = acc[valid] * (acc_dyn_mag[valid] / acc_mag[valid])[:, None]
    return linear_acc, "magnitude_baseline", {
        "method": "rolling_median_acc_magnitude",
        "error_median": np.nan,
        "gravity_magnitude": resolved_gravity_magnitude,
    }


def finite_difference(values, times):
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    out = np.full_like(values, np.nan, dtype=float)
    dt = np.diff(times, prepend=np.nan)
    diff = np.diff(values, axis=0, prepend=np.nan)
    valid = np.isfinite(dt) & (dt > 0)
    if values.ndim == 1:
        out[valid] = diff[valid] / dt[valid]
    else:
        out[valid] = diff[valid] / dt[valid, None]
    return out


def compute_orientation_change(df, times):
    if all(col in df.columns for col in QUAT_COLS):
        quat, valid = normalize_quat_wxyz(df[list(QUAT_COLS)].to_numpy(dtype=float))
        change = np.full(len(df), np.nan, dtype=float)
        for idx in range(1, len(df)):
            if not valid[idx] or not valid[idx - 1]:
                continue
            dot = float(np.clip(np.abs(np.dot(quat[idx], quat[idx - 1])), -1.0, 1.0))
            change[idx] = np.degrees(2.0 * np.arccos(dot))
    else:
        euler = df[["roll", "pitch", "yaw"]].to_numpy(dtype=float)
        change = np.linalg.norm(np.diff(euler, axis=0, prepend=np.nan), axis=1)

    velocity = np.full(len(df), np.nan, dtype=float)
    dt = np.diff(times, prepend=np.nan)
    valid_dt = np.isfinite(dt) & (dt > 0) & np.isfinite(change)
    velocity[valid_dt] = change[valid_dt] / dt[valid_dt]
    return change, velocity


def rolling_series(values, times):
    index = pd.to_timedelta(times, unit="s")
    return pd.Series(values, index=index)


def rolling_apply(values, times, window_s, func, min_periods=1):
    window = f"{int(round(window_s * 1000))}ms"
    return rolling_series(values, times).rolling(window, min_periods=min_periods).apply(func, raw=True).to_numpy(dtype=float)


def rolling_mean(values, times, window_s):
    window = f"{int(round(window_s * 1000))}ms"
    return rolling_series(values, times).rolling(window, min_periods=1).mean().to_numpy(dtype=float)


def rolling_std(values, times, window_s):
    window = f"{int(round(window_s * 1000))}ms"
    return rolling_series(values, times).rolling(window, min_periods=1).std().fillna(0.0).to_numpy(dtype=float)


def rolling_count(values, times, window_s):
    window = f"{int(round(window_s * 1000))}ms"
    return rolling_series(values, times).rolling(window, min_periods=1).count().to_numpy(dtype=float)


def rolling_range(values, times, window_s):
    def finite_range(window_values):
        window_values = window_values[np.isfinite(window_values)]
        if len(window_values) == 0:
            return np.nan
        return float(np.nanmax(window_values) - np.nanmin(window_values))

    return rolling_apply(values, times, window_s, finite_range)


def add_head_imu_features(
    df,
    window_s=config.HEAD_IMU_WINDOW_S,
    acc_mode="auto",
    gravity_magnitude=config.HEAD_IMU_GRAVITY_MAGNITUDE,
    low_gyro_threshold=config.HEAD_IMU_LOW_GYRO_THRESHOLD_DEG_S,
    low_acc_dyn_threshold=config.HEAD_IMU_LOW_ACC_DYN_THRESHOLD_M_S2,
    stability_eps=config.HEAD_IMU_STABILITY_EPS,
):
    out = df.copy()
    times = out["feature_time_s"].to_numpy(dtype=float)
    sample_rate = estimate_sample_rate(times)

    acc = out[list(ACC_COLS)].to_numpy(dtype=float)
    gyro = out[list(GYRO_COLS)].to_numpy(dtype=float)
    valid_acc = np.isfinite(acc).all(axis=1)
    valid_gyro = np.isfinite(gyro).all(axis=1)

    acc_mag = np.full(len(out), np.nan, dtype=float)
    gyro_mag = np.full(len(out), np.nan, dtype=float)
    acc_mag[valid_acc] = np.linalg.norm(acc[valid_acc], axis=1)
    gyro_mag[valid_gyro] = np.linalg.norm(gyro[valid_gyro], axis=1)

    linear_acc, resolved_acc_mode, gravity_info = resolve_dynamic_acceleration(
        out,
        acc,
        gyro_mag,
        sample_rate=sample_rate,
        acc_mode=acc_mode,
        gravity_magnitude=gravity_magnitude,
    )
    valid_linear = np.isfinite(linear_acc).all(axis=1)
    acc_dyn_mag = np.full(len(out), np.nan, dtype=float)
    acc_dyn_mag[valid_linear] = np.linalg.norm(linear_acc[valid_linear], axis=1)

    motion_intensity = acc_dyn_mag + gyro_mag

    if all(col in out.columns for col in QUAT_COLS):
        roll, pitch, yaw = quat_to_euler_deg_wxyz(out[list(QUAT_COLS)].to_numpy(dtype=float))
    else:
        roll = numeric_column(out, "roll").to_numpy(dtype=float)
        pitch = numeric_column(out, "pitch").to_numpy(dtype=float)
        yaw = numeric_column(out, "yaw").to_numpy(dtype=float)

    out["roll"] = roll
    out["pitch"] = pitch
    out["yaw"] = yaw
    orientation_change, orientation_velocity = compute_orientation_change(out, times)

    jerk = finite_difference(linear_acc, times)
    angular_acc = finite_difference(gyro, times)
    jerk_mag = np.full(len(out), np.nan, dtype=float)
    angular_acc_mag = np.full(len(out), np.nan, dtype=float)
    valid_jerk = np.isfinite(jerk).all(axis=1)
    valid_angular_acc = np.isfinite(angular_acc).all(axis=1)
    jerk_mag[valid_jerk] = np.linalg.norm(jerk[valid_jerk], axis=1)
    angular_acc_mag[valid_angular_acc] = np.linalg.norm(angular_acc[valid_angular_acc], axis=1)

    low_motion = (
        np.isfinite(gyro_mag)
        & np.isfinite(acc_dyn_mag)
        & (gyro_mag <= low_gyro_threshold)
        & (acc_dyn_mag <= low_acc_dyn_threshold)
    ).astype(float)

    window_sample_count = rolling_count(out["feature_time_s"].to_numpy(dtype=float), times, window_s)
    window_duration_s = np.minimum(window_s, times - np.nanmin(times))
    window_start_time_s = times - window_duration_s

    gyro_std = rolling_std(gyro_mag, times, window_s)
    features = pd.DataFrame(
        {
            "frame_idx": numeric_column(out, "frame_idx"),
            "time_s": numeric_column(out, "time_s"),
            "video_time_s": numeric_column(out, "video_time_s") if "video_time_s" in out.columns else out["feature_time_s"],
            "video_frame_idx": out["feature_frame_idx"],
            "window_start_time_s": window_start_time_s,
            "window_duration_s": window_duration_s,
            "window_sample_count": window_sample_count.astype(int),
            "acc_mag": acc_mag,
            "gyro_mag": gyro_mag,
            "linear_accel_x": linear_acc[:, 0],
            "linear_accel_y": linear_acc[:, 1],
            "linear_accel_z": linear_acc[:, 2],
            "acc_dyn_mag": acc_dyn_mag,
            "motion_intensity": motion_intensity,
            "gyro_energy": rolling_mean(gyro_mag ** 2, times, window_s),
            "acc_energy": rolling_mean(acc_dyn_mag ** 2, times, window_s),
            "roll": roll,
            "pitch": pitch,
            "yaw": yaw,
            "roll_range": rolling_range(roll, times, window_s),
            "pitch_range": rolling_range(pitch, times, window_s),
            "yaw_range": rolling_range(yaw, times, window_s),
            "orientation_change": orientation_change,
            "orientation_velocity": orientation_velocity,
            "gyro_std": gyro_std,
            "roll_std": rolling_std(roll, times, window_s),
            "pitch_std": rolling_std(pitch, times, window_s),
            "yaw_std": rolling_std(yaw, times, window_s),
            "low_motion_ratio": rolling_mean(low_motion, times, window_s),
            "head_stability_score": 1.0 / (gyro_std + stability_eps),
            "jerk_x": jerk[:, 0],
            "jerk_y": jerk[:, 1],
            "jerk_z": jerk[:, 2],
            "jerk_mag": jerk_mag,
            "angular_acc_x": angular_acc[:, 0],
            "angular_acc_y": angular_acc[:, 1],
            "angular_acc_z": angular_acc[:, 2],
            "angular_acc_mag": angular_acc_mag,
            "gravity_compensation_method": gravity_info["method"],
            "gravity_compensation_error_median": gravity_info["error_median"],
            "gravity_magnitude": gravity_info["gravity_magnitude"],
        }
    )
    features.attrs["sample_rate"] = sample_rate
    features.attrs["acc_mode"] = resolved_acc_mode
    return features


def output_path_for(csv_path, input_root, output_root):
    rel = Path(csv_path).relative_to(input_root)
    return Path(output_root) / rel.parts[0] / f"{rel.parts[1]}_head_imu_frame_features.csv"


def process_eye_imu_csv(
    csv_path,
    input_root,
    output_root,
    window_s=config.HEAD_IMU_WINDOW_S,
    acc_mode="auto",
    gravity_magnitude=config.HEAD_IMU_GRAVITY_MAGNITUDE,
    low_gyro_threshold=config.HEAD_IMU_LOW_GYRO_THRESHOLD_DEG_S,
    low_acc_dyn_threshold=config.HEAD_IMU_LOW_ACC_DYN_THRESHOLD_M_S2,
    overwrite=False,
):
    csv_path = Path(csv_path)
    input_root = Path(input_root)
    output_root = Path(output_root)
    participant_id, task_name, full_id = parse_recording_path(csv_path, input_root)
    out_path = output_path_for(csv_path, input_root, output_root)

    if out_path.exists() and not overwrite:
        existing = pd.read_csv(out_path, usecols=["video_frame_idx"])
        return {
            "participant_id": participant_id,
            "task_name": task_name,
            "full_id": full_id,
            "source_csv_path": str(csv_path),
            "frame_feature_csv": str(out_path),
            "row_count": int(len(existing)),
            "video_frame_count": int(existing["video_frame_idx"].nunique()),
            "sample_rate_hz": np.nan,
            "acc_mode": "existing",
            "gravity_compensation_method": "existing",
            "gravity_compensation_error_median": np.nan,
            "gravity_magnitude": np.nan,
            "skipped_existing": True,
        }

    raw_df = read_eye_imu_csv(csv_path)
    feature_df = add_head_imu_features(
        raw_df,
        window_s=window_s,
        acc_mode=acc_mode,
        gravity_magnitude=gravity_magnitude,
        low_gyro_threshold=low_gyro_threshold,
        low_acc_dyn_threshold=low_acc_dyn_threshold,
    )
    for column, value in (
        ("participant_id", participant_id),
        ("task_name", task_name),
        ("full_id", full_id),
    ):
        feature_df.insert(0, column, value)
    feature_df = feature_df[OUTPUT_COLUMNS]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(out_path, index=False)

    return {
        "participant_id": participant_id,
        "task_name": task_name,
        "full_id": full_id,
        "source_csv_path": str(csv_path),
        "frame_feature_csv": str(out_path),
        "row_count": int(len(feature_df)),
        "video_frame_count": int(feature_df["video_frame_idx"].nunique()),
        "sample_rate_hz": float(feature_df.attrs.get("sample_rate", np.nan)),
        "acc_mode": feature_df.attrs.get("acc_mode", ""),
        "gravity_compensation_method": (
            str(feature_df["gravity_compensation_method"].dropna().iloc[0])
            if feature_df["gravity_compensation_method"].notna().any()
            else ""
        ),
        "gravity_compensation_error_median": (
            float(feature_df["gravity_compensation_error_median"].dropna().iloc[0])
            if feature_df["gravity_compensation_error_median"].notna().any()
            else np.nan
        ),
        "gravity_magnitude": (
            float(feature_df["gravity_magnitude"].dropna().iloc[0])
            if feature_df["gravity_magnitude"].notna().any()
            else np.nan
        ),
        "skipped_existing": False,
    }


def extract_head_imu_features(
    input_root=config.VIDEO_ALIGNED_SENSOR_ROOT,
    output_root=config.HEAD_IMU_FRAME_OUTPUT_ROOT,
    index_path=None,
    summary_path=None,
    participant=None,
    full_id=None,
    window_s=config.HEAD_IMU_WINDOW_S,
    acc_mode="auto",
    gravity_magnitude=config.HEAD_IMU_GRAVITY_MAGNITUDE,
    low_gyro_threshold=config.HEAD_IMU_LOW_GYRO_THRESHOLD_DEG_S,
    low_acc_dyn_threshold=config.HEAD_IMU_LOW_ACC_DYN_THRESHOLD_M_S2,
    include_calibration=False,
    limit=None,
    overwrite=False,
):
    input_root = Path(input_root)
    output_root = Path(output_root)
    index_path = Path(index_path) if index_path is not None else output_root.parent / config.HEAD_IMU_INDEX.name
    summary_path = Path(summary_path) if summary_path is not None else output_root.parent / config.HEAD_IMU_SUMMARY.name
    if not input_root.exists():
        raise FileNotFoundError(input_root)

    paths = iter_eye_imu_csvs(
        input_root,
        participant=participant,
        full_id=full_id,
        include_calibration=include_calibration,
        limit=limit,
    )
    if not paths:
        raise FileNotFoundError(f"No {config.HEAD_IMU_MODALITY}.csv files found under {input_root}")

    rows = []
    iterator = paths
    if tqdm is not None:
        iterator = tqdm(paths, desc="Extracting head IMU features", unit="recording")

    for path in iterator:
        rows.append(
            process_eye_imu_csv(
                path,
                input_root=input_root,
                output_root=output_root,
                window_s=window_s,
                acc_mode=acc_mode,
                gravity_magnitude=gravity_magnitude,
                low_gyro_threshold=low_gyro_threshold,
                low_acc_dyn_threshold=low_acc_dyn_threshold,
                overwrite=overwrite,
            )
        )

    index_df = pd.DataFrame(rows)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(index_path, index=False)

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "index_csv": str(index_path),
        "recording_count": int(len(index_df)),
        "row_count": int(index_df["row_count"].sum()),
        "video_frame_count": int(index_df["video_frame_count"].sum()),
        "window_s": float(window_s),
        "acc_mode": acc_mode,
        "gravity_magnitude": gravity_magnitude,
        "resolved_gravity_magnitude_median": float(index_df["gravity_magnitude"].median()),
        "low_gyro_threshold_deg_s": float(low_gyro_threshold),
        "low_acc_dyn_threshold_m_s2": float(low_acc_dyn_threshold),
        "gravity_compensation_methods": index_df["gravity_compensation_method"].value_counts(dropna=False).to_dict(),
        "skipped_calibration_tasks": [] if include_calibration else sorted(SKIP_TASKS),
        "feature_policy": {
            "time_axis": "Uses video_time_s/video_frame_idx from aligned_sensor_with_YOLO when available.",
            "window": "All rolling statistics are causal over the current sample and the previous window_s seconds.",
            "acc_dyn_mag": "Uses quaternion gravity compensation when possible; otherwise falls back to rolling-median acceleration magnitude baseline.",
            "motion_intensity": "acc_dyn_mag + gyro_mag.",
            "head_stability_score": "1 / (gyro_std + eps), where gyro_std is computed over the causal window.",
        },
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return index_df, summary


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Extract upper-level head-motion features from eye_tracker__imu CSV files."
    )
    parser.add_argument("--input-root", default=str(config.VIDEO_ALIGNED_SENSOR_ROOT))
    parser.add_argument("--output-root", default=str(config.HEAD_IMU_FRAME_OUTPUT_ROOT))
    parser.add_argument("--index-path", default=None)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--participant", default=None, help="Optional participant id filter, e.g. IA_4_H_02.")
    parser.add_argument("--full-id", action="append", default=None, help="Optional full recording id filter. Can be repeated.")
    parser.add_argument("--window-s", type=float, default=config.HEAD_IMU_WINDOW_S, help="Causal rolling-window length in seconds.")
    parser.add_argument(
        "--acc-mode",
        choices=("auto", "quaternion", "magnitude_baseline"),
        default="auto",
        help="Acceleration preprocessing for acc_dyn_mag.",
    )
    parser.add_argument(
        "--gravity-magnitude",
        default=config.HEAD_IMU_GRAVITY_MAGNITUDE,
        help="Gravity magnitude for quaternion compensation, or 'auto' to infer it per recording.",
    )
    parser.add_argument("--low-gyro-threshold", type=float, default=config.HEAD_IMU_LOW_GYRO_THRESHOLD_DEG_S)
    parser.add_argument("--low-acc-dyn-threshold", type=float, default=config.HEAD_IMU_LOW_ACC_DYN_THRESHOLD_M_S2)
    parser.add_argument("--include-calibration", action="store_true", help="Include Eye1, Eye2, and Recalibration1 recordings.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of recordings.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing feature CSV files.")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    index_df, summary = extract_head_imu_features(
        input_root=args.input_root,
        output_root=args.output_root,
        index_path=args.index_path,
        summary_path=args.summary_path,
        participant=args.participant,
        full_id=args.full_id,
        window_s=args.window_s,
        acc_mode=args.acc_mode,
        gravity_magnitude=args.gravity_magnitude,
        low_gyro_threshold=args.low_gyro_threshold,
        low_acc_dyn_threshold=args.low_acc_dyn_threshold,
        include_calibration=args.include_calibration,
        limit=args.limit,
        overwrite=args.overwrite,
    )

    print("Head IMU feature extraction completed.")
    print(json.dumps(summary, indent=2))
    print("\nProcessed recordings:")
    print(index_df[["participant_id", "full_id", "row_count", "video_frame_count", "acc_mode"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
