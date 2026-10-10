import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from lower_feature_cleaning_extraction.config import (
    IMU_ARM_FEATURE_OUTPUT_ROOT,
    SKIP_TASKS,
    VIDEO_ALIGNED_CSV_OUTPUT_ROOT,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


ACC_COLS = ("accel_x", "accel_y", "accel_z")
GYRO_COLS = ("gyro_x", "gyro_y", "gyro_z")
ORIENTATION_COLS = ("pitch", "roll", "yaw", "quat_w", "quat_x", "quat_y", "quat_z")
WINDOW_COLUMNS = [
    "participant_id",
    "task_name",
    "full_id",
    "sensor_id",
    "start_time_s",
    "end_time_s",
    "start_frame",
    "end_frame",
    "samples",
    "acc_mode",
    "gravity_compensation_method",
    "gravity_compensation_error_median",
    "raw_acc_mag_mean",
    "linear_acc_mag_mean",
    "acc_mag_mean",
    "acc_mag_std",
    "acc_dyn_mean",
    "acc_dyn_max",
    "gyro_mag_mean",
    "gyro_mag_std",
    "gyro_mag_max",
    "jerk_mag_mean",
    "jerk_mag_max",
    "angular_accel_mag_mean",
    "angular_accel_mag_max",
    "orientation_delta_deg",
    "gyro_zero_crossing_rate",
    "intensity_score",
    "intensity_label",
    "idle_candidate",
    "grasp_candidate",
    "micro_motion_score",
    "micro_motion_candidate",
    "sudden_event_count",
]
EVENT_COLUMNS = [
    "participant_id",
    "task_name",
    "full_id",
    "sensor_id",
    "event_time_s",
    "event_frame",
    "event_type",
    "event_score",
    "jerk_mag",
    "angular_accel_mag",
    "acc_dyn",
    "gyro_mag",
]

GRAVITY_MAGNITUDE = 9.80665


def robust_center_scale(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return 0.0, 1.0

    center = float(np.nanmedian(values))
    mad = float(np.nanmedian(np.abs(values - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-9:
        scale = float(np.nanstd(values))
    if not np.isfinite(scale) or scale < 1e-9:
        scale = 1.0
    return center, scale


def robust_z(values):
    center, scale = robust_center_scale(values)
    return (np.asarray(values, dtype=float) - center) / scale


def sensor_id_from_path(path):
    return Path(path).stem.replace("movella__DOT_", "")


def parse_recording_path(csv_path, input_root):
    rel = Path(csv_path).relative_to(input_root)
    participant_id = rel.parts[0]
    full_id = rel.parts[1]
    prefix = f"{participant_id}_"
    task_name = full_id[len(prefix):] if full_id.startswith(prefix) else full_id
    return participant_id, task_name, full_id


def iter_imu_csvs(input_root, participant=None, full_id=None, include_calibration=False, limit=None):
    input_root = Path(input_root)
    paths = sorted(input_root.glob("*/*/movella__DOT_*.csv"))

    if participant:
        paths = [path for path in paths if path.relative_to(input_root).parts[0] == participant]

    if full_id:
        wanted = set(full_id)
        paths = [path for path in paths if path.relative_to(input_root).parts[1] in wanted]

    if not include_calibration:
        filtered = []
        for path in paths:
            participant_id, task_name, _ = parse_recording_path(path, input_root)
            if task_name not in SKIP_TASKS:
                filtered.append(path)
        paths = filtered

    if limit is not None:
        paths = paths[:limit]

    return paths


def read_imu_csv(path):
    df = pd.read_csv(path)
    required = set(ACC_COLS) | set(GYRO_COLS)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing IMU columns in {path}: {missing}")

    out = df.copy()
    for column in ["time_s", "video_time_s", "video_frame_idx", *ACC_COLS, *GYRO_COLS, *ORIENTATION_COLS]:
        if column in out.columns:
            out[column] = pd.to_numeric(out[column], errors="coerce")

    time_col = "video_time_s" if "video_time_s" in out.columns else "time_s"
    if time_col not in out.columns:
        raise ValueError(f"Missing time_s/video_time_s in {path}")

    out = out.dropna(subset=[time_col]).copy()
    out = out.sort_values(time_col).reset_index(drop=True)
    out["feature_time_s"] = out[time_col].astype(float)

    if "video_frame_idx" in out.columns:
        out["feature_frame_idx"] = out["video_frame_idx"].round()
    else:
        out["feature_frame_idx"] = np.round(out["feature_time_s"] * 30.0)

    out["feature_frame_idx"] = pd.to_numeric(out["feature_frame_idx"], errors="coerce")
    return out


def rolling_median(values, window_samples):
    series = pd.Series(values)
    return series.rolling(window_samples, center=True, min_periods=1).median().to_numpy()


def estimate_sample_rate(times):
    times = np.asarray(times, dtype=float)
    diffs = np.diff(times)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if len(diffs) == 0:
        return 30.0
    return float(1.0 / np.median(diffs))


def quat_conjugate(q):
    out = q.copy()
    out[:, 1:] *= -1.0
    return out


def normalize_quaternions(q):
    q = np.asarray(q, dtype=float)
    norms = np.linalg.norm(q, axis=1)
    valid = np.isfinite(norms) & (norms > 1e-6)
    out = np.full_like(q, np.nan, dtype=float)
    out[valid] = q[valid] / norms[valid, None]
    return out, valid


def quat_rotate(q, vectors):
    q = np.asarray(q, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    q_vec = q[:, 1:]
    q_w = q[:, :1]
    t = 2.0 * np.cross(q_vec, vectors)
    return vectors + q_w * t + np.cross(q_vec, t)


def low_motion_mask(gyro_mag, valid_acc, valid_quat):
    valid = valid_acc & valid_quat & np.isfinite(gyro_mag)
    if int(valid.sum()) < 10:
        return valid
    cutoff = np.nanquantile(gyro_mag[valid], 0.25)
    low = valid & (gyro_mag <= cutoff)
    if int(low.sum()) < 10:
        return valid
    return low


def gravity_compensated_acceleration(acc, quat, gyro_mag, gravity_magnitude=GRAVITY_MAGNITUDE):
    quat, valid_quat = normalize_quaternions(quat)
    valid_acc = np.all(np.isfinite(acc), axis=1)
    score_mask = low_motion_mask(gyro_mag, valid_acc, valid_quat)
    if int(score_mask.sum()) < 10:
        return None, {
            "method": "unavailable",
            "error_median": np.nan,
            "gravity_magnitude": float(gravity_magnitude),
        }

    candidates = []
    for z_sign in (1.0, -1.0):
        gravity_world = np.tile(np.array([0.0, 0.0, z_sign * gravity_magnitude]), (len(acc), 1))
        for rotation_name, q_for_rotation in (
            ("q_rotates_world_to_sensor", quat),
            ("q_rotates_sensor_to_world", quat_conjugate(quat)),
        ):
            gravity_sensor = quat_rotate(q_for_rotation, gravity_world)
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
        "gravity_magnitude": float(gravity_magnitude),
    }


def resolve_motion_acceleration(acc, gyro_mag, out, acc_mode, gravity_magnitude):
    mode = acc_mode
    if mode not in {"auto", "quaternion", "magnitude_baseline"}:
        raise ValueError(f"Unsupported acc_mode: {acc_mode}")

    has_quat = all(col in out.columns for col in ("quat_w", "quat_x", "quat_y", "quat_z"))
    if mode in {"auto", "quaternion"} and has_quat:
        quat = out[["quat_w", "quat_x", "quat_y", "quat_z"]].to_numpy(dtype=float)
        linear_acc, info = gravity_compensated_acceleration(acc, quat, gyro_mag, gravity_magnitude)
        if linear_acc is not None:
            return linear_acc, "quaternion", info
        if mode == "quaternion":
            raise ValueError("Quaternion gravity compensation requested, but valid quaternion data was not available.")

    return acc, "magnitude_baseline", {
        "method": "rolling_median_acc_magnitude",
        "error_median": np.nan,
        "gravity_magnitude": float(gravity_magnitude),
    }


def add_motion_signals(df, acc_mode="auto", gravity_magnitude=GRAVITY_MAGNITUDE):
    out = df.copy()
    times = out["feature_time_s"].to_numpy(dtype=float)
    sample_rate = estimate_sample_rate(times)
    rolling_samples = max(3, int(round(sample_rate * 1.0)))

    acc = out[list(ACC_COLS)].to_numpy(dtype=float)
    gyro = out[list(GYRO_COLS)].to_numpy(dtype=float)
    valid_acc = np.all(np.isfinite(acc), axis=1)
    valid_gyro = np.all(np.isfinite(gyro), axis=1)

    acc_mag = np.full(len(out), np.nan)
    gyro_mag = np.full(len(out), np.nan)
    acc_mag[valid_acc] = np.linalg.norm(acc[valid_acc], axis=1)
    gyro_mag[valid_gyro] = np.linalg.norm(gyro[valid_gyro], axis=1)

    motion_acc, resolved_acc_mode, gravity_info = resolve_motion_acceleration(
        acc,
        gyro_mag,
        out,
        acc_mode=acc_mode,
        gravity_magnitude=gravity_magnitude,
    )
    valid_motion_acc = np.all(np.isfinite(motion_acc), axis=1)
    linear_acc_mag = np.full(len(out), np.nan)
    linear_acc_mag[valid_motion_acc] = np.linalg.norm(motion_acc[valid_motion_acc], axis=1)

    gyro_baseline = rolling_median(gyro_mag, rolling_samples)
    if resolved_acc_mode == "quaternion":
        signal_acc_mag = linear_acc_mag
        acc_dyn = linear_acc_mag
    else:
        signal_acc_mag = acc_mag
        acc_baseline = rolling_median(acc_mag, rolling_samples)
        acc_dyn = np.abs(acc_mag - acc_baseline)
    gyro_highfreq = np.abs(gyro_mag - gyro_baseline)

    dt = np.diff(times, prepend=np.nan)
    dt[~np.isfinite(dt) | (dt <= 0)] = np.nan

    jerk = np.full(len(out), np.nan)
    angular_accel = np.full(len(out), np.nan)
    acc_diff = np.diff(motion_acc, axis=0, prepend=np.nan)
    gyro_diff = np.diff(gyro, axis=0, prepend=np.nan)
    valid_dt = np.isfinite(dt)
    jerk[valid_motion_acc & valid_dt] = np.linalg.norm(acc_diff[valid_motion_acc & valid_dt], axis=1) / dt[valid_motion_acc & valid_dt]
    angular_accel[valid_gyro & valid_dt] = np.linalg.norm(gyro_diff[valid_gyro & valid_dt], axis=1) / dt[valid_gyro & valid_dt]

    intensity_signal = np.maximum(0.0, robust_z(acc_dyn)) + np.maximum(0.0, robust_z(gyro_mag))
    event_signal = np.maximum(0.0, robust_z(jerk)) + 0.5 * np.maximum(0.0, robust_z(angular_accel))
    micro_signal = np.maximum(0.0, robust_z(gyro_highfreq)) + 0.5 * np.maximum(0.0, robust_z(acc_dyn))

    out["raw_acc_mag"] = acc_mag
    out["linear_accel_x"] = motion_acc[:, 0]
    out["linear_accel_y"] = motion_acc[:, 1]
    out["linear_accel_z"] = motion_acc[:, 2]
    out["linear_acc_mag"] = linear_acc_mag
    out["acc_mag"] = signal_acc_mag
    out["gyro_mag"] = gyro_mag
    out["acc_dyn"] = acc_dyn
    out["gyro_highfreq"] = gyro_highfreq
    out["jerk_mag"] = jerk
    out["angular_accel_mag"] = angular_accel
    out["intensity_signal"] = intensity_signal
    out["event_signal"] = event_signal
    out["micro_signal"] = micro_signal
    out.attrs["sample_rate"] = sample_rate
    out.attrs["acc_mode"] = resolved_acc_mode
    out.attrs["gravity_compensation_method"] = gravity_info["method"]
    out.attrs["gravity_compensation_error_median"] = gravity_info["error_median"]
    out.attrs["gravity_magnitude"] = gravity_info["gravity_magnitude"]
    return out


def orientation_delta_deg(window_df):
    if all(col in window_df.columns for col in ("quat_w", "quat_x", "quat_y", "quat_z")):
        q = window_df[["quat_w", "quat_x", "quat_y", "quat_z"]].dropna().to_numpy(dtype=float)
        if len(q) >= 2:
            q0 = q[0] / max(np.linalg.norm(q[0]), 1e-9)
            q1 = q[-1] / max(np.linalg.norm(q[-1]), 1e-9)
            dot = float(np.clip(np.abs(np.dot(q0, q1)), -1.0, 1.0))
            return float(np.degrees(2.0 * np.arccos(dot)))

    if all(col in window_df.columns for col in ("pitch", "roll", "yaw")):
        euler = window_df[["pitch", "roll", "yaw"]].dropna().to_numpy(dtype=float)
        if len(euler) >= 2:
            return float(np.linalg.norm(euler[-1] - euler[0]))

    return np.nan


def zero_crossing_rate(window_df):
    rates = []
    duration = float(window_df["feature_time_s"].iloc[-1] - window_df["feature_time_s"].iloc[0])
    if duration <= 0:
        return 0.0

    for column in GYRO_COLS:
        values = window_df[column].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if len(values) < 3:
            continue
        centered = values - np.nanmedian(values)
        signs = np.sign(centered)
        signs = signs[signs != 0]
        if len(signs) < 2:
            rates.append(0.0)
            continue
        rates.append(float(np.sum(signs[1:] != signs[:-1]) / duration))

    return float(np.mean(rates)) if rates else 0.0


def classify_intensity(score):
    if score < 0.8:
        return "idle"
    if score < 2.0:
        return "low"
    if score < 4.0:
        return "medium"
    return "high"


def extract_events(signal_df, participant_id, task_name, full_id, sensor_id, threshold=15.0, min_gap_s=1.0):
    signal = signal_df["event_signal"].to_numpy(dtype=float)
    times = signal_df["feature_time_s"].to_numpy(dtype=float)
    frames = signal_df["feature_frame_idx"].to_numpy(dtype=float)
    rows = []
    last_event_time = -np.inf

    for idx in range(1, len(signal) - 1):
        score = signal[idx]
        if not np.isfinite(score) or score < threshold:
            continue
        if score < signal[idx - 1] or score < signal[idx + 1]:
            continue
        if times[idx] - last_event_time < min_gap_s:
            continue

        last_event_time = times[idx]
        rows.append(
            {
                "participant_id": participant_id,
                "task_name": task_name,
                "full_id": full_id,
                "sensor_id": sensor_id,
                "event_time_s": float(times[idx]),
                "event_frame": int(round(frames[idx])) if np.isfinite(frames[idx]) else -1,
                "event_type": "pickup_putdown_candidate",
                "event_score": float(score),
                "jerk_mag": float(signal_df["jerk_mag"].iloc[idx]),
                "angular_accel_mag": float(signal_df["angular_accel_mag"].iloc[idx]),
                "acc_dyn": float(signal_df["acc_dyn"].iloc[idx]),
                "gyro_mag": float(signal_df["gyro_mag"].iloc[idx]),
            }
        )

    return rows


def extract_window_features(
    signal_df,
    events_df,
    participant_id,
    task_name,
    full_id,
    sensor_id,
    window_s,
    stride_s,
    micro_threshold=2.0,
):
    if signal_df.empty:
        return []

    start = float(signal_df["feature_time_s"].min())
    end = float(signal_df["feature_time_s"].max())
    event_times = events_df["event_time_s"].to_numpy(dtype=float) if not events_df.empty else np.array([])

    rows = []
    current = start
    while current + window_s <= end + 1e-9:
        window_end = current + window_s
        win = signal_df[(signal_df["feature_time_s"] >= current) & (signal_df["feature_time_s"] < window_end)]
        current += stride_s

        if len(win) < 3:
            continue

        intensity_score = float(np.nanmean(win["intensity_signal"]))
        intensity_label = classify_intensity(intensity_score)
        micro_score = float(np.nanmean(win["micro_signal"]) + 0.15 * zero_crossing_rate(win))
        sudden_count = int(np.sum((event_times >= win["feature_time_s"].iloc[0]) & (event_times < win["feature_time_s"].iloc[-1])))
        micro_candidate = bool(micro_score >= micro_threshold and intensity_label in {"low", "medium"} and sudden_count == 0)
        grasp_candidate = bool(intensity_label in {"medium", "high"} and np.nanmean(win["gyro_mag"]) > np.nanmedian(signal_df["gyro_mag"]))

        rows.append(
            {
                "participant_id": participant_id,
                "task_name": task_name,
                "full_id": full_id,
                "sensor_id": sensor_id,
                "start_time_s": float(win["feature_time_s"].iloc[0]),
                "end_time_s": float(win["feature_time_s"].iloc[-1]),
                "start_frame": int(round(win["feature_frame_idx"].iloc[0])) if np.isfinite(win["feature_frame_idx"].iloc[0]) else -1,
                "end_frame": int(round(win["feature_frame_idx"].iloc[-1])) if np.isfinite(win["feature_frame_idx"].iloc[-1]) else -1,
                "samples": int(len(win)),
                "acc_mode": signal_df.attrs.get("acc_mode", ""),
                "gravity_compensation_method": signal_df.attrs.get("gravity_compensation_method", ""),
                "gravity_compensation_error_median": signal_df.attrs.get("gravity_compensation_error_median", np.nan),
                "raw_acc_mag_mean": float(np.nanmean(win["raw_acc_mag"])),
                "linear_acc_mag_mean": float(np.nanmean(win["linear_acc_mag"])),
                "acc_mag_mean": float(np.nanmean(win["acc_mag"])),
                "acc_mag_std": float(np.nanstd(win["acc_mag"])),
                "acc_dyn_mean": float(np.nanmean(win["acc_dyn"])),
                "acc_dyn_max": float(np.nanmax(win["acc_dyn"])),
                "gyro_mag_mean": float(np.nanmean(win["gyro_mag"])),
                "gyro_mag_std": float(np.nanstd(win["gyro_mag"])),
                "gyro_mag_max": float(np.nanmax(win["gyro_mag"])),
                "jerk_mag_mean": float(np.nanmean(win["jerk_mag"])),
                "jerk_mag_max": float(np.nanmax(win["jerk_mag"])),
                "angular_accel_mag_mean": float(np.nanmean(win["angular_accel_mag"])),
                "angular_accel_mag_max": float(np.nanmax(win["angular_accel_mag"])),
                "orientation_delta_deg": orientation_delta_deg(win),
                "gyro_zero_crossing_rate": zero_crossing_rate(win),
                "intensity_score": intensity_score,
                "intensity_label": intensity_label,
                "idle_candidate": int(intensity_label == "idle"),
                "grasp_candidate": int(grasp_candidate),
                "micro_motion_score": micro_score,
                "micro_motion_candidate": int(micro_candidate),
                "sudden_event_count": sudden_count,
            }
        )

    return rows


def relative_recording_output_dir(csv_path, input_root, output_root):
    rel = Path(csv_path).relative_to(input_root)
    return Path(output_root) / rel.parts[0] / rel.parts[1]


def process_imu_csv(
    csv_path,
    input_root,
    output_root,
    window_s=1.0,
    stride_s=0.5,
    event_threshold=15.0,
    min_event_gap_s=1.0,
    micro_threshold=2.0,
    acc_mode="auto",
    gravity_magnitude=GRAVITY_MAGNITUDE,
    overwrite=False,
):
    csv_path = Path(csv_path)
    input_root = Path(input_root)
    output_root = Path(output_root)
    participant_id, task_name, full_id = parse_recording_path(csv_path, input_root)
    sensor_id = sensor_id_from_path(csv_path)
    out_dir = relative_recording_output_dir(csv_path, input_root, output_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    window_path = out_dir / f"imu__DOT_{sensor_id}_window_features.csv"
    event_path = out_dir / f"imu__DOT_{sensor_id}_events.csv"
    if window_path.exists() and event_path.exists() and not overwrite:
        return {
            "participant_id": participant_id,
            "task_name": task_name,
            "full_id": full_id,
            "sensor_id": sensor_id,
            "source_csv_path": str(csv_path),
            "window_features_csv": str(window_path),
            "events_csv": str(event_path),
            "window_count": len(pd.read_csv(window_path)),
            "event_count": len(pd.read_csv(event_path)),
            "idle_windows": np.nan,
            "grasp_candidate_windows": np.nan,
            "micro_motion_candidate_windows": np.nan,
            "acc_mode": "existing",
            "gravity_compensation_method": "existing",
            "gravity_compensation_error_median": np.nan,
            "skipped_existing": True,
        }

    raw_df = read_imu_csv(csv_path)
    signal_df = add_motion_signals(raw_df, acc_mode=acc_mode, gravity_magnitude=gravity_magnitude)
    event_rows = extract_events(
        signal_df,
        participant_id,
        task_name,
        full_id,
        sensor_id,
        threshold=event_threshold,
        min_gap_s=min_event_gap_s,
    )
    events_df = pd.DataFrame(event_rows, columns=EVENT_COLUMNS)
    window_rows = extract_window_features(
        signal_df,
        events_df,
        participant_id,
        task_name,
        full_id,
        sensor_id,
        window_s=window_s,
        stride_s=stride_s,
        micro_threshold=micro_threshold,
    )
    windows_df = pd.DataFrame(window_rows, columns=WINDOW_COLUMNS)

    windows_df.to_csv(window_path, index=False)
    events_df.to_csv(event_path, index=False)

    return {
        "participant_id": participant_id,
        "task_name": task_name,
        "full_id": full_id,
        "sensor_id": sensor_id,
        "source_csv_path": str(csv_path),
        "window_features_csv": str(window_path),
        "events_csv": str(event_path),
        "window_count": int(len(windows_df)),
        "event_count": int(len(events_df)),
        "idle_windows": int(windows_df["idle_candidate"].sum()) if len(windows_df) else 0,
        "grasp_candidate_windows": int(windows_df["grasp_candidate"].sum()) if len(windows_df) else 0,
        "micro_motion_candidate_windows": int(windows_df["micro_motion_candidate"].sum()) if len(windows_df) else 0,
        "acc_mode": signal_df.attrs.get("acc_mode", ""),
        "gravity_compensation_method": signal_df.attrs.get("gravity_compensation_method", ""),
        "gravity_compensation_error_median": signal_df.attrs.get("gravity_compensation_error_median", np.nan),
        "skipped_existing": False,
    }


def extract_imu_arm_features(
    input_root=VIDEO_ALIGNED_CSV_OUTPUT_ROOT,
    output_root=IMU_ARM_FEATURE_OUTPUT_ROOT,
    participant=None,
    full_id=None,
    window_s=1.0,
    stride_s=0.5,
    event_threshold=15.0,
    min_event_gap_s=1.0,
    micro_threshold=2.0,
    acc_mode="auto",
    gravity_magnitude=GRAVITY_MAGNITUDE,
    limit=None,
    include_calibration=False,
    overwrite=False,
):
    input_root = Path(input_root)
    output_root = Path(output_root)
    if not input_root.exists():
        raise FileNotFoundError(input_root)

    paths = iter_imu_csvs(
        input_root,
        participant=participant,
        full_id=full_id,
        include_calibration=include_calibration,
        limit=limit,
    )
    if not paths:
        raise FileNotFoundError(f"No Movella DOT IMU CSV files found under {input_root}")

    rows = []
    iterator = paths
    if tqdm is not None:
        iterator = tqdm(paths, desc="Extracting IMU arm features", unit="sensor")

    for path in iterator:
        rows.append(
            process_imu_csv(
                path,
                input_root=input_root,
                output_root=output_root,
                window_s=window_s,
                stride_s=stride_s,
                event_threshold=event_threshold,
                min_event_gap_s=min_event_gap_s,
                micro_threshold=micro_threshold,
                acc_mode=acc_mode,
                gravity_magnitude=gravity_magnitude,
                overwrite=overwrite,
            )
        )

    index_df = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / "imu_arm_feature_index.csv"
    index_df.to_csv(index_path, index=False)

    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "index_csv": str(index_path),
        "sensor_file_count": int(len(index_df)),
        "recording_count": int(index_df["full_id"].nunique()),
        "window_count": int(index_df["window_count"].sum()),
        "event_count": int(index_df["event_count"].sum()),
        "idle_windows": int(index_df.get("idle_windows", pd.Series(dtype=float)).fillna(0).sum()),
        "grasp_candidate_windows": int(index_df.get("grasp_candidate_windows", pd.Series(dtype=float)).fillna(0).sum()),
        "micro_motion_candidate_windows": int(index_df.get("micro_motion_candidate_windows", pd.Series(dtype=float)).fillna(0).sum()),
        "window_s": float(window_s),
        "stride_s": float(stride_s),
        "event_threshold": float(event_threshold),
        "min_event_gap_s": float(min_event_gap_s),
        "micro_threshold": float(micro_threshold),
        "acc_mode": acc_mode,
        "gravity_magnitude": float(gravity_magnitude),
        "gravity_compensation_methods": (
            index_df["gravity_compensation_method"].value_counts(dropna=False).to_dict()
            if "gravity_compensation_method" in index_df.columns
            else {}
        ),
        "feature_policy": {
            "acceleration": "Default acc_mode=auto uses quaternion orientation to subtract gravity from raw accel_x/y/z. If quaternion compensation is unavailable, it falls back to rolling-median acceleration-magnitude baseline.",
            "action_intensity": "Robust within-recording acc_dyn/linear acceleration + gyro magnitude score; labels idle/low/medium/high. grasp_candidate marks medium/high active windows.",
            "sudden_events": "Local peaks in robust linear-acceleration jerk + angular acceleration signal; event_type is pickup_putdown_candidate.",
            "micro_motion": "Short-window high-frequency gyro/acc signal with gyro zero-crossing rate, excluding sudden-event windows.",
            "sensor_mapping": "Outputs are per Movella DOT sensor_id because no left/right arm placement map is available in the CSV files.",
        },
        "skipped_calibration_tasks": [] if include_calibration else sorted(SKIP_TASKS),
    }

    summary_path = output_root / "imu_arm_feature_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return index_df, summary


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Extract action intensity, sudden-event, and micro-motion features from Movella DOT IMU CSV files."
    )
    parser.add_argument("--input-root", default=str(VIDEO_ALIGNED_CSV_OUTPUT_ROOT))
    parser.add_argument("--output-root", default=str(IMU_ARM_FEATURE_OUTPUT_ROOT))
    parser.add_argument("--participant", default=None, help="Optional participant id filter, e.g. IA_4_H_02.")
    parser.add_argument("--full-id", action="append", default=None, help="Optional full recording id filter. Can be repeated.")
    parser.add_argument("--window-s", type=float, default=1.0, help="Sliding-window length in seconds.")
    parser.add_argument("--stride-s", type=float, default=0.5, help="Sliding-window stride in seconds.")
    parser.add_argument("--event-threshold", type=float, default=15.0, help="Robust event peak threshold. Higher is more conservative.")
    parser.add_argument("--min-event-gap-s", type=float, default=1.0, help="Minimum seconds between sudden-event peaks.")
    parser.add_argument("--micro-threshold", type=float, default=2.0, help="Micro-motion window score threshold.")
    parser.add_argument(
        "--acc-mode",
        choices=("auto", "quaternion", "magnitude_baseline"),
        default="auto",
        help=(
            "Acceleration preprocessing. auto uses quaternion gravity compensation "
            "when possible and falls back to magnitude_baseline."
        ),
    )
    parser.add_argument("--gravity-magnitude", type=float, default=GRAVITY_MAGNITUDE, help="Gravity magnitude in m/s^2.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of sensor CSV files.")
    parser.add_argument("--include-calibration", action="store_true", help="Include Eye1, Eye2, and Recalibration1 recordings.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing feature CSV files.")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    index_df, summary = extract_imu_arm_features(
        input_root=args.input_root,
        output_root=args.output_root,
        participant=args.participant,
        full_id=args.full_id,
        window_s=args.window_s,
        stride_s=args.stride_s,
        event_threshold=args.event_threshold,
        min_event_gap_s=args.min_event_gap_s,
        micro_threshold=args.micro_threshold,
        acc_mode=args.acc_mode,
        gravity_magnitude=args.gravity_magnitude,
        limit=args.limit,
        include_calibration=args.include_calibration,
        overwrite=args.overwrite,
    )

    print("IMU arm feature extraction completed.")
    print(json.dumps(summary, indent=2))
    print("\nTop recordings by candidate events:")
    grouped = (
        index_df.groupby(["participant_id", "full_id"], as_index=False)
        .agg(
            sensor_files=("sensor_id", "count"),
            windows=("window_count", "sum"),
            events=("event_count", "sum"),
            grasp_windows=("grasp_candidate_windows", "sum"),
            micro_windows=("micro_motion_candidate_windows", "sum"),
        )
        .sort_values(["events", "grasp_windows", "micro_windows"], ascending=False)
    )
    print(grouped.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
