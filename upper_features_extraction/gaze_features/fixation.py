import json
from pathlib import Path

import numpy as np
import pandas as pd

from upper_features_extraction import config

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


GAZE_MODALITY = "eye_tracker__gaze"
EYE_IMU_MODALITY = "eye_tracker__imu"
REQUIRED_COLUMNS = ["time_s", "video_frame_idx", "x", "y"]
LEFT_AXIS_COLUMNS = ["optical_axis_left_x", "optical_axis_left_y", "optical_axis_left_z"]
RIGHT_AXIS_COLUMNS = ["optical_axis_right_x", "optical_axis_right_y", "optical_axis_right_z"]
HEAD_GYRO_COLUMNS = ["gyro_x", "gyro_y", "gyro_z"]
HEAD_GYRO_INTERPOLATION_MAX_GAP_S = 0.05


def read_gaze_index(index_path=config.GAZE_XY_FILLED_INDEX):
    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(
            f"Missing gaze nearest-fill index: {index_path}. "
            "Run lower_feature_cleaning_extraction.fill_gaze_xy_nearest first."
        )

    index_df = pd.read_csv(index_path)
    gaze_df = index_df[index_df["modality"] == GAZE_MODALITY].copy()
    if gaze_df.empty:
        raise ValueError(f"No {GAZE_MODALITY} rows found in {index_path}.")
    return gaze_df


def numeric_series(df, column):
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce")


def normalize_vectors(values):
    norms = np.linalg.norm(values, axis=1)
    out = np.full(values.shape, np.nan, dtype=float)
    valid = np.isfinite(values).all(axis=1) & (norms > 0)
    out[valid] = values[valid] / norms[valid, None]
    return out, valid


def average_optical_axis(df):
    left = df.reindex(columns=LEFT_AXIS_COLUMNS).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    right = df.reindex(columns=RIGHT_AXIS_COLUMNS).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    left_norm, left_valid = normalize_vectors(left)
    right_norm, right_valid = normalize_vectors(right)

    averaged = np.full(left.shape, np.nan, dtype=float)
    both = left_valid & right_valid
    left_only = left_valid & ~right_valid
    right_only = right_valid & ~left_valid

    averaged[both] = (left_norm[both] + right_norm[both]) / 2.0
    averaged[left_only] = left_norm[left_only]
    averaged[right_only] = right_norm[right_only]

    averaged_norm, averaged_valid = normalize_vectors(averaged)
    return averaged_norm, averaged_valid


def compute_xy_velocity(df):
    time_s = numeric_series(df, "time_s").to_numpy(dtype=float)
    x = numeric_series(df, "x").to_numpy(dtype=float)
    y = numeric_series(df, "y").to_numpy(dtype=float)

    dx = np.diff(x, prepend=np.nan)
    dy = np.diff(y, prepend=np.nan)
    dt = np.diff(time_s, prepend=np.nan)

    velocity = np.full(len(df), np.nan, dtype=float)
    valid = np.isfinite(dx) & np.isfinite(dy) & np.isfinite(dt) & (dt > 0)
    velocity[valid] = np.sqrt(dx[valid] ** 2 + dy[valid] ** 2) / dt[valid]
    return velocity


def compute_optical_axis_angular_velocity(df):
    axis, axis_valid = average_optical_axis(df)
    time_s = numeric_series(df, "time_s").to_numpy(dtype=float)
    dt = np.diff(time_s, prepend=np.nan)

    angular_velocity = np.full(len(df), np.nan, dtype=float)
    for i in range(1, len(df)):
        if not axis_valid[i] or not axis_valid[i - 1]:
            continue
        if not np.isfinite(dt[i]) or dt[i] <= 0:
            continue
        cosine = float(np.clip(np.dot(axis[i], axis[i - 1]), -1.0, 1.0))
        angle_deg = np.degrees(np.arccos(cosine))
        angular_velocity[i] = angle_deg / dt[i]

    return angular_velocity, axis_valid


def infer_eye_imu_path(gaze_path):
    return Path(gaze_path).parent / f"{EYE_IMU_MODALITY}.csv"


def read_eye_imu_for_gaze(gaze_path):
    imu_path = infer_eye_imu_path(gaze_path)
    if not imu_path.exists():
        return None

    imu_df = pd.read_csv(imu_path)
    required = {"time_s", *HEAD_GYRO_COLUMNS}
    if not required.issubset(imu_df.columns):
        return None

    for column in ["time_s", *HEAD_GYRO_COLUMNS]:
        imu_df[column] = pd.to_numeric(imu_df[column], errors="coerce")

    imu_df = imu_df.dropna(subset=["time_s", *HEAD_GYRO_COLUMNS]).copy()
    imu_df = imu_df.sort_values("time_s").drop_duplicates("time_s")
    if imu_df.empty:
        return None
    return imu_df


def interpolate_head_gyro_to_gaze(gaze_time_s, imu_df, max_gap_s=HEAD_GYRO_INTERPOLATION_MAX_GAP_S):
    gyro = np.full((len(gaze_time_s), 3), np.nan, dtype=float)
    valid = np.zeros(len(gaze_time_s), dtype=bool)
    if imu_df is None or imu_df.empty:
        return gyro, valid

    imu_time = imu_df["time_s"].to_numpy(dtype=float)
    if len(imu_time) < 2:
        return gyro, valid

    for axis_idx, column in enumerate(HEAD_GYRO_COLUMNS):
        values = imu_df[column].to_numpy(dtype=float)
        gyro[:, axis_idx] = np.interp(gaze_time_s, imu_time, values, left=np.nan, right=np.nan)

    pos = np.searchsorted(imu_time, gaze_time_s)
    left_pos = np.clip(pos - 1, 0, len(imu_time) - 1)
    right_pos = np.clip(pos, 0, len(imu_time) - 1)
    nearest_gap = np.minimum(np.abs(gaze_time_s - imu_time[left_pos]), np.abs(imu_time[right_pos] - gaze_time_s))

    valid = (
        np.isfinite(gaze_time_s)
        & np.isfinite(gyro).all(axis=1)
        & np.isfinite(nearest_gap)
        & (nearest_gap <= max_gap_s)
    )
    gyro[~valid] = np.nan
    return gyro, valid


def compute_head_fused_gaze_angular_velocity(df, imu_df=None):
    axis, axis_valid = average_optical_axis(df)
    time_s = numeric_series(df, "time_s").to_numpy(dtype=float)
    dt = np.diff(time_s, prepend=np.nan)

    optical_axis_angular_velocity = np.full(len(df), np.nan, dtype=float)
    fused_angular_velocity = np.full(len(df), np.nan, dtype=float)
    eye_in_head_angular_velocity = np.full(len(df), np.nan, dtype=float)
    head_gyro_magnitude = np.full(len(df), np.nan, dtype=float)

    head_gyro_deg_s, head_gyro_valid = interpolate_head_gyro_to_gaze(time_s, imu_df)
    head_gyro_rad_s = np.deg2rad(head_gyro_deg_s)
    head_gyro_magnitude[head_gyro_valid] = np.linalg.norm(head_gyro_deg_s[head_gyro_valid], axis=1)

    fused_valid = np.zeros(len(df), dtype=bool)

    for i in range(1, len(df)):
        if not axis_valid[i] or not axis_valid[i - 1]:
            continue
        if not np.isfinite(dt[i]) or dt[i] <= 0:
            continue

        cosine = float(np.clip(np.dot(axis[i], axis[i - 1]), -1.0, 1.0))
        angle_rad = np.arccos(cosine)
        optical_axis_angular_velocity[i] = np.degrees(angle_rad) / dt[i]

        dg_dt = (axis[i] - axis[i - 1]) / dt[i]
        eye_in_head_angular_velocity[i] = np.degrees(np.linalg.norm(dg_dt))

        if not head_gyro_valid[i]:
            fused_angular_velocity[i] = optical_axis_angular_velocity[i]
            continue

        # optical_axis is expressed in the head/eye-tracker frame. For a world
        # fixation estimate, combine eye-in-head direction change with head
        # angular velocity: d(g_world)/dt = R * (dg_head/dt + omega_head x g_head).
        head_rotation_component = np.cross(head_gyro_rad_s[i], axis[i])
        world_gaze_derivative = dg_dt + head_rotation_component
        fused_angular_velocity[i] = np.degrees(np.linalg.norm(world_gaze_derivative))
        fused_valid[i] = True

    return {
        "optical_axis_angular_velocity_deg_s": optical_axis_angular_velocity,
        "eye_in_head_angular_velocity_deg_s": eye_in_head_angular_velocity,
        "head_gyro_magnitude_deg_s": head_gyro_magnitude,
        "head_gyro_valid": head_gyro_valid,
        "head_fused_gaze_angular_velocity_deg_s": fused_angular_velocity,
        "head_fused_gaze_valid": fused_valid,
        "optical_axis_valid": axis_valid,
    }


def compute_valid_worn_mask(df):
    worn = numeric_series(df, "worn")
    if worn.notna().any():
        return worn.eq(1).to_numpy()
    return np.ones(len(df), dtype=bool)


def compute_blink_mask(df):
    aperture_cols = ["eyelid_aperture_left", "eyelid_aperture_right"]
    aperture_values = [
        numeric_series(df, col).to_numpy(dtype=float)
        for col in aperture_cols
        if col in df.columns
    ]

    pupil_values = [
        numeric_series(df, col).to_numpy(dtype=float)
        for col in ["pupil_diameter_left", "pupil_diameter_right"]
        if col in df.columns
    ]

    blink = np.zeros(len(df), dtype=bool)
    if aperture_values:
        aperture = np.vstack(aperture_values).T
        aperture_has_value = np.isfinite(aperture).any(axis=1)
        aperture_mean = np.full(len(df), np.nan, dtype=float)
        aperture_mean[aperture_has_value] = np.nanmean(aperture[aperture_has_value], axis=1)
        blink |= aperture_mean < config.BLINK_MIN_EYELID_APERTURE
        blink |= ~aperture_has_value

    if pupil_values:
        pupils = np.vstack(pupil_values).T
        blink |= ~np.isfinite(pupils).any(axis=1)

    return blink


def contiguous_true_segments(mask):
    segments = []
    start = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            segments.append((start, idx - 1))
            start = None
    if start is not None:
        segments.append((start, len(mask) - 1))
    return segments


def bridge_candidate_gaps(candidate_mask, time_s, max_gap_duration_ms):
    bridged = candidate_mask.copy()
    max_gap_s = max_gap_duration_ms / 1000.0
    false_segments = contiguous_true_segments(~candidate_mask)

    for start_idx, end_idx in false_segments:
        left_idx = start_idx - 1
        right_idx = end_idx + 1
        if left_idx < 0 or right_idx >= len(candidate_mask):
            continue
        if not candidate_mask[left_idx] or not candidate_mask[right_idx]:
            continue

        gap_start = time_s[start_idx]
        gap_end = time_s[end_idx]
        if not np.isfinite(gap_start) or not np.isfinite(gap_end):
            continue
        gap_duration_s = gap_end - gap_start
        if gap_duration_s <= max_gap_s:
            bridged[start_idx : end_idx + 1] = True

    return bridged


def segment_dispersion(segment_df):
    x = pd.to_numeric(segment_df["x"], errors="coerce")
    y = pd.to_numeric(segment_df["y"], errors="coerce")
    if x.empty or y.empty:
        return np.nan
    return float((x.max() - x.min()) + (y.max() - y.min()))


def segment_duration_ms(df, start_idx, end_idx):
    start_time = df["time_s"].iloc[start_idx]
    end_time = df["time_s"].iloc[end_idx]
    if not np.isfinite(start_time) or not np.isfinite(end_time):
        return np.nan
    return float((end_time - start_time) * 1000.0)


def find_fixation_segments_in_candidate_run(
    df,
    run_start_idx,
    run_end_idx,
    min_duration_ms,
    sliding_window_ms,
    max_dispersion_px,
):
    segments = []
    idx = run_start_idx

    while idx <= run_end_idx:
        window_end = idx
        while window_end <= run_end_idx:
            duration_ms = segment_duration_ms(df, idx, window_end)
            if np.isfinite(duration_ms) and duration_ms >= min_duration_ms:
                break
            window_end += 1

        if window_end > run_end_idx:
            break

        window = df.iloc[idx : window_end + 1]
        window_dispersion = segment_dispersion(window)
        if not np.isfinite(window_dispersion) or window_dispersion > max_dispersion_px:
            idx += 1
            continue

        segment_end = window_end
        while segment_end + 1 <= run_end_idx:
            candidate_duration_ms = segment_duration_ms(df, idx, segment_end + 1)
            if np.isfinite(candidate_duration_ms) and candidate_duration_ms > sliding_window_ms:
                break

            candidate = df.iloc[idx : segment_end + 2]
            candidate_dispersion = segment_dispersion(candidate)
            if not np.isfinite(candidate_dispersion) or candidate_dispersion > max_dispersion_px:
                break

            segment_end += 1

        # After finding a stable low-dispersion window, greedily extend while
        # the entire fixation remains compact. This keeps short fixations
        # responsive but still allows naturally longer stable looks.
        while segment_end + 1 <= run_end_idx:
            candidate = df.iloc[idx : segment_end + 2]
            candidate_dispersion = segment_dispersion(candidate)
            if not np.isfinite(candidate_dispersion) or candidate_dispersion > max_dispersion_px:
                break
            segment_end += 1

        segments.append((idx, segment_end))
        idx = segment_end + 1

    return segments


def extract_fixations_from_gaze(
    gaze_df,
    imu_df=None,
    min_duration_ms=config.FIXATION_MIN_DURATION_MS,
    sliding_window_ms=config.FIXATION_SLIDING_WINDOW_MS,
    max_gap_duration_ms=config.FIXATION_MAX_GAP_DURATION_MS,
    max_dispersion_px=config.FIXATION_MAX_DISPERSION_PX,
    max_gaze_velocity_px_s=config.FIXATION_MAX_GAZE_VELOCITY_PX_S,
    max_axis_angular_velocity_deg_s=config.FIXATION_MAX_OPTICAL_AXIS_ANGULAR_VELOCITY_DEG_S,
):
    df = gaze_df.copy().reset_index(drop=True)
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"gaze CSV is missing required column: {col}")

    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    df["video_frame_idx"] = pd.to_numeric(df["video_frame_idx"], errors="coerce")
    df["x"] = pd.to_numeric(df["x"], errors="coerce")
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    worn_valid = compute_valid_worn_mask(df)
    blink = compute_blink_mask(df)
    xy_valid = np.isfinite(df["x"].to_numpy(dtype=float)) & np.isfinite(df["y"].to_numpy(dtype=float))

    xy_velocity = compute_xy_velocity(df)
    angular_velocity = compute_head_fused_gaze_angular_velocity(df, imu_df=imu_df)
    axis_angular_velocity = angular_velocity["optical_axis_angular_velocity_deg_s"]
    fused_angular_velocity = angular_velocity["head_fused_gaze_angular_velocity_deg_s"]

    layer1_valid = worn_valid & ~blink & xy_valid
    raw_velocity_candidate = (
        layer1_valid
        & np.isfinite(xy_velocity)
        & (xy_velocity <= max_gaze_velocity_px_s)
        & np.isfinite(fused_angular_velocity)
        & (fused_angular_velocity <= max_axis_angular_velocity_deg_s)
    )

    # Keep the first valid sample of a low-velocity run when angular velocity is
    # undefined only because it has no previous valid optical-axis sample.
    first_valid_idx = np.flatnonzero(layer1_valid)
    if len(first_valid_idx):
        raw_velocity_candidate[first_valid_idx[0]] = True

    time_s = df["time_s"].to_numpy(dtype=float)
    velocity_candidate = bridge_candidate_gaps(
        raw_velocity_candidate,
        time_s=time_s,
        max_gap_duration_ms=max_gap_duration_ms,
    )

    df["worn_valid"] = worn_valid.astype(int)
    df["blink_like"] = blink.astype(int)
    df["layer1_valid"] = layer1_valid.astype(int)
    df["gaze_velocity_px_s"] = xy_velocity
    df["optical_axis_valid"] = angular_velocity["optical_axis_valid"].astype(int)
    df["optical_axis_angular_velocity_deg_s"] = axis_angular_velocity
    df["eye_in_head_angular_velocity_deg_s"] = angular_velocity["eye_in_head_angular_velocity_deg_s"]
    df["head_gyro_valid"] = angular_velocity["head_gyro_valid"].astype(int)
    df["head_gyro_magnitude_deg_s"] = angular_velocity["head_gyro_magnitude_deg_s"]
    df["head_fused_gaze_valid"] = angular_velocity["head_fused_gaze_valid"].astype(int)
    df["head_fused_gaze_angular_velocity_deg_s"] = fused_angular_velocity
    df["fixation_candidate_raw"] = raw_velocity_candidate.astype(int)
    df["fixation_candidate"] = velocity_candidate.astype(int)
    df["fixation_id"] = pd.NA

    fixation_rows = []
    fixation_id = 0
    for run_start_idx, run_end_idx in contiguous_true_segments(velocity_candidate):
        for start_idx, end_idx in find_fixation_segments_in_candidate_run(
            df,
            run_start_idx=run_start_idx,
            run_end_idx=run_end_idx,
            min_duration_ms=min_duration_ms,
            sliding_window_ms=sliding_window_ms,
            max_dispersion_px=max_dispersion_px,
        ):
            segment = df.iloc[start_idx : end_idx + 1].copy()
            if len(segment) < 2:
                continue

            start_time = float(segment["time_s"].iloc[0])
            end_time = float(segment["time_s"].iloc[-1])
            duration_ms = (end_time - start_time) * 1000.0
            dispersion_px = segment_dispersion(segment)

            if duration_ms < min_duration_ms:
                continue
            if not np.isfinite(dispersion_px) or dispersion_px > max_dispersion_px:
                continue

            fixation_id += 1
            df.loc[start_idx:end_idx, "fixation_id"] = fixation_id
            fixation_rows.append(
                {
                    "fixation_id": fixation_id,
                    "method": "velocity_then_sliding_dispersion",
                    "start_sample_idx": int(start_idx),
                    "end_sample_idx": int(end_idx),
                    "sample_count": int(len(segment)),
                    "start_time_s": start_time,
                    "end_time_s": end_time,
                    "duration_ms": float(duration_ms),
                    "start_video_frame_idx": int(segment["video_frame_idx"].iloc[0]),
                    "end_video_frame_idx": int(segment["video_frame_idx"].iloc[-1]),
                    "center_x": float(segment["x"].mean()),
                    "center_y": float(segment["y"].mean()),
                    "median_x": float(segment["x"].median()),
                    "median_y": float(segment["y"].median()),
                    "dispersion_px": float(dispersion_px),
                    "mean_gaze_velocity_px_s": float(segment["gaze_velocity_px_s"].mean()),
                    "mean_optical_axis_angular_velocity_deg_s": float(
                        segment["optical_axis_angular_velocity_deg_s"].mean()
                    ),
                    "mean_head_gyro_magnitude_deg_s": float(segment["head_gyro_magnitude_deg_s"].mean()),
                    "mean_head_fused_gaze_angular_velocity_deg_s": float(
                        segment["head_fused_gaze_angular_velocity_deg_s"].mean()
                    ),
                }
            )

    return df, pd.DataFrame(fixation_rows)


def output_paths(participant_id, full_id):
    sample_path = (
        config.GAZE_FIXATION_SAMPLE_OUTPUT_ROOT
        / participant_id
        / f"{full_id}_gaze_fixation_samples.csv"
    )
    segment_path = (
        config.GAZE_FIXATION_SEGMENT_OUTPUT_ROOT
        / participant_id
        / f"{full_id}_gaze_fixation_segments.csv"
    )
    return sample_path, segment_path


def run_fixation_extraction(gaze_index_path=config.GAZE_XY_FILLED_INDEX):
    gaze_index_df = read_gaze_index(gaze_index_path)
    config.GAZE_FIXATION_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    config.GAZE_FIXATION_SAMPLE_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    config.GAZE_FIXATION_SEGMENT_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    records = list(gaze_index_df.to_dict("records"))
    iterator = records
    if tqdm is not None:
        iterator = tqdm(records, desc="Extracting gaze fixations", unit="recording")

    index_rows = []
    for record in iterator:
        participant_id = record["participant_id"]
        task_name = record["task_name"]
        full_id = record["full_id"]
        gaze_path = Path(record["output_csv_path"])
        imu_df = read_eye_imu_for_gaze(gaze_path)

        gaze_df = pd.read_csv(gaze_path)
        sample_df, segment_df = extract_fixations_from_gaze(gaze_df, imu_df=imu_df)

        sample_path, segment_path = output_paths(participant_id, full_id)
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        segment_path.parent.mkdir(parents=True, exist_ok=True)
        sample_df.to_csv(sample_path, index=False)
        segment_df.to_csv(segment_path, index=False)

        index_rows.append(
            {
                "participant_id": participant_id,
                "task_name": task_name,
                "full_id": full_id,
                "gaze_csv_path": str(gaze_path),
                "eye_imu_csv_path": str(infer_eye_imu_path(gaze_path)),
                "eye_imu_used": int(imu_df is not None and not imu_df.empty),
                "sample_output_csv_path": str(sample_path),
                "segment_output_csv_path": str(segment_path),
                "sample_count": int(len(sample_df)),
                "layer1_valid_count": int(sample_df["layer1_valid"].sum()),
                "head_gyro_valid_count": int(sample_df["head_gyro_valid"].sum()),
                "fixation_candidate_count": int(sample_df["fixation_candidate"].sum()),
                "fixation_count": int(len(segment_df)),
                "total_fixation_duration_ms": float(segment_df["duration_ms"].sum())
                if not segment_df.empty
                else 0.0,
            }
        )

    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(config.GAZE_FIXATION_INDEX, index=False)

    summary = {
        "gaze_index_path": str(gaze_index_path),
        "output_root": str(config.GAZE_FIXATION_OUTPUT_ROOT),
        "index_csv": str(config.GAZE_FIXATION_INDEX),
        "recording_count": int(len(index_df)),
        "fixation_count": int(index_df["fixation_count"].sum()) if not index_df.empty else 0,
        "parameters": {
            "min_duration_ms": config.FIXATION_MIN_DURATION_MS,
            "sliding_window_ms": config.FIXATION_SLIDING_WINDOW_MS,
            "max_dispersion_px": config.FIXATION_MAX_DISPERSION_PX,
            "max_gap_duration_ms": config.FIXATION_MAX_GAP_DURATION_MS,
            "max_gaze_velocity_px_s": config.FIXATION_MAX_GAZE_VELOCITY_PX_S,
            "max_head_fused_gaze_angular_velocity_deg_s": config.FIXATION_MAX_OPTICAL_AXIS_ANGULAR_VELOCITY_DEG_S,
            "head_gyro_interpolation_max_gap_s": HEAD_GYRO_INTERPOLATION_MAX_GAP_S,
            "blink_min_eyelid_aperture": config.BLINK_MIN_EYELID_APERTURE,
        },
        "policy": (
            "Layer 1 filters out not-worn/blink-like samples. Layer 2 marks "
            "low x-y velocity and low head-fused gaze angular velocity samples "
            "as fixation candidates. The fused angular velocity combines optical "
            "axis change in the head frame with eye_tracker__imu head gyro because "
            "optical_axis is expressed in the head/eye-tracker coordinate frame. "
            "Layer 3 confirms contiguous candidates by minimum duration and "
            "maximum x/y dispersion."
        ),
    }
    with open(config.GAZE_FIXATION_SUMMARY, "w") as f:
        json.dump(summary, f, indent=2)

    return index_df, summary


def main():
    index_df, summary = run_fixation_extraction()
    print("Gaze fixation extraction completed.")
    print(json.dumps(summary, indent=2))
    if not index_df.empty:
        print("\nFixations by recording:")
        print(
            index_df[
                [
                    "full_id",
                    "sample_count",
                    "layer1_valid_count",
                    "fixation_candidate_count",
                    "fixation_count",
                    "total_fixation_duration_ms",
                ]
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()
