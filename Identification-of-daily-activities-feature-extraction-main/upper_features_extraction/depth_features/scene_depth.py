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


DEPTH_COLS = ("min_depth", "max_depth", "mean_depth", "median_depth", "valid_ratio")
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
    "min_depth",
    "max_depth",
    "mean_depth",
    "median_depth",
    "valid_ratio",
    "depth_range",
    "near_scene_score",
    "foreground_contrast",
    "relative_foreground_contrast",
    "delta_min_depth",
    "delta_median_depth",
    "min_depth_velocity",
    "median_depth_velocity",
    "depth_velocity",
    "min_depth_acceleration",
    "median_depth_acceleration",
    "depth_acceleration",
    "min_depth_std",
    "median_depth_std",
    "mean_depth_std",
    "valid_ratio_std",
    "depth_change_peak",
    "min_depth_change_peak",
    "median_depth_change_peak",
    "foreground_contrast_std",
    "foreground_contrast_mean",
    "relative_foreground_contrast_mean",
]
SKIP_TASKS = {"Eye1", "Eye2", "Recalibration1"}


def parse_recording_path(csv_path, input_root):
    rel = Path(csv_path).relative_to(input_root)
    participant_id = rel.parts[0]
    full_id = rel.parts[1]
    prefix = f"{participant_id}_"
    task_name = full_id[len(prefix):] if full_id.startswith(prefix) else full_id
    return participant_id, task_name, full_id


def iter_depth_csvs(input_root, participant=None, full_id=None, include_calibration=False, limit=None):
    input_root = Path(input_root)
    paths = sorted(input_root.glob(f"*/*/{config.DEPTH_MODALITY}.csv"))

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


def read_depth_csv(path):
    df = pd.read_csv(path)
    missing = sorted(set(DEPTH_COLS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing depth columns in {path}: {missing}")

    out = df.copy()
    for column in ["frame_idx", "time_s", "video_time_s", "video_frame_idx", *DEPTH_COLS]:
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


def finite_difference(values, times):
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype=float)
    out = np.full(len(values), np.nan, dtype=float)
    dt = np.diff(times, prepend=np.nan)
    diff = np.diff(values, prepend=np.nan)
    valid = np.isfinite(diff) & np.isfinite(dt) & (dt > 0)
    out[valid] = diff[valid] / dt[valid]
    return out


def delta(values):
    values = np.asarray(values, dtype=float)
    return np.diff(values, prepend=np.nan)


def rolling_series(values, times):
    return pd.Series(values, index=pd.to_timedelta(times, unit="s"))


def rolling_mean(values, times, window_s):
    window = f"{int(round(window_s * 1000))}ms"
    return rolling_series(values, times).rolling(window, min_periods=1).mean().to_numpy(dtype=float)


def rolling_std(values, times, window_s):
    window = f"{int(round(window_s * 1000))}ms"
    return rolling_series(values, times).rolling(window, min_periods=1).std().fillna(0.0).to_numpy(dtype=float)


def rolling_count(values, times, window_s):
    window = f"{int(round(window_s * 1000))}ms"
    return rolling_series(values, times).rolling(window, min_periods=1).count().to_numpy(dtype=float)


def rolling_abs_peak(values, times, window_s):
    def finite_abs_peak(window_values):
        window_values = window_values[np.isfinite(window_values)]
        if len(window_values) == 0:
            return np.nan
        return float(np.nanmax(np.abs(window_values)))

    window = f"{int(round(window_s * 1000))}ms"
    return (
        rolling_series(values, times)
        .rolling(window, min_periods=1)
        .apply(finite_abs_peak, raw=True)
        .to_numpy(dtype=float)
    )


def add_depth_features(df, window_s=config.DEPTH_FEATURE_WINDOW_S, eps=config.DEPTH_EPS):
    out = df.copy()
    times = out["feature_time_s"].to_numpy(dtype=float)

    min_depth = numeric_column(out, "min_depth").to_numpy(dtype=float)
    max_depth = numeric_column(out, "max_depth").to_numpy(dtype=float)
    mean_depth = numeric_column(out, "mean_depth").to_numpy(dtype=float)
    median_depth = numeric_column(out, "median_depth").to_numpy(dtype=float)
    valid_ratio = numeric_column(out, "valid_ratio").to_numpy(dtype=float)

    depth_range = max_depth - min_depth
    near_scene_score = 1.0 / (min_depth + eps)
    foreground_contrast = median_depth - min_depth
    relative_foreground_contrast = foreground_contrast / (median_depth + eps)

    delta_min_depth = delta(min_depth)
    delta_median_depth = delta(median_depth)
    min_depth_velocity = finite_difference(min_depth, times)
    median_depth_velocity = finite_difference(median_depth, times)
    min_depth_acceleration = finite_difference(min_depth_velocity, times)
    median_depth_acceleration = finite_difference(median_depth_velocity, times)

    window_sample_count = rolling_count(times, times, window_s)
    window_duration_s = np.minimum(window_s, times - np.nanmin(times))
    window_start_time_s = times - window_duration_s
    min_depth_change_peak = rolling_abs_peak(delta_min_depth, times, window_s)
    median_depth_change_peak = rolling_abs_peak(delta_median_depth, times, window_s)

    return pd.DataFrame(
        {
            "frame_idx": numeric_column(out, "frame_idx"),
            "time_s": numeric_column(out, "time_s"),
            "video_time_s": numeric_column(out, "video_time_s") if "video_time_s" in out.columns else out["feature_time_s"],
            "video_frame_idx": out["feature_frame_idx"],
            "window_start_time_s": window_start_time_s,
            "window_duration_s": window_duration_s,
            "window_sample_count": window_sample_count.astype(int),
            "min_depth": min_depth,
            "max_depth": max_depth,
            "mean_depth": mean_depth,
            "median_depth": median_depth,
            "valid_ratio": valid_ratio,
            "depth_range": depth_range,
            "near_scene_score": near_scene_score,
            "foreground_contrast": foreground_contrast,
            "relative_foreground_contrast": relative_foreground_contrast,
            "delta_min_depth": delta_min_depth,
            "delta_median_depth": delta_median_depth,
            "min_depth_velocity": min_depth_velocity,
            "median_depth_velocity": median_depth_velocity,
            "depth_velocity": median_depth_velocity,
            "min_depth_acceleration": min_depth_acceleration,
            "median_depth_acceleration": median_depth_acceleration,
            "depth_acceleration": median_depth_acceleration,
            "min_depth_std": rolling_std(min_depth, times, window_s),
            "median_depth_std": rolling_std(median_depth, times, window_s),
            "mean_depth_std": rolling_std(mean_depth, times, window_s),
            "valid_ratio_std": rolling_std(valid_ratio, times, window_s),
            "depth_change_peak": np.fmax(min_depth_change_peak, median_depth_change_peak),
            "min_depth_change_peak": min_depth_change_peak,
            "median_depth_change_peak": median_depth_change_peak,
            "foreground_contrast_std": rolling_std(foreground_contrast, times, window_s),
            "foreground_contrast_mean": rolling_mean(foreground_contrast, times, window_s),
            "relative_foreground_contrast_mean": rolling_mean(relative_foreground_contrast, times, window_s),
        }
    )


def output_path_for(csv_path, input_root, output_root):
    rel = Path(csv_path).relative_to(input_root)
    return Path(output_root) / rel.parts[0] / f"{rel.parts[1]}_depth_scene_frame_features.csv"


def process_depth_csv(csv_path, input_root, output_root, window_s=config.DEPTH_FEATURE_WINDOW_S, overwrite=False):
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
            "valid_depth_row_count": np.nan,
            "skipped_existing": True,
        }

    raw_df = read_depth_csv(csv_path)
    feature_df = add_depth_features(raw_df, window_s=window_s)
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
        "valid_depth_row_count": int(feature_df["min_depth"].notna().sum()),
        "skipped_existing": False,
    }


def extract_depth_scene_features(
    input_root=config.VIDEO_ALIGNED_SENSOR_ROOT,
    output_root=config.DEPTH_FEATURE_FRAME_OUTPUT_ROOT,
    index_path=None,
    summary_path=None,
    participant=None,
    full_id=None,
    window_s=config.DEPTH_FEATURE_WINDOW_S,
    include_calibration=False,
    limit=None,
    overwrite=False,
):
    input_root = Path(input_root)
    output_root = Path(output_root)
    index_path = Path(index_path) if index_path is not None else output_root.parent / config.DEPTH_FEATURE_INDEX.name
    summary_path = Path(summary_path) if summary_path is not None else output_root.parent / config.DEPTH_FEATURE_SUMMARY.name
    if not input_root.exists():
        raise FileNotFoundError(input_root)

    paths = iter_depth_csvs(
        input_root,
        participant=participant,
        full_id=full_id,
        include_calibration=include_calibration,
        limit=limit,
    )
    if not paths:
        raise FileNotFoundError(f"No {config.DEPTH_MODALITY}.csv files found under {input_root}")

    rows = []
    iterator = paths
    if tqdm is not None:
        iterator = tqdm(paths, desc="Extracting depth scene features", unit="recording")

    for path in iterator:
        rows.append(
            process_depth_csv(
                path,
                input_root=input_root,
                output_root=output_root,
                window_s=window_s,
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
        "valid_depth_row_count": int(index_df["valid_depth_row_count"].fillna(0).sum()),
        "window_s": float(window_s),
        "depth_unit": "same as zed__depth input, documented as mm",
        "skipped_calibration_tasks": [] if include_calibration else sorted(SKIP_TASKS),
        "feature_policy": {
            "time_axis": "Uses video_time_s/video_frame_idx from aligned_sensor_with_YOLO when available.",
            "window": "All rolling statistics are causal over the current sample and the previous window_s seconds.",
            "foreground_contrast": "median_depth - min_depth.",
            "relative_foreground_contrast": "(median_depth - min_depth) / (median_depth + eps).",
            "depth_velocity": "Alias for median_depth_velocity; min_depth_velocity is also retained.",
            "depth_acceleration": "Alias for median_depth_acceleration; min_depth_acceleration is also retained.",
            "depth_change_peak": "Maximum absolute min/median depth delta inside the causal window.",
        },
    }

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return index_df, summary


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Extract upper-level scene depth features from zed__depth CSV files."
    )
    parser.add_argument("--input-root", default=str(config.VIDEO_ALIGNED_SENSOR_ROOT))
    parser.add_argument("--output-root", default=str(config.DEPTH_FEATURE_FRAME_OUTPUT_ROOT))
    parser.add_argument("--index-path", default=None)
    parser.add_argument("--summary-path", default=None)
    parser.add_argument("--participant", default=None, help="Optional participant id filter, e.g. IA_4_H_02.")
    parser.add_argument("--full-id", action="append", default=None, help="Optional full recording id filter. Can be repeated.")
    parser.add_argument("--window-s", type=float, default=config.DEPTH_FEATURE_WINDOW_S, help="Causal rolling-window length in seconds.")
    parser.add_argument("--include-calibration", action="store_true", help="Include Eye1, Eye2, and Recalibration1 recordings.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of recordings.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing feature CSV files.")
    return parser


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    index_df, summary = extract_depth_scene_features(
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

    print("Depth scene feature extraction completed.")
    print(json.dumps(summary, indent=2))
    print("\nProcessed recordings:")
    print(index_df[["participant_id", "full_id", "row_count", "video_frame_count", "valid_depth_row_count"]].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
