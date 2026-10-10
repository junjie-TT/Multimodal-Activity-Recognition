"""Shared helpers for the causal SVM current-action baseline."""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = Path(__file__).resolve().parent
INTERMEDIATE_DIR = BASELINE_DIR / "intermediate"
RESULTS_DIR = BASELINE_DIR / "results"

DEFAULT_FPS = 30.0
DEFAULT_GAZE_DIR = Path(
    "/home/julius/datasets/seminar_object_detection/upper_feature_outcome/"
    "gaze_object_features/probabilistic_heatmap_frame_aggregation"
)
DEFAULT_HAND_DIR = Path(
    "/home/julius/datasets/seminar_object_detection/upper_feature_outcome/"
    "hand_object_interaction_features"
)
DEFAULT_DEPTH_DIR = Path(
    "/home/julius/datasets/seminar_object_detection/upper_feature_outcome/"
    "depth_scene_features"
)
DEFAULT_HEAD_IMU_DIR = Path(
    "/home/julius/datasets/seminar_object_detection/upper_feature_outcome/"
    "head_imu_features"
)
DEFAULT_IMU_DIR = Path(
    "/home/julius/datasets/seminar_object_detection/upper_feature_outcome/"
    "imu_joint_kinematics_features"
)
DEFAULT_SKEL_DIR = Path(
    "/home/julius/datasets/seminar_object_detection/upper_feature_outcome/"
    "skeleton_upper_limb_kinematics_features"
)
DEFAULT_ANNOTATIONS_DIR = Path(
    "/home/julius/datasets/seminar_object_detection/"
    "iAssistADL_dataset_export_2026-05-18/annotations"
)

FRAME_LEVEL_PATH = INTERMEDIATE_DIR / "frame_level_multimodal_features.pkl"

PARTICIPANT_RE = re.compile(r"IA_\d+_[A-Za-z]_\d+")
CALIBRATION_TASKS = {"Eye1", "Eye2", "Recalibration1"}
TARGET_MODES = ("task", "step", "task_step")

FRAME_ALIASES = ("frame_id", "video_frame_idx", "video_frame_id", "frame_idx", "frame")
TIMESTAMP_ALIASES = (
    "timestamp",
    "aligned_timestamp",
    "video_time_s",
    "time_sec",
    "time_s",
    "t",
)
KEY_COLUMNS = {
    "participant_id",
    "task_name",
    "video_id",
    "video_path",
    "filename",
    "frame_id",
}


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def existing_dir(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.exists():
        raise argparse.ArgumentTypeError(f"Directory does not exist: {path}")
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"Not a directory: {path}")
    return path


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def format_seconds_for_name(value: float) -> str:
    text = f"{value:g}".replace(".", "p")
    return text


def window_feature_path(target_mode: str, window_sec: float = 1.5, stride_sec: float = 0.5) -> Path:
    window_tag = format_seconds_for_name(window_sec)
    stride_tag = format_seconds_for_name(stride_sec)
    return INTERMEDIATE_DIR / f"window_features_{window_tag}s_stride{stride_tag}s_{target_mode}.pkl"


def infer_participant_id(path: Path, row_value: object | None = None) -> str | None:
    if row_value is not None and pd.notna(row_value):
        match = PARTICIPANT_RE.search(str(row_value))
        if match:
            return match.group(0)
    match = PARTICIPANT_RE.search(str(path))
    return match.group(0) if match else None


def infer_recording_name(path: Path, participant_id: str | None = None) -> str | None:
    stem = path.name
    for suffix in (
        ".annotations.json",
        ".alignment.json",
        ".objects.json",
        ".segments.csv",
        ".parquet",
        ".csv",
    ):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break

    if participant_id and stem.startswith(participant_id + "_"):
        stem = stem[len(participant_id) + 1 :]

    suffix_patterns = (
        r"_gaze_yolo_probabilistic_heatmap_frame_objects$",
        r"_gaze_yolo_probabilistic_heatmap_frame$",
        r"_hand_object_frame_features$",
        r"_hoi_v2_frame_topk$",
        r"_hand_object_pair_features$",
        r"_depth_scene_frame_features$",
        r"_head_imu_frame_features$",
        r"_imu_joint_kinematics_frame_features$",
        r"_Body_\d+_skeleton_upper_limb_kinematics_frame_features$",
        r"_skeleton_upper_limb_kinematics_frame_features$",
    )
    for pattern in suffix_patterns:
        stem = re.sub(pattern, "", stem)

    return stem or None


def normalize_label(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return re.sub(r"\s+", "_", text)


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported table format: {path}")


def read_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataframe format: {path}")


def write_dataframe(df: pd.DataFrame, path: Path) -> None:
    ensure_parent(path)
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(path, index=False)
        return
    if suffix in {".pkl", ".pickle"}:
        df.to_pickle(path)
        return
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return
    raise ValueError(f"Unsupported dataframe format: {path}")


def iter_feature_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in {".csv", ".parquet"}:
            yield path
