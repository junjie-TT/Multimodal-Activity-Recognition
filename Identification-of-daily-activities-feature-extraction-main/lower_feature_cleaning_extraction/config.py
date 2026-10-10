from pathlib import Path


DATASET_ROOT = Path("/home/julius/datasets/seminar_object_detection")
EXPORT_ROOT = DATASET_ROOT / "iAssistADL_dataset_export_2026-05-18"

ALIGNED_H5_ROOT = EXPORT_ROOT / "aligned_h5"
ANNOTATIONS_ROOT = EXPORT_ROOT / "annotations"
WORLD_CAMERA_VIDEO_ROOT = EXPORT_ROOT / "world_camera_videos"
OUTPUT_ROOT = DATASET_ROOT / "lower_feature_outcome" / "clean_sensor_csv"
ALIGNED_SENSOR_WITH_YOLO_ROOT = DATASET_ROOT / "lower_feature_outcome" / "aligned_sensor_with_YOLO"
YOLO_HAND_OUTCOME_ROOT = DATASET_ROOT / "lower_feature_outcome" / "yolo_hand_outcome_csv"
YOLO_LEFT_RIGHT_HAND_OUTCOME_ROOT = DATASET_ROOT / "lower_feature_outcome" / "yolo_left_right_hand_outcome_csv"
HAND_YOLO_MODEL_ROOT = Path.home() / "models" / "hand_yolo"
HAND_YOLO_MODEL_PATH = HAND_YOLO_MODEL_ROOT / "hand_yolov8s.pt"
LEFT_RIGHT_HAND_YOLO_MODEL_ROOT = Path.home() / "models" / "EgoHands_YOLO" / "left_right_hand_yolov8s_img640_b4"
LEFT_RIGHT_HAND_YOLO_MODEL_PATH = LEFT_RIGHT_HAND_YOLO_MODEL_ROOT / "weights" / "best.pt"
VIDEO_FPS = 30

SKIP_TASKS = {"Eye1", "Eye2", "Recalibration1"}

MANIFEST_OUTPUT = OUTPUT_ROOT / "clean_manifest.csv"
EXCLUDED_OUTPUT = OUTPUT_ROOT / "excluded_calibration_manifest.csv"
SUMMARY_OUTPUT = OUTPUT_ROOT / "cleaning_summary.json"
ALIGNMENT_MANIFEST_OUTPUT = OUTPUT_ROOT / "alignment_manifest.csv"

CLEAN_CSV_OUTPUT_ROOT = OUTPUT_ROOT
CLEAN_CSV_INDEX_OUTPUT = CLEAN_CSV_OUTPUT_ROOT / "clean_sensor_csv_index.csv"
VIDEO_ALIGNED_CSV_OUTPUT_ROOT = ALIGNED_SENSOR_WITH_YOLO_ROOT
VIDEO_ALIGNED_CSV_INDEX_OUTPUT = VIDEO_ALIGNED_CSV_OUTPUT_ROOT / "video_aligned_sensor_csv_index.csv"
MISSING_ALIGNMENT_OUTPUT = VIDEO_ALIGNED_CSV_OUTPUT_ROOT / "missing_alignment.csv"
GAZE_XY_FILLED_OUTPUT_ROOT = DATASET_ROOT / "lower_feature_outcome" / "gaze_xy_filled_sensor_csv"
GAZE_XY_FILLED_INDEX_OUTPUT = GAZE_XY_FILLED_OUTPUT_ROOT / "gaze_xy_filled_index.csv"
IMU_ARM_FEATURE_OUTPUT_ROOT = DATASET_ROOT / "lower_feature_outcome" / "imu_arm_features"
IMU_ARM_FEATURE_INDEX_OUTPUT = IMU_ARM_FEATURE_OUTPUT_ROOT / "imu_arm_feature_index.csv"

# In the aligned H5 files, skeleton keypoint coordinates can use -1 as an
# invalid marker. IMU values can legitimately be negative, so -1 is not treated
# as invalid for all modalities globally.
MINUS_ONE_INVALID_MODALITY_PREFIXES = ("zed__skeleton__",)
MINUS_ONE_INVALID_COLUMNS = {
    "eye_tracker__gaze": ("x", "y"),
}
