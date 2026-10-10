from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent

DATASET_ROOT = Path("/home/julius/datasets/seminar_object_detection")
VIDEO_ROOT = (
    DATASET_ROOT
    / "iAssistADL_dataset_export_2026-05-18"
    / "world_camera_videos"
)
OUTPUT_ROOT = Path("/home/julius/datasets/seminar_object_detection"
                   "/lower_feature_outcome/yolo_object_outcome_csv")

MODEL_PATH = Path("/home/julius/models/yolov8n.pt") 
# Adjust the path to your YOLO model if necessary
# a separate folder for models is recommended
CONFIDENCE_THRESHOLD = 0.25 # Adjust this threshold based on your requirements  

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
SKIP_VIDEO_KEYWORDS = ("Eye1", "Eye2", "Recalibration1")
