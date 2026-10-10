# YOLO Features Extraction

This folder contains the active YOLO pipeline for extracting object-detection
features from world-camera videos.

The current maintained path uses a COCO YOLO model and exports one CSV per
video.

## Active Files

```text
yolo_features_extraction/
  config.py
  object_detection_csv_output.py
  run_yolo11s_coco_output.sh
```

`config.py` defines the default dataset paths, output folder, model path,
detection confidence threshold, supported video extensions, and calibration
videos to skip.

`object_detection_csv_output.py` loads the YOLO model, scans videos
recursively, runs detection frame by frame, and writes detection results to CSV.

`run_yolo11s_coco_output.sh` is the main convenience entry point for running
YOLO11s COCO detection in the `object_detection` conda environment.

## Run

From the project root:

```bash
./yolo_features_extraction/run_yolo11s_coco_output.sh
```

Useful overrides:

```bash
MODEL=yolo11s.pt ./yolo_features_extraction/run_yolo11s_coco_output.sh
LIMIT=3 ./yolo_features_extraction/run_yolo11s_coco_output.sh
OVERWRITE=0 ./yolo_features_extraction/run_yolo11s_coco_output.sh
```

The script writes CSV files to:

```text
/home/julius/datasets/seminar_object_detection/lower_feature_outcome/yolo_object_outcome_csv
```

## Output Columns

Each row is one detected object in one frame:

```text
video
frame_id
class_id
confidence
x1, y1, x2, y2
cx, cy
width, height
area
```

The bounding box coordinates are image pixel coordinates. `cx` and `cy` are the
bounding-box center, and `area` is `width * height`.

## Archived ADL14 Experiment

The ADL14 custom-object model work was not continued as the main pipeline. Its
training, inference, and COCO+ADL14 merge scripts were moved to:

```text
yolo_features_extraction/archive/adl14_experiment/
```

Keep that folder for reference unless the experiment is intentionally revived.
