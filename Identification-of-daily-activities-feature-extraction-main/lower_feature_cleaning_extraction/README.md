# Lower Feature Cleaning Extraction

This folder contains the first cleaning stage for the aligned H5 sensor data.

Current stage:

1. scan `aligned_h5`
2. parse `participant_id`, `task_name`, and `full_id`
3. exclude calibration recordings:
   - `Eye1`
   - `Eye2`
   - `Recalibration1`
4. validate that each H5 file opens
5. summarize sample count, duration, and modality availability
6. write clean/excluded manifests to `lower_feature_outcome`

Run:

```bash
conda run --no-capture-output -n object_detection python -m lower_feature_cleaning_extraction.build_clean_manifest
```

Outputs:

```text
/home/julius/datasets/seminar_object_detection/lower_feature_outcome/clean_manifest.csv
/home/julius/datasets/seminar_object_detection/lower_feature_outcome/excluded_calibration_manifest.csv
/home/julius/datasets/seminar_object_detection/lower_feature_outcome/cleaning_summary.json
```

This stage does not copy or modify raw H5 files.

## Alignment Manifest

Collect all `*.alignment.json` files from `annotations/` and parse them by
participant/task:

```bash
conda run --no-capture-output -n object_detection python -m lower_feature_cleaning_extraction.build_alignment_manifest
```

Output:

```text
/home/julius/datasets/seminar_object_detection/lower_feature_outcome/clean_sensor_csv/alignment_manifest.csv
```

## Export Clean Sensor CSV

After building the manifest, export each kept H5 modality to CSV:

```bash
conda run --no-capture-output -n object_detection python -m lower_feature_cleaning_extraction.export_clean_sensor_csv
```

Output layout:

```text
/home/julius/datasets/seminar_object_detection/lower_feature_outcome/clean_sensor_csv/
  clean_sensor_csv_index.csv
  clean_sensor_csv_summary.json
  {participant_id}/
    {full_id}/
      eye_tracker__gaze.csv
      eye_tracker__imu.csv
      movella__DOT_....csv
      zed__depth.csv
      zed__skeleton__Body_....csv
```

Invalid value policy:

* `NaN`, `inf`, and `-inf` are exported as CSV missing values.
* `-1` is treated as invalid for `zed__skeleton__*` modalities.
* `-1` is treated as invalid for `eye_tracker__gaze` columns `x` and `y`.
* IMU values are not globally converted when equal to `-1`, because negative sensor readings can be valid.

## Align Sensor CSV to Video Time

After building `alignment_manifest.csv` and exporting clean sensor CSV files,
apply each recording's alignment offset:

```bash
conda run --no-capture-output -n object_detection python -m lower_feature_cleaning_extraction.apply_video_alignment_to_csv
```

The alignment transform is:

```text
video_time_s = sensor time_s - offset_s
video_frame_idx = round(video_time_s * VIDEO_FPS)
```

Output layout:

```text
/home/julius/datasets/seminar_object_detection/lower_feature_outcome/aligned_sensor_with_YOLO/
  video_aligned_sensor_csv_index.csv
  missing_alignment.csv
  video_aligned_sensor_csv_summary.json
  {participant_id}/{full_id}/{modality}.csv
```

Not every recording has an alignment JSON. If an alignment JSON is missing, the
script assumes there is no offset:

```text
offset_s = 0.0
video_time_s = sensor time_s
```

Those zero-offset assumptions are recorded in `missing_alignment.csv` for audit
purposes, but the CSV files are still exported.

## Fill Missing Gaze X/Y

After video alignment, fill missing gaze `x` and `y` coordinates with the
nearest valid gaze point in time:

```bash
conda run --no-capture-output -n object_detection python -m lower_feature_cleaning_extraction.fill_gaze_xy_nearest
```

Output:

```text
/home/julius/datasets/seminar_object_detection/lower_feature_outcome/gaze_xy_filled_sensor_csv/
  gaze_xy_filled_index.csv
  gaze_xy_filled_summary.json
  {participant_id}/{full_id}/{modality}.csv
```

Only `eye_tracker__gaze.csv` is modified. Other modalities are copied unchanged.
The gaze CSV keeps two audit columns:

```text
gaze_xy_was_missing
gaze_xy_filled_nearest
```

## IMU Arm Motion Features

Extract three heuristic feature groups from Movella DOT IMU CSV files:

* action intensity: `idle`, `low`, `medium`, `high`, plus `grasp_candidate`
* sudden events: `pickup_putdown_candidate` peaks from jerk/angular acceleration
* micro motion: small high-frequency interaction candidates

Run a quick check on one participant:

```bash
conda run --no-capture-output -n object_detection python -m lower_feature_cleaning_extraction.extract_imu_arm_features \
  --participant IA_4_H_02 \
  --overwrite
```

Run all recordings:

```bash
conda run --no-capture-output -n object_detection python -m lower_feature_cleaning_extraction.extract_imu_arm_features \
  --overwrite
```

Useful options:

```bash
--full-id IA_4_H_02_Brief_schreiben1
--window-s 1.0
--stride-s 0.5
--event-threshold 15.0
--min-event-gap-s 1.0
--micro-threshold 2.0
--acc-mode auto
--gravity-magnitude 9.80665
--limit 10
```

Output:

```text
/home/julius/datasets/seminar_object_detection/lower_feature_outcome/imu_arm_features/
  imu_arm_feature_index.csv
  imu_arm_feature_summary.json
  {participant_id}/{full_id}/
    imu__DOT_{sensor_id}_window_features.csv
    imu__DOT_{sensor_id}_events.csv
```

`*_window_features.csv` contains per-window movement features and labels:

```text
acc_mode,gravity_compensation_method,raw_acc_mag_mean,linear_acc_mag_mean,
intensity_score,intensity_label,idle_candidate,grasp_candidate,
micro_motion_score,micro_motion_candidate,sudden_event_count
```

`*_events.csv` contains sudden motion peaks:

```text
event_time_s,event_frame,event_type,event_score,jerk_mag,angular_accel_mag
```

The current output is per Movella DOT `sensor_id`. The CSV files do not include
a stable sensor-placement map, so left/right arm naming should be added only
after confirming which DOT ID was worn on which arm.

Acceleration preprocessing defaults to `--acc-mode auto`. In this mode, the
script uses `quat_w/x/y/z` to subtract gravity from raw `accel_x/y/z`, selecting
the quaternion rotation convention that minimizes low-gyro residual acceleration.
If quaternion compensation is unavailable, it falls back to the previous
rolling-median acceleration-magnitude baseline. Use `--acc-mode
magnitude_baseline` to force the old behavior for comparison.

## Hand YOLO Detection

Run a hand-specific Ultralytics YOLO model on the world camera videos:

Prerequisite:

```bash
conda run --no-capture-output -n yolo pip install ultralytics
```

```bash
conda run --no-capture-output -n yolo python -m lower_feature_cleaning_extraction.detect_hands_yolo \
  --model /home/julius/models/hand_yolo/hand_yolov8s.pt
```

Because this is the default model path, this is equivalent:

```bash
conda run --no-capture-output -n yolo python -m lower_feature_cleaning_extraction.detect_hands_yolo
```

The script skips calibration recordings (`Eye1`, `Eye2`, `Recalibration1`) by
default. Add `--include-calibration` if those videos should be processed too.
It uses GPU device `0` by default. Add `--device cpu` to force CPU inference.

Optional filters are useful for quick checks:

```bash
conda run --no-capture-output -n yolo python -m lower_feature_cleaning_extraction.detect_hands_yolo \
  --model /home/julius/models/hand_yolo \
  --participant IA_4_H_02 \
  --limit 1 \
  --overwrite
```

If the model contains non-hand classes, keep only hand classes by name or id:

```bash
conda run --no-capture-output -n yolo python -m lower_feature_cleaning_extraction.detect_hands_yolo \
  --model /home/julius/models/hand_yolo/hand_yolov8s.pt \
  --class-name hand,left_hand,right_hand
```

Output:

```text
/home/julius/datasets/seminar_object_detection/lower_feature_outcome/yolo_hand_outcome_csv/
  yolo_hand_outcome_index.csv
  yolo_hand_outcome_summary.json
  {full_id}_world_camera.csv
```

Each per-video CSV contains:

```text
frame_id,class_id,class_name,confidence,x1,y1,x2,y2
```

## Left/Right Hand YOLO Detection

After training the EgoHands left/right model, run it with the wrapper module:

```bash
conda run --no-capture-output -n object_detection python -m lower_feature_cleaning_extraction.detect_left_right_hands_yolo \
  --participant IA_4_H_02 \
  --limit 1 \
  --overwrite
```

Full run:

```bash
conda run --no-capture-output -n object_detection python -m lower_feature_cleaning_extraction.detect_left_right_hands_yolo \
  --overwrite
```

Defaults:

```text
model: /home/julius/models/EgoHands_YOLO/left_right_hand_yolov8s_img640_b4/weights/best.pt
output: /home/julius/datasets/seminar_object_detection/lower_feature_outcome/yolo_left_right_hand_outcome_csv/
device: 0
```

Each per-video CSV keeps the same columns, but `class_name` should be
`left_hand` or `right_hand`:

```text
frame_id,class_id,class_name,confidence,x1,y1,x2,y2
```

## Train Left/Right Hand YOLO

To train a project-specific model from EgoHands, download the official/Kaggle
EgoHands archive and convert `polygons.mat` annotations to YOLO boxes:

```bash
mkdir -p /home/julius/datasets/EgoHands_YOLO

curl -L "https://www.kaggle.com/api/v1/datasets/download/himaniishah/egohands" \
  -o /home/julius/datasets/EgoHands_YOLO/egohands_kaggle.zip

mkdir -p /home/julius/datasets/EgoHands_YOLO/raw

unzip -q /home/julius/datasets/EgoHands_YOLO/egohands_kaggle.zip \
  -d /home/julius/datasets/EgoHands_YOLO/raw

conda run --no-capture-output -n object_detection python -m lower_feature_cleaning_extraction.convert_egohands_mat_to_yolo \
  --source-root /home/julius/datasets/EgoHands_YOLO/raw \
  --output-root /home/julius/datasets/EgoHands_YOLO/left_right_yolo \
  --other-mode ignore \
  --link-mode symlink
```

This maps:

```text
myleft  -> left_hand
myright -> right_hand
yourleft/yourright -> ignored
```

If you want to use other people's hands too, replace `--other-mode ignore`
with:

```bash
--other-mode merge_by_side
```

The prepared dataset is:

```text
/home/julius/datasets/EgoHands_YOLO/left_right_yolo/
  data.yaml
  convert_summary.json
  train/images/
  train/labels/
  val/images/
  val/labels/
  test/images/
  test/labels/
```

If you already have a Roboflow YOLO-format EgoHands export, it should look like:

```text
egohands_yolo/
  data.yaml
  train/images/
  train/labels/
  valid/images/    # or val/images/
  valid/labels/    # or val/labels/
  test/images/
  test/labels/
```

For that YOLO-format source, prepare a simplified left/right dataset. By
default, this maps:

```text
own left hand  -> left_hand
own right hand -> right_hand
other left/right hand -> ignored
```

Run:

```bash
conda run --no-capture-output -n object_detection python -m lower_feature_cleaning_extraction.prepare_egohands_yolo \
  --source-root /home/julius/datasets/egohands_yolo \
  --output-root /home/julius/datasets/egohands_left_right_yolo \
  --other-mode ignore
```

If you want to keep other people's hands by side, use:

```bash
--other-mode merge_by_side
```

This writes:

```text
/home/julius/datasets/egohands_left_right_yolo/
  data.yaml
  prepare_summary.json
  train/images/
  train/labels/
  val/images/
  val/labels/
  test/images/
  test/labels/
```

Train YOLO:

```bash
conda run --no-capture-output -n object_detection python -m lower_feature_cleaning_extraction.train_hand_yolo \
  --data-yaml /home/julius/datasets/egohands_left_right_yolo/data.yaml \
  --model yolov8s.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device 0
```

The best weights are saved under:

```text
/home/julius/models/hand_yolo_left_right/egohands_left_right/weights/best.pt
```

Use the trained left/right model for your ADL videos:

```bash
conda run --no-capture-output -n object_detection python -m lower_feature_cleaning_extraction.detect_hands_yolo \
  --model /home/julius/models/hand_yolo_left_right/egohands_left_right/weights/best.pt \
  --output-root /home/julius/datasets/seminar_object_detection/lower_feature_outcome/yolo_left_right_hand_outcome_csv \
  --overwrite
```

The resulting CSV `class_name` should be `left_hand` or `right_hand` if the
trained model learned the simplified classes.
