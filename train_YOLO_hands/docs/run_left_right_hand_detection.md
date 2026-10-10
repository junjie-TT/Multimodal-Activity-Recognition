# Run YOLO11s Left/Right Hand Detection

Use the trained model:

```text
/home/julius/models/hand_detection/yolo11s_egohands_real_adl_mixed/weights/best.pt
```

Input videos:

```text
/home/julius/datasets/seminar_object_detection/iAssistADL_dataset_export_2026-05-18/world_camera_videos
```

Default output:

```text
/home/julius/datasets/seminar_object_detection/lower_feature_outcome/yolo11s_left_right_hand_outcome_csv
```

Each video gets one CSV with columns:

```text
frame_id,class_id,class_name,confidence,x1,y1,x2,y2
```

Class ids:

```text
0 = left_hand
1 = right_hand
```

## Quick Test

Run one video first:

```bash
cd /home/julius/projects/Identification-of-daily-activities-feature-extraction/train_YOLO_hands

python scripts/detect_world_camera_left_right_hands.py \
  --limit 1
```

## Run All Non-Calibration Videos

By default, `Eye1`, `Eye2`, and `Recalibration1` are skipped.

```bash
cd /home/julius/projects/Identification-of-daily-activities-feature-extraction/train_YOLO_hands

python scripts/detect_world_camera_left_right_hands.py
```

## Include Calibration Videos

```bash
python scripts/detect_world_camera_left_right_hands.py \
  --include-calibration
```

## Run One Participant

```bash
python scripts/detect_world_camera_left_right_hands.py \
  --participant IA_4_H_02
```

## Run One Recording

```bash
python scripts/detect_world_camera_left_right_hands.py \
  --full-id IA_4_H_02_Chaos_several_Tasks_combine1
```

## Faster Pass

Process every 5th frame:

```bash
python scripts/detect_world_camera_left_right_hands.py \
  --vid-stride 5
```

## Overwrite Existing CSVs

```bash
python scripts/detect_world_camera_left_right_hands.py \
  --overwrite
```
