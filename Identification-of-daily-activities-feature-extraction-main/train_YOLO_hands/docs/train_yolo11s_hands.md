# Train YOLO11s Left/Right Hand Detector

This directory prepares a first-person left/right hand detector with:

- Source dataset: EgoHands left/right YOLO train split
- Target dataset: manually labeled Real ADL frames
- Classes:
  - `0: left_hand`
  - `1: right_hand`
- Model: Ultralytics YOLO11s

All scripts are located at:

```text
/home/julius/projects/Identification-of-daily-activities-feature-extraction/train_YOLO_hands/scripts
```

Before running commands, enter the training helper directory:

```bash
cd /home/julius/projects/Identification-of-daily-activities-feature-extraction/train_YOLO_hands
```

Do not run these commands from the project root unless you manually adjust script paths.

## Paths

- Raw Real ADL: `/home/julius/datasets/real_adl_hands_raw`
- Split Real ADL: `/home/julius/datasets/real_adl_hands_split_v2`
- EgoHands: `/home/julius/datasets/EgoHands_YOLO/left_right_yolo`
- Mixed dataset: `/home/julius/datasets/mixed_egohands_real_adl_v1`
- Model/runs output: `/home/julius/models/hand_detection`

The scripts do not delete, move, or modify source data. Images without a matching `.txt` label file are skipped and are not used for training or evaluation.

## Step 0: Create Model Output Directory

```bash
mkdir -p /home/julius/models/hand_detection
```

Check that this directory exists before training. YOLO commands below set `project=/home/julius/models/hand_detection` so outputs do not go into the default `runs/` directory.

## Step 1: Prepare Raw Real ADL Dataset

This copies only images that already have matching YOLO `.txt` labels into a standard raw dataset. Images without label files are skipped.

```bash
python scripts/prepare_real_adl_raw_labels.py \
  --image_dir /home/julius/datasets/real_adl_hands/frames_pilot_500 \
  --label_dir /home/julius/datasets/real_adl_hands/frames_pilot_500_labeled \
  --out_root /home/julius/datasets/real_adl_hands_raw
```

Check the summary for found images, copied labeled images, skipped missing-label images, left/right hand box counts, and warnings.

## Step 2: Check Raw Real ADL Dataset

```bash
python scripts/check_yolo_hand_dataset.py \
  --image_dir /home/julius/datasets/real_adl_hands_raw/images \
  --label_dir /home/julius/datasets/real_adl_hands_raw/labels
```

Review missing/extra labels and the first error samples. Any missing labels at this stage should be investigated because unlabeled images are not intended to be used.

## Step 3: Split Real ADL Train/Val/Test

The splitter groups by participant or video-like filename stem to reduce adjacent-frame leakage. It does not do pure random per-frame splitting. If any image is missing a label file, the splitter skips that image.

```bash
python scripts/split_real_adl_dataset.py \
  --image_dir /home/julius/datasets/real_adl_hands_raw/images \
  --label_dir /home/julius/datasets/real_adl_hands_raw/labels \
  --out_root /home/julius/datasets/real_adl_hands_split_v2 \
  --train_count 900 \
  --val_count 250 \
  --seed 42
```

Check `data.yaml` and the printed train/val/test counts. Real ADL val/test must remain target-only evaluation data.

## Step 4: Visualize Real ADL Train Labels

```bash
python scripts/visualize_yolo_hand_labels.py \
  --image_dir /home/julius/datasets/real_adl_hands_split_v2/images/train \
  --label_dir /home/julius/datasets/real_adl_hands_split_v2/labels/train \
  --out_dir /home/julius/datasets/real_adl_hands_split_v2/quality_check_train \
  --num_samples 100 \
  --seed 42
```

Inspect the output images for swapped left/right hands, boxes that are too loose or too tight, missed hands, and boxes covering objects or arms.

## Step 5: Build EgoHands + Real ADL Mixed Dataset

Rules:

- Mixed train = Real ADL train + sampled EgoHands train
- Mixed val = Real ADL val only
- Mixed test = Real ADL test only
- EgoHands is never added to val/test
- EgoHands is sampled, defaulting to 2500 images, to avoid overwhelming target data
- Images without label files are skipped

```bash
python scripts/build_mixed_ego2hands_real_adl.py \
  --ego_images /home/julius/datasets/EgoHands_YOLO/left_right_yolo/train/images \
  --ego_labels /home/julius/datasets/EgoHands_YOLO/left_right_yolo/train/labels \
  --real_root /home/julius/datasets/real_adl_hands_split_v2 \
  --out_root /home/julius/datasets/mixed_egohands_real_adl_v1 \
  --ego_sample 2500 \
  --seed 42
```

Confirm that mixed val/test counts match Real ADL val/test counts exactly.

## Step 6: Check Mixed Train Dataset

```bash
python scripts/check_yolo_hand_dataset.py \
  --image_dir /home/julius/datasets/mixed_egohands_real_adl_v1/images/train \
  --label_dir /home/julius/datasets/mixed_egohands_real_adl_v1/labels/train
```

Check class counts and label validity before training.

## Step 7: YOLO11n Sanity Check

```bash
yolo detect train \
  model=yolo11n.pt \
  data=/home/julius/datasets/mixed_egohands_real_adl_v1/data.yaml \
  imgsz=640 \
  epochs=10 \
  batch=4 \
  device=0 \
  workers=4 \
  project=/home/julius/models/hand_detection \
  name=yolo11n_egohands_mixed_sanity
```

Use this short run to catch path, class, or GPU/environment problems before the larger run.

## Step 8: YOLO11s Mixed Training

```bash
yolo detect train \
  model=yolo11s.pt \
  data=/home/julius/datasets/mixed_egohands_real_adl_v1/data.yaml \
  imgsz=640 \
  epochs=80 \
  batch=4 \
  device=0 \
  workers=4 \
  patience=20 \
  project=/home/julius/models/hand_detection \
  name=yolo11s_egohands_real_adl_mixed
```

Monitor Real ADL val performance, not just training loss.

## Step 9: Real ADL Fine-Tuning

```bash
yolo detect train \
  model=/home/julius/models/hand_detection/yolo11s_egohands_real_adl_mixed/weights/best.pt \
  data=/home/julius/datasets/real_adl_hands_split_v2/data.yaml \
  imgsz=640 \
  epochs=30 \
  batch=4 \
  device=0 \
  workers=4 \
  patience=10 \
  lr0=0.001 \
  project=/home/julius/models/hand_detection \
  name=yolo11s_real_adl_finetune
```

This stage adapts the detector to the Real ADL target domain. Keep val/test isolated from training.

## Step 10: Real ADL Test Evaluation

```bash
yolo detect val \
  model=/home/julius/models/hand_detection/yolo11s_real_adl_finetune/weights/best.pt \
  data=/home/julius/datasets/real_adl_hands_split_v2/data.yaml \
  split=test \
  imgsz=640 \
  batch=4 \
  device=0
```

Use the test split only after model selection is finished.

## Step 11: Prediction Visualization

```bash
yolo detect predict \
  model=/home/julius/models/hand_detection/yolo11s_real_adl_finetune/weights/best.pt \
  source=/home/julius/datasets/real_adl_hands_split_v2/images/test \
  imgsz=640 \
  conf=0.25 \
  device=0 \
  save=True \
  project=/home/julius/models/hand_detection \
  name=real_adl_test_predictions
```

Inspect these predictions for left/right confusion, missed hands, false positives on objects, and robustness across participants/tasks.
