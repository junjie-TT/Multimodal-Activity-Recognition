#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-object_detection}"

VIDEO_ROOT="${VIDEO_ROOT:-/home/julius/datasets/seminar_object_detection/iAssistADL_dataset_export_2026-05-18/world_camera_videos}"
FINAL_OUTPUT_ROOT="${FINAL_OUTPUT_ROOT:-/home/julius/datasets/seminar_object_detection/lower_feature_outcome/yolo_object_outcome_csv}"
COCO_OUTPUT_ROOT="${COCO_OUTPUT_ROOT:-/home/julius/datasets/seminar_object_detection/lower_feature_outcome/yolo11s_coco_object_outcome_csv}"
ADL14_OUTPUT_ROOT="${ADL14_OUTPUT_ROOT:-/home/julius/datasets/seminar_object_detection/lower_feature_outcome/yolo_real_adl_14_object_outcome_csv}"

COCO_MODEL="${COCO_MODEL:-yolo11s.pt}"
ADL14_MODEL="${ADL14_MODEL:-runs/real_adl_hands/yolo11s_real_adl_14cls_gpu/weights/best.pt}"
CONF="${CONF:-0.25}"
CLASS_ID_OFFSET="${CLASS_ID_OFFSET:-1000}"
DUPLICATE_IOU_THRESHOLD="${DUPLICATE_IOU_THRESHOLD:-0.60}"

LIMIT_ARG=()
if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARG=(--limit "$LIMIT")
fi

export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/Ultralytics}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

echo "Step 1/3: YOLO11s COCO output -> $COCO_OUTPUT_ROOT"
conda run --no-capture-output -n "$CONDA_ENV" python -m yolo_features_extraction.object_detection_csv_output \
  --model "$COCO_MODEL" \
  --input-root "$VIDEO_ROOT" \
  --output-root "$COCO_OUTPUT_ROOT" \
  --conf "$CONF" \
  --overwrite \
  "${LIMIT_ARG[@]}"

echo "Step 2/3: ADL14 enhancement output -> $ADL14_OUTPUT_ROOT"
conda run --no-capture-output -n "$CONDA_ENV" python -m yolo_features_extraction.archive.adl14_experiment.adl_14_object_detection_csv_output \
  --model "$ADL14_MODEL" \
  --input-root "$VIDEO_ROOT" \
  --output-root "$ADL14_OUTPUT_ROOT" \
  --conf "$CONF" \
  --class-id-offset "$CLASS_ID_OFFSET" \
  --overwrite \
  "${LIMIT_ARG[@]}"

echo "Step 3/3: merge COCO + ADL14 -> $FINAL_OUTPUT_ROOT"
conda run --no-capture-output -n "$CONDA_ENV" python -m yolo_features_extraction.archive.adl14_experiment.merge_coco_and_adl14_yolo_csv \
  --coco-root "$COCO_OUTPUT_ROOT" \
  --adl14-root "$ADL14_OUTPUT_ROOT" \
  --output-root "$FINAL_OUTPUT_ROOT" \
  --duplicate-iou-threshold "$DUPLICATE_IOU_THRESHOLD" \
  --overwrite

echo "Done. Final merged output:"
echo "$FINAL_OUTPUT_ROOT"
