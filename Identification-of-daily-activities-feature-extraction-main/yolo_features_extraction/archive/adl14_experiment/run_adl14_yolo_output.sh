#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-object_detection}"
MODEL="${MODEL:-runs/real_adl_hands/yolo11s_real_adl_14cls_gpu/weights/best.pt}"
INPUT_ROOT="${INPUT_ROOT:-/home/julius/datasets/seminar_object_detection/iAssistADL_dataset_export_2026-05-18/world_camera_videos}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/home/julius/datasets/seminar_object_detection/lower_feature_outcome/yolo_real_adl_14_object_outcome_csv}"
CONF="${CONF:-0.25}"
CLASS_ID_OFFSET="${CLASS_ID_OFFSET:-1000}"
LIMIT_ARG=()
OVERWRITE_ARG=()

if [[ -n "${LIMIT:-}" ]]; then
  LIMIT_ARG=(--limit "$LIMIT")
fi

if [[ "${OVERWRITE:-0}" == "1" ]]; then
  OVERWRITE_ARG=(--overwrite)
fi

export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/Ultralytics}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

conda run --no-capture-output -n "$CONDA_ENV" python -m yolo_features_extraction.archive.adl14_experiment.adl_14_object_detection_csv_output \
  --model "$MODEL" \
  --input-root "$INPUT_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --conf "$CONF" \
  --class-id-offset "$CLASS_ID_OFFSET" \
  "${LIMIT_ARG[@]}" \
  "${OVERWRITE_ARG[@]}"
