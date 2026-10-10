#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash baseline_svm_current_action/run_svm_pipeline.sh frame
  bash baseline_svm_current_action/run_svm_pipeline.sh window
  bash baseline_svm_current_action/run_svm_pipeline.sh train
  bash baseline_svm_current_action/run_svm_pipeline.sh train_gpu
  bash baseline_svm_current_action/run_svm_pipeline.sh all

Steps:
  frame   Build frame-level multimodal features and labels.
  window  Build causal 1.5s SVM window features with fast CPU rolling aggregation.
  train   Train/evaluate SVM with participant split using sklearn CPU backend.
  train_gpu Train/evaluate SVM with participant split using cuML GPU backend.
  all     Run frame, window, then CPU train.
EOF
}

run_frame() {
  python baseline_svm_current_action/01_build_frame_level_dataset.py \
    --fps 30
}

run_window() {
  python baseline_svm_current_action/02_build_window_features.py \
    --fps 30 \
    --window_sec 1.5 \
    --stride_sec 0.5 \
    --label_tail_sec 0.5 \
    --target_mode task_step \
    --min_label_purity 0.6 \
    --aggregation_mode fast
}

run_train() {
  python baseline_svm_current_action/03_train_svm_baseline.py \
    --target_mode task_step \
    --classifier_mode hierarchical \
    --split_mode participant \
    --svm_kernel rbf \
    --C 1.0 \
    --gamma scale \
    --backend sklearn
}

run_train_gpu() {
  python baseline_svm_current_action/03_train_svm_baseline.py \
    --target_mode task_step \
    --classifier_mode hierarchical \
    --split_mode participant \
    --svm_kernel rbf \
    --C 1.0 \
    --gamma scale \
    --backend cuml
}

step="${1:-}"
case "$step" in
  frame)
    run_frame
    ;;
  window)
    run_window
    ;;
  train)
    run_train
    ;;
  train_gpu)
    run_train_gpu
    ;;
  all)
    run_frame
    run_window
    run_train
    ;;
  -h|--help|help|"")
    usage
    ;;
  *)
    echo "Unknown step: $step" >&2
    usage >&2
    exit 2
    ;;
esac
