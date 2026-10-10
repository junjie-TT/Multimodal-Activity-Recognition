# ADL Training Workflows

This directory is the standard entry point for model training and evaluation.
It keeps the older scripts in place and adds one non-destructive runner:

```bash
python train/run_training_workflow.py --list
```

The runner does not delete, move, or rewrite existing experiment files. It only
calls the existing scripts with consistent paths.

## Output Layout

Standard outputs are grouped outside the repository under:

```text
/home/julius/datasets/adl_training_workflows/
```

```text
/home/julius/datasets/adl_training_workflows/intermediate/
  frame_level_multimodal_features.pkl
  window_features_1p5s_stride0p5s_task_step.pkl
  goi_hoi_label_merged_features.csv

/home/julius/datasets/adl_training_workflows/results/
  baseline_current_action/
  goi_hoi/
```

The older `baseline_svm_current_action/intermediate/` and
`baseline_svm_current_action/results/` paths still work when running those
scripts directly, but new training runs should prefer the unified runner.

## Workflow Families

### Baseline Current-Action Workflows

These workflows use multimodal upper-feature CSV/Parquet outputs, build causal
1.5 second windows every 0.5 seconds, and predict `task_step` by default.

Feature preparation only:

```bash
python train/run_training_workflow.py baseline_features
```

Classical baselines:

```bash
python train/run_training_workflow.py baseline_classical
```

This runs:

```text
baseline_frame -> baseline_window -> baseline_svm -> baseline_decision_tree -> baseline_random_forest
```

Neural baselines:

```bash
python train/run_training_workflow.py baseline_neural
```

This runs:

```text
baseline_frame -> baseline_window -> baseline_mlp -> baseline_lstm
```

All baseline models plus MLP permutation importance:

```bash
python train/run_training_workflow.py baseline_all
```

### GOI/HOI Workflows

These workflows merge gaze-object interaction, hand-object interaction, and
annotation labels into one frame-level table, then train GOI+HOI classifiers.

Merge features only:

```bash
python train/run_training_workflow.py goi_hoi_features
```

Classical GOI/HOI models:

```bash
python train/run_training_workflow.py goi_hoi_classical
```

This runs:

```text
goi_hoi_merge -> goi_hoi_extratrees -> goi_hoi_sgd_svm
```

Realtime MLP:

```bash
python train/run_training_workflow.py goi_hoi_neural
```

All GOI/HOI models plus six-source ablation:

```bash
python train/run_training_workflow.py goi_hoi_all
```

## Recommended Runs

For a fast, standard CPU baseline:

```bash
python train/run_training_workflow.py baseline_classical --skip-existing
```

For the GOI/HOI compact-feature experiments:

```bash
python train/run_training_workflow.py goi_hoi_classical --skip-existing
```

For a full run:

```bash
python train/run_training_workflow.py all --skip-existing
```

Use `--dry-run` before launching a long job:

```bash
python train/run_training_workflow.py baseline_all --dry-run
```

## Important Options

Shared:

```text
--fps 30
--target-mode task_step
--classifier-mode hierarchical
--random-state 42
--device auto
```

Baseline windows:

```text
--baseline-window-sec 1.5
--baseline-stride-sec 0.5
--baseline-label-tail-sec 0.5
--min-label-purity 0.6
--baseline-split-mode participant
```

GOI/HOI:

```text
--goi-csv /path/to/gaze_object_compact_train_features.csv
--hoi-csv /path/to/hand_object_compact_train_features.csv
--goi-hoi-window-sec 0.5
--goi-hoi-split-mode participant
```

## Legacy Script Map

The runner wraps these existing files without removing them:

```text
baseline_svm_current_action/01_build_frame_level_dataset.py
baseline_svm_current_action/02_build_window_features.py
baseline_svm_current_action/03_train_svm_baseline.py
baseline_svm_current_action/04_train_lstm_baseline.py
baseline_svm_current_action/05_train_mlp_baseline.py
baseline_svm_current_action/06_train_decision_tree_baseline.py
baseline_svm_current_action/07_train_random_forest_baseline.py
baseline_svm_current_action/08_mlp_permutation_importance.py

train/merge_goi_hoi_features.py
train/train_goi_hoi_window_extratrees.py
train/train_goi_hoi_window_sgd_svm.py
train/train_goi_hoi_realtime_step_mlp.py
train/feature_importance_ablation_six_sources_extratrees.py
```
