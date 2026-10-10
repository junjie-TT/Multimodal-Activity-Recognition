# SVM Current ADL Action Baseline

This baseline performs current ADL action recognition using causal 1.5-second multimodal windows. Predictions are generated every 0.5 seconds. Each window is aggregated into statistical features and classified using an SVM. The default target is `task_step`.

## Pipeline

1. `01_build_frame_level_dataset.py`

   Recursively reads CSV/Parquet frame-level feature files from the gaze, hand-object, depth, head IMU, body IMU/joint, and skeleton feature directories. It normalizes frame/time keys, prefixes modality feature columns, outer-merges available modalities, parses annotation JSON and segment CSV files, and writes:

   ```text
   baseline_svm_current_action/intermediate/frame_level_multimodal_features.pkl
   baseline_svm_current_action/intermediate/frame_level_multimodal_features_sample.csv
   baseline_svm_current_action/intermediate/frame_level_label_stats.csv
   ```

2. `02_build_window_features.py`

   Builds causal windows:

   ```text
   feature window = [t - 1.5s, t]
   label window = [t - 0.5s, t]
   label = majority action label inside the label window
   ```

   The default `--aggregation_mode fast` uses vectorized CPU rolling statistics per sequence. The original exact loop implementation is still available with `--aggregation_mode loop` for debugging.

   Default output:

   ```text
   baseline_svm_current_action/intermediate/window_features_1p5s_stride0p5s_task_step.pkl
   baseline_svm_current_action/intermediate/window_label_stats_task_step.csv
   ```

3. `03_train_svm_baseline.py`

   Trains with either sklearn CPU SVM or RAPIDS cuML GPU SVM. For the default
   `task_step` target it now uses a hierarchical classifier:

   ```text
   task SVM -> predicted task -> task-specific step SVM -> task__step prediction
   ```

   Each SVM uses:

   ```text
   SimpleImputer(strategy="median")
   StandardScaler()
   SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced")
   ```

   Splitting is group-based by `participant_id` by default: all participant IDs are
   sorted, randomly shuffled with `--random_state`, then assigned to train/val/test
   by ratio. Windows are never randomly split across train/val/test. The selected
   groups are written to `split_groups.csv` in the result directory. You can still
   use `--split_mode video` for a less strict video-level split. The default backend
   is sklearn CPU. Use `--backend cuml` only if you later want RAPIDS GPU training.

4. `04_train_lstm_baseline.py`

   Trains a PyTorch LSTM on the same causal windows, but keeps the window content
   as frame-level sequences instead of SVM statistical aggregates. It reuses the
   same target labels, label purity rule, and participant/video group split logic.
   For the default `task_step` target it follows the same hierarchical setup:

   ```text
   task LSTM -> predicted task -> task-specific step LSTM -> task__step prediction
   ```

5. `05_train_mlp_baseline.py`

   Trains a PyTorch MLP on the same window-level statistical features used by
   the SVM. This keeps the existing window feature dataset and group split logic,
   while replacing only the classifier. For the default `task_step` target it
   uses the same hierarchical structure:

   ```text
   task MLP -> predicted task -> task-specific step MLP -> task__step prediction
   ```

6. `06_train_decision_tree_baseline.py`

   Trains a sklearn decision tree on the same window-level statistical features
   used by the SVM and MLP, with the same group split and optional hierarchical
   `task_step` setup.

7. `07_train_random_forest_baseline.py`

   Trains a sklearn random forest on the same window-level statistical features.
   This is the stronger tree-family baseline compared with a single decision
   tree, while keeping the same target and split logic.

8. `08_mlp_permutation_importance.py`

   Computes permutation feature importance for the MLP baseline. It loads the
   existing `model.pt` from the matching MLP result directory when available;
   otherwise it trains an MLP with the same split and hyperparameter arguments.
   Each feature is shuffled on the selected evaluation split and ranked by the
   drop in accuracy, macro F1, or weighted F1. Outputs include:

   ```text
   baseline_svm_current_action/results/mlp_task_step_hierarchical_1p5s_stride0p5s_participant/permutation_importance/permutation_importance.csv
   baseline_svm_current_action/results/mlp_task_step_hierarchical_1p5s_stride0p5s_participant/permutation_importance/permutation_importance_by_family.csv
   baseline_svm_current_action/results/mlp_task_step_hierarchical_1p5s_stride0p5s_participant/permutation_importance/permutation_importance_summary.json
   ```

## Manual Run

Run one step at a time for easier debugging:

```bash
bash baseline_svm_current_action/run_svm_pipeline.sh frame
bash baseline_svm_current_action/run_svm_pipeline.sh window
bash baseline_svm_current_action/run_svm_pipeline.sh train
```

`train` uses the sklearn CPU backend. Optional GPU backend:

```bash
bash baseline_svm_current_action/run_svm_pipeline.sh train_gpu
```

To run everything in sequence after debugging:

```bash
bash baseline_svm_current_action/run_svm_pipeline.sh all
```

Or run each Python step manually:

```bash
python baseline_svm_current_action/01_build_frame_level_dataset.py --fps 30
python baseline_svm_current_action/02_build_window_features.py --target_mode task_step --aggregation_mode fast
python baseline_svm_current_action/03_train_svm_baseline.py --target_mode task_step --classifier_mode hierarchical --split_mode participant --backend sklearn
python baseline_svm_current_action/04_train_lstm_baseline.py --target_mode task_step --classifier_mode hierarchical --split_mode participant
python baseline_svm_current_action/05_train_mlp_baseline.py --target_mode task_step --classifier_mode hierarchical --split_mode participant
python baseline_svm_current_action/06_train_decision_tree_baseline.py --target_mode task_step --classifier_mode hierarchical --split_mode participant
python baseline_svm_current_action/07_train_random_forest_baseline.py --target_mode task_step --classifier_mode hierarchical --split_mode participant
python baseline_svm_current_action/08_mlp_permutation_importance.py --target_mode task_step --classifier_mode hierarchical --split_mode participant --eval_split test --n_repeats 5 --importance_metric accuracy
```

All scripts support `--help` for paths and parameters.
