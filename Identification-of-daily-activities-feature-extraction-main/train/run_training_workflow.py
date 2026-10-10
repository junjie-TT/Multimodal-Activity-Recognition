#!/usr/bin/env python3
"""Run standard ADL training workflows without moving or deleting legacy scripts."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = PROJECT_ROOT / "train"
BASELINE_DIR = PROJECT_ROOT / "baseline_svm_current_action"

DEFAULT_DATASET_ROOT = Path("/home/julius/datasets/seminar_object_detection")
DEFAULT_INPUT_TRAIN_DIR = DEFAULT_DATASET_ROOT / "input_train"
DEFAULT_WORKFLOW_OUTPUT_ROOT = Path("/home/julius/datasets/adl_training_workflows")
DEFAULT_INTERMEDIATE_DIR = DEFAULT_WORKFLOW_OUTPUT_ROOT / "intermediate"
DEFAULT_RESULTS_DIR = DEFAULT_WORKFLOW_OUTPUT_ROOT / "results"

DEFAULT_GOI_CSV = DEFAULT_INPUT_TRAIN_DIR / "gaze_object_compact_train_features.csv"
DEFAULT_HOI_CSV = DEFAULT_INPUT_TRAIN_DIR / "hand_object_compact_train_features.csv"
DEFAULT_GOI_HOI_MERGED_CSV = DEFAULT_INTERMEDIATE_DIR / "goi_hoi_label_merged_features.csv"
DEFAULT_FRAME_FEATURES = DEFAULT_INTERMEDIATE_DIR / "frame_level_multimodal_features.pkl"
DEFAULT_WINDOW_FEATURES = DEFAULT_INTERMEDIATE_DIR / "window_features_1p5s_stride0p5s_task_step.pkl"


@dataclass(frozen=True)
class Stage:
    name: str
    description: str
    command: Callable[[argparse.Namespace], list[str]]
    output_marker: Callable[[argparse.Namespace], Path | None] = lambda _args: None


def py(script: Path, *args: object) -> list[str]:
    return [sys.executable, str(script), *(str(arg) for arg in args)]


def baseline_frame_command(args: argparse.Namespace) -> list[str]:
    return py(
        BASELINE_DIR / "01_build_frame_level_dataset.py",
        "--fps",
        args.fps,
        "--output",
        args.frame_features,
    )


def baseline_window_command(args: argparse.Namespace) -> list[str]:
    return py(
        BASELINE_DIR / "02_build_window_features.py",
        "--input",
        args.frame_features,
        "--fps",
        args.fps,
        "--window_sec",
        args.baseline_window_sec,
        "--stride_sec",
        args.baseline_stride_sec,
        "--label_tail_sec",
        args.baseline_label_tail_sec,
        "--target_mode",
        args.target_mode,
        "--min_label_purity",
        args.min_label_purity,
        "--aggregation_mode",
        args.aggregation_mode,
        "--output",
        args.window_features,
    )


def baseline_train_command(script_name: str, model_name: str) -> Callable[[argparse.Namespace], list[str]]:
    def build(args: argparse.Namespace) -> list[str]:
        cmd = py(
            BASELINE_DIR / script_name,
            "--input",
            args.window_features,
            "--target_mode",
            args.target_mode,
            "--classifier_mode",
            args.classifier_mode,
            "--split_mode",
            args.baseline_split_mode,
            "--train_ratio",
            args.train_ratio,
            "--val_ratio",
            args.val_ratio,
            "--test_ratio",
            args.test_ratio,
            "--random_state",
            args.random_state,
            "--results_dir",
            args.baseline_results_dir,
        )
        if model_name == "svm":
            cmd.extend(["--backend", args.svm_backend, "--svm_kernel", args.svm_kernel])
        return cmd

    return build


def baseline_lstm_command(args: argparse.Namespace) -> list[str]:
    return py(
        BASELINE_DIR / "04_train_lstm_baseline.py",
        "--input",
        args.frame_features,
        "--target_mode",
        args.target_mode,
        "--classifier_mode",
        args.classifier_mode,
        "--split_mode",
        args.baseline_split_mode,
        "--fps",
        args.fps,
        "--window_sec",
        args.baseline_window_sec,
        "--stride_sec",
        args.baseline_stride_sec,
        "--label_tail_sec",
        args.baseline_label_tail_sec,
        "--min_label_purity",
        args.min_label_purity,
        "--random_state",
        args.random_state,
        "--results_dir",
        args.baseline_results_dir,
        "--device",
        args.device,
    )


def baseline_mlp_importance_command(args: argparse.Namespace) -> list[str]:
    return py(
        BASELINE_DIR / "08_mlp_permutation_importance.py",
        "--input",
        args.window_features,
        "--target_mode",
        args.target_mode,
        "--classifier_mode",
        args.classifier_mode,
        "--split_mode",
        args.baseline_split_mode,
        "--random_state",
        args.random_state,
        "--results_dir",
        args.baseline_results_dir,
        "--eval_split",
        args.importance_eval_split,
        "--n_repeats",
        args.importance_repeats,
        "--importance_metric",
        args.importance_metric,
    )


def goi_hoi_merge_command(args: argparse.Namespace) -> list[str]:
    return py(
        TRAIN_DIR / "merge_goi_hoi_features.py",
        "--goi-csv",
        args.goi_csv,
        "--hoi-csv",
        args.hoi_csv,
        "--output-csv",
        args.goi_hoi_merged_csv,
        "--annotations-dir",
        args.annotations_dir,
    )


def goi_hoi_train_command(script_name: str, output_name: str) -> Callable[[argparse.Namespace], list[str]]:
    def build(args: argparse.Namespace) -> list[str]:
        return py(
            TRAIN_DIR / script_name,
            "--input-csv",
            args.goi_hoi_merged_csv,
            "--output-dir",
            args.goi_hoi_results_dir / output_name,
            "--fps",
            args.fps,
            "--window-sec",
            args.goi_hoi_window_sec,
            "--split-mode",
            args.goi_hoi_split_mode,
            "--test-size",
            args.test_size,
            "--random-state",
            args.random_state,
        )

    return build


def goi_hoi_ablation_command(args: argparse.Namespace) -> list[str]:
    return py(
        TRAIN_DIR / "feature_importance_ablation_six_sources_extratrees.py",
        "--output-dir",
        args.goi_hoi_results_dir / "six_source_ablation_extratrees",
        "--test-size",
        args.test_size,
        "--random-state",
        args.random_state,
        "--n-jobs",
        args.n_jobs,
    )


STAGES: dict[str, Stage] = {
    "baseline_frame": Stage(
        "baseline_frame",
        "Build multimodal frame-level features and labels.",
        baseline_frame_command,
        lambda args: args.frame_features,
    ),
    "baseline_window": Stage(
        "baseline_window",
        "Build causal statistical windows from baseline frame features.",
        baseline_window_command,
        lambda args: args.window_features,
    ),
    "baseline_svm": Stage(
        "baseline_svm",
        "Train the hierarchical/current-action SVM baseline.",
        baseline_train_command("03_train_svm_baseline.py", "svm"),
    ),
    "baseline_decision_tree": Stage(
        "baseline_decision_tree",
        "Train the decision-tree baseline.",
        baseline_train_command("06_train_decision_tree_baseline.py", "decision_tree"),
    ),
    "baseline_random_forest": Stage(
        "baseline_random_forest",
        "Train the random-forest baseline.",
        baseline_train_command("07_train_random_forest_baseline.py", "random_forest"),
    ),
    "baseline_mlp": Stage(
        "baseline_mlp",
        "Train the MLP baseline on statistical windows.",
        baseline_train_command("05_train_mlp_baseline.py", "mlp"),
    ),
    "baseline_lstm": Stage(
        "baseline_lstm",
        "Train the LSTM baseline on frame-level causal windows.",
        baseline_lstm_command,
    ),
    "baseline_mlp_importance": Stage(
        "baseline_mlp_importance",
        "Compute MLP permutation importance.",
        baseline_mlp_importance_command,
    ),
    "goi_hoi_merge": Stage(
        "goi_hoi_merge",
        "Merge GOI, HOI, and annotation labels into one frame-level table.",
        goi_hoi_merge_command,
        lambda args: args.goi_hoi_merged_csv,
    ),
    "goi_hoi_extratrees": Stage(
        "goi_hoi_extratrees",
        "Train GOI+HOI ExtraTrees window classifiers.",
        goi_hoi_train_command("train_goi_hoi_window_extratrees.py", "window_extratrees"),
    ),
    "goi_hoi_sgd_svm": Stage(
        "goi_hoi_sgd_svm",
        "Train GOI+HOI streaming SGD-SVM window classifiers.",
        goi_hoi_train_command("train_goi_hoi_window_sgd_svm.py", "window_sgd_svm"),
    ),
    "goi_hoi_mlp": Stage(
        "goi_hoi_mlp",
        "Train GOI+HOI realtime MLP classifiers.",
        goi_hoi_train_command("train_goi_hoi_realtime_step_mlp.py", "realtime_step_mlp"),
    ),
    "goi_hoi_ablation": Stage(
        "goi_hoi_ablation",
        "Run six-source ExtraTrees feature importance and ablation.",
        goi_hoi_ablation_command,
    ),
}


WORKFLOWS: dict[str, list[str]] = {
    "baseline_features": ["baseline_frame", "baseline_window"],
    "baseline_classical": [
        "baseline_frame",
        "baseline_window",
        "baseline_svm",
        "baseline_decision_tree",
        "baseline_random_forest",
    ],
    "baseline_neural": ["baseline_frame", "baseline_window", "baseline_mlp", "baseline_lstm"],
    "baseline_all": [
        "baseline_frame",
        "baseline_window",
        "baseline_svm",
        "baseline_decision_tree",
        "baseline_random_forest",
        "baseline_mlp",
        "baseline_lstm",
        "baseline_mlp_importance",
    ],
    "goi_hoi_features": ["goi_hoi_merge"],
    "goi_hoi_classical": ["goi_hoi_merge", "goi_hoi_extratrees", "goi_hoi_sgd_svm"],
    "goi_hoi_neural": ["goi_hoi_merge", "goi_hoi_mlp"],
    "goi_hoi_all": [
        "goi_hoi_merge",
        "goi_hoi_extratrees",
        "goi_hoi_sgd_svm",
        "goi_hoi_mlp",
        "goi_hoi_ablation",
    ],
}
WORKFLOWS["all"] = WORKFLOWS["baseline_all"] + WORKFLOWS["goi_hoi_all"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Unified, non-destructive runner for baseline and GOI/HOI ADL training "
            "workflows. Existing scripts remain usable on their own."
        )
    )
    parser.add_argument("workflow", nargs="?", choices=sorted(WORKFLOWS), help="Workflow to run.")
    parser.add_argument("--list", action="store_true", help="List available workflows and stages.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip feature-building stages when their output marker already exists.",
    )

    parser.add_argument("--intermediate-dir", type=Path, default=DEFAULT_INTERMEDIATE_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--frame-features", type=Path, default=DEFAULT_FRAME_FEATURES)
    parser.add_argument("--window-features", type=Path, default=DEFAULT_WINDOW_FEATURES)
    parser.add_argument("--baseline-results-dir", type=Path, default=DEFAULT_RESULTS_DIR / "baseline_current_action")
    parser.add_argument("--goi-hoi-results-dir", type=Path, default=DEFAULT_RESULTS_DIR / "goi_hoi")
    parser.add_argument("--goi-hoi-merged-csv", type=Path, default=DEFAULT_GOI_HOI_MERGED_CSV)

    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--target-mode", default="task_step", choices=["task", "step", "task_step"])
    parser.add_argument("--classifier-mode", default="hierarchical", choices=["auto", "single", "hierarchical"])
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--n-jobs", default="-1")

    parser.add_argument("--baseline-window-sec", type=float, default=1.5)
    parser.add_argument("--baseline-stride-sec", type=float, default=0.5)
    parser.add_argument("--baseline-label-tail-sec", type=float, default=0.5)
    parser.add_argument("--min-label-purity", type=float, default=0.6)
    parser.add_argument("--aggregation-mode", default="fast", choices=["fast", "loop"])
    parser.add_argument("--baseline-split-mode", default="participant", choices=["participant", "video"])
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--svm-backend", default="sklearn", choices=["sklearn", "cuml"])
    parser.add_argument("--svm-kernel", default="rbf", choices=["rbf", "linear"])
    parser.add_argument("--importance-eval-split", default="test", choices=["val", "test", "val_test"])
    parser.add_argument("--importance-repeats", type=int, default=5)
    parser.add_argument(
        "--importance-metric",
        default="accuracy",
        choices=["accuracy", "macro_f1", "weighted_f1"],
    )

    parser.add_argument("--goi-csv", type=Path, default=DEFAULT_GOI_CSV)
    parser.add_argument("--hoi-csv", type=Path, default=DEFAULT_HOI_CSV)
    parser.add_argument(
        "--annotations-dir",
        type=Path,
        default=DEFAULT_DATASET_ROOT / "iAssistADL_dataset_export_2026-05-18" / "annotations",
    )
    parser.add_argument("--goi-hoi-window-sec", type=float, default=0.5)
    parser.add_argument("--goi-hoi-split-mode", default="participant", choices=["participant", "row"])
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def print_available() -> None:
    print("Workflows:")
    for name, stages in sorted(WORKFLOWS.items()):
        print(f"  {name}: {', '.join(stages)}")
    print("\nStages:")
    for name, stage in sorted(STAGES.items()):
        print(f"  {name}: {stage.description}")


def ensure_output_roots(args: argparse.Namespace) -> None:
    args.intermediate_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.baseline_results_dir.mkdir(parents=True, exist_ok=True)
    args.goi_hoi_results_dir.mkdir(parents=True, exist_ok=True)
    args.frame_features.parent.mkdir(parents=True, exist_ok=True)
    args.window_features.parent.mkdir(parents=True, exist_ok=True)
    args.goi_hoi_merged_csv.parent.mkdir(parents=True, exist_ok=True)


def format_command(cmd: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in cmd)


def run_stage(stage: Stage, args: argparse.Namespace) -> None:
    marker = stage.output_marker(args)
    if args.skip_existing and marker is not None and marker.exists():
        print(f"[skip] {stage.name}: {marker} already exists")
        return

    cmd = stage.command(args)
    print(f"\n[{stage.name}] {stage.description}")
    print(format_command(cmd))
    if args.dry_run:
        return

    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)


def main() -> None:
    args = parse_args()
    if args.list:
        print_available()
        return
    if not args.workflow:
        raise SystemExit("Choose a workflow or pass --list.")

    ensure_output_roots(args)
    for stage_name in WORKFLOWS[args.workflow]:
        run_stage(STAGES[stage_name], args)


if __name__ == "__main__":
    main()
