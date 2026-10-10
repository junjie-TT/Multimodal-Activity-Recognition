import json
from pathlib import Path
import sys

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from lower_feature_cleaning_extraction.config import (
    ALIGNMENT_MANIFEST_OUTPUT,
    ANNOTATIONS_ROOT,
    OUTPUT_ROOT,
)


def parse_alignment_name(path):
    participant_id = path.parent.name
    stem = path.name.replace(".alignment.json", "")
    prefix = f"{participant_id}_"
    task_name = stem[len(prefix):] if stem.startswith(prefix) else stem
    return participant_id, task_name, stem


def read_alignment(path):
    with open(path, "r") as f:
        data = json.load(f)

    details = data.get("details", {})
    participant_id, task_name, full_id = parse_alignment_name(path)

    return {
        "participant_id": participant_id,
        "task_name": task_name,
        "full_id": full_id,
        "alignment_path": str(path),
        "offset_s": data.get("offset_s"),
        "method": data.get("method", ""),
        "video_anchor_s": details.get("video_anchor_s"),
        "onset_h5_s": details.get("onset_h5_s"),
        "sensor_key": details.get("sensor_key", ""),
        "metric": details.get("metric", ""),
        "threshold": details.get("threshold"),
        "baseline_mean": details.get("baseline_mean"),
        "baseline_std": details.get("baseline_std"),
    }


def build_alignment_manifest():
    rows = []
    alignment_files = sorted(ANNOTATIONS_ROOT.glob("*/*.alignment.json"))
    iterator = alignment_files
    if tqdm is not None:
        iterator = tqdm(alignment_files, desc="Building alignment manifest", unit="file")

    for path in iterator:
        rows.append(read_alignment(path))

    df = pd.DataFrame(rows)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    df.to_csv(ALIGNMENT_MANIFEST_OUTPUT, index=False)
    return df


def main():
    df = build_alignment_manifest()
    print(f"Alignment files: {len(df)}")
    print(f"Saved: {ALIGNMENT_MANIFEST_OUTPUT}")
    if not df.empty:
        print("\nAlignment count by participant:")
        print(df["participant_id"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
