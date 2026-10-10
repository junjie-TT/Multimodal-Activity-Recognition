import json
from pathlib import Path
import sys

import h5py
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from lower_feature_cleaning_extraction.config import (
    CLEAN_CSV_OUTPUT_ROOT,
    MANIFEST_OUTPUT,
    MINUS_ONE_INVALID_COLUMNS,
    MINUS_ONE_INVALID_MODALITY_PREFIXES,
)

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


def sanitize_name(name):
    return name.replace("/", "__").replace(" ", "_")


def read_modality_as_df(h5_file, modality_name):
    group = h5_file["modalities"][modality_name]
    data = {}

    for key, dataset in group.items():
        if isinstance(dataset, h5py.Dataset):
            data[key] = dataset[:]

    return pd.DataFrame(data)


def should_replace_minus_one(modality_name, column_name):
    if any(modality_name.startswith(prefix) for prefix in MINUS_ONE_INVALID_MODALITY_PREFIXES):
        return True

    invalid_columns = MINUS_ONE_INVALID_COLUMNS.get(modality_name, ())
    return column_name in invalid_columns


def clean_dataframe(df, modality_name):
    clean = df.copy()

    for column in clean.columns:
        if not pd.api.types.is_numeric_dtype(clean[column]):
            continue

        values = clean[column].astype(float)
        values = values.replace([np.inf, -np.inf], np.nan)

        if should_replace_minus_one(modality_name, column):
            values = values.mask(values == -1, np.nan)

        clean[column] = values

    return clean


def export_clean_csvs(manifest_path=MANIFEST_OUTPUT, output_root=CLEAN_CSV_OUTPUT_ROOT):
    manifest_path = Path(manifest_path)
    output_root = Path(output_root)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Missing clean manifest: {manifest_path}. "
            "Run build_clean_manifest.py first."
        )

    manifest = pd.read_csv(manifest_path)
    output_root.mkdir(parents=True, exist_ok=True)

    rows = []
    records = list(manifest.to_dict("records"))
    iterator = records
    if tqdm is not None:
        iterator = tqdm(records, desc="Exporting clean sensor CSV", unit="sample")

    for sample in iterator:
        h5_path = Path(sample["h5_path"])
        participant_id = sample["participant_id"]
        full_id = sample["full_id"]
        sample_dir = output_root / participant_id / full_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        with h5py.File(h5_path, "r") as f:
            time_s = None
            if "metadata/global_time_seconds" in f:
                raw_t = f["metadata/global_time_seconds"][:]
                time_s = raw_t - raw_t[0] if len(raw_t) else raw_t

            for modality_name in f["modalities"].keys():
                df = read_modality_as_df(f, modality_name)
                df = clean_dataframe(df, modality_name)

                df.insert(0, "frame_idx", np.arange(len(df)))
                if time_s is not None and len(time_s) == len(df):
                    df.insert(1, "time_s", time_s)

                output_path = sample_dir / f"{sanitize_name(modality_name)}.csv"
                df.to_csv(output_path, index=False)

                rows.append(
                    {
                        "participant_id": participant_id,
                        "task_name": sample["task_name"],
                        "full_id": full_id,
                        "modality": modality_name,
                        "rows": int(len(df)),
                        "columns": int(len(df.columns)),
                        "csv_path": str(output_path),
                    }
                )

    index_df = pd.DataFrame(rows)
    index_path = output_root / "clean_sensor_csv_index.csv"
    index_df.to_csv(index_path, index=False)

    summary = {
        "manifest": str(manifest_path),
        "output_root": str(output_root),
        "sample_count": int(manifest["full_id"].nunique()),
        "csv_file_count": int(len(index_df)),
        "index_csv": str(index_path),
        "invalid_value_policy": {
            "nan_and_inf": "converted to CSV NaN/empty values for all numeric columns",
            "minus_one": {
                "modality_prefixes": list(MINUS_ONE_INVALID_MODALITY_PREFIXES),
                "columns_by_modality": {
                    key: list(value) for key, value in MINUS_ONE_INVALID_COLUMNS.items()
                },
            },
        },
    }

    export_summary_path = output_root / "clean_sensor_csv_summary.json"
    with open(export_summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    return index_df, summary


def main():
    index_df, summary = export_clean_csvs()
    print("Clean sensor CSV export completed.")
    print(json.dumps(summary, indent=2))
    print("\nCSV files by modality:")
    print(index_df["modality"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
