"""
Part 2 - Looseness prediction
Builds a unified samples index for:
- train labeled subset (data/ with metadata labels)
- train unlabeled remainder (data/ without labels)
- test samples (test_data/ with test_metadata)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

from tractian_cm.io.metadata_part2 import (
    load_test_metadata_part2,
    load_train_metadata_part2,
)


def _list_csv_stems(directory: str | Path) -> set[str]:
    p = Path(directory)
    if not p.exists():
        raise FileNotFoundError(f"Directory not found: {p}")
    return {f.stem for f in p.glob("*.csv")}


def build_part2_samples_index(
    data_dir: str | Path,
    test_data_dir: str | Path,
    train_metadata_path: str | Path,
    test_metadata_path: str | Path,
) -> pd.DataFrame:
    """
    Returns a master index DataFrame with minimal fields needed for Phase 1.

    Columns:
    - sample_id
    - split: train_labeled | train_unlabeled | test
    - filepath
    - rpm
    - sensor_id (nullable)
    - asset (nullable)
    - label (nullable bool)
    - orientation (dict)
    """
    data_dir = Path(data_dir)
    test_data_dir = Path(test_data_dir)

    train_md = load_train_metadata_part2(str(train_metadata_path))
    test_md = load_test_metadata_part2(str(test_metadata_path))

    data_files = _list_csv_stems(data_dir)
    test_files = _list_csv_stems(test_data_dir)

    labeled_ids = set(train_md["sample_id"].astype(str))
    available_data_ids = set(map(str, data_files))

    # Intersection to avoid metadata pointing to missing files
    labeled_present = labeled_ids & available_data_ids
    missing_from_data = labeled_ids - available_data_ids
    if missing_from_data:
        # fail early: metadata references files not present
        raise FileNotFoundError(
            f"Train metadata references {len(missing_from_data)} sample_id not found in data_dir. "
            f"Examples: {sorted(list(missing_from_data))[:10]}"
        )

    unlabeled_ids = sorted(list(available_data_ids - labeled_ids))
    labeled_ids_sorted = sorted(list(labeled_present))

    # Build labeled index
    labeled_df = train_md.copy()
    labeled_df = labeled_df[labeled_df["sample_id"].isin(labeled_ids_sorted)].copy()
    labeled_df["split"] = "train_labeled"
    labeled_df["filepath"] = labeled_df["sample_id"].map(lambda sid: str(data_dir / f"{sid}.csv"))
    labeled_df["asset"] = None

    # Build unlabeled index (from data_dir files not in metadata)
    # For unlabeled, we keep rpm/sensor/orientation unknown for now (None) because metadata is not provided.
    # If you later decide to infer these or compute defaults, do it in a later commit.
    unlabeled_df = pd.DataFrame(
        {
            "sample_id": unlabeled_ids,
            "split": "train_unlabeled",
            "filepath": [str(data_dir / f"{sid}.csv") for sid in unlabeled_ids],
            "rpm": None,
            "sensor_id": None,
            "orientation": None,
            "label": None,
            "condition": None,
            "asset": None,
        }
    )

    # Build test index
    test_md2 = test_md.copy()
    missing_test_files = set(test_md2["sample_id"]) - set(map(str, test_files))
    if missing_test_files:
        raise FileNotFoundError(
            f"Test metadata references {len(missing_test_files)} sample_id not found in test_data_dir. "
            f"Examples: {sorted(list(missing_test_files))[:10]}"
        )

    test_md2["split"] = "test"
    test_md2["filepath"] = test_md2["sample_id"].map(lambda sid: str(test_data_dir / f"{sid}.csv"))
    test_md2["sensor_id"] = None
    test_md2["label"] = None
    test_md2["condition"] = None

    # Align columns
    cols = ["sample_id", "split", "filepath", "rpm", "sensor_id", "asset", "label", "condition", "orientation"]
    labeled_df = labeled_df[cols]
    unlabeled_df = unlabeled_df[cols]
    test_md2 = test_md2[cols]

    idx = pd.concat([labeled_df, unlabeled_df, test_md2], ignore_index=True)
    idx = idx.sort_values(["split", "sample_id"]).reset_index(drop=True)
    return idx
