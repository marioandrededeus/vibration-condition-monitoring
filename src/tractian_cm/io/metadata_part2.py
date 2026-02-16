"""
Part 2 - Looseness prediction
Metadata loader and validation utilities.

Files:
- part_3_metadata.csv: train metadata (subset labeled)
- test_metadata.csv: test metadata
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd


AXES = {"axisX", "axisY", "axisZ"}
ORIENTATIONS = {"horizontal", "vertical", "axial"}
CONDITIONS = {"healthy", "structural_looseness"}


class MetadataValidationError(ValueError):
    pass


def parse_orientation(value: Any) -> Dict[str, str]:
    """
    Parses the `orientation` field which may come as:
    - dict already
    - string representation of dict (e.g. "{'axisX': 'horizontal', ...}")
    """
    if isinstance(value, dict):
        orientation = value
    elif isinstance(value, str):
        s = value.strip()
        # orientation in provided CSV is usually a Python dict string, not JSON
        try:
            orientation = ast.literal_eval(s)
        except Exception as e:
            raise MetadataValidationError(f"Failed to parse orientation string: {value}") from e
    else:
        raise MetadataValidationError(f"Unsupported orientation type: {type(value)}")

    if not isinstance(orientation, dict):
        raise MetadataValidationError(f"Parsed orientation is not a dict: {orientation}")

    # Normalize keys/values to str
    orientation = {str(k): str(v) for k, v in orientation.items()}
    validate_orientation(orientation)
    return orientation


def validate_orientation(orientation: Dict[str, str]) -> None:
    """
    Validates orientation mapping:
    - keys must be axisX/axisY/axisZ
    - values must include exactly one of each: horizontal/vertical/axial
    """
    keys = set(orientation.keys())
    if keys != AXES:
        raise MetadataValidationError(
            f"Orientation keys must be {AXES}, got {keys}. Value: {orientation}"
        )

    values = list(orientation.values())
    values_set = set(values)
    if not values_set.issubset(ORIENTATIONS):
        raise MetadataValidationError(
            f"Orientation values must be subset of {ORIENTATIONS}, got {values_set}. Value: {orientation}"
        )

    # Must contain exactly one of each orientation (no duplicates)
    if sorted(values) != sorted(ORIENTATIONS):
        raise MetadataValidationError(
            "Orientation must map to exactly one of each: horizontal, vertical, axial. "
            f"Got values={values}. Full mapping: {orientation}"
        )


def _ensure_columns(df: pd.DataFrame, required: set[str], file_label: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise MetadataValidationError(f"{file_label}: missing columns {missing}. Got columns={list(df.columns)}")


def load_train_metadata_part2(path: str) -> pd.DataFrame:
    """
    Loads labeled subset metadata (part_3_metadata.csv) and returns a normalized DF.

    Expected columns (from case):
    - sample_id
    - condition (healthy/structural_looseness)
    - rpm
    - sensor_id
    - orientation (dict-like string)
    """
    df = pd.read_csv(path)

    required = {"sample_id", "condition", "rpm", "sensor_id", "orientation"}
    _ensure_columns(df, required, "train_metadata_part2")

    # Normalize
    df = df.copy()
    df["sample_id"] = df["sample_id"].astype(str)

    # Validate condition
    df["condition"] = df["condition"].astype(str)
    invalid_conditions = set(df["condition"].unique()) - CONDITIONS
    if invalid_conditions:
        raise MetadataValidationError(f"Invalid conditions found: {invalid_conditions}. Allowed={CONDITIONS}")

    df["label"] = df["condition"].map(lambda x: x == "structural_looseness")

    # rpm numeric
    df["rpm"] = pd.to_numeric(df["rpm"], errors="coerce")
    if df["rpm"].isna().any():
        bad = df[df["rpm"].isna()][["sample_id", "rpm"]].head(10)
        raise MetadataValidationError(f"Found NaN rpm values. Examples:\n{bad}")

    # sensor_id keep as str (safe)
    df["sensor_id"] = df["sensor_id"].astype(str)

    # Parse and validate orientation dict
    df["orientation"] = df["orientation"].apply(parse_orientation)

    # Keep a stable column order
    cols = ["sample_id", "label", "condition", "rpm", "sensor_id", "orientation"]
    return df[cols].sort_values("sample_id").reset_index(drop=True)


def load_test_metadata_part2(path: str) -> pd.DataFrame:
    """
    Loads test metadata (test_metadata.csv) and returns a normalized DF.

    Expected columns (from case):
    - sample_id
    - rpm
    - asset
    - orientation
    """
    df = pd.read_csv(path)

    required = {"sample_id", "rpm", "asset", "orientation"}
    _ensure_columns(df, required, "test_metadata_part2")

    df = df.copy()
    df["sample_id"] = df["sample_id"].astype(str)

    df["rpm"] = pd.to_numeric(df["rpm"], errors="coerce")
    if df["rpm"].isna().any():
        bad = df[df["rpm"].isna()][["sample_id", "rpm"]].head(10)
        raise MetadataValidationError(f"Found NaN rpm values in test metadata. Examples:\n{bad}")

    df["asset"] = df["asset"].astype(str)
    df["orientation"] = df["orientation"].apply(parse_orientation)

    cols = ["sample_id", "rpm", "asset", "orientation"]
    return df[cols].sort_values("sample_id").reset_index(drop=True)
