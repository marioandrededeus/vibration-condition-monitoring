from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from tractian_cm.part1.schemas import Wave
from tractian_cm.utils.signal import (
    assert_strictly_increasing,
    assert_uniform_sampling,
    infer_sampling_rate,
)


def load_part1_wave_csv(path: Path) -> tuple[Wave, float]:
    """
    EN: Load Part 1 waveform CSV ('t', 'data') into a strict schema.
    EN: We validate the time axis because spectral analysis depends on fs and uniform sampling.
    """
    df = pd.read_csv(path)

    required = {"t", "data"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing required columns. Expected: {sorted(required)}")

    time = df["t"].astype(float).to_numpy()
    signal = df["data"].astype(float).to_numpy()

    # EN: Physical consistency checks for vibration signals.
    assert_strictly_increasing(time)
    assert_uniform_sampling(time, tolerance=1e-4)

    fs = infer_sampling_rate(time)

    wave = Wave(time=time.tolist(), signal=signal.tolist())
    return wave, fs

class LoaderError(ValueError):
    pass


@dataclass(frozen=True)
class RawTriAxial:
    t: np.ndarray
    axisX: np.ndarray
    axisY: np.ndarray
    axisZ: np.ndarray
    fs_est: float
    n_samples: int
    schema: str  # "part2_data" or "part2_test"


def _estimate_fs(t: np.ndarray) -> float:
    if t.ndim != 1 or len(t) < 3:
        raise LoaderError("Time vector must be 1D with at least 3 points.")
    dt = np.diff(t)
    if np.any(~np.isfinite(dt)):
        raise LoaderError("Non-finite dt found while estimating fs.")
    median_dt = float(np.median(dt))
    if median_dt <= 0:
        raise LoaderError(f"Non-positive median dt: {median_dt}")
    return 1.0 / median_dt


def _is_monotonic_increasing(t: np.ndarray) -> bool:
    return bool(np.all(np.diff(t) > 0))


def load_raw_triaxial_part2_csv(path: str | Path) -> RawTriAxial:
    """
    Loads a raw tri-axial vibration CSV from Part 2 datasets and normalizes schema to:
    t, axisX, axisY, axisZ.

    Supported schemas:
    - data/: columns: 'X-Axis', 'Ch1 Y-Axis', 'Ch2 Y-Axis', 'Ch3 Y-Axis'
      where Ch1/Ch2/Ch3 correspond to axes X/Y/Z respectively (per case statement).
    - test_data/: columns: 't', 'x', 'y', 'z' where x/y/z correspond to axes X/Y/Z respectively.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    cols = set(df.columns)

    # Schema A: data/
    schema_a = {"X-Axis", "Ch1 Y-Axis", "Ch2 Y-Axis", "Ch3 Y-Axis"}
    # Schema B: test_data/
    schema_b = {"t", "x", "y", "z"}

    if schema_a.issubset(cols):
        t = df["X-Axis"].to_numpy(dtype=float)
        axisX = df["Ch1 Y-Axis"].to_numpy(dtype=float)
        axisY = df["Ch2 Y-Axis"].to_numpy(dtype=float)
        axisZ = df["Ch3 Y-Axis"].to_numpy(dtype=float)
        schema = "part2_data"
    elif schema_b.issubset(cols):
        t = df["t"].to_numpy(dtype=float)
        axisX = df["x"].to_numpy(dtype=float)
        axisY = df["y"].to_numpy(dtype=float)
        axisZ = df["z"].to_numpy(dtype=float)
        schema = "part2_test"
    else:
        raise LoaderError(
            f"Unsupported CSV schema in {path.name}. "
            f"Columns={list(df.columns)}. Expected either {sorted(schema_a)} or {sorted(schema_b)}"
        )

    if not np.all(np.isfinite(t)):
        raise LoaderError(f"Non-finite time values found in {path.name}")

    if not _is_monotonic_increasing(t):
        raise LoaderError(f"Time vector is not strictly increasing in {path.name}")

    for name, arr in [("axisX", axisX), ("axisY", axisY), ("axisZ", axisZ)]:
        if not np.all(np.isfinite(arr)):
            raise LoaderError(f"Non-finite values found in {name} for {path.name}")

    fs_est = _estimate_fs(t)
    n_samples = int(len(t))
    if any(len(arr) != n_samples for arr in [axisX, axisY, axisZ]):
        raise LoaderError(f"Axis arrays have different length than time in {path.name}")

    return RawTriAxial(
        t=t,
        axisX=axisX,
        axisY=axisY,
        axisZ=axisZ,
        fs_est=fs_est,
        n_samples=n_samples,
        schema=schema,
    )
