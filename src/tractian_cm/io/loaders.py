from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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
    """Raised when a CSV cannot be loaded or validated."""
    pass


@dataclass(frozen=True)
class RawTriAxial:
    """
    Normalized raw tri-axial waveform for Part 2.

    Always returns:
      - t (seconds)
      - axisX/axisY/axisZ (acceleration in g)
      - fs_est inferred from time vector
    """
    t: np.ndarray
    axisX: np.ndarray
    axisY: np.ndarray
    axisZ: np.ndarray
    fs_est: float
    n_samples: int
    schema: str  # "part2_data" | "part2_test"


def load_raw_triaxial_part2_csv(
    path: str | Path,
    *,
    uniform_tolerance: float = 1e-4,
) -> RawTriAxial:
    """
    Loads a raw tri-axial vibration CSV from Part 2 datasets and normalizes schema to:
    t, axisX, axisY, axisZ.

    Supported schemas (per case statement):
    - data/: columns:
        'X-Axis' (time, s),
        'Ch1 Y-Axis', 'Ch2 Y-Axis', 'Ch3 Y-Axis' (acc in g)
      where Ch1/Ch2/Ch3 correspond to axes X/Y/Z respectively.
    - test_data/: columns:
        't' (time, s),
        'x', 'y', 'z' (acc in g) corresponding to axes X/Y/Z.

    Validations:
    - strictly increasing time
    - approximately uniform sampling (tolerance configurable)
    - finite values, consistent lengths
    - fs_est inferred from time vector
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    df = pd.read_csv(path)
    cols = set(df.columns)

    schema_a = {"X-Axis", "Ch1 Y-Axis", "Ch2 Y-Axis", "Ch3 Y-Axis"}  # data/
    schema_b = {"t", "x", "y", "z"}  # test_data/

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
            f"Columns={list(df.columns)}. Expected either {sorted(schema_a)} or {sorted(schema_b)}."
        )

    # Finite checks
    if not np.all(np.isfinite(t)):
        raise LoaderError(f"Non-finite time values found in {path.name}")
    for name, arr in [("axisX", axisX), ("axisY", axisY), ("axisZ", axisZ)]:
        if not np.all(np.isfinite(arr)):
            raise LoaderError(f"Non-finite values found in {name} for {path.name}")

    # Length checks
    n_samples = int(len(t))
    if any(len(arr) != n_samples for arr in [axisX, axisY, axisZ]):
        raise LoaderError(f"Axis arrays have different length than time in {path.name}")

    # Physical time-axis checks (reuse common utilities)
    try:
        assert_strictly_increasing(t)
        assert_uniform_sampling(t, tolerance=uniform_tolerance)
        fs_est = float(infer_sampling_rate(t))
    except Exception as e:
        raise LoaderError(f"Time-axis validation failed for {path.name}: {e}") from e

    return RawTriAxial(
        t=t,
        axisX=axisX,
        axisY=axisY,
        axisZ=axisZ,
        fs_est=fs_est,
        n_samples=n_samples,
        schema=schema,
    )
