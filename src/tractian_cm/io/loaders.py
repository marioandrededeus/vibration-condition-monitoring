from __future__ import annotations

from pathlib import Path

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
