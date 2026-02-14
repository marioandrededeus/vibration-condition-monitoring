from __future__ import annotations

import numpy as np


def assert_strictly_increasing(time: np.ndarray) -> None:
    """
    EN: Ensures time axis is strictly increasing.
    EN: Non-monotonic time breaks frequency-domain interpretation.
    """
    if np.any(np.diff(time) <= 0):
        raise ValueError("Time axis is not strictly increasing.")


def infer_sampling_rate(time: np.ndarray) -> float:
    """
    EN: Estimate sampling frequency from median time step.
    EN: Median is used to reduce sensitivity to jitter.
    """
    dt = np.diff(time)

    if len(dt) == 0:
        raise ValueError("Time array too short to infer sampling rate.")

    dt_median = float(np.median(dt))

    if dt_median <= 0:
        raise ValueError("Invalid time step detected.")

    fs = 1.0 / dt_median
    return fs


def assert_uniform_sampling(time: np.ndarray, tolerance: float = 1e-4) -> None:
    """
    EN: Validates approximate uniform sampling.
    EN: Large variation in dt introduces spectral distortion.
    """
    dt = np.diff(time)
    dt_median = np.median(dt)

    relative_variation = np.abs(dt - dt_median) / dt_median

    if np.max(relative_variation) > tolerance:
        raise ValueError(
            f"Sampling is not uniform within tolerance {tolerance}. "
            f"Max relative deviation: {np.max(relative_variation)}"
        )
