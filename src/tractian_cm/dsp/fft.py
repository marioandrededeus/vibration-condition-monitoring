# src/tractian_cm/dsp/fft.py
from __future__ import annotations

import numpy as np


def fft_magnitude_single_sided(x: np.ndarray, fs_hz: float) -> tuple[np.ndarray, np.ndarray]:
    """
    EN: Compute single-sided FFT magnitude for a real-valued time signal.
    Returns frequency vector (Hz) and magnitude (a.u.).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        raise ValueError("Signal length must be >= 2.")

    # EN: real FFT for efficiency
    X = np.fft.rfft(x)
    mag = np.abs(X) / n

    f = np.fft.rfftfreq(n, d=1.0 / float(fs_hz))
    return f, mag
