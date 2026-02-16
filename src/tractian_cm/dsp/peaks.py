# src/tractian_cm/dsp/peaks.py
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def robust_threshold(m: np.ndarray, k: float = 6.0) -> float:
    """
    EN: Robust peak threshold based on MAD.
    """
    m = np.asarray(m, dtype=float)
    med = float(np.median(m))
    mad = float(np.median(np.abs(m - med))) + 1e-12
    return med + k * mad


def extract_peaks_in_band(
    f: np.ndarray,
    m: np.ndarray,
    fmin: float,
    fmax: float,
    k: float = 6.0,
    min_distance_hz: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    EN: Extract peaks in a frequency band using a robust amplitude threshold.

    Returns:
      - peak frequencies (Hz)
      - peak magnitudes
      - threshold used
    """
    f = np.asarray(f, dtype=float)
    m = np.asarray(m, dtype=float)

    sel = (f >= fmin) & (f <= fmax)
    fb = f[sel]
    mb = m[sel]

    if fb.size < 3:
        return np.array([], dtype=float), np.array([], dtype=float), float("nan")

    thr = robust_threshold(mb, k=k)

    # EN: convert distance in Hz to bins (assumes near-uniform spacing)
    df = float(np.median(np.diff(fb)))
    dist_bins = max(1, int(round(min_distance_hz / df)))

    idx, _ = find_peaks(mb, height=thr, distance=dist_bins)

    return fb[idx], mb[idx], float(thr)
