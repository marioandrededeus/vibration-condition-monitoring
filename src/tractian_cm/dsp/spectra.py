from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import welch


@dataclass(frozen=True)
class WelchConfig:
    """
    EN: Centralizes Welch parameters to keep PSD computation reproducible.
    EN: Stable PSD is important for broadband (noise-floor) detection such as 'carpet'.
    """
    nperseg: int = 2048
    noverlap: int = 1024
    window: str = "hann"
    scaling: str = "density"


def welch_psd(x: np.ndarray, fs: float, cfg: WelchConfig = WelchConfig()) -> tuple[np.ndarray, np.ndarray]:
    """
    EN: Compute Welch PSD for 1D vibration signals.
    """
    if x.ndim != 1:
        raise ValueError("Input signal must be 1D.")
    if fs <= 0:
        raise ValueError("Sampling rate fs must be positive.")
    if cfg.noverlap >= cfg.nperseg:
        raise ValueError("noverlap must be smaller than nperseg.")

    f, pxx = welch(
        x,
        fs=fs,
        nperseg=cfg.nperseg,
        noverlap=cfg.noverlap,
        window=cfg.window,
        scaling=cfg.scaling,
    )
    return f, pxx


def to_db(pxx: np.ndarray, eps: float = 1e-20) -> np.ndarray:
    """
    EN: Convert linear power to dB safely (avoid log(0)).
    """
    return 10.0 * np.log10(pxx + eps)
