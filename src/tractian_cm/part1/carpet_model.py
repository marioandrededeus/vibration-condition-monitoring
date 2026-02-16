from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from tractian_cm.dsp.spectra import WelchConfig, to_db, welch_psd
from tractian_cm.part1.schemas import CarpetRegion, Wave


@dataclass(frozen=True)
class CarpetDetectorConfig:
    """
    EN: Baseline carpet detector configuration.
    EN: Carpet is modeled as broadband elevation above a robust spectral baseline (noise floor).
    """
    min_freq_hz: float = 1000.0
    threshold_db: float = 4.0
    median_window_bins: int = 61  # EN: more rigid baseline to avoid tracking broad structures
    min_region_width_hz: float = 200.0
    welch: WelchConfig = WelchConfig()

    # EN: Detection mode
    mode: str = "band"  # "point" or "band"

    # EN: Band-consistency parameters (broadband vs isolated peaks)
    band_width_hz: float = 200.0
    frac_min: float = 0.30


def _rolling_median(x: np.ndarray, window: int) -> np.ndarray:
    """
    EN: Rolling median approximates spectral noise floor and is robust to narrow peaks.
    """
    if window < 3 or window % 2 == 0:
        raise ValueError("median_window_bins must be an odd integer >= 3")

    pad = window // 2
    x_pad = np.pad(x, pad_width=pad, mode="edge")

    out = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        out[i] = np.median(x_pad[i : i + window])
    return out


def _mask_to_regions(freq: np.ndarray, mask: np.ndarray) -> List[Tuple[float, float]]:
    """
    EN: Convert boolean mask over frequency bins to contiguous [start_hz, end_hz] intervals.
    """
    if len(freq) != len(mask):
        raise ValueError("freq and mask must have the same length")

    regions: List[Tuple[float, float]] = []
    in_region = False
    start_hz = 0.0

    for i in range(len(mask)):
        if mask[i] and not in_region:
            in_region = True
            start_hz = float(freq[i])
        elif (not mask[i]) and in_region:
            end_hz = float(freq[i - 1])
            regions.append((start_hz, end_hz))
            in_region = False

    if in_region:
        regions.append((start_hz, float(freq[-1])))

    return regions


def detect_carpet_regions(wave: Wave, fs: float, cfg: CarpetDetectorConfig = CarpetDetectorConfig()) -> List[CarpetRegion]:
    """
    EN: Baseline carpet detection using relative elevation above noise floor in dB.

    Steps:
      1) Welch PSD -> stable spectrum for broadband phenomena
      2) Convert PSD to dB
      3) Restrict to f >= min_freq_hz (case requirement)
      4) Estimate baseline via rolling median in dB
      5) Detect bins where (PSD_dB - baseline_dB) > threshold_db
      6) Convert contiguous bins to regions
      7) Filter by minimum region width
    """
    x = np.asarray(wave.signal, dtype=float)

    f, pxx = welch_psd(x, fs=fs, cfg=cfg.welch)
    pxx_db = to_db(pxx)

    band = f >= cfg.min_freq_hz
    f_band = f[band]
    p_band = pxx_db[band]

    if len(f_band) < cfg.median_window_bins:
        raise ValueError("Not enough frequency bins above min_freq_hz for the configured median window.")

    baseline = _rolling_median(p_band, cfg.median_window_bins)
    delta_db = p_band - baseline

    if cfg.mode not in {"point", "band"}:
        raise ValueError("cfg.mode must be 'point' or 'band'")

    if cfg.mode == "point":
        # EN: Pointwise threshold (prone to peak-driven detections)
        mask = delta_db > cfg.threshold_db
    else:
        # EN: Band-consistency threshold (targets broadband elevation)
        df = float(np.median(np.diff(f_band)))
        bins_per_band = max(1, int(round(cfg.band_width_hz / df)))

        above = (delta_db > cfg.threshold_db).astype(float)
        frac = np.convolve(above, np.ones(bins_per_band), mode="same") / bins_per_band

        mask = frac >= cfg.frac_min

    raw_regions = _mask_to_regions(f_band, mask)

    out: List[CarpetRegion] = []
    for start_hz, end_hz in raw_regions:
        if (end_hz - start_hz) >= cfg.min_region_width_hz:
            out.append(CarpetRegion(start_hz=start_hz, end_hz=end_hz))

    return out
