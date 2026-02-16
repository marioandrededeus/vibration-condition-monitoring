from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from tractian_cm.part1.schemas import Wave


class FeatureError(ValueError):
    pass


# -----------------------------
# Core signal helpers
# -----------------------------

def _to_numpy_wave(w: Wave) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(w.time, dtype=float)
    x = np.asarray(w.signal, dtype=float)
    if t.ndim != 1 or x.ndim != 1:
        raise FeatureError("Wave time/signal must be 1D.")
    if t.size != x.size:
        raise FeatureError("Wave time and signal must have same length.")
    if t.size < 8:
        raise FeatureError("Wave too short for spectral features.")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(x)):
        raise FeatureError("Non-finite values in Wave.")
    return t, x


def _infer_fs_from_time(t: np.ndarray) -> float:
    dt = np.diff(t)
    median_dt = float(np.median(dt))
    if median_dt <= 0:
        raise FeatureError("Non-positive dt while inferring fs.")
    return 1.0 / median_dt


def _rfft_amplitude(x: np.ndarray, fs: float, window: str = "hann") -> tuple[np.ndarray, np.ndarray]:
    """
    Returns frequency bins and single-sided amplitude spectrum (magnitude).
    Amplitude scaling is consistent across signals of same length/window.
    """
    n = x.size
    x_d = x - np.mean(x)

    if window == "hann":
        w = np.hanning(n)
    elif window == "rect":
        w = np.ones(n)
    else:
        raise FeatureError(f"Unsupported window: {window}")

    xw = x_d * w

    # FFT
    X = np.fft.rfft(xw)
    mag = np.abs(X)

    # Frequency bins
    f = np.fft.rfftfreq(n, d=1.0 / fs)

    # Scale to get an amplitude-like quantity
    # (Note: exact calibration is not essential here; consistency matters for ML.)
    # Compensate window RMS-ish effect:
    scale = 2.0 / np.sum(w)
    amp = mag * scale

    return f, amp


def _bandpower_from_amp(f: np.ndarray, amp: np.ndarray, fmin: float, fmax: float) -> float:
    """
    Approximate band power from amplitude spectrum.
    We integrate squared amplitude over band.
    """
    if fmin >= fmax:
        return 0.0
    mask = (f >= fmin) & (f < fmax)
    if not np.any(mask):
        return 0.0
    # integrate amp^2 df
    df = float(np.median(np.diff(f)))
    return float(np.sum((amp[mask] ** 2)) * df)


def _peak_amp_near(f: np.ndarray, amp: np.ndarray, f0: float, tol_hz: float) -> float:
    """
    Returns max amplitude within [f0 - tol, f0 + tol].
    """
    if f0 <= 0 or tol_hz <= 0:
        return 0.0
    mask = (f >= (f0 - tol_hz)) & (f <= (f0 + tol_hz))
    if not np.any(mask):
        return 0.0
    return float(np.max(amp[mask]))


# -----------------------------
# Feature definitions
# -----------------------------

@dataclass(frozen=True)
class LoosenessFeatureParams:
    """
    Parameters controlling spectral feature extraction.

    harmonic_tol_hz:
      tolerance window around each harmonic frequency. Keep it > spectral resolution.
      A robust default is max(0.5 Hz, 0.02 * f_rot).
    """
    window: str = "hann"
    low_bands_hz: Tuple[Tuple[float, float], ...] = ((0.0, 50.0), (50.0, 200.0), (200.0, 500.0))
    include_subharmonic: bool = True
    max_harmonic: int = 3
    base_tol_hz: float = 2
    rel_tol_rot: float = 0.02  # 2% of f_rot


def extract_looseness_features_single_wave(
    wave: Wave,
    rpm: float,
    params: LoosenessFeatureParams = LoosenessFeatureParams(),
) -> Dict[str, float]:
    """
    Extract physics-driven features from a single Wave (one direction).
    Focus:
    - low-frequency band energy
    - rotation-synchronous peaks: 1x, 2x, 3x (and optional 0.5x)
    - harmonic ratios
    """
    if rpm <= 0:
        raise FeatureError(f"Invalid rpm: {rpm}")

    t, x = _to_numpy_wave(wave)
    fs = _infer_fs_from_time(t)

    f, amp = _rfft_amplitude(x, fs=fs, window=params.window)

    f_rot = float(rpm) / 60.0

    # spectral resolution
    df_hz = float(np.median(np.diff(f)))

    # make tolerance robust to FFT resolution
    tol = float(
        max(
            params.base_tol_hz,
            params.rel_tol_rot * f_rot,
            2.0 * df_hz,
        )
    )

    feats: Dict[str, float] = {}
    feats["df_hz"] = df_hz

    # --- Low-frequency bandpowers (absolute Hz bands; safe even with different fs)
    total_power = _bandpower_from_amp(f, amp, 0.0, float(f[-1]))
    feats["spec_power_total"] = total_power

    for (a, b) in params.low_bands_hz:
        bp = _bandpower_from_amp(f, amp, float(a), float(b))
        feats[f"bp_{int(a)}_{int(b)}"] = bp
        feats[f"bp_{int(a)}_{int(b)}_ratio"] = (bp / total_power) if total_power > 0 else 0.0

    # --- Rotation-synchronous peak amplitudes
    # 1x..max_harmonic
    amp_1x = _peak_amp_near(f, amp, f_rot, tol)
    feats["amp_1x"] = amp_1x

    eps = 1e-8

    for k in range(2, params.max_harmonic + 1):
        ak = _peak_amp_near(f, amp, k * f_rot, tol)
        feats[f"amp_{k}x"] = ak
        feats[f"amp_{k}x_over_1x"] = ak / (amp_1x + eps)

    if params.include_subharmonic:
        a05 = _peak_amp_near(f, amp, 0.5 * f_rot, tol)
        feats["amp_0_5x"] = a05
        feats["amp_0_5x_over_1x"] = a05 / (amp_1x + eps)

    # --- Simple time-domain impulsiveness (optional but helpful)
    # These can capture intermittent contact due to looseness.
    rms = float(np.sqrt(np.mean(x ** 2)))
    peak = float(np.max(np.abs(x)))
    feats["rms"] = rms
    feats["peak"] = peak
    feats["crest_factor"] = (peak / rms) if rms > 0 else 0.0

    # kurtosis (excess or raw? Use raw kurtosis (Pearson) for simplicity)
    m2 = float(np.mean((x - np.mean(x)) ** 2))
    m4 = float(np.mean((x - np.mean(x)) ** 4))
    feats["kurtosis"] = (m4 / (m2 ** 2)) if m2 > 0 else 0.0

    # Add meta-derived values (useful for debugging / interpretability)
    feats["fs_est_hz"] = fs
    feats["f_rot_hz"] = f_rot
    feats["harm_tol_hz"] = tol

    return feats


def _prefix_feats(prefix: str, feats: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}__{k}": float(v) for k, v in feats.items()}


def extract_looseness_features_hva(
    wave_hor: Wave,
    wave_ver: Wave,
    wave_axi: Wave,
    rpm: float,
    params: LoosenessFeatureParams = LoosenessFeatureParams(),
) -> Dict[str, float]:
    """
    Extract features for horizontal/vertical/axial waves and return a single flat dict.

    Output keys include:
      hor__*, ver__*, axi__*
    plus aggregated features (max/mean across directions) for key metrics.
    """
    fh = extract_looseness_features_single_wave(wave_hor, rpm, params)
    fv = extract_looseness_features_single_wave(wave_ver, rpm, params)
    fa = extract_looseness_features_single_wave(wave_axi, rpm, params)

    out: Dict[str, float] = {}
    out.update(_prefix_feats("hor", fh))
    out.update(_prefix_feats("ver", fv))
    out.update(_prefix_feats("axi", fa))

    # --- Aggregations for a few key indicators
    # helps robustness when looseness expresses stronger in one direction
    def agg(keys: List[str], name: str) -> None:
        vals = []
        for k in keys:
            vals.append(out[f"hor__{k}"])
            vals.append(out[f"ver__{k}"])
            vals.append(out[f"axi__{k}"])
        out[f"agg__{name}_max"] = float(np.max(vals))
        out[f"agg__{name}_mean"] = float(np.mean(vals))

    agg(["amp_1x"], "amp_1x")
    for k in range(2, params.max_harmonic + 1):
        agg([f"amp_{k}x_over_1x"], f"amp_{k}x_over_1x")
    if params.include_subharmonic:
        agg(["amp_0_5x_over_1x"], "amp_0_5x_over_1x")

    # Low-frequency bandpower ratios (more comparable across sensors)
    for (a, b) in params.low_bands_hz:
        agg([f"bp_{int(a)}_{int(b)}_ratio"], f"bp_{int(a)}_{int(b)}_ratio")

    agg(["crest_factor"], "crest_factor")
    agg(["kurtosis"], "kurtosis")

    return out


def dict_to_feature_vector(feat_dict: Dict[str, float]) -> tuple[np.ndarray, List[str]]:
    """
    Convert dict features to a stable (sorted) vector representation.
    """
    keys = sorted(feat_dict.keys())
    x = np.asarray([feat_dict[k] for k in keys], dtype=float)
    return x, keys
