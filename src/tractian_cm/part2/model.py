from __future__ import annotations

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
import numpy as np


class Wave(BaseModel):
    time: List[float] = Field(..., description="Time points of the wave")
    signal: List[float] = Field(..., description="Signal values")


class LoosenessModel:
    """
    Part 2 model interface required by the case PDF.

    Implements:
      - __init__(**params)
      - predict(wave_hor, wave_ver, wave_axi) -> bool
      - score(wave_hor, wave_ver, wave_axi) -> float (0..1)
    """

    def __init__(self, **params: Any):
        # Store hyperparameters / configuration
        self.params: Dict[str, Any] = dict(params)

        # Reasonable defaults (can be overwritten via **params)
        self.rpm: Optional[float] = self.params.get("rpm", None)

        # Threshold on score to output boolean label
        self.threshold: float = float(self.params.get("threshold", 0.5))

        # Scoring weights (heuristic v2)
        self.w_2x: float = float(self.params.get("w_2x", 0.45))
        self.w_3x: float = float(self.params.get("w_3x", 0.35))
        self.w_crest: float = float(self.params.get("w_crest", 0.20))

        # Feature scaling / centers (tuned from your analysis; keep configurable)
        self.center_2x: float = float(self.params.get("center_2x", 0.9))
        self.scale_2x: float = float(self.params.get("scale_2x", 0.25))

        self.center_3x: float = float(self.params.get("center_3x", 0.6))
        self.scale_3x: float = float(self.params.get("scale_3x", 0.25))

        self.center_crest: float = float(self.params.get("center_crest", 3.5))
        self.scale_crest: float = float(self.params.get("scale_crest", 0.7))

        # Harmonic extraction params
        self.base_tol_hz: float = float(self.params.get("base_tol_hz", 2.0))
        self.rel_tol_rot: float = float(self.params.get("rel_tol_rot", 0.02))  # 2% of f_rot
        self.min_nyquist_factor: float = float(self.params.get("min_nyquist_factor", 1.2))

    # ------------------------
    # Public API (as required)
    # ------------------------

    def predict(self, wave_hor: Wave, wave_ver: Wave, wave_axi: Wave) -> bool:
        """
        Predicts looseness presence from horizontal, vertical and axial waves.
        Returns True if looseness is detected.
        """
        s = self.score(wave_hor, wave_ver, wave_axi)
        return bool(s >= self.threshold)

    def score(self, wave_hor: Wave, wave_ver: Wave, wave_axi: Wave) -> float:
        """
        Computes confidence score (0..1) for structural looseness.
        """
        # Convert to numpy and validate shapes
        t_h, x_h = self._to_np(wave_hor)
        t_v, x_v = self._to_np(wave_ver)
        t_a, x_a = self._to_np(wave_axi)

        # Infer fs and validate time sampling consistency (lightweight)
        fs_h = self._infer_fs(t_h)
        fs_v = self._infer_fs(t_v)
        fs_a = self._infer_fs(t_a)

        # Use a robust fs estimate (median of the three)
        fs = float(np.median([fs_h, fs_v, fs_a]))

        # rpm is required to anchor harmonics; load it from params
        if self.rpm is None:
            raise ValueError(
                "LoosenessModel requires 'rpm' in params to anchor harmonic frequencies. "
                "Instantiate as LoosenessModel(rpm=<value>, ...)."
            )

        f_rot = float(self.rpm) / 60.0  # Hz

        # Basic Nyquist guard (if rpm implies harmonics above Nyquist, scoring degrades)
        nyq = fs / 2.0
        if (3.0 * f_rot) > (nyq / self.min_nyquist_factor):
            # Not enough bandwidth to reliably evaluate 3x
            # We still score using available signals, but down-weight 3x
            w_3x = 0.0
            w_2x = self.w_2x + self.w_3x  # reallocate
        else:
            w_3x = self.w_3x
            w_2x = self.w_2x

        # Aggregate directional spectrum features (H/V/A)
        # Compute 1x,2x,3x amplitudes per axis and combine robustly
        a1 = self._harmonic_amp(x_h, fs, f_rot, self._tol_hz(fs, f_rot))
        a2 = self._harmonic_amp(x_h, fs, 2.0 * f_rot, self._tol_hz(fs, f_rot))
        a3 = self._harmonic_amp(x_h, fs, 3.0 * f_rot, self._tol_hz(fs, f_rot))

        b1 = self._harmonic_amp(x_v, fs, f_rot, self._tol_hz(fs, f_rot))
        b2 = self._harmonic_amp(x_v, fs, 2.0 * f_rot, self._tol_hz(fs, f_rot))
        b3 = self._harmonic_amp(x_v, fs, 3.0 * f_rot, self._tol_hz(fs, f_rot))

        c1 = self._harmonic_amp(x_a, fs, f_rot, self._tol_hz(fs, f_rot))
        c2 = self._harmonic_amp(x_a, fs, 2.0 * f_rot, self._tol_hz(fs, f_rot))
        c3 = self._harmonic_amp(x_a, fs, 3.0 * f_rot, self._tol_hz(fs, f_rot))

        # Robust combine across axes (median reduces outlier axis effects)
        amp_1x = float(np.median([a1, b1, c1]))
        amp_2x = float(np.median([a2, b2, c2]))
        amp_3x = float(np.median([a3, b3, c3]))

        # Harmonic ratios (with epsilon for stability)
        eps = 1e-12
        r2 = amp_2x / (amp_1x + eps)
        r3 = amp_3x / (amp_1x + eps)

        # Crest factor (median across axes)
        crest = float(np.median([self._crest_factor(x_h), self._crest_factor(x_v), self._crest_factor(x_a)]))

        # Soft scoring via sigmoid over each feature
        s2 = self._sigmoid((r2 - self.center_2x) / (self.scale_2x + eps))
        s3 = self._sigmoid((r3 - self.center_3x) / (self.scale_3x + eps))
        sc = self._sigmoid((crest - self.center_crest) / (self.scale_crest + eps))

        # Weighted sum → normalized to [0,1]
        score = (w_2x * s2) + (w_3x * s3) + (self.w_crest * sc)

        # Ensure final score within [0,1]
        return float(np.clip(score, 0.0, 1.0))

    # ------------------------
    # Internal helpers
    # ------------------------

    def _to_np(self, wave: Wave) -> tuple[np.ndarray, np.ndarray]:
        t = np.asarray(wave.time, dtype=float)
        x = np.asarray(wave.signal, dtype=float)
        if t.ndim != 1 or x.ndim != 1:
            raise ValueError("Wave.time and Wave.signal must be 1D.")
        if len(t) != len(x):
            raise ValueError("Wave.time and Wave.signal must have the same length.")
        if len(t) < 8:
            raise ValueError("Wave must contain enough samples for spectral analysis.")
        return t, x

    def _infer_fs(self, t: np.ndarray) -> float:
        dt = np.diff(t)
        if np.any(dt <= 0):
            raise ValueError("Invalid time vector: non-positive time differences detected.")
        dt_med = float(np.median(dt))
        return 1.0 / dt_med

    def _tol_hz(self, fs: float, f_rot: float) -> float:
        # resolution-aware tolerance
        # approximate bin width using N (we compute FFT on full signal)
        # For robustness, use max(base, rel*f_rot, 2*df) where df ~ fs/N
        # Here, df computed later in _harmonic_amp; we approximate with 2*fs/N there.
        # We still return base and relative component here; df handled inside.
        return max(self.base_tol_hz, self.rel_tol_rot * f_rot)

    def _harmonic_amp(self, x: np.ndarray, fs: float, f_target: float, tol_base: float) -> float:
        n = len(x)
        # FFT magnitude (rfft)
        X = np.fft.rfft(x - np.mean(x))
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        mag = np.abs(X)

        # resolution-aware tolerance
        df = fs / n
        tol = max(tol_base, 2.0 * df)

        # find bins within tolerance
        mask = (freqs >= (f_target - tol)) & (freqs <= (f_target + tol))
        if not np.any(mask):
            return 0.0

        return float(np.max(mag[mask]))

    def _crest_factor(self, x: np.ndarray) -> float:
        peak = float(np.max(np.abs(x)))
        rms = float(np.sqrt(np.mean(np.square(x))) + 1e-12)
        return peak / rms

    def _sigmoid(self, z: float) -> float:
        # numerically stable sigmoid
        if z >= 0:
            ez = np.exp(-z)
            return float(1.0 / (1.0 + ez))
        else:
            ez = np.exp(z)
            return float(ez / (1.0 + ez))
        
    def explain(self, wave_hor: Wave, wave_ver: Wave, wave_axi: Wave) -> dict:
        """
        Returns intermediate physical features for transparency.
        """
        ...
        return {
            "ratio_2x_1x": r2,
            "ratio_3x_1x": r3,
            "crest_factor": crest
        }

    def explain(self, wave_hor: Wave, wave_ver: Wave, wave_axi: Wave) -> dict:
        """
        Returns physical intermediate features used for scoring.
        """

        # Recompute same internal quantities used in score()
        t_h, x_h = self._to_np(wave_hor)
        t_v, x_v = self._to_np(wave_ver)
        t_a, x_a = self._to_np(wave_axi)

        fs = float(np.median([
            self._infer_fs(t_h),
            self._infer_fs(t_v),
            self._infer_fs(t_a),
        ]))

        if self.rpm is None:
            raise ValueError("LoosenessModel requires 'rpm' in params.")

        f_rot = float(self.rpm) / 60.0

        tol = self._tol_hz(fs, f_rot)

        # Harmonic amplitudes
        a1 = self._harmonic_amp(x_h, fs, f_rot, tol)
        a2 = self._harmonic_amp(x_h, fs, 2.0 * f_rot, tol)
        a3 = self._harmonic_amp(x_h, fs, 3.0 * f_rot, tol)

        b1 = self._harmonic_amp(x_v, fs, f_rot, tol)
        b2 = self._harmonic_amp(x_v, fs, 2.0 * f_rot, tol)
        b3 = self._harmonic_amp(x_v, fs, 3.0 * f_rot, tol)

        c1 = self._harmonic_amp(x_a, fs, f_rot, tol)
        c2 = self._harmonic_amp(x_a, fs, 2.0 * f_rot, tol)
        c3 = self._harmonic_amp(x_a, fs, 3.0 * f_rot, tol)

        amp_1x = float(np.median([a1, b1, c1]))
        amp_2x = float(np.median([a2, b2, c2]))
        amp_3x = float(np.median([a3, b3, c3]))

        eps = 1e-12
        r2 = amp_2x / (amp_1x + eps)
        r3 = amp_3x / (amp_1x + eps)

        crest = float(np.median([
            self._crest_factor(x_h),
            self._crest_factor(x_v),
            self._crest_factor(x_a),
        ]))

        return {
            "agg__amp_2x_over_1x_max": r2,
            "agg__amp_3x_over_1x_max": r3,
            "agg__crest_factor_max": crest,
        }
