# src/tractian_cm/pipelines/fft_carpet.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from tractian_cm.io.loaders import load_part1_wave_csv
from tractian_cm.dsp.fft import fft_magnitude_single_sided
from tractian_cm.dsp.peaks import extract_peaks_in_band
from tractian_cm.part1.fft_carpet import (
    PeakCluster,
    cluster_peaks_by_gap,
    cluster_metrics_and_score,
    sample_severity_from_clusters,
)

# Reuse your existing FFT + peak + cluster functions (already in notebook 02)
# You should move these implementations into src (or import from existing modules):
# - fft_magnitude_single_sided
# - extract_peaks_in_band
# - PeakCluster + cluster_peaks_by_gap
# - cluster_metrics_and_score
# - sample_severity_from_clusters


class CarpetRegionFFT(BaseModel):
    """EN: A detected carpet-like region defined by a peak cluster."""
    start_hz: float
    end_hz: float
    n_peaks: int
    density_per_khz: float
    spacing_cv: float
    dominance: float
    score: float


class SingleSampleResult(BaseModel):
    """EN: Result for a single uploaded sample."""
    file_name: str
    fs_hz: float
    rms: float

    n_peaks: int
    n_clusters: int

    severity_score: float = Field(..., description="Raw severity score (higher = more carpet-like).")
    severity_index_0_1: float = Field(..., ge=0.0, le=1.0, description="Normalized index for UI readability.")

    stability_0_1: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional robustness indicator based on gap sensitivity.",
    )

    regions: list[CarpetRegionFFT]
    params: dict[str, Any]


class BatchResult(BaseModel):
    """EN: Result for a batch of samples."""
    n_samples: int
    n_regions_total: int
    top3: list[SingleSampleResult]


@dataclass(frozen=True)
class FFTCarpetConfig:
    """EN: Central config used by pipeline and app."""
    fmin_hz: float = 1000.0
    k_peak: float = 6.0
    min_distance_hz: float = 5.0

    gap_max_hz: float = 35.0
    min_peaks_cluster: int = 8

    top_k_clusters: int = 1  # EN: worst symptom / strongest evidence
    compute_stability: bool = True


def _rms(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.sqrt(np.mean(x * x)))


def _robust_index_0_1(score: float, p10: float, p90: float) -> float:
    """EN: Robust min-max using percentiles; clipped to [0,1]."""
    if p90 <= p10:
        return 0.0
    z = (score - p10) / (p90 - p10)
    return float(np.clip(z, 0.0, 1.0))


def analyze_csv_path(path, cfg: FFTCarpetConfig) -> dict:
    """
    EN: Analyze a single CSV path and return raw artifacts for UI/plots/export.
    Returns dict with: result model + FFT arrays + peaks and clusters.
    """
    wave, fs = load_part1_wave_csv(path)
    x = np.asarray(wave.signal, dtype=float)

    f, mag = fft_magnitude_single_sided(x, fs)

    mask = f >= cfg.fmin_hz
    fb = f[mask]
    mb = mag[mask]

    fpk, mpk, thr = extract_peaks_in_band(
        fb, mb,
        fmin=cfg.fmin_hz,
        fmax=float(fb.max()),
        k=cfg.k_peak,
        min_distance_hz=cfg.min_distance_hz,
    )

    clusters = cluster_peaks_by_gap(
        fpk, mpk,
        gap_max_hz=cfg.gap_max_hz,
        min_peaks=cfg.min_peaks_cluster,
    )

    cluster_scores = [cluster_metrics_and_score(c) for c in clusters]
    severity = sample_severity_from_clusters(cluster_scores, top_k=cfg.top_k_clusters)

    regions = [
        CarpetRegionFFT(
            start_hz=float(d["f_start_hz"]),
            end_hz=float(d["f_end_hz"]),
            n_peaks=int(d["n_peaks"]),
            density_per_khz=float(d["density_per_khz"]),
            spacing_cv=float(d["spacing_cv"]),
            dominance=float(d["dominance"]),
            score=float(d["score"]),
        )
        for d in sorted(cluster_scores, key=lambda z: z["score"], reverse=True)
        if np.isfinite(d["score"])
    ]

    return {
        "file_name": getattr(path, "name", str(path)),
        "fs_hz": float(fs),
        "rms": _rms(x),
        "n_peaks": int(len(fpk)),
        "n_clusters": int(len(clusters)),
        "severity_score": float(severity),
        "regions": regions,
        "params": {
            "fmin_hz": cfg.fmin_hz,
            "k_peak": cfg.k_peak,
            "min_distance_hz": cfg.min_distance_hz,
            "gap_max_hz": cfg.gap_max_hz,
            "min_peaks_cluster": cfg.min_peaks_cluster,
            "top_k_clusters": cfg.top_k_clusters,
            "peak_threshold_used": float(thr),
        },
        # artifacts for plot
        "f": fb,
        "mag": mb,
        "peaks_f": fpk,
        "peaks_m": mpk,
        "clusters": clusters,
    }


def compute_stability_index(scores: list[float]) -> float:
    """
    EN: Heuristic stability from parameter sweep scores.
    Higher = more stable.
    """
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return 0.0
    mu = float(np.mean(s))
    if mu <= 0:
        return 0.0
    cv = float(np.std(s) / mu)
    return float(np.clip(1.0 - cv, 0.0, 1.0))


def batch_analyze(paths: list, cfg: FFTCarpetConfig) -> tuple[BatchResult, list[SingleSampleResult], dict]:
    """
    EN: Analyze a batch and compute robust 0-1 indices inside the batch.
    Returns: batch summary, per-sample results, and artifacts for plotting/export.
    """
    raw = [analyze_csv_path(p, cfg) for p in paths]
    scores = [r["severity_score"] for r in raw]
    p10, p90 = np.percentile(scores, [10, 90]) if len(scores) >= 3 else (min(scores), max(scores))

    sample_models: list[SingleSampleResult] = []
    artifacts: dict[str, dict] = {}

    for r in raw:
        idx = _robust_index_0_1(r["severity_score"], float(p10), float(p90))

        stability = None
        if cfg.compute_stability:
            # EN: quick sensitivity check (kept small)
            gaps = [25.0, cfg.gap_max_hz, 50.0]
            sens_scores = []
            for g in gaps:
                cfg2 = FFTCarpetConfig(**{**cfg.__dict__, "gap_max_hz": g, "compute_stability": False})
                rr = analyze_csv_path(next(p for p in paths if getattr(p, "name", str(p)) == r["file_name"]), cfg2)
                sens_scores.append(float(rr["severity_score"]))
            stability = compute_stability_index(sens_scores)

        m = SingleSampleResult(
            file_name=r["file_name"],
            fs_hz=r["fs_hz"],
            rms=r["rms"],
            n_peaks=r["n_peaks"],
            n_clusters=r["n_clusters"],
            severity_score=r["severity_score"],
            severity_index_0_1=idx,
            stability_0_1=stability,
            regions=r["regions"],
            params=r["params"],
        )
        sample_models.append(m)
        artifacts[m.file_name] = r

    sample_models_sorted = sorted(sample_models, key=lambda x: x.severity_score, reverse=True)
    top3 = sample_models_sorted[:3]

    batch = BatchResult(
        n_samples=len(sample_models_sorted),
        n_regions_total=int(sum(len(s.regions) for s in sample_models_sorted)),
        top3=top3,
    )
    return batch, sample_models_sorted, artifacts
