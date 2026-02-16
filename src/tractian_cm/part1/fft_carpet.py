# src/tractian_cm/part1/fft_carpet.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PeakCluster:
    """EN: A cluster of peaks in frequency domain."""
    f_start_hz: float
    f_end_hz: float
    f_peaks_hz: np.ndarray
    m_peaks: np.ndarray


def cluster_peaks_by_gap(
    f_peaks_hz: np.ndarray,
    m_peaks: np.ndarray,
    gap_max_hz: float = 35.0,
    min_peaks: int = 8,
) -> list[PeakCluster]:
    """
    EN: Group peaks into clusters by maximum allowed frequency gap.
    """
    fpk = np.asarray(f_peaks_hz, dtype=float)
    mpk = np.asarray(m_peaks, dtype=float)
    if fpk.size == 0:
        return []

    order = np.argsort(fpk)
    fpk = fpk[order]
    mpk = mpk[order]

    clusters: list[PeakCluster] = []
    start = 0

    for i in range(1, fpk.size):
        if (fpk[i] - fpk[i - 1]) > gap_max_hz:
            if (i - start) >= min_peaks:
                fp = fpk[start:i]
                mp = mpk[start:i]
                clusters.append(PeakCluster(float(fp.min()), float(fp.max()), fp, mp))
            start = i

    # tail
    if (fpk.size - start) >= min_peaks:
        fp = fpk[start:]
        mp = mpk[start:]
        clusters.append(PeakCluster(float(fp.min()), float(fp.max()), fp, mp))

    return clusters


def cluster_metrics_and_score(cluster: PeakCluster) -> dict[str, Any]:
    """
    EN: Compute interpretable metrics and a severity score for one cluster.
    """
    fpk = cluster.f_peaks_hz
    mpk = cluster.m_peaks

    width_hz = float(cluster.f_end_hz - cluster.f_start_hz)
    width_khz = width_hz / 1000.0 if width_hz > 0 else np.nan

    density = float(fpk.size / width_khz) if np.isfinite(width_khz) and width_khz > 0 else float("nan")

    if fpk.size >= 3:
        df = np.diff(fpk)
        mu = float(np.mean(df))
        spacing_cv = float(np.std(df) / mu) if mu > 0 else float("nan")
    else:
        spacing_cv = float("nan")

    if mpk.size:
        med = float(np.median(mpk))
        dominance = float(np.max(mpk) / (med + 1e-12))
    else:
        dominance = float("nan")

    score = (
        float(density * spacing_cv / (dominance + 1e-12))
        if np.isfinite(density) and np.isfinite(spacing_cv) and np.isfinite(dominance)
        else float("nan")
    )

    return {
        "f_start_hz": cluster.f_start_hz,
        "f_end_hz": cluster.f_end_hz,
        "n_peaks": int(fpk.size),
        "width_hz": width_hz,
        "density_per_khz": density,
        "spacing_cv": spacing_cv,
        "dominance": dominance,
        "score": score,
    }


def sample_severity_from_clusters(cluster_scores: list[dict[str, Any]], top_k: int = 1) -> float:
    """
    EN: Aggregate sample severity as the maximum (top_k=1) or sum of top-k cluster scores.
    """
    scores = [d["score"] for d in cluster_scores if np.isfinite(d["score"])]
    if not scores:
        return 0.0
    scores_sorted = sorted(scores, reverse=True)
    return float(np.sum(scores_sorted[:top_k]))
