# src/tractian_cm/viz/plotly_fft.py
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from tractian_cm.part1.fft_carpet import PeakCluster


def fft_clusters_figure(
    f: np.ndarray,
    mag: np.ndarray,
    clusters: list[PeakCluster],
    peaks_f: np.ndarray | None = None,
    peaks_m: np.ndarray | None = None,
    title: str = "FFT magnitude with clustered regions",
    fmin: float = 1000.0,
    fmax: float | None = None,
) -> go.Figure:
    f = np.asarray(f, dtype=float)
    mag = np.asarray(mag, dtype=float)

    if fmax is None:
        fmax = float(np.max(f))

    sel = (f >= fmin) & (f <= fmax)
    fp = f[sel]
    mp = mag[sel]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fp, y=mp, mode="lines", name="FFT magnitude"))

    if peaks_f is not None and peaks_m is not None:
        fig.add_trace(go.Scatter(
            x=peaks_f, y=peaks_m,
            mode="markers",
            marker=dict(size=6),
            name="Detected peaks",
        ))

    # shaded rectangles for clusters
    for c in clusters:
        a, b = float(c.f_start_hz), float(c.f_end_hz)
        if b < fmin or a > fmax:
            continue
        fig.add_vrect(x0=max(a, fmin), x1=min(b, fmax), opacity=0.15, line_width=0)

    fig.update_layout(
        title=title,
        xaxis_title="Frequency (Hz)",
        yaxis_title="FFT magnitude (a.u.)",
        legend=dict(orientation="h"),
    )
    return fig
