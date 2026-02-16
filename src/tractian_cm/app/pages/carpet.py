# src/tractian_cm/app/pages/carpet.py
from __future__ import annotations

from pathlib import Path

import streamlit as st

from tractian_cm.app.ui_shared import download_json_button, save_uploaded_to_temp, traffic_badge
from tractian_cm.pipelines.fft_carpet import FFTCarpetConfig, batch_analyze
from tractian_cm.viz.plotly_fft import fft_clusters_figure


def _traffic_light(max_index: float, stability: float | None, n_regions_total: int) -> tuple[str, str]:
    """
    EN: Heuristic traffic light (not a probability).
    """
    if n_regions_total == 0:
        return ("GREEN", "No carpet-like regions detected in the uploaded sample(s).")

    stab_ok = True if stability is None else (stability >= 0.60)

    if (max_index >= 0.70) and stab_ok:
        return ("RED", "Strong carpet-like indication in at least one sample.")
    if max_index >= 0.40:
        return ("YELLOW", "Some carpet-like indication detected; review bands and diagnostics.")
    return ("YELLOW", "Detected regions are weak; review diagnostics for confirmation.")


def render_carpet() -> None:
    st.subheader("Carpet detection (FFT)")
    st.caption(
        "FFT-based carpet indication aligned with the challenge definition. "
        "Outputs are heuristic severity indicators (not calibrated probabilities)."
    )

    with st.sidebar:
        st.header("Parameters (Carpet detection)")
        cfg = FFTCarpetConfig(
            fmin_hz=float(st.number_input("fmin_hz", value=1000.0, step=100.0)),
            k_peak=float(st.number_input("k_peak (MAD multiplier)", value=6.0, step=0.5)),
            min_distance_hz=float(st.number_input("min_distance_hz", value=5.0, step=1.0)),
            gap_max_hz=float(st.number_input("gap_max_hz", value=35.0, step=5.0)),
            min_peaks_cluster=int(st.number_input("min_peaks_cluster", value=8, step=1)),
            top_k_clusters=1,
            compute_stability=bool(st.checkbox("Compute stability (gap 25/35/50)", value=True)),
        )

        st.divider()
        st.header("Data Input (Part 1)")
        uploads = st.file_uploader(
            "Upload one or more CSV files",
            type=["csv"],
            accept_multiple_files=True,
        )

    if not uploads:
        st.info("Upload CSV files in the sidebar to run Part 1 analysis.")
        return

    temp_and_names = [save_uploaded_to_temp(u) for u in uploads]
    paths = [p for (p, _) in temp_and_names]
    basename_map = {Path(p).name: orig for (p, orig) in temp_and_names}

    batch, samples_sorted, artifacts = batch_analyze(paths, cfg)

    # EN: replace temp basenames with original upload names (UI + export)
    renamed_samples = [s.model_copy(update={"file_name": basename_map.get(s.file_name, s.file_name)}) for s in samples_sorted]

    remapped_artifacts = {}
    for k, v in artifacts.items():
        remapped_artifacts[basename_map.get(k, k)] = v
    artifacts = remapped_artifacts

    samples_sorted = sorted(renamed_samples, key=lambda x: x.severity_score, reverse=True)
    batch = batch.model_copy(update={"top3": samples_sorted[:3]})

    max_index = max(s.severity_index_0_1 for s in samples_sorted) if samples_sorted else 0.0
    stability_top = samples_sorted[0].stability_0_1 if samples_sorted else None
    light, message = _traffic_light(max_index=max_index, stability=stability_top, n_regions_total=batch.n_regions_total)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(traffic_badge(light), unsafe_allow_html=True)
    with c2:
        st.metric("Samples Loaded", batch.n_samples)
    with c3:
        st.metric("Carpet-like Regions (Total)", batch.n_regions_total)
    with c4:
        st.metric("Max Severity Index (0–1)", f"{max_index:.2f}")

    st.write(message)

    st.subheader("Top 3 Samples by Carpet Severity")
    st.dataframe(
        [
            {
                "file_name": s.file_name,
                "severity_score": float(s.severity_score),
                "severity_index_0_1": float(s.severity_index_0_1),
                "stability_0_1": None if s.stability_0_1 is None else float(s.stability_0_1),
                "n_clusters": int(s.n_clusters),
                "n_peaks": int(s.n_peaks),
                "fs_hz": float(s.fs_hz),
                "rms": float(s.rms),
                "n_regions": int(len(s.regions)),
            }
            for s in batch.top3
        ],
        use_container_width=True,
    )

    st.subheader("Diagnostics")
    sample_names = [s.file_name for s in samples_sorted]
    selected = st.selectbox("Select a sample", sample_names, index=0)
    s = next(x for x in samples_sorted if x.file_name == selected)
    a = artifacts[selected]

    left, right = st.columns([2, 1])
    with left:
        fig = fft_clusters_figure(
            f=a["f"],
            mag=a["mag"],
            clusters=a["clusters"],
            peaks_f=a["peaks_f"],
            peaks_m=a["peaks_m"],
            title=f"FFT magnitude with clusters — {selected}",
            fmin=cfg.fmin_hz,
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.markdown("### Selected Sample Metrics")
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Severity score (raw)", f"{s.severity_score:.3f}")
            st.metric("Clusters detected", s.n_clusters)
            st.metric("fs (Hz)", f"{s.fs_hz:.1f}")
            if s.stability_0_1 is not None:
                st.metric("Stability (0–1)", f"{s.stability_0_1:.2f}")
        with m2:
            st.metric("Severity index (0–1)", f"{s.severity_index_0_1:.2f}")
            st.metric("Peaks detected", s.n_peaks)
            st.metric("RMS", f"{s.rms:.3f}")
            st.metric("Regions", len(s.regions))

    st.markdown("### Regions (by cluster score)")
    if s.regions:
        st.dataframe([r.model_dump() for r in s.regions], use_container_width=True)
    else:
        st.caption("No carpet-like regions detected for this sample.")

    st.subheader("Export")
    export_obj = {
        "mode": "carpet_fft_carpet",
        "batch": batch.model_dump(),
        "samples": [x.model_dump() for x in samples_sorted],
        "config": cfg.__dict__,
        "notes": {"interpretation": "Heuristic indication based on FFT peak clustering; not a calibrated probability."},
    }
    download_json_button(export_obj, filename="carpet_report.json")
