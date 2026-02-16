# src/tractian_cm/app/pages/carpet.py
from __future__ import annotations

from pathlib import Path

import streamlit as st

from tractian_cm.app.ui_shared import download_json_button, save_uploaded_to_temp, traffic_badge
from tractian_cm.pipelines.fft_carpet import FFTCarpetConfig, batch_analyze
from tractian_cm.viz.plotly_fft import fft_clusters_figure
from tractian_cm.app.ui_shared import load_uploads_to_temp_paths  # add import at top


def _traffic_light(
    max_index: float,
    stability: float | None,
    n_regions_total: int,
    med_thr: float,
    high_thr: float,
    use_stability_gate: bool,
    stability_thr: float,
                    ) -> tuple[str, str]:
    
    """
    EN: Heuristic traffic light (not a probability).
    """
    if n_regions_total == 0:
        return ("GREEN", "No carpet-like regions detected in the uploaded sample(s).")

    stab_ok = True
    if use_stability_gate and (stability is not None):
        stab_ok = stability >= stability_thr

    if (max_index >= high_thr) and stab_ok:
        return ("RED", "Strong carpet-like indication in at least one sample.")
    if max_index >= med_thr:
        return ("YELLOW", "Some carpet-like indication detected; review bands and diagnostics.")
    return ("YELLOW", "Detected regions are weak; review diagnostics for confirmation.")



def render_carpet() -> None:
    st.subheader("Carpet Detection (FFT)")
    st.caption(
        "FFT-based carpet indication aligned with the challenge definition. "
        "Outputs are heuristic severity indicators (not calibrated probabilities)."
    )

    with st.sidebar:
        with st.expander("Parameters (Carpet Detection)", expanded = False):
            cfg = FFTCarpetConfig(
                fmin_hz=float(st.number_input("fmin_hz", value=1000.0, step=100.0)),
                k_peak=float(st.number_input("k_peak (MAD multiplier)", value=6.0, step=0.5)),
                min_distance_hz=float(st.number_input("min_distance_hz", value=5.0, step=1.0)),
                gap_max_hz=float(st.number_input("gap_max_hz", value=35.0, step=5.0)),
                min_peaks_cluster=int(st.number_input("min_peaks_cluster", value=8, step=1)),
                top_k_clusters=1,
                compute_stability=bool(st.checkbox("Compute stability (gap 25/35/50)", value=True)),
            )

        with st.expander("Indication thresholds", expanded = False):
            metric_name = st.selectbox(
                "Metric used for indication",
                ["severity_index_0_1"],
                index=0,
                help="Used to classify the batch/sample as LOW/MEDIUM/HIGH. "
                    "severity_index_0_1 is a robust 0–1 normalization computed within the uploaded batch (not a probability).",
            )

            high_thr = float(st.number_input(
                "HIGH threshold (RED)",
                value=0.70,
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                help="If max(severity_index_0_1) >= this value (and stability passes, if enabled), the indication becomes HIGH.",
            ))

            med_thr = float(st.number_input(
                "MEDIUM threshold (YELLOW)",
                value=0.40,
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                help="If max(severity_index_0_1) is between MEDIUM and HIGH thresholds, the indication becomes MEDIUM.",
            ))

            use_stability_gate = bool(st.checkbox(
                "Require stability for HIGH",
                value=True,
                help="If enabled, HIGH is only triggered when stability_0_1 >= the stability threshold.",
            ))

            stability_thr = float(st.number_input(
                "Stability threshold",
                value=0.60,
                min_value=0.0,
                max_value=1.0,
                step=0.05,
                help="Minimum stability_0_1 required to confirm HIGH indication when the stability gate is enabled.",
            ))


        with st.expander("Data Input (Carpet Detection)", expanded = True):
            uploads = st.file_uploader(
                "Upload one or more CSV files (or a ZIP containing CSV files)",
                type=["csv", "zip"],
                accept_multiple_files=True,
            )


    if not uploads:
        st.info("Upload CSV files in the sidebar to run Carpet Detection analysis.")
        return

    paths, basename_map = load_uploads_to_temp_paths(uploads)
    if not paths:
        st.warning("No CSV files were found in the uploaded input.")
        return


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

    light, message = _traffic_light(
    max_index=max_index,
    stability=stability_top,
    n_regions_total=batch.n_regions_total,
    med_thr=med_thr,
    high_thr=high_thr,
    use_stability_gate=use_stability_gate,
    stability_thr=stability_thr,
)


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
            st.metric("Severity score (raw)", f"{s.severity_score:.1f}")
            st.metric("Clusters detected", s.n_clusters)
            st.metric("fs (Hz)", f"{s.fs_hz:.0f}")
            if s.stability_0_1 is not None:
                st.metric("Stability (0–1)", f"{s.stability_0_1:.2f}")
        with m2:
            st.metric("Severity index (0–1)", f"{s.severity_index_0_1:.2f}")
            st.metric("Peaks detected", s.n_peaks)
            st.metric("RMS", f"{s.rms:.2f}")
            st.metric("Regions", len(s.regions))
    
    show_help = st.toggle("Show metric definitions", value=False)
    if show_help:
        st.markdown(
            """
        **Metric definitions**

        Severity_score (raw): Sum of the top-k cluster scores (higher = more carpet-like).
        Severity_index_0_1: Robust 0–1 normalization of severity_score within the uploaded batch (not a probability).
        Stability_0_1: Heuristic robustness indicator based on score variation under gap sensitivity (25/35/50).
        Clusters detected: Number of peak clusters passing minimum requirements.
            """.strip()
            )

    st.markdown("### Regions (by cluster score)")
    if s.regions:
        st.dataframe([r.model_dump() for r in s.regions], use_container_width=True)
    else:
        st.caption("No carpet-like regions detected for this sample.")

    st.subheader("Export")
    export_obj = {
            "indication": {
            "metric": metric_name,
            "medium_threshold": med_thr,
            "high_threshold": high_thr,
            "use_stability_gate": use_stability_gate,
            "stability_threshold": stability_thr,
        },
    }

    download_json_button(export_obj, filename="carpet_report.json")
