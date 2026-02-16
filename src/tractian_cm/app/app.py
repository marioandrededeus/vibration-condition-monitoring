# src/tractian_cm/app/app.py
from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import streamlit as st

from tractian_cm.pipelines.fft_carpet import FFTCarpetConfig, batch_analyze
from tractian_cm.viz.plotly_fft import fft_clusters_figure


def _save_uploaded_to_temp(uploaded: st.runtime.uploaded_file_manager.UploadedFile) -> tuple[Path, str]:
    """
    EN: Persist Streamlit UploadedFile to a temporary file.
    Returns (temp_path, original_file_name).
    """
    with NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded.getbuffer())
        return Path(tmp.name), str(uploaded.name)


def _traffic_light(max_index: float, stability: float | None, n_regions_total: int) -> tuple[str, str]:
    """
    EN: Simple heuristic traffic light for carpet indication (not a probability).
    """
    if n_regions_total == 0:
        return ("GREEN", "No carpet-like regions detected in the uploaded sample(s).")

    stab_ok = True if stability is None else (stability >= 0.60)

    if (max_index >= 0.70) and stab_ok:
        return ("RED", "Strong carpet-like indication in at least one sample.")
    if max_index >= 0.40:
        return ("YELLOW", "Some carpet-like indication detected; review bands and diagnostics.")
    return ("YELLOW", "Detected regions are weak; review diagnostics for confirmation.")


def _traffic_badge(light: str) -> str:
    """
    EN: Render a colored severity badge.
    """
    color = {"GREEN": "#2E7D32", "YELLOW": "#F9A825", "RED": "#C62828"}.get(light, "#616161")
    label = {"GREEN": "LOW", "YELLOW": "MEDIUM", "RED": "HIGH"}.get(light, "N/A")
    return f"""
    <div style="
        background-color:{color};
        color:white;
        padding:16px;
        border-radius:12px;
        text-align:center;
        font-weight:700;
        font-size:18px;">
        CARPET INDICATION: {label}
    </div>
    """


def main() -> None:
    st.set_page_config(page_title="Carpet Indicator (FFT)", layout="wide")
    st.title("Condition Monitoring – Carpet Indicator (FFT)")
    st.caption(
        "FFT-based carpet indication aligned with the challenge definition. "
        "Outputs are heuristic severity indicators (not calibrated probabilities)."
    )

    # --- Sidebar controls
    with st.sidebar:

        st.header("Data Input")
        uploads = st.file_uploader(
            "Upload one or more CSV files",
            type=["csv"],
            accept_multiple_files=True,
        )

        st.divider()

        st.header("Parameters (Advanced)")
        st.markdown("Defaults match the final notebook configuration.")

        cfg = FFTCarpetConfig(
            fmin_hz=float(st.number_input("fmin_hz", value=1000.0, step=100.0)),
            k_peak=float(st.number_input("k_peak (MAD multiplier)", value=6.0, step=0.5)),
            min_distance_hz=float(st.number_input("min_distance_hz", value=5.0, step=1.0)),
            gap_max_hz=float(st.number_input("gap_max_hz", value=35.0, step=5.0)),
            min_peaks_cluster=int(st.number_input("min_peaks_cluster", value=8, step=1)),
            top_k_clusters=1,  # EN: worst symptom / strongest evidence
            compute_stability=bool(st.checkbox("Compute stability (gap 25/35/50)", value=True)),
        )

    if not uploads:
        st.info("Upload CSV files in the sidebar to run the FFT-based carpet indicator.")
        return

    # --- Persist uploads to temp files while preserving original names
    temp_and_names = [_save_uploaded_to_temp(u) for u in uploads]
    paths = [p for (p, _) in temp_and_names]
    # Map: temp basename -> original name
    basename_map = {Path(p).name: orig for (p, orig) in temp_and_names}

    batch, samples_sorted, artifacts = batch_analyze(paths, cfg)

    # --- Replace temp basenames with original upload names (UI + export)
    # EN: Pydantic models are immutable by default; use model_copy(update=...) to rename.
    renamed_samples = []
    for s in samples_sorted:
        new_name = basename_map.get(s.file_name, s.file_name)
        renamed_samples.append(s.model_copy(update={"file_name": new_name}))

    # EN: artifacts keys were produced from path.name; remap keys too
    remapped_artifacts = {}
    for k, v in artifacts.items():
        remapped_artifacts[basename_map.get(k, k)] = v
    artifacts = remapped_artifacts

    samples_sorted = sorted(renamed_samples, key=lambda x: x.severity_score, reverse=True)
    batch = batch.model_copy(update={"top3": samples_sorted[:3]})

    # --- Summary stats for UI
    max_index = max(s.severity_index_0_1 for s in samples_sorted) if samples_sorted else 0.0
    stability_top = samples_sorted[0].stability_0_1 if samples_sorted else None
    light, message = _traffic_light(max_index=max_index, stability=stability_top, n_regions_total=batch.n_regions_total)

    # --- Home summary (badge + metrics)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(_traffic_badge(light), unsafe_allow_html=True)
    with c2:
        st.metric("Samples Loaded", batch.n_samples)
    with c3:
        st.metric("Carpet-like Regions (Total)", batch.n_regions_total)
    with c4:
        st.metric("Max Severity Index (0–1)", f"{max_index:.2f}")

    st.write(message)

    # --- Top 3 table
    st.subheader("Top 3 Samples by Carpet Severity")
    rows = []
    for s in batch.top3:
        rows.append(
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
        )
    st.dataframe(rows, use_container_width=True)

    # --- Per-sample diagnostics
    st.subheader("Diagnostics")
    sample_names = [s.file_name for s in samples_sorted]
    selected = st.selectbox("Select a sample", sample_names, index=0)

    s = next(x for x in samples_sorted if x.file_name == selected)

    # EN: artifacts were remapped to original names; access by selected
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

    # --- Regions table below the plot (full width)
    st.markdown("### Regions (by cluster score)")
    if s.regions:
        st.dataframe([r.model_dump() for r in s.regions], use_container_width=True)
    else:
        st.caption("No carpet-like regions detected for this sample.")

    # --- Export JSON
    st.subheader("Export")
    export_obj = {
        "batch": batch.model_dump(),
        "samples": [x.model_dump() for x in samples_sorted],
        "config": cfg.__dict__,
        "notes": {
            "interpretation": "Heuristic carpet indication based on FFT peak clustering; not a calibrated probability.",
        },
    }
    export_json = json.dumps(export_obj, indent=2)

    st.download_button(
        "Download JSON report",
        data=export_json,
        file_name="carpet_indicator_report.json",
        mime="application/json",
    )


if __name__ == "__main__":
    main()
