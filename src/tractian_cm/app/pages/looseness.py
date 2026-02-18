from __future__ import annotations

import io
import zipfile
from typing import Tuple, Dict, List

import pandas as pd
import streamlit as st

from tractian_cm.part2.model import Wave, LoosenessModel
from tractian_cm.app.ui_shared import traffic_badge


# -------------------------
# IO helpers
# -------------------------

def _read_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)


def _extract_zip(uploaded_zip) -> Dict[str, bytes]:
    raw = uploaded_zip.read()
    with zipfile.ZipFile(io.BytesIO(raw), "r") as zf:
        return {n: zf.read(n) for n in zf.namelist() if not n.endswith("/")}


def _pick_axis_files(names: list[str]) -> Tuple[str, str, str]:
    low = [n.lower() for n in names]

    def find_one(keys):
        for k in keys:
            for orig, ln in zip(names, low):
                if k in ln:
                    return orig
        return None

    f_h = find_one(["hor", "horizontal", "_h", "h.csv"])
    f_v = find_one(["ver", "vertical", "_v", "v.csv"])
    f_a = find_one(["axi", "axial", "_a", "a.csv"])

    if not (f_h and f_v and f_a):
        csvs = [n for n in names if n.lower().endswith(".csv")]
        if len(csvs) >= 3:
            return csvs[0], csvs[1], csvs[2]

    if not (f_h and f_v and f_a):
        raise ValueError(
            "Could not identify H/V/A files in ZIP. "
            "Rename files to include hor/ver/axi (or _h/_v/_a)."
        )

    return f_h, f_v, f_a


# -------------------------
# Parsing to Wave
# -------------------------

def _infer_time_column(df: pd.DataFrame) -> str:
    """
    Case files often use:
      - X-Axis (time)
      - time
      - t
    """
    cols = {c.lower(): c for c in df.columns}
    for key in ["x-axis", "time", "t"]:
        if key in cols:
            return cols[key]
    raise ValueError("Could not infer time column. Expected one of: X-Axis, time, t.")


def _make_wave(df: pd.DataFrame, time_col: str, sig_col: str) -> Wave:
    t = df[time_col].astype(float).to_list()
    x = df[sig_col].astype(float).to_list()
    return Wave(time=t, signal=x)


def _df_to_tri_waves(df: pd.DataFrame, col_h: str, col_v: str, col_a: str) -> tuple[Wave, Wave, Wave]:
    time_col = _infer_time_column(df)
    return (
        _make_wave(df, time_col, col_h),
        _make_wave(df, time_col, col_v),
        _make_wave(df, time_col, col_a),
    )


def _traffic_light_looseness(score: float, threshold: float, yellow_margin: float = 0.08) -> tuple[str, str]:
    """
    Heuristic traffic light (not a calibrated probability), aligned with Part 1 UI.
    """
    if score >= (threshold + yellow_margin):
        return ("RED", "Looseness detected with high confidence.")
    if score <= (threshold - yellow_margin):
        return ("GREEN", "No looseness detected (signal consistent with normal condition).")
    return ("YELLOW", "Borderline indication; review features and diagnostics.")


# -------------------------
# Page
# -------------------------

def render_looseness() -> None:
    st.title("Part 2 — Structural Looseness")

    with st.expander("This page runs...", expanded=False):
        st.markdown(
            """
This page runs the **physics-informed LoosenessModel** (Part 2), following the case template:
- `Wave(time, signal)`
- `LoosenessModel.predict(...) -> bool`
- `LoosenessModel.score(...) -> float` (0..1)

✅ The case dataset typically provides **3 axes in a single CSV** (time + 3 channels).  
This page supports that format.
            """.strip()
        )

    with st.expander("Input format", expanded=False):
        st.markdown(
            """
**Supported upload modes**
1. **Single CSV (3 axes)** — recommended for the official dataset  
2. 3 CSVs (H/V/A)  
3. ZIP containing 3 CSVs  

**Time column**
- Usually `X-Axis`, but `time` or `t` are also accepted.
            """.strip()
        )

    c1, c2 = st.columns(2)
    with c1:
        rpm = st.number_input("RPM", min_value=1.0, value=1598.0, step=1.0)
    with c2:
        threshold = st.slider("Decision threshold", 0.0, 1.0, 0.50, 0.01)

    mode = st.radio("Upload mode", ["Single CSV (3 axes)", "3 CSVs", "ZIP"], horizontal=True)

    wave_h = wave_v = wave_a = None
    explanation = None  # ALWAYS defined to avoid UnboundLocalError

    try:
        if mode == "Single CSV (3 axes)":
            f = st.file_uploader("CSV with time + 3 axis signals", type=["csv"], key="p2_single")

            if f:
                df = _read_csv(f)

                time_col = _infer_time_column(df)
                sig_cols: List[str] = [c for c in df.columns if c != time_col]

                if len(sig_cols) < 3:
                    raise ValueError(
                        f"Expected at least 3 signal columns besides '{time_col}'. "
                        f"Found: {sig_cols}"
                    )

                st.caption(f"Detected time column: **{time_col}**")

                default_h = sig_cols[0]
                default_v = sig_cols[1] if len(sig_cols) > 1 else sig_cols[0]
                default_a = sig_cols[2] if len(sig_cols) > 2 else sig_cols[0]

                cc1, cc2, cc3 = st.columns(3)
                col_h = cc1.selectbox("Horizontal axis column (H)", sig_cols, index=sig_cols.index(default_h))
                col_v = cc2.selectbox("Vertical axis column (V)", sig_cols, index=sig_cols.index(default_v))
                col_a = cc3.selectbox("Axial axis column (A)", sig_cols, index=sig_cols.index(default_a))

                if len({col_h, col_v, col_a}) < 3:
                    st.warning("H, V and A are using repeated columns. Prefer selecting 3 different channels.")

                wave_h, wave_v, wave_a = _df_to_tri_waves(df, col_h, col_v, col_a)

        elif mode == "3 CSVs":
            f_h = st.file_uploader("Horizontal CSV", type=["csv"], key="p2_h")
            f_v = st.file_uploader("Vertical CSV", type=["csv"], key="p2_v")
            f_a = st.file_uploader("Axial CSV", type=["csv"], key="p2_a")

            def parse_single_axis(uploaded):
                df = _read_csv(uploaded)
                time_col = _infer_time_column(df)
                sig_cols = [c for c in df.columns if c != time_col]
                if len(sig_cols) != 1:
                    raise ValueError(f"Axis CSV must have exactly 1 signal column besides '{time_col}'. Got: {sig_cols}")
                return _make_wave(df, time_col, sig_cols[0])

            if f_h and f_v and f_a:
                wave_h = parse_single_axis(f_h)
                wave_v = parse_single_axis(f_v)
                wave_a = parse_single_axis(f_a)

        else:  # ZIP
            f_zip = st.file_uploader("ZIP file", type=["zip"], key="p2_zip")

            if f_zip:
                extracted = _extract_zip(f_zip)
                names = list(extracted.keys())
                n_h, n_v, n_a = _pick_axis_files(names)

                def parse_zip_member(name):
                    df = pd.read_csv(io.BytesIO(extracted[name]))
                    time_col = _infer_time_column(df)
                    sig_cols = [c for c in df.columns if c != time_col]
                    if len(sig_cols) != 1:
                        raise ValueError(f"{name}: expected 1 signal column besides '{time_col}'. Got: {sig_cols}")
                    return _make_wave(df, time_col, sig_cols[0])

                wave_h = parse_zip_member(n_h)
                wave_v = parse_zip_member(n_v)
                wave_a = parse_zip_member(n_a)

        disabled = wave_h is None or wave_v is None or wave_a is None

        # -------------------------
        # Physical features preview
        # -------------------------
        if not disabled:
            try:
                model_preview = LoosenessModel(rpm=rpm, threshold=threshold)
                explanation = model_preview.explain(wave_h, wave_v, wave_a)

                st.markdown("### Extracted Physical Features (Preview)")
                f1, f2, f3 = st.columns(3)
                f1.metric("2x / 1x", f"{explanation['agg__amp_2x_over_1x_max']:.3f}")
                f2.metric("3x / 1x", f"{explanation['agg__amp_3x_over_1x_max']:.3f}")
                f3.metric("Crest factor", f"{explanation['agg__crest_factor_max']:.3f}")

            except Exception as e:
                st.warning(f"Could not compute preview features: {e}")
                explanation = None

        st.markdown("---")

        # Button AFTER the physical features
        run = st.button("Run inference", type="primary", disabled=disabled)

        if run and not disabled:
            with st.spinner("Running looseness model..."):
                model = LoosenessModel(rpm=rpm, threshold=threshold)
                score = model.score(wave_h, wave_v, wave_a)
                pred = model.predict(wave_h, wave_v, wave_a)

                if explanation is None:
                    explanation = model.explain(wave_h, wave_v, wave_a)

            st.markdown("### Inference Result")

            light, message = _traffic_light_looseness(score=score, threshold=threshold)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(traffic_badge(light), unsafe_allow_html=True)
            with c2:
                st.metric("RPM", f"{rpm:.0f}")
            with c3:
                st.metric("Score (0–1)", f"{score:.3f}")
            with c4:
                st.metric("Prediction", "Looseness" if pred else "Normal")

            st.write(message)

            st.markdown("### Structured Output")

            output_row = {
                "sample_id": "uploaded_sample",
                "rpm": float(rpm),
                "prediction_looseness": bool(pred),
                "score": float(score),
                **(explanation or {}),
            }

            output_df = pd.DataFrame([output_row])
            st.dataframe(output_df, use_container_width=True)

    except Exception as e:
        st.error(f"Input error: {e}")
