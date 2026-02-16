# src/tractian_cm/app/app.py
from __future__ import annotations

import streamlit as st

from tractian_cm.app.pages.carpet import render_carpet
from tractian_cm.app.pages.looseness import render_looseness


def main() -> None:
    st.set_page_config(page_title="Condition Monitoring", layout="wide")
    st.title("Condition Monitoring – Interactive Diagnostics")

    with st.sidebar:
        st.header("Mode")
        mode = st.radio(
            "Select analysis",
            ["Carpet detection (FFT)", "Looseness classification"],
            index=0,
        )

    if mode.startswith("Carpet"):
        render_carpet()
    else:
        render_looseness()


if __name__ == "__main__":
    main()
