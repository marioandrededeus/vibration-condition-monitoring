# src/tractian_cm/app/pages/looseness.py
from __future__ import annotations

import streamlit as st


def render_looseness() -> None:
    st.subheader("Looseness classification")
    st.info(
        "Looseness module not implemented yet. "
        "This page will reuse the same upload/validation pattern and provide a looseness classifier output."
    )
