import streamlit as st

st.set_page_config(page_title="Condition Monitoring", layout="wide")

st.title("Condition Monitoring – Interactive Diagnostics")
st.markdown(
    """
This Streamlit app provides two modules:

- **Carpet Detection (Part 1)** — FFT-based broadband carpet indication  
- **Looseness Classification (Part 2)** — structural looseness diagnosis
"""
)
st.info("Use the left sidebar to navigate between pages.")
