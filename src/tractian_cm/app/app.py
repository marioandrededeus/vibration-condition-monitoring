from __future__ import annotations
import streamlit as st

def main() -> None:
    st.set_page_config(page_title="Condition Monitoring", layout="wide")
    st.title("Condition Monitoring – Interactive Diagnostics")
    st.info("Use Streamlit multipage navigation (left sidebar).")

if __name__ == "__main__":
    main()
