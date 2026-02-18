import streamlit as st
from tractian_cm.app.pages.looseness import render_looseness

st.set_page_config(page_title="Looseness Classification", layout="wide")
render_looseness()
