import streamlit as st
from tractian_cm.app.pages.carpet import render_carpet

st.set_page_config(page_title="Carpet Detection", layout="wide")
render_carpet()
