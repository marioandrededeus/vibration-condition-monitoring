# src/tractian_cm/app/ui_shared.py
from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import streamlit as st


def save_uploaded_to_temp(uploaded: st.runtime.uploaded_file_manager.UploadedFile) -> tuple[Path, str]:
    """
    EN: Persist UploadedFile to a temp file and return (temp_path, original_name).
    """
    with NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded.getbuffer())
        return Path(tmp.name), str(uploaded.name)


def traffic_badge(light: str) -> str:
    """
    EN: Render a colored severity badge as HTML.
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
        INDICATION: {label}
    </div>
    """


def download_json_button(obj: dict[str, Any], filename: str) -> None:
    """
    EN: Provide a JSON download button.
    """
    payload = json.dumps(obj, indent=2)
    st.download_button(
        "Download JSON report",
        data=payload,
        file_name=filename,
        mime="application/json",
    )
