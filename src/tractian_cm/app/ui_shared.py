# src/tractian_cm/app/ui_shared.py
from __future__ import annotations

import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

import streamlit as st
import io
import zipfile



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

def _is_safe_member(name: str) -> bool:
    """
    EN: Basic Zip Slip protection. Reject absolute paths and parent traversal.
    """
    p = Path(name)
    if p.is_absolute():
        return False
    return ".." not in p.parts


def load_uploads_to_temp_paths(
    uploads: list[st.runtime.uploaded_file_manager.UploadedFile],
) -> tuple[list[Path], dict[str, str]]:
    """
    EN: Accept .csv and .zip uploads and return:
      - paths: list of temp csv paths
      - basename_map: temp_basename -> original_name (for UI/export)
    """
    paths: list[Path] = []
    basename_map: dict[str, str] = {}

    for up in uploads:
        name = str(up.name).lower()

        if name.endswith(".csv"):
            tmp_path, orig = save_uploaded_to_temp(up)
            paths.append(tmp_path)
            basename_map[tmp_path.name] = orig
            continue

        if name.endswith(".zip"):
            data = up.getbuffer()
            zf = zipfile.ZipFile(io.BytesIO(data))

            # EN: extract only CSV members safely
            for member in zf.infolist():
                if member.is_dir():
                    continue
                if not member.filename.lower().endswith(".csv"):
                    continue
                if not _is_safe_member(member.filename):
                    continue

                # EN: keep only file basename for display; preserve internal folder name if needed
                orig_name = Path(member.filename).name

                # EN: write to temp
                with NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                    tmp.write(zf.read(member))
                    tmp_path = Path(tmp.name)

                paths.append(tmp_path)
                basename_map[tmp_path.name] = orig_name

            continue

        # EN: ignore unsupported types silently (uploader already filters, but keep defensive)
        continue

    return paths, basename_map
