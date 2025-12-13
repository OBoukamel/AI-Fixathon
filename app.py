"""Streamlit app: chat with LLM, explore CSVs/ZIPs, generate insights and project plans."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from utils_analysis import generate_project_plan, simple_priority_score, summarize_tables
from utils_chat import add_message, init_chat_state, render_chat_history
from utils_data import build_table_preview_markdown, load_uploaded_csvs
from utils_llm import llm_chat_with_history
from utils_zip import extract_zip_to_dir, render_tree, zip_to_tree


st.set_page_config(page_title="Clinical Research AI Assistant", page_icon=":earth_africa:", layout="wide")
st.title("Impact AI Assistant – Hackathon Demo")

# --- SIDEBAR ---
st.sidebar.header("Configuration")

# --- MAIN TABS ---
tab_zip, tab_ideas = st.tabs(
    ["Survey Explorer and data summary", "Exploring data based on research questions"]
)

  

# ==============================
# TAB 3 — ZIP EXPLORER
# ==============================
with tab_zip:
    st.subheader("ZIP Explorer — upload and browse archives")
    uploaded_zips = st.file_uploader(
        "Upload one or more ZIP files",
        type=["zip"],
        accept_multiple_files=True,
        key="zip_uploader",
    )

    zip_root = Path("survey")
    zip_root.mkdir(parents=True, exist_ok=True)
    st.caption(f"ZIP storage folder: {zip_root.resolve()}")

    if not uploaded_zips:
        st.info("Upload ZIP files to see their internal file tree and extraction folder.")
    else:
        for up in uploaded_zips:
            zip_name = Path(up.name).stem
            archive_path = zip_root / up.name
            archive_path.write_bytes(up.read())

            with zipfile.ZipFile(archive_path, mode="r") as zf:
                tree = zip_to_tree(zf)

                st.subheader(f"Archive: {up.name}")
                render_tree(tree)

                out_dir = zip_root / zip_name
                out_dir.mkdir(parents=True, exist_ok=True)
                extract_zip_to_dir(zf, out_dir)
                st.success(f"Extracted to: {out_dir}")

        st.divider()
        st.write(
            "Tip: you can add file previews or download buttons for extracted files from the storage folder."
        )

        st.markdown("### Summarize uploaded data")
        if st.button("Generate summary"):
            with st.spinner("Analyzing…"):
                from utils_analysis import analyze_survey_folder
                summary = analyze_survey_folder(folder_path = "survey/", max_rows = 500, max_chars_meta = 4000)
            st.markdown(summary)
        

# ==============================
# TAB 4 — IDEAS & PROJECT PLAN
# ==============================
with tab_ideas:
    st.markdown("Let's make research!")
    from utils_analysis import collect_survey_data_files_ai
    with st.spinner("Identifying survey data…"):
        data_list = collect_survey_data_files_ai(folder_path = 'survey', output_dir= "data", model = "green-l")
        list_of_data, string = collect_survey_data_files_ai(folder_path = 'survey', output_dir= "data", model = "green-l")
        st.write(list_of_data.keys())
        st.markdown(string)