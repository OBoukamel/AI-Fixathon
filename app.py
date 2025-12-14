"""Streamlit app: chat with LLM, explore CSVs/ZIPs, generate insights and project plans."""
from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import streamlit as st

from utils_analysis import (
    generate_project_plan,
    load_survey_tables,
    simple_priority_score,
    summarize_tables,
)
from utils_chat import add_message, init_chat_state, render_chat_history
from utils_data import build_table_preview_markdown, load_uploaded_csvs
from utils_llm import llm_chat_with_history
from utils_understanding_data import understand_survey_columns
from utils_zip import extract_zip_to_dir, render_tree, zip_to_tree
from meta_data_extract import getting_meta_data_all


st.set_page_config(page_title="Clinical Research AI Assistant", page_icon=":earth_africa:", layout="wide")
st.title("Impact AI Assistant – Hackathon Demo")

# Cache heavy data operations to avoid re-running on every interaction
@st.cache_data(show_spinner=False)
def load_survey_tables_cached(folder: str = "survey", output_dir: str = "data"):
    return load_survey_tables(folder=folder, output_dir=output_dir)


@st.cache_data(show_spinner=False)
def load_schema(folder: str, cols: dict):
    try:
        return understand_survey_columns(folder, cols, max_rows=500, max_chars_meta=4000)
    except Exception as exc:  # noqa: BLE001
        return f"Schema generation failed: {exc}"


@st.cache_data(show_spinner=False)
def load_metadata(folder: str):
    return getting_meta_data_all(path=folder)

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
                summary = analyze_survey_folder(folder_path = "survey", max_rows = 500, max_chars_meta = 4000)
            st.markdown(summary)
        

# ==============================
# TAB 4 — IDEAS & PROJECT PLAN
# ==============================
with tab_ideas:
    st.markdown("#### Let's make research!")
    st.write("### Using AI to find survey data and gather this in a list of dataframes ready for analysis")
    with st.spinner("Identifying survey data..."):
        list_of_data, status_msg = load_survey_tables_cached(folder="survey", output_dir="data")
        st.write(list_of_data.keys())
        st.markdown(status_msg)

    if not list_of_data:
        st.warning("No data tables found in the survey folder.")
    else:
        st.write("### Generate data definition")
        with st.spinner("Generating data definition..."):
            cols = {
                name: info.get("columns", [])
                for name, info in list_of_data.items()
                if isinstance(info, dict)
            }
            st.write("Print of all column names in the data files:")
            st.write(cols)
            schema_json = load_schema("survey", cols)
            st.write(schema_json)
        
        st.write("### Generating the folder meta-data")
        with st.spinner("Preparing the detailed meta-data"):
            meta_data = load_metadata("survey")
            st.write(meta_data)

        st.markdown("### Generate analysis plan based on research question")
        research_question = st.text_area("Describe your research idea or question:", key="research_idea", height=100)
        if st.button("Generate pre-processing"):

            system_prompt  = f"""You are a data analyst research assistant.You are analyzing a survey dataset contained in a folder 
        to anwer the research question: {research_question}
        The data to use is provided in {list_of_data}
        use the column name and schema definitions provided in {schema_json} to understand the meaning of the data
        use the meta-data provided in {meta_data} to understand the context of the data and include it in your analysis
        provide elements to the research question {research_question}
Tasks: 

- suggest a data cleaning and preprocessing plan to prepare the data for analysis.
- provide elements to the research question based on the data provided.
If you need to make assumptions about the data, make them explicit in your answer.
If the cleaning process provided by the user is not related to the data provided, politely inform the user that you will proceed with a generic cleaning process.
""" 

            plan_response = llm_chat_with_history(
                system_prompt=system_prompt,
                messages=[{"role": "user", "content": "Provide the analysis plan now."}],
                model="green-l",
            )
            
            st.markdown(plan_response)

        st.markdown("### Customize data cleaning process")
        describe_cleaning_process = st.text_area("Also describe data cleaning and preprocessing steps", value="")

        # Initialize session state keys for code editing
        if "generated_code" not in st.session_state:
            st.session_state.generated_code = ""
        if "editable_code" not in st.session_state:
            st.session_state.editable_code = ""

        # Generate Python code
        if st.button("Generate Python code for data cleaning"):
            st.session_state.generated_code = f"""import pandas as pd

print("Loading data...")
print({list(list_of_data.keys())})
"""
            st.session_state.editable_code = st.session_state.generated_code

        # Show and optionally edit the code
        if st.session_state.generated_code:
            edit_mode = st.toggle("Edit code ?", key="edit_code_toggle")

            if edit_mode:
                st.session_state.editable_code = st.text_area(
                    "You can edit the generated Python code below:",
                    value=st.session_state.editable_code,
                    height=400,
                    key="editable_code_area",
                )
                st.code(st.session_state.editable_code, language="python")
            else:
                st.code(st.session_state.generated_code, language="python")

            final_python_code = (
                st.session_state.editable_code if edit_mode else st.session_state.generated_code
            )
        else:
            st.info("Click the button to generate code first.")

        st.markdown("### Execute data cleaning code")
        
