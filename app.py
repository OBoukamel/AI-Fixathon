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


st.set_page_config(page_title="Clinical Research AI Assistant", page_icon="🩺", layout="wide")
st.title("🩺 AI Survey Data Assistant – Fixckathon Demo")

# Cache heavy data operations to avoid re-running on every interaction
@st.cache_data(show_spinner=False)
def load_survey_tables_cached(folder: str = "survey", output_dir: str = "data"):
    result = load_survey_tables(folder=folder, output_dir=output_dir)
    # Backward-compatibility: handle both (data, msg) and (data, msg, db_path)
    if isinstance(result, tuple):
        if len(result) == 3:
            return result
        if len(result) == 2:
            data, msg = result
            return data, msg, None
    return result


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
st.sidebar.header("⚙️ Configuration")
uploaded_zips = st.sidebar.file_uploader(
    "Upload one or more ZIP files",
    type=["zip"],
    accept_multiple_files=True,
    key="zip_uploader",
)

# --- MAIN TABS ---
tab_zip, tab_ideas = st.tabs(
    ["📦 Survey Explorer & Summary", "💡 Research Questions"]
)


# ==============================
# TAB 3 — ZIP EXPLORER
# ==============================
with tab_zip:
    st.subheader("📦 ZIP Explorer — upload and browse archives")
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
                from meta_data_extract import rewrite_file_in_place
                with st.spinner("🔒 Anonymising extracted files..."):
                                    for file_path in out_dir.rglob("*"):
                                        if file_path.is_file():
                                            rewrite_file_in_place(file_path)
                                            st.success(f"🔒 Anonymisation complete for {file_path}.")
        st.divider()
        st.write(
            "Tip: you can add file previews or download buttons for extracted files from the storage folder."
        )

        st.markdown("### 📑 Summarize uploaded data")
        if st.button("Generate summary"):
            with st.spinner("Analyzing…"):
                from utils_analysis import analyze_survey_folder
                summary = analyze_survey_folder(folder_path = "survey", max_rows = 500, max_chars_meta = 4000)
            st.markdown(summary)
        

# ==============================
# TAB 4 — IDEAS & PROJECT PLAN
# ==============================
with tab_ideas:
    st.markdown("#### 🧠🔬 Let's make research!")
    st.write("### 🤖 Using AI to find survey data and gather this in a list of dataframes ready for analysis")
    tab1, tab2 = st.tabs(["📊 Data understanding", "🧭 Research question analysis"])
    with tab1:
        with st.spinner("Identifying survey data..."):
            list_of_data, status_msg, db_path = load_survey_tables_cached(folder="survey", output_dir="data")
            st.write(list_of_data.keys())
            st.markdown(status_msg)
            st.caption(f"SQLite database path: {db_path}")

        if not list_of_data:
            st.warning("No data tables found in the survey folder.")
        else:
            with st.expander("📋 Detailed data table previews"):
                st.write("### 🧾 Generate data definition")
                with st.spinner("Generating data definition..."):
                    cols = {
                        name: info.get("columns", [])
                        for name, info in list_of_data.items()
                        if isinstance(info, dict)
                        }
                    st.write("Print of all column names in the data files:")
                    #st.write(cols)
                    schema_json = load_schema("survey", cols)
                    st.write(schema_json)
            with st.expander("🗂️ Folder meta-data"):
                st.write("### ℹ️ Generating the folder meta-data")
                with st.spinner("Preparing the detailed meta-data"):
                    meta_data = load_metadata("survey")
                    st.write(meta_data)
    with tab2:
        st.markdown("### 🧭 Generate analysis plan based on research question")
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

        st.markdown("### 🧹 Customize data cleaning process")
        describe_cleaning_process = st.text_area("Also describe data cleaning and preprocessing steps", value="")

        # Initialize session state keys for code editing and query results
        defaults = {
            "generated_code": "",
            "editable_code": "",
            "edit_mode": False,
            "query_result_df": None,
            "query_result_excel_bytes": None,
            "last_sql": "",
        }
        for key, default in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = default

        # Generate SQL for data cleaning
        import pandas as pd
        import sqlite3

        sqlite_path = "./data/survey.db"

        # ===============================
        # 1) CONNECT DB (SAFE)
        # ===============================
        try:
            conn_sqlite = sqlite3.connect(sqlite_path)
            st.success(f"✅ Connected to database: {sqlite_path}")
        except Exception as e:
            st.error(f"❌ Cannot connect to DB: {e}")
            st.stop()

        # ===============================
        # 2) LIST TABLES (READ-ONLY)
        # ===============================
        tables_df = pd.read_sql_query(
            "SELECT name AS table_name FROM sqlite_master WHERE type='table' ORDER BY name;",
            conn_sqlite,
        )

        if tables_df.empty:
            st.warning("⚠️ No tables found in the database.")
        else:
            st.markdown("### 📂 Tables available in the database")
            st.dataframe(tables_df, use_container_width=True)

        # ===============================
        # 3) BUTTON = GENERATE CODE (WRITE STATE)
        # ===============================
        import pandas as pd
        import sqlite3
        from io import BytesIO
        import time

        # ---------- session state ----------
        if "generated_code" not in st.session_state:
            st.session_state.generated_code = ""
        if "editable_code" not in st.session_state:
            st.session_state.editable_code = ""
        if "edit_mode" not in st.session_state:
            st.session_state.edit_mode = False
        if "query_result_df" not in st.session_state:
            st.session_state.query_result_df = None
        if "query_result_excel_bytes" not in st.session_state:
            st.session_state.query_result_excel_bytes = None

        # ---------- DB connection ----------
        conn_sqlite = sqlite3.connect("./data/survey.db")

        # ---------- generate SQL ----------
        if st.button("Generate SQL code for data cleaning"):
            st.session_state.generated_code = """SELECT facility_id,
    difficulty_breathing,
    sex,
    AVG(age_years) AS mean_age
FROM zipe_ari_meta5_data_submissions
GROUP BY facility_id, sex;
"""
            st.session_state.editable_code = st.session_state.generated_code

        # ---------- display + edit ----------
        if st.session_state.generated_code:
            st.session_state.edit_mode = st.toggle("Edit code ?", value=st.session_state.edit_mode)

            if st.session_state.edit_mode:
                st.session_state.editable_code = st.text_area(
                    "Edit SQL code:",
                    value=st.session_state.editable_code,
                    height=300,
                )
                final_sql_code = st.session_state.editable_code
            else:
                st.code(st.session_state.generated_code, language="sql")
                final_sql_code = st.session_state.generated_code

            # ---------- run SQL ----------
            if st.button("Run code"):
                try:
                    sql = (final_sql_code or "").strip().rstrip(";")
                    if not sql:
                        st.warning("Please paste a SQL query first.")
                    else:
                        if "limit" not in sql.lower():
                            sql += " LIMIT 100"

                        df = pd.read_sql_query(sql, conn_sqlite)
                        st.session_state.query_result_df = df

                        buffer = BytesIO()
                        with pd.ExcelWriter(buffer) as writer:
                            df.to_excel(writer, index=False, sheet_name="QueryResult")
                        st.session_state.query_result_excel_bytes = buffer.getvalue()

                        st.success("Query executed successfully")

                except Exception as e:
                    st.error(f"⚠️ Error executing SQL: {e}")

        # ---------- always show result + download ----------
        if st.session_state.query_result_df is not None:
            st.markdown("### 📊 Query result")
            st.dataframe(st.session_state.query_result_df, use_container_width=True)

            if st.session_state.query_result_excel_bytes:
                st.download_button(
                    "Download query result as Excel",
                    data=st.session_state.query_result_excel_bytes,
                    file_name="query_result.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
            else:
                st.info("No exportable bytes found yet. Run a query first.")

            if st.button("Clear results"):
                st.session_state.generated_code = ""
                st.session_state.editable_code = ""
                st.session_state.query_result_df = None
                st.session_state.query_result_excel_bytes = None
                st.rerun()

        conn_sqlite.close()
