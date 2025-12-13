"""Streamlit app: chat with LLM, explore CSVs, generate insights and project plans."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from utils_analysis import generate_project_plan, simple_priority_score, summarize_tables
from utils_chat import add_message, init_chat_state, render_chat_history
from utils_data import build_table_preview_markdown, load_uploaded_csvs
from utils_llm import llm_chat_with_history


st.set_page_config(
    page_title="Impact AI Assistant",
    page_icon="🌍",
    layout="wide",
)

st.title("🌍 Impact AI Assistant – Hackathon Demo")

# --- SIDEBAR ---
st.sidebar.header("⚙️ Configuration")

model = st.sidebar.selectbox(
    "LLM model",
    ["gpt-4.1-mini", "gpt-4.1", "gpt-4.1-preview"],
    index=0,
)

st.sidebar.markdown("### 📂 Upload multiple CSV files")
uploaded_files = st.sidebar.file_uploader(
    "Choose one or more CSV files",
    type=["csv"],
    accept_multiple_files=True,
)

if uploaded_files:
    tables = load_uploaded_csvs(uploaded_files)
    st.session_state["tables"] = tables
    st.sidebar.success(f"{len(tables)} table(s) loaded.")
else:
    tables = st.session_state.get("tables", {})

if tables:
    st.sidebar.markdown("### Loaded tables")
    for name in tables.keys():
        st.sidebar.write(f"- `{name}`")

# --- MAIN TABS ---
tab_chat, tab_data, tab_ideas = st.tabs(["🤖 Chat with data", "📊 Data Explorer", "💡 Ideas & Project Plan"])

# ==============================
# TAB 1 — CHAT WITH DATA
# ==============================
with tab_chat:
    st.subheader("🤖 Chat with your contextualized LLM")
    init_chat_state()
    render_chat_history()

    data_context = build_table_preview_markdown(tables, max_rows=5)

    default_system_prompt = f"""
You are an AI assistant for a hackathon on the Sustainable Development Goals (SDGs).

Here are previews of the available tables:

{data_context}

Rules:
- Use the data whenever relevant.
- Refer to table and column names explicitly.
    """

    with st.expander("Advanced: view/modify system prompt"):
        system_prompt = st.text_area(
            "System prompt",
            value=default_system_prompt,
            height=200,
        )

    if "system_prompt_override" not in st.session_state:
        st.session_state.system_prompt_override = default_system_prompt

    if system_prompt != st.session_state.system_prompt_override:
        st.session_state.system_prompt_override = system_prompt

    user_msg = st.chat_input("Ask a question about your data…")
    if user_msg:
        add_message("user", user_msg)
        with st.chat_message("user"):
            st.markdown(user_msg)

        with st.chat_message("assistant"):
            with st.spinner("Thinking with your data…"):
                reply = llm_chat_with_history(
                    system_prompt=st.session_state.system_prompt_override,
                    messages=st.session_state.messages,
                    model=model,
                )
                st.markdown(reply)
        add_message("assistant", reply)

# ==============================
# TAB 2 — DATA EXPLORER
# ==============================
with tab_data:
    st.subheader("📊 Explore your tables")
    if not tables:
        st.info("Upload CSV files from the sidebar.")
    else:
        table_names = list(tables.keys())
        selected_table = st.selectbox("Select a table", table_names)
        df = tables[selected_table]

        st.markdown(f"### Preview of `{selected_table}`")
        st.dataframe(df.head(50))

        st.markdown("#### Statistics")
        st.write(df.describe(include="all"))

        with st.expander("Priority scoring tool (optional)"):
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

            if not numeric_cols:
                st.write("No numeric columns available.")
            else:
                selected_cols = st.multiselect("Select numeric columns to include", numeric_cols)

                if selected_cols:
                    weights = {}
                    for col in selected_cols:
                        # Simple manual weighting; ensures deterministic behavior.
                        w = st.number_input(
                            f"Weight for {col}",
                            min_value=0.0,
                            max_value=1.0,
                            value=0.3,
                        )
                        weights[col] = w

                    if st.button("Compute priority score"):
                        scored = simple_priority_score(df, weights)
                        st.success("Computed scoring:")
                        st.dataframe(scored.head(50))

# ==============================
# TAB 3 — IDEAS & PROJECT PLAN
# ==============================
with tab_ideas:
    st.subheader("💡 Summary & Impact Project Generator")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 1️⃣ Summarize uploaded data")
        if st.button("Generate summary"):
            if not tables:
                st.warning("No tables available.")
            else:
                with st.spinner("Analyzing…"):
                    summary = summarize_tables(tables)
                st.markdown(summary)

    with col2:
        st.markdown("### 2️⃣ Create SDG project plan")
        problem_description = st.text_area(
            "Describe the challenge or SDG problem you want to solve",
            height=150,
        )
        if st.button("Generate project plan"):
            if not problem_description.strip():
                st.warning("Please describe the problem first.")
            else:
                with st.spinner("Generating project…"):
                    plan = generate_project_plan(problem_description)
                st.markdown(plan)
