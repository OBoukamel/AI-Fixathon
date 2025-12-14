
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from utils_llm import llm_chat


def understand_survey_columns(
    folder_path,
    survey_columns,
    max_rows  = 500,
    max_chars_meta = 4000,
) -> str:
    """
    Inspect a folder, separate metadata files from survey data files, and provide definitions of the survey variables.

    Heuristics:
    - Metadata files: .md, .txt, .json, .yaml, .yml (read as text, truncated)
    - Data files: .csv (loaded as DataFrames with row cap)
    """
    root = Path(folder_path)
    if not root.exists() or not root.is_dir():
        return f"Folder not found: {root}"

    meta_suffixes = {".md", ".txt", ".json", ".yaml", ".yml"}
    data_suffixes = {".csv"}

    meta_blobs: List[str] = []
    data_tables: Dict[str, pd.DataFrame] = {}

    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in meta_suffixes:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
                meta_blobs.append(f"FILE: {path.name}\n{text}")
            except Exception as exc:  # noqa: BLE001
                meta_blobs.append(f"FILE: {path.name}\n<unable to read: {exc}>")
        elif suffix in data_suffixes:
            try:
                df = pd.read_csv(path, nrows=max_rows)
                data_tables[path.name] = df
            except Exception as exc:  # noqa: BLE001
                meta_blobs.append(f"FILE: {path.name}\n<unable to load CSV: {exc}>")

    if not data_tables:
        return "No survey data (.csv) found in the folder."

    meta_context = "\n\n".join(meta_blobs)[:max_chars_meta] if meta_blobs else "No metadata files found."

    snippets = []
    for name, df in data_tables.items():
        csv_snippet = df.head(10).to_csv(index=False)
        snippets.append(f"DATA FILE: {name}\n{csv_snippet}")
    data_context = "\n\n".join(snippets)[:8000]

    prompt = f"""
You are identifying the column names definitions in the data from the folder from the meta data files.

Metadata excerpts:
{meta_context}

Survey data samples (CSV):
{data_context}

Tasks:
- Identify the column provided in arguments as {survey_columns} and in which dataset they are found.
- Provide a concise definition for each column based on the metadata and data samples.
- Return the result as a JSON object where keys are column names and values are their definitions.
Return concise bullet points.
If you can't find a definition for a column, respond with "Definition not found". 
If you can't find the {survey_columns} just say it.
"""

    return llm_chat(
        system_prompt="You are a data analyst skilled at using metadata to contextualize survey data.",
        user_message=prompt,
        model="green-l",
    )

