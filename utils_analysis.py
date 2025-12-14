"""Analysis helpers: LLM summarization, project planning, scoring, and survey handling."""
from __future__ import annotations

import json
import re
import shutil
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import pandas as pd

from utils_llm import llm_chat


def summarize_tables(tables: Dict[str, pd.DataFrame]) -> str:
    """
    Summarize available tables using the LLM.

    Each table is truncated to a small CSV snippet to avoid blowing context.
    """
    if not tables:
        return "No tables available."

    snippets = []
    for name, df in tables.items():
        sample = df.head(10)
        csv_snippet = sample.to_csv(index=False)
        snippets.append(f"TABLE: {name}\n{csv_snippet}")

    joined = "\n\n".join(snippets)[:8000]

    prompt = f"""
Below are excerpts from several tables (CSV format):

{joined}

Please provide:
1. A short description of the data.
2. 3–5 potential insights or interesting SDG-related questions.
3. 3 simple analysis or visualization ideas.

Respond in concise bullet points.
"""

    return llm_chat(
        system_prompt="You are a researcher in medicine and epidemiology and you need to analyse your data for your research.",
        user_message=prompt,
        model="green-l",
    )


def generate_project_plan(problem_description: str) -> str:
    """
    Generate a mini SDG project plan from a short problem description.
    """
    prompt = f"""
We are in a hackathon setting focused on the Sustainable Development Goals.

Problem to solve:
{problem_description}

Please generate a concrete project plan including:
- A clear objective
- 3–5 core activities
- Data to collect or leverage
- 3 measurable KPIs
- A realistic use of AI/LLMs

Respond with clear headings and bullet points.
"""

    return llm_chat(
        system_prompt="You design practical development and impact-oriented data projects.",
        user_message=prompt,
        model="green-l",
    )


def simple_priority_score(df: pd.DataFrame, cols_weights: Dict[str, float]) -> pd.DataFrame:
    """
    Create a priority score = weighted sum of normalized numeric columns.
    """
    if df.empty:
        return df

    result = df.copy()
    score = np.zeros(len(result))

    for col, weight in cols_weights.items():
        if col in result.columns:
            col_values = pd.to_numeric(result[col], errors="coerce").fillna(0)
            denom = (col_values.max() - col_values.min()) or 1.0
            col_norm = (col_values - col_values.min()) / denom
            score += weight * col_norm

    result["priority_score"] = score
    return result.sort_values("priority_score", ascending=False)


def analyze_survey_folder(
    folder_path: str | Path,
    max_rows: int = 500,
    max_chars_meta: int = 4000,
) -> str:
    """
    Inspect a folder, separate metadata files from survey data files, and summarize data with context.

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
You are analyzing a survey dataset contained in a folder.

Metadata excerpts:
{meta_context}

Survey data samples (CSV):
{data_context}

Tasks:
- Identify which files describe the data (metadata) and which contain the survey responses.
- Summarize the survey content using the metadata for context.
- List all variables available.
- List 3-5 key insights or checks to perform.
- Suggest 3 quality checks or cleaning steps based on metadata and data.
Return concise bullet points.
"""

    return llm_chat(
        system_prompt="You are a data analyst skilled at using metadata to contextualize survey data.",
        user_message=prompt,
        model="green-l",
    )


def collect_survey_data_files_ai(
    folder_path: str | Path, output_dir: str | Path = "data", model: str = "green-l"
) -> Tuple[Dict[str, pd.DataFrame], str]:
    """
    Use an LLM to recognize which files are survey data (vs metadata), copy them to output_dir
    (keeping the original file name), and return a dict of DataFrames keyed by the original path.

    Returns (dataframes_dict, status_message).
    """
    root = Path(folder_path)
    if not root.exists() or not root.is_dir():
        return {}, f"Folder not found: {root}"

    candidates = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        # Consider common text-like files; LLM will decide.
        if suffix in {".csv", ".tsv", ".txt", ".json", ".yaml", ".yml", ".md"}:
            preview = ""
            try:
                if suffix in {".csv", ".tsv"}:
                    df = pd.read_csv(path, nrows=5)
                    preview = df.to_csv(index=False)
                else:
                    preview = path.read_text(encoding="utf-8", errors="ignore")[:2000]
            except Exception as exc:  # noqa: BLE001
                preview = f"<error reading file: {exc}>"
            candidates.append(
                {
                    "name": path.name,
                    "suffix": suffix,
                    "size_bytes": path.stat().st_size,
                    "preview": preview[:2000],
                }
            )

    if not candidates:
        return {}, "No candidate files found to classify."

    prompt = f"""
You are given file summaries from a survey folder. For each file, decide if it contains SURVEY DATA (actual responses / tabular records) or METADATA/DOCUMENTATION.

Return ONLY a JSON array of objects with:
- file_name: exact file name
- is_data: true/false

File summaries:
{json.dumps(candidates, ensure_ascii=False, indent=2)}
"""

    llm_response = llm_chat(
        system_prompt="Classify files as survey data vs metadata and respond ONLY with the requested JSON array.",
        user_message=prompt,
        model=model,
    )

    try:
        parsed = json.loads(llm_response)
        if not isinstance(parsed, list):
            raise ValueError("LLM response is not a list.")
    except Exception:
        # Fallback: assume CSV/TSV are data with generic names.
        parsed = [
            {"file_name": c["name"], "is_data": True}
            for c in candidates
            if c["suffix"] in {".csv", ".tsv"}
        ]

    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    loaded: Dict[str, pd.DataFrame] = {}

    for item in parsed:
        if not isinstance(item, dict):
            continue
        if not item.get("is_data"):
            continue

        file_name = item.get("file_name")
        if not file_name:
            continue

        source_path = root / file_name
        if not source_path.exists():
            matches = list(root.rglob(file_name))
            if not matches:
                continue
            source_path = matches[0]

        try:
            relative_path = source_path.relative_to(root)
        except ValueError:
            relative_path = Path(file_name)

        target = dest / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)

        suffix_counter = 0
        while target.exists():
            suffix_counter += 1
            target = target.with_name(f"{target.stem}_{suffix_counter}{target.suffix}")
        try:
            shutil.copy2(source_path, target)
            df = pd.read_csv(target)
            # Keep the dict key identical to the original source path for simpler lookups.
            loaded[str(source_path)] = df
        except Exception:
            continue

    if not loaded:
        return {}, "LLM did not identify any survey data files to copy."

    return loaded, f"Copied and loaded {len(loaded)} survey data table(s) into {dest.resolve()}"


def collect_survey_data_files_simple(
    folder_path: str | Path, output_dir: str | Path = "data"
) -> Tuple[Dict[str, pd.DataFrame], str]:
    """
    Scan a folder for CSV files, copy them into output_dir (preserving relative paths),
    and return a dict of DataFrames keyed by the original source path. No LLM involved.
    """
    root = Path(folder_path)
    if not root.exists() or not root.is_dir():
        return {}, f"Folder not found: {root}"

    csv_files = [p for p in root.rglob("*.csv") if p.is_file()]
    if not csv_files:
        return {}, "No CSV files found to copy."

    dest = Path(output_dir)
    loaded: Dict[str, pd.DataFrame] = {}

    for source_path in csv_files:
        try:
            relative_path = source_path.relative_to(root)
        except ValueError:
            relative_path = source_path.name

        target = dest / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)

        suffix_counter = 0
        while target.exists():
            suffix_counter += 1
            target = target.with_name(f"{target.stem}_{suffix_counter}{target.suffix}")

        try:
            shutil.copy2(source_path, target)
            df = pd.read_csv(target)
            # Keep the dict key identical to the original source path for simpler lookups.
            loaded[str(source_path)] = df
        except Exception:
            continue

    if not loaded:
        return {}, "Failed to copy or load any CSV files."

    return loaded, f"Copied and loaded {len(loaded)} CSV file(s) into {dest.resolve()}"


def _safe_table_name(name: str) -> str:
    """Normalize a string to a SQLite-friendly table name."""
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name[:60] or "table"


def _read_table_file(p: Path, max_rows: int | None = None) -> pd.DataFrame | None:
    """Load CSV/XLS/XLSX into a DataFrame (string-typed to reduce surprises)."""
    try:
        if p.suffix.lower() == ".csv":
            try:
                return pd.read_csv(p, dtype=str, nrows=max_rows, encoding="utf-8")
            except Exception:
                return pd.read_csv(p, dtype=str, nrows=max_rows, encoding="latin1")
        if p.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(p, sheet_name=0, dtype=str, nrows=max_rows)
    except Exception:
        return None
    return None


def load_survey_tables(
    folder: Union[str, Path] = "survey",
    output_dir: Union[str, Path] = "data",
    db_name: str = "surveys.db",
    *,
    recursive: bool = True,
    if_exists: str = "replace",
    max_rows_per_file: int | None = None,
) -> Tuple[Dict[str, Dict[str, Any]], str]:
    """
    Load CSV/XLS/XLSX files from `folder`, write them to SQLite at `output_dir/db_name`,
    and return (tables_metadata, status_message). Keys are table names written to SQLite.
    """
    root = Path(folder).expanduser().resolve()
    dest = Path(output_dir).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)

    db_path = dest / db_name

    if not root.exists():
        return {}, f"❌ Folder not found: `{root}`"

    file_iter = root.rglob("*") if recursive else root.glob("*")
    files = [p for p in file_iter if p.is_file() and p.suffix.lower() in {".csv", ".xlsx", ".xls"}]
    if not files:
        return {}, f"WARNING: No CSV/XLSX files found under `{root}`."

    conn = sqlite3.connect(db_path)
    written: Dict[str, Dict[str, Any]] = {}
    errors: List[str] = []

    try:
        for path in files:
            df = _read_table_file(path, max_rows=max_rows_per_file)
            if df is None or df.empty:
                continue

            rel = str(path.relative_to(root)).replace("\\", "/")
            base = _safe_table_name(Path(rel).with_suffix("").as_posix())

            table_name = base
            suffix_idx = 2
            while table_name in written:
                table_name = f"{base}_{suffix_idx}"
                suffix_idx += 1

            try:
                df.to_sql(table_name, conn, if_exists=if_exists, index=False)
                written[table_name] = {
                    "source_path": str(path),
                    "n_rows": int(df.shape[0]),
                    "n_cols": int(df.shape[1]),
                    "columns": list(df.columns),
                }
            except Exception as exc:  # noqa: BLE001
                errors.append(f"- `{path}` → {exc}")
    finally:
        conn.close()

    if not written:
        return {}, f"WARNING: No tables were written to `{db_path}`."

    table_names = list(written.keys())
    status_lines = [
        f"Wrote {len(table_names)} table(s) to SQLite DB: `{db_path}`",
        "",
        "**Table names:**",
        *[f"- `{t}`" for t in table_names],
    ]
    if errors:
        status_lines += ["", "**Errors:**", *errors]

    return written, "\n".join(status_lines)
