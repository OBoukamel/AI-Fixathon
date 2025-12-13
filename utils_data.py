"""Helpers for loading datasets from URLs or file uploads and building previews."""
from __future__ import annotations

from io import StringIO
from typing import Dict, Iterable

import pandas as pd
import requests


def load_csv_from_url(url: str) -> pd.DataFrame:
    """Load a CSV file from a public URL."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return pd.read_csv(StringIO(resp.text))


def load_uploaded_csvs(files: Iterable) -> Dict[str, pd.DataFrame]:
    """
    Convert a set of uploaded CSV files (Streamlit) into a name->DataFrame mapping.

    Table names are inferred from file names (without extension).
    """
    tables: Dict[str, pd.DataFrame] = {}
    for f in files:
        name = f.name.rsplit(".", 1)[0]
        df = pd.read_csv(f)
        tables[name] = df
    return tables


def build_table_preview_markdown(tables: Dict[str, pd.DataFrame], max_rows: int = 5) -> str:
    """
    Create a markdown summary for each table (head only) to inject into prompts.
    """
    if not tables:
        return "No tables provided."

    parts = []
    for name, df in tables.items():
        preview = df.head(max_rows)
        parts.append(f"### TABLE: {name}\n{preview.to_markdown(index=False)}\n")

    return "\n".join(parts)
