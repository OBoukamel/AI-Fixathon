"""Analysis helpers: LLM summarization, project planning, and lightweight scoring."""
from __future__ import annotations

from typing import Dict

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
        system_prompt="You are a data analyst for social impact projects.",
        user_message=prompt,
        model="gpt-4.1-mini",
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
        model="gpt-4.1-mini",
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
            # Convert to numeric and normalize to 0-1 to make weights meaningful.
            col_values = pd.to_numeric(result[col], errors="coerce").fillna(0)
            denom = (col_values.max() - col_values.min()) or 1.0
            col_norm = (col_values - col_values.min()) / denom
            score += weight * col_norm

    result["priority_score"] = score
    return result.sort_values("priority_score", ascending=False)
