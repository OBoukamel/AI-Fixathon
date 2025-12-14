from pathlib import Path
from typing import Dict, List, Union
import pandas as pd
from utils_llm import llm_chat

def understand_survey_columns(
    folder_path: Union[str, Path],
    survey_schema: Dict[str, List[str]],   # <-- dict dataset -> colonnes attendues
    max_rows: int = 200,
    max_chars_meta: int = 2500,
    max_chars_data: int = 6000,
    max_meta_files: int = 15,
    max_csv_files: int = 60,
) -> str:
    root = Path(folder_path)
    if not root.exists() or not root.is_dir():
        return f"Folder not found: {root}"

    meta_suffixes = {".md", ".txt", ".json", ".yaml", ".yml"}
    data_suffixes = {".csv"}

    all_files = [p for p in root.rglob("*") if p.is_file()]
    meta_files = [p for p in all_files if p.suffix.lower() in meta_suffixes][:max_meta_files]
    csv_files = [p for p in all_files if p.suffix.lower() in data_suffixes][:max_csv_files]

    # ---- Metadata (capé) ----
    meta_blobs: List[str] = []
    for p in meta_files:
        try:
            txt = p.read_text(encoding="utf-8", errors="ignore")[:800]
            meta_blobs.append(f"FILE: {p.relative_to(root)}\n{txt}")
        except Exception as exc:
            meta_blobs.append(f"FILE: {p.relative_to(root)}\n<unable to read: {exc}>")

    meta_context = ("\n\n".join(meta_blobs)[:max_chars_meta]) if meta_blobs else "No metadata files found."

    if not csv_files:
        return "No survey data (.csv) found in the folder."

    # ---- Header scan : trouver quelles colonnes de chaque dataset existent dans chaque CSV ----
    # Important: utiliser le chemin relatif pour distinguer les data.csv
    file_headers: Dict[str, List[str]] = {}
    dataset_hits: Dict[str, Dict[str, List[str]]] = {ds: {} for ds in survey_schema}

    for p in csv_files:
        rel = str(p.relative_to(root))
        try:
            df0 = pd.read_csv(p, nrows=0)
            headers = [c.strip() for c in df0.columns.tolist()]
            file_headers[rel] = headers
            header_set = set(headers)

            for ds_name, expected_cols in survey_schema.items():
                present = [c for c in expected_cols if c in header_set]
                if present:
                    dataset_hits[ds_name][rel] = present

        except Exception:
            continue

    # Si aucun match, on renvoie un diagnostic utile
    found_any = any(len(files) > 0 for files in dataset_hits.values())
    if not found_any:
        return (
            "No expected columns were found in scanned CSV headers.\n"
            f"Datasets expected: {list(survey_schema.keys())}\n"
            f"Scanned files (capped): {[str(p.relative_to(root)) for p in csv_files]}"
        )

    # ---- Construire un échantillon data_context : uniquement des fichiers pertinents ----
    relevant_files = []
    for ds_name, files_map in dataset_hits.items():
        for rel in files_map.keys():
            relevant_files.append(rel)
    # unique + ordre stable
    relevant_files = list(dict.fromkeys(relevant_files))[:10]  # cap dur: 10 fichiers au LLM

    snippets: List[str] = []
    for rel in relevant_files:
        p = root / rel
        try:
            df = pd.read_csv(p, nrows=max_rows)
            df_small = df.head(10)
            csv_snippet = df_small.to_csv(index=False)[:1400]
            snippets.append(f"DATA FILE: {rel}\n{csv_snippet}")
        except Exception as exc:
            snippets.append(f"DATA FILE: {rel}\n<unable to load sample: {exc}>")

    data_context = ("\n\n".join(snippets)[:max_chars_data]) if snippets else ""

    prompt = f"""
You are identifying the column definitions in survey data using metadata.

Expected schema (dataset -> columns):
{survey_schema}

Where columns were found (header scan):
{dataset_hits}

Metadata excerpts:
{meta_context}

Survey data samples:
{data_context}

Tasks:
- For each dataset, define each expected column concisely.
- Return JSON with top-level keys = dataset names.
- Each dataset value is a JSON object: {{column_name: definition}}.
- If definition missing, return "Definition not found".
"""

    return llm_chat(
        system_prompt="You are a data analyst skilled at using metadata to contextualize survey data.",
        user_message=prompt,
        model="green-l",
    )
