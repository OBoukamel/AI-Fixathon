from __future__ import annotations

from pathlib import Path
import re
import datetime as dt
from typing import Any, Dict, List, Tuple, Union, Optional

import pandas as pd

# Optional PDF support (PyMuPDF / fitz)
try:
    import fitz  # type: ignore
    FITZ_AVAILABLE = True
except Exception:
    fitz = None
    FITZ_AVAILABLE = False


# =============================
# CONFIG
# =============================
SUPPORTED_EXTENSIONS = {".pdf", ".xlsx", ".xls", ".csv"}
ANON_MARKER = "[ANONYMIZED_BY_SCRIPT]"

# Sidecar marker to avoid corrupting CSV/XLSX by inserting marker text into the file
ANON_SIDECAR_SUFFIX = ".anonmarker"


# =============================
# PATIENT + GEO DETECTION
# =============================
_SENSITIVE_PATTERNS = [
    # Patient identity
    r"\bName\b",
    r"\bPatient\b",
    r"\bFirst Name\b",
    r"\bLast Name\b",
    r"\bDOB\b",
    r"\bDate of Birth\b",
    r"\b\d{2}[/-]\d{2}[/-]\d{4}\b",
    r"\b\d{9}\b",  # BSN-like
    r"\bMRN\b",
    r"\bPatient ID\b",
    # Geographic data
    r"\bAddress\b",
    r"\bStreet\b",
    r"\bCity\b",
    r"\bZip\b",
    r"\bPostcode\b",
    r"\bNetherlands\b",
    r"\bAmsterdam\b",
    r"\bLatitude\b",
    r"\bLongitude\b",
]


def contains_sensitive_data(text: str) -> bool:
    return any(re.search(p, text, re.IGNORECASE) for p in _SENSITIVE_PATTERNS)


def sidecar_marker_path(file_path: Path) -> Path:
    return file_path.with_suffix(file_path.suffix + ANON_SIDECAR_SUFFIX)


def is_marked_anonymized(file_path: Path) -> bool:
    """
    We do NOT insert the marker into CSV/XLSX content because it breaks parsing.
    Instead, we create a sidecar file: <filename>.<ext>.anonmarker
    """
    return sidecar_marker_path(file_path).exists()


# =============================
# TEXT EXTRACTION (safe / bounded)
# =============================
def _truncate(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n…[TRUNCATED]…"


def extract_text(path: Path, *, max_rows: int = 2000, max_chars: int = 200_000) -> str:
    """
    Best-effort extraction for detecting sensitive patterns + building metadata.
    Bounded by max_rows/max_chars to keep Streamlit responsive.
    """
    ext = path.suffix.lower()

    if ext == ".pdf":
        if not FITZ_AVAILABLE:
            return ""  # gracefully skip if fitz not available
        return _truncate(extract_text_from_pdf(path), max_chars)

    if ext in {".xlsx", ".xls"}:
        return _truncate(extract_text_from_excel(path, max_rows=max_rows), max_chars)

    if ext == ".csv":
        return _truncate(extract_text_from_csv(path, max_rows=max_rows), max_chars)

    return ""


def extract_text_from_pdf(path: Path) -> str:
    try:
        doc = fitz.open(path)  # type: ignore
        return "\n".join(page.get_text() for page in doc)
    except Exception:
        return ""


def extract_text_from_excel(path: Path, *, max_rows: int = 2000) -> str:
    # First sheet only, cast to str so regex works consistently
    df = pd.read_excel(path, sheet_name=0, dtype=str, nrows=max_rows)
    return df.to_csv(index=False)


def extract_text_from_csv(path: Path, *, max_rows: int = 2000) -> str:
    # Try utf-8 then fallback
    try:
        df = pd.read_csv(path, dtype=str, nrows=max_rows, encoding="utf-8")
    except Exception:
        df = pd.read_csv(path, dtype=str, nrows=max_rows, encoding="latin1")
    return df.to_csv(index=False)


# =============================
# ANONYMISATION
# =============================
def anonymize_text_blob(text: str) -> str:
    """
    Anonymize a plain text blob (e.g., extracted PDF text).
    We do NOT add marker lines here; marker is tracked via sidecar file.
    """
    text = re.sub(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", "[REDACTED_NAME]", text)
    text = re.sub(r"\b\d{2}[/-]\d{2}[/-]\d{4}\b", "[REDACTED_DATE]", text)
    text = re.sub(r"\b\d{9}\b", "[REDACTED_ID]", text)
    text = re.sub(r"\b\d{1,5}\s+\w+\s+\w+\b", "[REDACTED_ADDRESS]", text)
    return text


def anonymize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    def _anon_cell(x: Any) -> Any:
        if pd.isna(x):
            return x
        s = str(x)
        s = re.sub(r"\b[A-Z][a-z]+ [A-Z][a-z]+\b", "[REDACTED_NAME]", s)
        s = re.sub(r"\b\d{2}[/-]\d{2}[/-]\d{4}\b", "[REDACTED_DATE]", s)
        s = re.sub(r"\b\d{9}\b", "[REDACTED_ID]", s)
        s = re.sub(r"\b\d{1,5}\s+\w+\s+\w+\b", "[REDACTED_ADDRESS]", s)
        return s

    return df.applymap(_anon_cell)


def mark_anonymized(file_path: Path) -> None:
    sidecar_marker_path(file_path).write_text(ANON_MARKER, encoding="utf-8")


def rewrite_file_in_place(file_path: Path) -> None:
    """
    Rewrite file in place (optional behavior, off by default).
    CSV/XLSX: anonymize cell values and preserve valid format.
    PDF: replaces the PDF with a simple one-page text PDF (best-effort).
    """
    ext = file_path.suffix.lower()

    if ext == ".csv":
        try:
            df = pd.read_csv(file_path, dtype=str, encoding="utf-8")
        except Exception:
            df = pd.read_csv(file_path, dtype=str, encoding="latin1")

        df2 = anonymize_dataframe(df)
        df2.to_csv(file_path, index=False, encoding="utf-8")
        mark_anonymized(file_path)
        return

    if ext in {".xlsx", ".xls"}:
        df = pd.read_excel(file_path, sheet_name=0, dtype=str)
        df2 = anonymize_dataframe(df)
        df2.to_excel(file_path, index=False)
        mark_anonymized(file_path)
        return

    if ext == ".pdf":
        if not FITZ_AVAILABLE:
            return  # can't rewrite PDFs without fitz

        original = extract_text_from_pdf(file_path)
        anon = anonymize_text_blob(original)

        doc = fitz.open()  # type: ignore
        page = doc.new_page()
        # Insert marker + truncated anonymized text (PDF text rendering is limited)
        page.insert_text((72, 72), (ANON_MARKER + "\n" + anon)[:5000])
        doc.save(file_path)
        mark_anonymized(file_path)
        return


# =============================
# METADATA
# =============================
def get_metadata(path: Path, text: str, *, root: Path) -> Dict[str, Any]:
    stat = path.stat()
    return {
        "filename": path.name,
        "relative_path": str(path.relative_to(root)),
        "absolute_path": str(path.resolve()),
        "extension": path.suffix.lower(),
        "size_bytes": stat.st_size,
        "created_at": dt.datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "modified_at": dt.datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "num_chars": len(text),
        "num_words": len(text.split()) if text else 0,
        "contains_sensitive_data": contains_sensitive_data(text) if text else False,
        "already_anonymized": is_marked_anonymized(path),
        "snippet": text[:300] if text else "",
    }


# =============================
# MAIN PIPELINE (Streamlit-friendly)
# =============================
from typing import Union, Dict, Any, List, Tuple
from pathlib import Path

def getting_meta_data_all(path: Union[str, Path]) -> Dict[str, Any]:
    # ✅ always convert to Path first (prevents 'str' has no attribute glob)
    root = Path(path).expanduser().resolve()

    all_files = [p for p in root.rglob("*") if p.is_file()]

    processed = {}
    errors: List[Tuple[str, str]] = []
    anonymized_files = []
    skipped_anonymization = []

    for f in all_files:
        if f.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            text = extract_text(f)

            if contains_sensitive_data(text):
                if is_marked_anonymized(f):
                    skipped_anonymization.append(str(f))
                else:
                    # only rewrite if you want anonymization-in-place
                    # rewrite_file_in_place(f)
                    # mark_anonymized(f)
                    anonymized_files.append(str(f))

            processed[str(f)] = get_metadata(f, text, root=root)

        except Exception as e:
            errors.append((str(f), str(e)))

    return {
        "processed": processed,
        "summary": {
            "root": str(root),
            "total_files_found": len(all_files),
            "files_processed": len(processed),
            "flagged_sensitive": len(anonymized_files),
            "already_anonymized": len(skipped_anonymization),
            "errors": len(errors),
        },
        "errors": errors,
    }
