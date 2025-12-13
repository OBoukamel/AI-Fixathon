"""ZIP utilities: build directory trees, render them in Streamlit, and extract safely."""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Dict, List

import streamlit as st


def _add_path_to_tree(tree: Dict[str, dict], parts: List[str]) -> None:
    """Mutate `tree` by inserting path components, e.g., ['a', 'b', 'file.csv']."""
    node = tree
    for part in parts:
        node = node.setdefault(part, {})


def zip_to_tree(zf: zipfile.ZipFile) -> Dict[str, dict]:
    """
    Build a nested dict representing the zip content.
    Directories are dicts; files are empty dicts {} as leaf nodes.
    """
    tree: Dict[str, dict] = {}
    for info in zf.infolist():
        # Normalize and skip empty entries
        name = info.filename.strip("/")
        if not name:
            continue
        parts = [p for p in name.split("/") if p]
        _add_path_to_tree(tree, parts)
    return tree


def render_tree(tree: Dict[str, dict], level: int = 0) -> None:
    """Render nested dict `tree` as a clickable arborescence in Streamlit."""
    # Sort folders before files (folders have children)
    items = sorted(tree.items(), key=lambda kv: (0 if kv[1] else 1, kv[0].lower()))

    for name, children in items:
        is_folder = bool(children)
        if is_folder:
            with st.expander(f"📁 {name}", expanded=(level < 1)):
                render_tree(children, level + 1)
        else:
            st.write(f"📄 {name}")


def extract_zip_to_dir(zf: zipfile.ZipFile, out_dir: Path) -> None:
    """
    Extract zip into out_dir safely (basic zip-slip prevention).
    """
    out_dir = out_dir.resolve()
    for member in zf.infolist():
        member_path = (out_dir / member.filename).resolve()
        if not str(member_path).startswith(str(out_dir)):
            # Skip entries attempting directory traversal
            continue
        zf.extract(member, out_dir)
