"""Streamlit helpers to manage and display chat history."""
from __future__ import annotations

import streamlit as st


def init_chat_state() -> None:
    """Initialize chat history in Streamlit session_state."""
    if "messages" not in st.session_state:
        st.session_state.messages = []


def add_message(role: str, content: str) -> None:
    """Append a message (user or assistant) to chat history."""
    st.session_state.messages.append({"role": role, "content": content})


def render_chat_history() -> None:
    """Render all chat messages in Streamlit."""
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
