"""Lightweight wrappers around the OpenAI client for hackathon-speed development."""
from __future__ import annotations

import os
from typing import List, Dict

from dotenv import load_dotenv
from openai import OpenAI
import streamlit as st

# Load environment variables from .env so OPENAI_API_KEY is available locally.
load_dotenv()
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])



def llm_chat(system_prompt: str, user_message: str, model: str = "gpt-4.1-mini") -> str:
    """
    Basic LLM call using the chat completion API.

    Keeps temperature low for predictable outputs during demos.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content


def llm_chat_with_history(
    system_prompt: str, messages: List[Dict[str, str]], model: str = "gpt-4.1-mini"
) -> str:
    """
    LLM call that includes the full conversation history.

    The system prompt is prepended to keep behavior consistent across turns.
    """
    full_messages = [{"role": "system", "content": system_prompt}] + messages

    response = client.chat.completions.create(
        model=model,
        messages=full_messages,
        temperature=0.2,
    )

    return response.choices[0].message.content
