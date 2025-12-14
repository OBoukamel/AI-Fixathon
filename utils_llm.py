import os
import json
import requests
from typing import Dict, List
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GREENPT_API_KEY") or st.secrets.get("GREENPT_API_KEY")
if not api_key:
    raise RuntimeError("GREENPT_API_KEY is missing.")

BASE_URL = (
    os.getenv("GREENPT_API_URL")
    or st.secrets.get("GREENPT_API_URL")
    or "https://api.greenpt.ai/v1/chat/completions"
)

# ---- allowed GreenPT models ----
KNOWN_MODELS = {
    "green-r",  # lightweight, fast
    "green-l",  # large 120B open-source
}

# Capability: does the model accept a dedicated system prompt role?
MODEL_CAPABILITIES: Dict[str, Dict[str, bool]] = {
    "green-r": {"system_prompt": False},
    "green-l": {"system_prompt": False},
}

def _validate_model(model: str) -> None:
    if model not in KNOWN_MODELS:
        raise ValueError(
            f"Unsupported model '{model}'. Available models: {', '.join(sorted(KNOWN_MODELS))}"
        )

def _post_json(payload: Dict, timeout = 60) -> Dict:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        response = requests.post(BASE_URL, json=payload, headers=headers, timeout= timeout)
        response.raise_for_status()
    except requests.exceptions.HTTPError as exc:
        # Preserve the original server message for easier debugging
        raise RuntimeError(
            f"GreenPT returned HTTP {response.status_code}: {response.text}"
        ) from exc
    except requests.exceptions.ConnectionError as exc:
        raise RuntimeError(
            f"Cannot reach GreenPT at {BASE_URL}. Check network/DNS."
        ) from exc
    except requests.exceptions.Timeout:
        raise RuntimeError("GreenPT request timed out.")
    return response.json()

def _build_messages_single(system_prompt: str, user_message: str, model: str) -> List[Dict[str, str]]:
    """
    Build a GreenPT-compatible message list. If the model does not support the
    system role, we prepend the system instructions into the user message.
    """
    if MODEL_CAPABILITIES[model]["system_prompt"]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]
    merged = f"System instructions:\n{system_prompt}\n\nUser question:\n{user_message}"
    return [{"role": "user", "content": merged}]

def _build_messages_history(system_prompt: str, messages: List[Dict[str, str]], model: str) -> List[Dict[str, str]]:
    """
    Build messages including history. If the model cannot take a system role,
    we flatten everything into a single user message.
    """
    if MODEL_CAPABILITIES[model]["system_prompt"]:
        return [{"role": "system", "content": system_prompt}] + messages

    history_lines = []
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        history_lines.append(f"[{role}] {content}")
    merged = f"System instructions:\n{system_prompt}\n\nConversation so far:\n" + "\n".join(history_lines)
    return [{"role": "user", "content": merged}]


def llm_chat(
    system_prompt: str,
    user_message: str,
    model: str = "green-l",
    temperature: float = 0.2,
) -> str:
    _validate_model(model)
    payload = {
        "model": model,
        "messages": _build_messages_single(system_prompt, user_message, model),
        "temperature": temperature,
    }
    result = _post_json(payload)
    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(
            f"Unexpected response format: {json.dumps(result, indent=2)}"
        ) from exc


def llm_chat_with_history(
    system_prompt: str,
    messages: List[Dict[str, str]],
    model: str = "green-l",
    temperature: float = 0.2,
) -> str:
    _validate_model(model)
    if not all(isinstance(m, dict) and {"role", "content"} <= m.keys() for m in messages):
        raise ValueError("`messages` must be a list of {'role':..., 'content':...} dicts.")

    payload = {
        "model": model,
        "messages": _build_messages_history(system_prompt, messages, model),
        "temperature": temperature,
    }
    result = _post_json(payload)
    try:
        return result["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(
            f"Unexpected response format: {json.dumps(result, indent=2)}"
        ) from exc