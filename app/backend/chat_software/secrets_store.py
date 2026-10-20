"""Local store for provider API keys and endpoint settings entered in the app.

Values live in ``<state_dir>/secrets.json`` with owner-only permissions (0600). They are never
returned by the API (only "set"/"not set" and where it came from) and never written to runtime
logs. An environment variable of the same name is the fallback, so existing ``.env`` setups keep
working. This is plaintext at rest, like ``.env``; use full-disk encryption on shared machines.
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Optional

from chat_software.paths import state_dir


SECRET_NAMES = {
    "OPENAI_API_KEY": "OpenAI (ChatGPT) API key",
    "ANTHROPIC_API_KEY": "Anthropic (Claude) API key",
    "COMPATIBLE_API_KEY": "OpenAI-compatible endpoint key (OpenRouter, Groq, Together, ...)",
    "BRAVE_SEARCH_API_KEY": "Brave Search API key (deep research)",
    "TAVILY_API_KEY": "Tavily API key (deep research)",
    "HF_TOKEN": "Hugging Face token (gated model downloads)",
}
SETTING_NAMES = {
    "OPENAI_BASE_URL": "OpenAI base URL override",
    "ANTHROPIC_BASE_URL": "Anthropic base URL override",
    "COMPATIBLE_BASE_URL": "OpenAI-compatible endpoint base URL",
    "COMPATIBLE_MODEL": "Default model for the OpenAI-compatible endpoint",
    "SEARXNG_URL": "SearXNG instance URL (deep research)",
    "RESEARCH_SEARCH_PROVIDER": "Web search provider for deep research",
    "AIRLLM_PYTHON": "AirLLM runtime Python (isolated venv)",
    "AIRLLM_HOME": "AirLLM folder for models and logs",
    "AIRLLM_BACKEND": "AirLLM backend: cpu, metal or cuda",
}
_URL_SETTINGS = {"OPENAI_BASE_URL", "ANTHROPIC_BASE_URL", "COMPATIBLE_BASE_URL", "SEARXNG_URL"}
_LOCK = threading.Lock()


def _path() -> Optional[Path]:
    directory = state_dir()
    if str(directory) == ":memory:":
        return None
    return directory / "secrets.json"


def _load() -> dict[str, str]:
    path = _path()
    if path is None or not path.exists():
        return {}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in data.items() if isinstance(v, str)}


def _save(data: dict[str, str]) -> None:
    path = _path()
    if path is None:
        raise RuntimeError("Secrets cannot be stored with an in-memory state directory.")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
    os.chmod(tmp, 0o600)
    os.replace(tmp, path)


def _check_name(name: str) -> None:
    if name not in SECRET_NAMES and name not in SETTING_NAMES:
        raise KeyError(f"Unknown setting: {name}")


def get(name: str) -> Optional[str]:
    """Stored value first, then the environment."""

    _check_name(name)
    stored = _load().get(name)
    if stored:
        return stored
    value = os.getenv(name, "").strip()
    return value or None


def source(name: str) -> str:
    _check_name(name)
    if _load().get(name):
        return "app"
    if os.getenv(name, "").strip():
        return "environment"
    return "none"


def set_value(name: str, value: Optional[str]) -> None:
    _check_name(name)
    cleaned = (value or "").strip()
    if cleaned and name in _URL_SETTINGS and not re.match(r"^https?://[^\s]+$", cleaned):
        raise ValueError(f"{name} must be an http(s) URL.")
    if cleaned and len(cleaned) > 4096:
        raise ValueError(f"{name} is too long.")
    with _LOCK:
        data = _load()
        if cleaned:
            data[name] = cleaned
        else:
            data.pop(name, None)
        _save(data)


def describe() -> dict[str, dict[str, object]]:
    """UI-safe view: secret values are never included; settings values are."""

    out: dict[str, dict[str, object]] = {}
    for name, label in SECRET_NAMES.items():
        out[name] = {"label": label, "secret": True, "set": bool(get(name)), "source": source(name)}
    for name, label in SETTING_NAMES.items():
        out[name] = {
            "label": label,
            "secret": False,
            "set": bool(get(name)),
            "source": source(name),
            "value": get(name),
        }
    return out
