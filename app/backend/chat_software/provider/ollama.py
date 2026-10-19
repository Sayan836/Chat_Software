"""Ollama native API helpers for local model management."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Optional

from .config import MODEL_REGISTRY, get_provider_settings, normalize_ollama_native_base_url


CATALOG_MODELS = [
    "qwen3:8b",
    "qwen2.5:7b",
    "llama3.2:3b",
    "llama3.1:8b",
    "mistral:7b",
    "gemma3:4b",
    "phi4-mini:latest",
    "codellama:7b",
]


def native_base_url() -> str:
    configured = os.getenv("OLLAMA_NATIVE_BASE_URL")
    if configured:
        return configured.rstrip("/")

    settings = get_provider_settings(provider="ollama")
    return normalize_ollama_native_base_url(settings.base_url)


def catalog_models() -> list[str]:
    registry_models = [
        capability.model
        for capability in MODEL_REGISTRY.values()
        if capability.provider == "ollama"
    ]
    return sorted(dict.fromkeys([*registry_models, *CATALOG_MODELS]))


def model_name(model: Any) -> str:
    if not isinstance(model, dict):
        return ""
    return str(model.get("name") or model.get("model") or "").strip()


def sort_installed_models(models: Any) -> list[dict[str, Any]]:
    """Return unique installed models with the most recently modified first."""

    if not isinstance(models, list):
        return []
    unique: dict[str, dict[str, Any]] = {}
    for model in models:
        name = model_name(model)
        if name and name not in unique:
            unique[name] = model
    return sorted(
        unique.values(),
        key=lambda model: (str(model.get("modified_at") or ""), model_name(model)),
        reverse=True,
    )


def select_installed_model(
    preferred: Optional[str],
    models: Any,
) -> Optional[str]:
    """Select an installed model, falling back to the newest local model."""

    installed = sort_installed_models(models)
    names = [model_name(model) for model in installed]
    selected = (preferred or "").strip()
    if selected in names:
        return selected
    if selected and ":" not in selected:
        matching_tag = next((name for name in names if name.split(":", 1)[0] == selected), None)
        if matching_tag:
            return matching_tag
    return names[0] if names else None


def _request(
    method: str,
    path: str,
    payload: Optional[dict[str, Any]] = None,
    timeout: float = 60,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(
        f"{native_base_url()}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            text = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Ollama at {native_base_url()}: {exc}") from exc

    return json.loads(text) if text else {}


def list_models() -> dict[str, Any]:
    body = _request("GET", "/api/tags", timeout=10)
    models = sort_installed_models(body.get("models", []))
    return {
        "base_url": native_base_url(),
        "models": models,
        "model_names": [model_name(model) for model in models],
        "recommended_model": select_installed_model(None, models),
    }


def pull_model(model: str) -> dict[str, Any]:
    body = _request("POST", "/api/pull", {"model": model, "stream": False}, timeout=1800)
    return {
        "base_url": native_base_url(),
        "model": model,
        "result": body,
    }


def delete_model(model: str) -> dict[str, Any]:
    body = _request("DELETE", "/api/delete", {"model": model}, timeout=60)
    return {
        "base_url": native_base_url(),
        "model": model,
        "result": body,
    }
