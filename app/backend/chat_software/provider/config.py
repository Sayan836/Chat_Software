"""Provider configuration.

This module is seeded by the uploaded `edur/utils/llm_provider.py`, but adapted
for this project:

- local-first default is Ollama, not OpenAI
- settings are read from env without project-specific imports
- Ollama and OpenAI share the OpenAI-compatible chat path
- Anthropic uses its native SDK path
- model capability metadata is explicit
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


SUPPORTED_PROVIDERS = {"ollama", "openai", "anthropic"}
DEFAULT_PROVIDER = "ollama"


@dataclass(frozen=True)
class ModelCapability:
    provider: str
    model: str
    supports_tools: bool
    tool_reliability: str
    context_window: int
    is_local: bool
    notes: str = ""

    @property
    def registry_key(self) -> str:
        return f"{self.provider}/{self.model}"


@dataclass(frozen=True)
class ProviderSettings:
    provider: str
    model: str
    api_key: Optional[str]
    base_url: Optional[str]
    temperature: float = 0.2
    max_tokens: Optional[int] = None


MODEL_REGISTRY: dict[str, ModelCapability] = {
    "ollama/qwen3:8b": ModelCapability(
        provider="ollama",
        model="qwen3:8b",
        supports_tools=True,
        tool_reliability="candidate",
        context_window=32768,
        is_local=True,
        notes="First candidate for local tool-call reliability bakeoff.",
    ),
    "ollama/qwen2.5:7b": ModelCapability(
        provider="ollama",
        model="qwen2.5:7b",
        supports_tools=True,
        tool_reliability="candidate",
        context_window=32768,
        is_local=True,
        notes="Fallback candidate with strong local tool-calling support.",
    ),
    "ollama/llama3.1": ModelCapability(
        provider="ollama",
        model="llama3.1",
        supports_tools=True,
        tool_reliability="fallback",
        context_window=8192,
        is_local=True,
        notes="Seed-file default retained as a fallback, not the product default.",
    ),
    "openai/gpt-4o-mini": ModelCapability(
        provider="openai",
        model="gpt-4o-mini",
        supports_tools=True,
        tool_reliability="high",
        context_window=128000,
        is_local=False,
        notes="Cloud opt-in small model.",
    ),
    "anthropic/claude-3-5-haiku-latest": ModelCapability(
        provider="anthropic",
        model="claude-3-5-haiku-latest",
        supports_tools=True,
        tool_reliability="high",
        context_window=200000,
        is_local=False,
        notes="Cloud opt-in fast Claude model through the native Anthropic SDK.",
    ),
}


def load_project_env(start: Optional[Path] = None) -> None:
    """Load `.env` if python-dotenv is installed.

    Importing dotenv is optional so tests and read-only tools can run without
    installing runtime dependencies.
    """

    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    root = start or Path.cwd()
    for candidate in [root / ".env", root.parent / ".env"]:
        try:
            exists = candidate.exists()
        except OSError:
            continue
        if exists:
            try:
                load_dotenv(candidate)
            except OSError:
                continue
            return


def get_provider() -> str:
    provider = (os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    if provider not in SUPPORTED_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_PROVIDERS))
        raise ValueError(f"Unsupported LLM_PROVIDER={provider!r}. Use one of: {supported}.")
    return provider


def resolve_model(
    requested_model: Optional[str] = None,
    env_var: Optional[str] = None,
    provider: Optional[str] = None,
) -> str:
    active_provider = provider or get_provider()
    if requested_model:
        return requested_model

    if env_var:
        env_model = os.getenv(env_var)
        if env_model:
            return env_model

    if active_provider == "ollama":
        return os.getenv("OLLAMA_MODEL") or os.getenv("LLM_MODEL") or "qwen3:8b"
    if active_provider == "anthropic":
        return (
            os.getenv("ANTHROPIC_MODEL")
            or os.getenv("LLM_MODEL")
            or "claude-3-5-haiku-latest"
        )
    return os.getenv("OPENAI_MODEL") or os.getenv("LLM_MODEL") or "gpt-4o-mini"


def get_model_capability(provider: str, model: str) -> ModelCapability:
    key = f"{provider}/{model}"
    if key in MODEL_REGISTRY:
        return MODEL_REGISTRY[key]
    return ModelCapability(
        provider=provider,
        model=model,
        supports_tools=provider in {"openai", "anthropic", "ollama"},
        tool_reliability="unknown",
        context_window=8192,
        is_local=provider == "ollama",
        notes="Unregistered model; validate before granting autonomy.",
    )


def normalize_ollama_openai_base_url(base_url: Optional[str]) -> str:
    """Return Ollama's OpenAI-compatible API base URL for chat calls."""

    selected = (base_url or "http://localhost:11434").rstrip("/")
    if not selected.endswith("/v1"):
        selected = f"{selected}/v1"
    return selected.rstrip("/")


def normalize_ollama_native_base_url(base_url: Optional[str]) -> str:
    """Return Ollama's native API base URL for model management calls."""

    selected = (base_url or "http://localhost:11434").rstrip("/")
    if selected.endswith("/v1"):
        selected = selected[:-3]
    return selected.rstrip("/")


def get_provider_settings(
    provider: Optional[str] = None,
    requested_model: Optional[str] = None,
    env_var: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
) -> ProviderSettings:
    load_project_env()
    active_provider = (provider or get_provider()).strip().lower()
    if active_provider not in SUPPORTED_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_PROVIDERS))
        raise ValueError(f"Unsupported provider={active_provider!r}. Use one of: {supported}.")
    model = resolve_model(
        requested_model=requested_model,
        env_var=env_var,
        provider=active_provider,
    )

    if active_provider == "ollama":
        return ProviderSettings(
            provider=active_provider,
            model=model,
            api_key=api_key or os.getenv("OLLAMA_API_KEY", "ollama"),
            base_url=normalize_ollama_openai_base_url(
                base_url or os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_NATIVE_BASE_URL")
            ),
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if active_provider == "anthropic":
        return ProviderSettings(
            provider=active_provider,
            model=model,
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            base_url=base_url or os.getenv("ANTHROPIC_BASE_URL"),
            temperature=temperature,
            max_tokens=max_tokens,
        )
    return ProviderSettings(
        provider=active_provider,
        model=model,
        api_key=api_key or os.getenv("OPENAI_API_KEY"),
        base_url=base_url or os.getenv("OPENAI_BASE_URL"),
        temperature=temperature,
        max_tokens=max_tokens,
    )
