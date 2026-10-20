"""Model client factory.

Ollama and OpenAI use the OpenAI SDK because Ollama exposes an OpenAI-compatible
chat endpoint. Claude uses the native Anthropic SDK. Tests can use the
deterministic echo client.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Sequence

from .config import ProviderSettings, get_provider_settings


Message = Mapping[str, str]


class ChatClient(Protocol):
    settings: ProviderSettings

    def complete(
        self,
        messages: Sequence[Message],
        tools: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> str:
        ...


@dataclass
class EchoChatClient:
    settings: ProviderSettings

    def complete(
        self,
        messages: Sequence[Message],
        tools: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> str:
        last_user = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        return f"Offline echo mode ({self.settings.provider}/{self.settings.model}): {last_user}"


@dataclass
class OpenAICompatibleChatClient:
    settings: ProviderSettings

    def complete(
        self,
        messages: Sequence[Message],
        tools: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> str:
        client = get_openai_compatible_client(self.settings)
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "messages": list(messages),
            "temperature": self.settings.temperature,
        }
        if self.settings.max_tokens is not None:
            kwargs["max_tokens"] = self.settings.max_tokens
        if tools:
            kwargs["tools"] = list(tools)

        response = client.chat.completions.create(**kwargs)
        content = response.choices[0].message.content
        return content or ""


@dataclass
class AnthropicChatClient:
    settings: ProviderSettings

    def complete(
        self,
        messages: Sequence[Message],
        tools: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> str:
        if tools:
            raise RuntimeError("Anthropic tool calling is not wired in this phase.")
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError("The anthropic package is required for Claude calls.") from exc

        client_kwargs: dict[str, Any] = {}
        if self.settings.api_key:
            client_kwargs["api_key"] = self.settings.api_key
        if self.settings.base_url:
            client_kwargs["base_url"] = self.settings.base_url

        system_parts: list[str] = []
        chat_messages: list[dict[str, str]] = []
        for message in messages:
            role = message.get("role")
            content = message.get("content", "")
            if role == "system":
                system_parts.append(content)
            elif role in {"user", "assistant"}:
                chat_messages.append({"role": role, "content": content})

        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "messages": chat_messages,
            "temperature": self.settings.temperature,
            "max_tokens": self.settings.max_tokens or 1024,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)

        response = Anthropic(**client_kwargs).messages.create(**kwargs)
        return "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )


def get_chat_client(
    provider: Optional[str] = None,
    requested_model: Optional[str] = None,
    env_var: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: Optional[int] = None,
    offline_echo: bool = False,
) -> ChatClient:
    settings = get_provider_settings(
        provider=provider,
        requested_model=requested_model,
        env_var=env_var,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if offline_echo:
        return EchoChatClient(settings=settings)
    if settings.provider == "anthropic":
        return AnthropicChatClient(settings=settings)
    if settings.provider == "airllm":
        from .airllm import MANAGER, AirLLMChatClient

        if not settings.model:
            raise ValueError("No AirLLM model yet: download or register one in Settings > AirLLM.")
        return AirLLMChatClient(settings=settings, manager=MANAGER)
    if settings.provider == "compatible" and not settings.base_url:
        raise ValueError("Set COMPATIBLE_BASE_URL (Settings > Providers) to use the compatible provider.")
    return OpenAICompatibleChatClient(settings=settings)


def get_openai_compatible_client(settings: Optional[ProviderSettings] = None) -> Any:
    """Return an OpenAI SDK client for OpenAI-compatible providers.

    This preserves the useful seed-file behavior for direct SDK integrations.
    Anthropic uses AnthropicChatClient instead.
    """

    active = settings or get_provider_settings()
    if active.provider == "anthropic":
        raise ValueError("Anthropic is not OpenAI-compatible; use AnthropicChatClient.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("The openai package is required for direct OpenAI clients.") from exc

    kwargs: dict[str, Any] = {"api_key": active.api_key}
    if active.base_url:
        kwargs["base_url"] = active.base_url
    return OpenAI(**kwargs)
