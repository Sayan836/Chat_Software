"""Simple model router for Phase 0/1."""

from __future__ import annotations

from enum import Enum

from .config import ModelCapability, get_model_capability, get_provider_settings


class TaskKind(str, Enum):
    CHAT = "chat"
    FILE_SEARCH = "file_search"
    SUMMARIZE = "summarize"
    DEVELOPMENT_READONLY = "development_readonly"


def choose_model_for_task(task: TaskKind) -> ModelCapability:
    """Return the active model capability.

    Phase 0 keeps routing conservative and transparent: user/provider settings
    win. Later phases can add hardware-aware and escalation-aware routing.
    """

    settings = get_provider_settings()
    capability = get_model_capability(settings.provider, settings.model)
    if task in {TaskKind.FILE_SEARCH, TaskKind.DEVELOPMENT_READONLY}:
        return capability
    return capability
