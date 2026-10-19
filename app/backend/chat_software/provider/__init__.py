"""Provider selection and model client construction."""

from .config import ModelCapability, ProviderSettings, get_provider_settings, resolve_model
from .factory import ChatClient, get_chat_client, get_openai_compatible_client
from .router import TaskKind, choose_model_for_task

__all__ = [
    "ChatClient",
    "ModelCapability",
    "ProviderSettings",
    "TaskKind",
    "choose_model_for_task",
    "get_chat_client",
    "get_openai_compatible_client",
    "get_provider_settings",
    "resolve_model",
]
