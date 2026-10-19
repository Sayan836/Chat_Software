"""Dependency bundle passed to LangGraph agent nodes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from chat_software.provider.config import ProviderSettings
from chat_software.provider.factory import ChatClient
from chat_software.safety_store import SafetyStore

from .session_store import SessionStore


@dataclass(frozen=True)
class AgentDeps:
    chat_client: ChatClient
    settings: ProviderSettings
    session_store: SessionStore
    safety_store: SafetyStore | None
    roots: tuple[str, ...]
    working_path: str = "."
    chat_handler: Optional[Callable[[str], str]] = None
    file_ops_handler: Optional[Callable[[str, Optional[str], bool], tuple[str, list[Any]]]] = None

    @classmethod
    def from_values(
        cls,
        *,
        chat_client: ChatClient,
        settings: ProviderSettings,
        session_store: SessionStore,
        safety_store: SafetyStore | None,
        roots: Iterable[str | Path],
        working_path: str | Path = ".",
        chat_handler: Optional[Callable[[str], str]] = None,
        file_ops_handler: Optional[Callable[[str, Optional[str], bool], tuple[str, list[Any]]]] = None,
    ) -> "AgentDeps":
        return cls(
            chat_client=chat_client,
            settings=settings,
            session_store=session_store,
            safety_store=safety_store,
            roots=tuple(str(Path(root).expanduser()) for root in roots),
            working_path=str(working_path or ".").strip() or ".",
            chat_handler=chat_handler,
            file_ops_handler=file_ops_handler,
        )
