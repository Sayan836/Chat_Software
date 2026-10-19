"""Agent registry for the Phase 3 graph router."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .deps import AgentDeps
from .agents import build_chat_agent, build_file_ops_agent
from .routing import requires_scoped_tool


@dataclass(frozen=True)
class AgentSpec:
    name: str
    description: str
    matches: Callable[[str], bool]
    build: Callable[[AgentDeps], Any]
    requires_roots: bool = True
    writes: bool = False


AGENTS: dict[str, AgentSpec] = {}


def register(spec: AgentSpec) -> None:
    if spec.name in AGENTS:
        raise ValueError(f"Agent already registered: {spec.name}")
    AGENTS[spec.name] = spec


def register_default_agents() -> None:
    if "chat" not in AGENTS:
        register(
            AgentSpec(
                name="chat",
                description="Ordinary non-file conversation.",
                matches=lambda _message: True,
                build=build_chat_agent,
                requires_roots=False,
                writes=False,
            )
        )
    if "file_ops" not in AGENTS:
        register(
            AgentSpec(
                name="file_ops",
                description="Approved-root file, folder, git, and organize preview operations.",
                matches=lambda message: requires_scoped_tool(message.lower()),
                build=build_file_ops_agent,
                requires_roots=True,
                writes=True,
            )
        )
