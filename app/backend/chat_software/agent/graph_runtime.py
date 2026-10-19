"""LangGraph runtime helpers for Phase 3 orchestration."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from chat_software.paths import ensure_state_dir, state_dir


GRAPH_CHECKPOINT_DB = "graph_checkpoints.sqlite3"


def graph_checkpoint_path() -> Path:
    return state_dir() / GRAPH_CHECKPOINT_DB


def build_checkpointer() -> Any:
    """Build the Phase 3 checkpointer without coupling it to safety/session data."""

    if str(state_dir()) == ":memory:":
        try:
            from langgraph.checkpoint.memory import InMemorySaver
        except ImportError as exc:
            raise RuntimeError("Install langgraph to use the in-memory graph checkpointer.") from exc

        return InMemorySaver()

    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        raise RuntimeError(
            "Install langgraph-checkpoint-sqlite>=3.1 to use persistent graph checkpoints."
        ) from exc

    ensure_state_dir()
    conn = sqlite3.connect(str(graph_checkpoint_path()), check_same_thread=False)
    saver = SqliteSaver(conn)
    setattr(saver, "_chat_software_conn", conn)
    return saver
