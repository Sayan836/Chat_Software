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


def delete_thread_checkpoints(thread_id: str) -> int:
    """Remove persisted graph checkpoints for one conversation thread (session delete)."""

    path = graph_checkpoint_path()
    if str(state_dir()) == ":memory:" or not path.exists():
        return 0
    removed = 0
    conn = sqlite3.connect(str(path))
    try:
        tables = [row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")]
        for table in tables:
            columns = {row[1] for row in conn.execute(f'PRAGMA table_info("{table}")')}
            if "thread_id" in columns:
                removed += conn.execute(f'DELETE FROM "{table}" WHERE thread_id = ?', (thread_id,)).rowcount
        conn.commit()
    finally:
        conn.close()
    return removed
