"""SQLite-backed session checkpointing."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


SESSION_KINDS = {"chat", "documents", "research", "team"}


@dataclass(frozen=True)
class SessionSummary:
    id: str
    title: str
    kind: str
    created_at: float
    updated_at: float
    archived_at: Optional[float]
    pinned: bool
    message_count: int
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.id,
            "title": self.title,
            "kind": self.kind,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "archived": self.archived_at is not None,
            "archived_at": self.archived_at,
            "pinned": self.pinned,
            "message_count": self.message_count,
            "meta": self.meta,
        }


@dataclass(frozen=True)
class StoredMessage:
    role: str
    content: str
    metadata: dict[str, Any]
    created_at: float


class SessionStore:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._memory_conn: Optional[sqlite3.Connection] = None
        if str(db_path) == ":memory:":
            self._memory_conn = sqlite3.connect(":memory:")
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        if self._memory_conn is not None:
            return self._memory_conn
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(session_id) REFERENCES sessions(id)
                )
                """
            )
            columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
            for name, ddl in (
                ("title", "TEXT"),
                ("kind", "TEXT NOT NULL DEFAULT 'chat'"),
                ("archived_at", "REAL"),
                ("pinned", "INTEGER NOT NULL DEFAULT 0"),
                ("meta_json", "TEXT NOT NULL DEFAULT '{}'"),
            ):
                if name not in columns:
                    conn.execute(f"ALTER TABLE sessions ADD COLUMN {name} {ddl}")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, id)"
            )

    def create_session(
        self,
        session_id: Optional[str] = None,
        *,
        kind: str = "chat",
        title: Optional[str] = None,
        meta: Optional[dict[str, Any]] = None,
    ) -> str:
        if kind not in SESSION_KINDS:
            raise ValueError(f"Unknown session kind: {kind}")
        selected = session_id or str(uuid.uuid4())
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO sessions (id, created_at, updated_at, kind, title, meta_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (selected, now, now, kind, title, json.dumps(meta or {})),
            )
        return selected

    def session_exists(self, session_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return row is not None

    def get_session(self, session_id: str) -> SessionSummary:
        rows = self._summaries("WHERE s.id = ?", (session_id,))
        if not rows:
            raise KeyError(f"Session not found: {session_id}")
        return rows[0]

    def list_sessions(
        self,
        *,
        include_archived: bool = False,
        archived_only: bool = False,
        kind: Optional[str] = None,
        query: Optional[str] = None,
        limit: int = 200,
    ) -> list[SessionSummary]:
        clauses: list[str] = []
        params: list[Any] = []
        if archived_only:
            clauses.append("s.archived_at IS NOT NULL")
        elif not include_archived:
            clauses.append("s.archived_at IS NULL")
        if kind:
            clauses.append("s.kind = ?")
            params.append(kind)
        if query:
            clauses.append(
                "(s.title LIKE ? OR EXISTS (SELECT 1 FROM messages m2 WHERE m2.session_id = s.id "
                "AND m2.content LIKE ?))"
            )
            params.extend([f"%{query}%", f"%{query}%"])
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        return self._summaries(where, tuple(params), limit=limit)

    def update_session(
        self,
        session_id: str,
        *,
        title: Optional[str] = None,
        archived: Optional[bool] = None,
        pinned: Optional[bool] = None,
        meta: Optional[dict[str, Any]] = None,
    ) -> SessionSummary:
        current = self.get_session(session_id)
        assignments: list[str] = []
        params: list[Any] = []
        if title is not None:
            assignments.append("title = ?")
            params.append(title.strip()[:200] or None)
        if archived is not None:
            assignments.append("archived_at = ?")
            params.append(time.time() if archived else None)
        if pinned is not None:
            assignments.append("pinned = ?")
            params.append(1 if pinned else 0)
        if meta is not None:
            assignments.append("meta_json = ?")
            params.append(json.dumps({**current.meta, **meta}))
        if assignments:
            with self._connect() as conn:
                conn.execute(
                    f"UPDATE sessions SET {', '.join(assignments)} WHERE id = ?",
                    (*params, session_id),
                )
        return self.get_session(session_id)

    def delete_session(self, session_id: str) -> bool:
        """Permanently delete a session and its messages. Returns False if it did not exist."""

        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            deleted = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,)).rowcount
        return bool(deleted)

    def recent_messages(self, session_id: str, limit: int = 20) -> list[StoredMessage]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, metadata_json, created_at FROM messages
                WHERE session_id = ? ORDER BY id DESC LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        return [
            StoredMessage(role=r[0], content=r[1], metadata=json.loads(r[2]), created_at=r[3])
            for r in reversed(rows)
        ]

    def _summaries(
        self,
        where: str,
        params: tuple[Any, ...],
        limit: int = 200,
    ) -> list[SessionSummary]:
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT s.id, s.title, s.kind, s.created_at, s.updated_at, s.archived_at, s.pinned,
                       s.meta_json,
                       (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id),
                       (SELECT content FROM messages m WHERE m.session_id = s.id AND m.role = 'user'
                        ORDER BY m.id ASC LIMIT 1)
                FROM sessions s {where}
                ORDER BY s.pinned DESC, s.updated_at DESC
                LIMIT ?
                """,
                (*params, max(1, min(int(limit), 1000))),
            ).fetchall()
        return [
            SessionSummary(
                id=row[0],
                title=row[1] or _default_title(row[9]),
                kind=row[2] or "chat",
                created_at=row[3],
                updated_at=row[4],
                archived_at=row[5],
                pinned=bool(row[6]),
                meta=json.loads(row[7] or "{}"),
                message_count=row[8],
            )
            for row in rows
        ]

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.create_session(session_id)
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages (session_id, role, content, metadata_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (session_id, role, content, json.dumps(metadata or {}), now),
            )
            conn.execute("UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id))

    def get_messages(self, session_id: str) -> list[StoredMessage]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, metadata_json, created_at
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
                """,
                (session_id,),
            ).fetchall()
        return [
            StoredMessage(
                role=row[0],
                content=row[1],
                metadata=json.loads(row[2]),
                created_at=row[3],
            )
            for row in rows
        ]


def _default_title(first_user_message: Optional[str]) -> str:
    text = " ".join((first_user_message or "").split())
    if not text:
        return "New conversation"
    return text if len(text) <= 60 else text[:57].rstrip() + "..."
