"""SQLite-backed safety state for filesystem authority and write auditing."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


PLAN_STATUSES = {"planned", "applying", "applied", "partial", "undone", "failed"}
ACTION_STATUSES = {"planned", "running", "applied", "undone", "skipped", "failed"}
JOURNAL_STATUSES = {"started", "completed", "undone", "failed"}


@dataclass(frozen=True)
class ApprovedRoot:
    id: str
    path: Path
    name: str
    created_at: float
    updated_at: float
    revoked_at: Optional[float] = None


class SafetyStore:
    """Authoritative server-side state for filesystem permissions and journals."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._memory_conn: Optional[sqlite3.Connection] = None
        if str(db_path) == ":memory:":
            self._memory_conn = self._new_connection(":memory:")
        else:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _new_connection(self, db_path: str | Path) -> sqlite3.Connection:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    def _connect(self) -> sqlite3.Connection:
        if self._memory_conn is not None:
            return self._memory_conn
        return self._new_connection(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS approved_roots (
                    id TEXT PRIMARY KEY,
                    path TEXT NOT NULL UNIQUE,
                    name TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    revoked_at REAL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS organize_plans (
                    id TEXT PRIMARY KEY,
                    root_id TEXT NOT NULL,
                    strategy TEXT NOT NULL,
                    prompt TEXT,
                    status TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    applied_at REAL,
                    undone_at REAL,
                    FOREIGN KEY(root_id) REFERENCES approved_roots(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS organize_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plan_id TEXT NOT NULL,
                    action_index INTEGER NOT NULL,
                    op TEXT NOT NULL,
                    from_path TEXT,
                    to_path TEXT,
                    path TEXT,
                    reason TEXT,
                    status TEXT NOT NULL DEFAULT 'planned',
                    action_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    UNIQUE(plan_id, action_index),
                    FOREIGN KEY(plan_id) REFERENCES organize_plans(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS operation_journal (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT NOT NULL,
                    plan_id TEXT,
                    action_id INTEGER,
                    op TEXT NOT NULL,
                    forward_json TEXT NOT NULL,
                    reverse_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    completed_at REAL,
                    error TEXT,
                    FOREIGN KEY(plan_id) REFERENCES organize_plans(id),
                    FOREIGN KEY(action_id) REFERENCES organize_actions(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    actor TEXT NOT NULL DEFAULT 'system',
                    root_id TEXT,
                    plan_id TEXT,
                    path TEXT,
                    metadata_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(root_id) REFERENCES approved_roots(id),
                    FOREIGN KEY(plan_id) REFERENCES organize_plans(id)
                )
                """
            )

    def approve_root(self, requested_path: str | Path) -> ApprovedRoot:
        root = Path(requested_path).expanduser().resolve()
        if not root.exists():
            raise FileNotFoundError(f"Root does not exist: {root}")
        if not root.is_dir():
            raise NotADirectoryError(f"Root is not a directory: {root}")

        now = time.time()
        name = root.name or str(root)
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM approved_roots WHERE path = ?",
                (str(root),),
            ).fetchone()
            if existing is None:
                root_id = str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO approved_roots (id, path, name, created_at, updated_at, revoked_at)
                    VALUES (?, ?, ?, ?, ?, NULL)
                    """,
                    (root_id, str(root), name, now, now),
                )
                self._insert_audit(
                    conn,
                    event_type="root.approved",
                    root_id=root_id,
                    path=str(root),
                    metadata={"name": name},
                    created_at=now,
                )
                row = conn.execute(
                    "SELECT * FROM approved_roots WHERE id = ?",
                    (root_id,),
                ).fetchone()
            else:
                conn.execute(
                    """
                    UPDATE approved_roots
                    SET name = ?, updated_at = ?, revoked_at = NULL
                    WHERE id = ?
                    """,
                    (name, now, existing["id"]),
                )
                self._insert_audit(
                    conn,
                    event_type="root.reapproved",
                    root_id=existing["id"],
                    path=str(root),
                    metadata={"name": name},
                    created_at=now,
                )
                row = conn.execute(
                    "SELECT * FROM approved_roots WHERE id = ?",
                    (existing["id"],),
                ).fetchone()

        if row is None:
            raise RuntimeError("Approved root was not persisted.")
        return self._approved_root_from_row(row)

    def list_approved_roots(self, active_only: bool = True) -> list[ApprovedRoot]:
        query = "SELECT * FROM approved_roots"
        params: tuple[object, ...] = ()
        if active_only:
            query += " WHERE revoked_at IS NULL"
        query += " ORDER BY updated_at DESC, path ASC"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._approved_root_from_row(row) for row in rows]

    def get_approved_root(self, root_id: str) -> ApprovedRoot:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM approved_roots WHERE id = ? AND revoked_at IS NULL",
                (root_id,),
            ).fetchone()
        if row is None:
            raise PermissionError(f"Root is not approved: {root_id}")
        return self._approved_root_from_row(row)

    def authorize_roots(self, roots: Iterable[str | Path]) -> list[ApprovedRoot]:
        approved_by_path = {str(root.path): root for root in self.list_approved_roots(active_only=True)}
        authorized: list[ApprovedRoot] = []
        for requested in roots:
            resolved = Path(requested).expanduser().resolve()
            approved = approved_by_path.get(str(resolved))
            if approved is None:
                raise PermissionError(f"Root is not approved by this server: {resolved}")
            authorized.append(approved)
        if not authorized:
            raise PermissionError("At least one server-approved root is required.")
        return authorized

    def create_organize_plan(
        self,
        *,
        root_id: str,
        strategy: str,
        plan: dict[str, Any],
        prompt: Optional[str] = None,
        plan_id: Optional[str] = None,
        status: str = "planned",
    ) -> str:
        if status not in PLAN_STATUSES:
            raise ValueError(f"Unsupported plan status: {status}")
        root = self.get_approved_root(root_id)
        selected_plan_id = plan_id or str(uuid.uuid4())
        now = time.time()
        stored_plan = {
            **plan,
            "plan_id": selected_plan_id,
            "root_id": root.id,
            "status": status,
        }
        actions = list(stored_plan.get("actions", []))
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO organize_plans
                    (id, root_id, strategy, prompt, status, plan_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    selected_plan_id,
                    root.id,
                    strategy,
                    prompt,
                    status,
                    json.dumps(stored_plan, sort_keys=True),
                    now,
                    now,
                ),
            )
            for index, action in enumerate(actions):
                conn.execute(
                    """
                    INSERT INTO organize_actions
                        (plan_id, action_index, op, from_path, to_path, path, reason,
                         status, action_json, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, 'planned', ?, ?, ?)
                    """,
                    (
                        selected_plan_id,
                        index,
                        str(action.get("op") or ""),
                        action.get("from"),
                        action.get("to"),
                        action.get("path"),
                        action.get("reason"),
                        json.dumps(action, sort_keys=True),
                        now,
                        now,
                    ),
                )
            self._insert_audit(
                conn,
                event_type="organize.plan_created",
                root_id=root.id,
                plan_id=selected_plan_id,
                path=str(root.path),
                metadata={"strategy": strategy, "actions": len(actions)},
                created_at=now,
            )
        return selected_plan_id

    def get_organize_plan(self, plan_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT plans.*, roots.path AS root_path, roots.name AS root_name
                FROM organize_plans AS plans
                JOIN approved_roots AS roots ON roots.id = plans.root_id
                WHERE plans.id = ?
                """,
                (plan_id,),
            ).fetchone()
        if row is None:
            raise KeyError(f"Unknown organize plan: {plan_id}")

        plan = json.loads(row["plan_json"])
        plan.update(
            {
                "plan_id": row["id"],
                "root_id": row["root_id"],
                "root": row["root_path"],
                "root_name": row["root_name"],
                "strategy": row["strategy"],
                "status": row["status"],
                "created_at": plan.get("created_at") or row["created_at"],
                "updated_at": row["updated_at"],
                "applied_at": row["applied_at"],
                "undone_at": row["undone_at"],
            }
        )
        action_rows = self.get_organize_actions(plan_id)
        if action_rows:
            plan["actions"] = [
                {
                    **action_row["action"],
                    "action_id": action_row["id"],
                    "action_index": action_row["action_index"],
                    "status": action_row["status"],
                }
                for action_row in action_rows
            ]
        return plan

    def get_organize_actions(self, plan_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM organize_actions
                WHERE plan_id = ?
                ORDER BY action_index ASC
                """,
                (plan_id,),
            ).fetchall()
        return [self._action_from_row(row) for row in rows]

    def update_plan_status(self, plan_id: str, status: str) -> None:
        if status not in PLAN_STATUSES:
            raise ValueError(f"Unsupported plan status: {status}")
        now = time.time()
        applied_at = now if status == "applied" else None
        undone_at = now if status == "undone" else None
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status, applied_at, undone_at FROM organize_plans WHERE id = ?",
                (plan_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown organize plan: {plan_id}")
            conn.execute(
                """
                UPDATE organize_plans
                SET status = ?,
                    updated_at = ?,
                    applied_at = COALESCE(?, applied_at),
                    undone_at = COALESCE(?, undone_at)
                WHERE id = ?
                """,
                (status, now, applied_at, undone_at, plan_id),
            )

    def update_action_status(self, action_id: int, status: str) -> None:
        if status not in ACTION_STATUSES:
            raise ValueError(f"Unsupported action status: {status}")
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                "UPDATE organize_actions SET status = ?, updated_at = ? WHERE id = ?",
                (status, now, action_id),
            )

    def start_journal_entry(
        self,
        *,
        batch_id: str,
        plan_id: str,
        action_id: Optional[int],
        op: str,
        forward: dict[str, Any],
        reverse: dict[str, Any],
    ) -> int:
        now = time.time()
        with self._connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO operation_journal
                    (batch_id, plan_id, action_id, op, forward_json, reverse_json, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, 'started', ?)
                """,
                (
                    batch_id,
                    plan_id,
                    action_id,
                    op,
                    json.dumps(forward, sort_keys=True),
                    json.dumps(reverse, sort_keys=True),
                    now,
                ),
            )
            return int(cursor.lastrowid)

    def complete_journal_entry(
        self,
        journal_id: int,
        *,
        status: str = "completed",
        reverse: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        if status not in JOURNAL_STATUSES:
            raise ValueError(f"Unsupported journal status: {status}")
        now = time.time()
        with self._connect() as conn:
            if reverse is not None:
                conn.execute(
                    """
                    UPDATE operation_journal
                    SET status = ?, reverse_json = ?, completed_at = ?, error = ?
                    WHERE id = ?
                    """,
                    (status, json.dumps(reverse, sort_keys=True), now, error, journal_id),
                )
            else:
                conn.execute(
                    """
                    UPDATE operation_journal
                    SET status = ?, completed_at = ?, error = ?
                    WHERE id = ?
                    """,
                    (status, now, error, journal_id),
                )

    def get_journal_entries(self, plan_id: str, reverse_order: bool = False) -> list[dict[str, Any]]:
        order = "DESC" if reverse_order else "ASC"
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT *
                FROM operation_journal
                WHERE plan_id = ?
                ORDER BY id {order}
                """,
                (plan_id,),
            ).fetchall()
        return [self._journal_from_row(row) for row in rows]

    def record_audit(
        self,
        event_type: str,
        *,
        actor: str = "system",
        root_id: Optional[str] = None,
        plan_id: Optional[str] = None,
        path: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        now = time.time()
        with self._connect() as conn:
            self._insert_audit(
                conn,
                event_type=event_type,
                actor=actor,
                root_id=root_id,
                plan_id=plan_id,
                path=path,
                metadata=metadata or {},
                created_at=now,
            )

    def table_names(self) -> set[str]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        return {row["name"] for row in rows}

    def _insert_audit(
        self,
        conn: sqlite3.Connection,
        *,
        event_type: str,
        actor: str = "system",
        root_id: Optional[str] = None,
        plan_id: Optional[str] = None,
        path: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
        created_at: Optional[float] = None,
    ) -> None:
        conn.execute(
            """
            INSERT INTO audit_log
                (event_type, actor, root_id, plan_id, path, metadata_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                event_type,
                actor,
                root_id,
                plan_id,
                path,
                json.dumps(metadata or {}, sort_keys=True),
                created_at or time.time(),
            ),
        )

    def _approved_root_from_row(self, row: sqlite3.Row) -> ApprovedRoot:
        return ApprovedRoot(
            id=row["id"],
            path=Path(row["path"]),
            name=row["name"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            revoked_at=row["revoked_at"],
        )

    def _action_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "plan_id": row["plan_id"],
            "action_index": row["action_index"],
            "op": row["op"],
            "from_path": row["from_path"],
            "to_path": row["to_path"],
            "path": row["path"],
            "reason": row["reason"],
            "status": row["status"],
            "action": json.loads(row["action_json"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def _journal_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "id": row["id"],
            "batch_id": row["batch_id"],
            "plan_id": row["plan_id"],
            "action_id": row["action_id"],
            "op": row["op"],
            "forward": json.loads(row["forward_json"]),
            "reverse": json.loads(row["reverse_json"]),
            "status": row["status"],
            "created_at": row["created_at"],
            "completed_at": row["completed_at"],
            "error": row["error"],
        }
