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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_bindings (
                    root_id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL,
                    credential_ref TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    revoked_at REAL,
                    FOREIGN KEY(root_id) REFERENCES approved_roots(id)
                )
                """
            )
            binding_columns = {
                row["name"] for row in conn.execute("PRAGMA table_info(knowledge_bindings)").fetchall()
            }
            if "ingest_credential_ref" not in binding_columns:
                conn.execute("ALTER TABLE knowledge_bindings ADD COLUMN ingest_credential_ref TEXT")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_ingest_submissions (
                    id TEXT PRIMARY KEY,
                    root_id TEXT NOT NULL,
                    workspace_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    approved_at REAL,
                    FOREIGN KEY(root_id) REFERENCES approved_roots(id)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_ingest_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    submission_id TEXT NOT NULL,
                    rel_path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    sha256 TEXT NOT NULL,
                    status TEXT NOT NULL,
                    asset_id TEXT,
                    document_id TEXT,
                    chunks_indexed INTEGER,
                    error TEXT,
                    updated_at REAL NOT NULL,
                    UNIQUE(submission_id, rel_path),
                    FOREIGN KEY(submission_id) REFERENCES knowledge_ingest_submissions(id)
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

    def revoke_root(self, root_id: str) -> ApprovedRoot:
        """Revoke a root and every knowledge binding attached to it.

        Bindings stay revoked if the root is later re-approved: indexed content
        must be re-bound explicitly rather than silently becoming reachable again.
        """

        now = time.time()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM approved_roots WHERE id = ? AND revoked_at IS NULL",
                (root_id,),
            ).fetchone()
            if row is None:
                raise PermissionError(f"Root is not approved: {root_id}")
            conn.execute(
                "UPDATE approved_roots SET revoked_at = ?, updated_at = ? WHERE id = ?",
                (now, now, root_id),
            )
            conn.execute(
                """
                UPDATE knowledge_bindings SET revoked_at = ?, updated_at = ?
                WHERE root_id = ? AND revoked_at IS NULL
                """,
                (now, now, root_id),
            )
            self._insert_audit(
                conn,
                event_type="root.revoked",
                root_id=root_id,
                path=row["path"],
                metadata={"name": row["name"]},
                created_at=now,
            )
            row = conn.execute("SELECT * FROM approved_roots WHERE id = ?", (root_id,)).fetchone()
        return self._approved_root_from_row(row)

    def bind_knowledge(self, root_id: str, workspace_id: str, credential_ref: str) -> dict[str, Any]:
        """Associate an active root with one retrieval-backend workspace."""

        root = self.get_approved_root(root_id)
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO knowledge_bindings
                    (root_id, workspace_id, credential_ref, created_at, updated_at, revoked_at)
                VALUES (?, ?, ?, ?, ?, NULL)
                ON CONFLICT(root_id) DO UPDATE SET
                    workspace_id = excluded.workspace_id,
                    credential_ref = excluded.credential_ref,
                    updated_at = excluded.updated_at,
                    revoked_at = NULL
                """,
                (root.id, workspace_id, credential_ref, now, now),
            )
            self._insert_audit(
                conn,
                event_type="knowledge.bound",
                root_id=root.id,
                path=str(root.path),
                metadata={"workspace_id": workspace_id, "credential_ref": credential_ref},
                created_at=now,
            )
        return self.get_knowledge_binding(root.id)

    def unbind_knowledge(self, root_id: str) -> None:
        now = time.time()
        with self._connect() as conn:
            updated = conn.execute(
                """
                UPDATE knowledge_bindings SET revoked_at = ?, updated_at = ?
                WHERE root_id = ? AND revoked_at IS NULL
                """,
                (now, now, root_id),
            ).rowcount
            if not updated:
                raise KeyError(f"No active knowledge binding for root: {root_id}")
            self._insert_audit(
                conn,
                event_type="knowledge.unbound",
                root_id=root_id,
                metadata={},
                created_at=now,
            )

    def get_knowledge_binding(self, root_id: str) -> dict[str, Any]:
        """Return the active binding for an active root, or raise PermissionError.

        Called before every retrieval and source-open so revocation is immediate.
        """

        root = self.get_approved_root(root_id)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM knowledge_bindings WHERE root_id = ? AND revoked_at IS NULL",
                (root.id,),
            ).fetchone()
        if row is None:
            raise PermissionError(f"Root has no active knowledge binding: {root_id}")
        return {
            "root_id": row["root_id"],
            "root_path": str(root.path),
            "root_name": root.name,
            "workspace_id": row["workspace_id"],
            "credential_ref": row["credential_ref"],
            "ingest_credential_ref": row["ingest_credential_ref"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def set_ingest_credential(self, root_id: str, credential_ref: Optional[str]) -> dict[str, Any]:
        binding = self.get_knowledge_binding(root_id)
        now = time.time()
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE knowledge_bindings SET ingest_credential_ref = ?, updated_at = ?
                WHERE root_id = ? AND revoked_at IS NULL
                """,
                (credential_ref, now, root_id),
            )
            self._insert_audit(
                conn,
                event_type="knowledge.ingest_enabled" if credential_ref else "knowledge.ingest_disabled",
                root_id=root_id,
                metadata={"workspace_id": binding["workspace_id"], "credential_ref": credential_ref},
                created_at=now,
            )
        return self.get_knowledge_binding(root_id)

    def create_ingest_submission(
        self,
        root_id: str,
        workspace_id: str,
        files: Iterable[dict[str, Any]],
    ) -> str:
        submission_id = str(uuid.uuid4())
        now = time.time()
        files = list(files)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO knowledge_ingest_submissions
                    (id, root_id, workspace_id, status, created_at, updated_at, approved_at)
                VALUES (?, ?, ?, 'awaiting_approval', ?, ?, NULL)
                """,
                (submission_id, root_id, workspace_id, now, now),
            )
            for item in files:
                conn.execute(
                    """
                    INSERT INTO knowledge_ingest_files
                        (submission_id, rel_path, size_bytes, sha256, status, updated_at)
                    VALUES (?, ?, ?, ?, 'awaiting_approval', ?)
                    """,
                    (submission_id, item["rel_path"], int(item["size_bytes"]), item["sha256"], now),
                )
            self._insert_audit(
                conn,
                event_type="knowledge.ingest_previewed",
                root_id=root_id,
                metadata={
                    "submission_id": submission_id,
                    "workspace_id": workspace_id,
                    "files": len(files),
                },
                created_at=now,
            )
        return submission_id

    def get_ingest_submission(self, submission_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM knowledge_ingest_submissions WHERE id = ?",
                (submission_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Ingest submission not found: {submission_id}")
            files = conn.execute(
                "SELECT * FROM knowledge_ingest_files WHERE submission_id = ? ORDER BY id ASC",
                (submission_id,),
            ).fetchall()
        return {
            "submission_id": row["id"],
            "root_id": row["root_id"],
            "workspace_id": row["workspace_id"],
            "status": row["status"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "approved_at": row["approved_at"],
            "files": [
                {
                    "id": item["id"],
                    "rel_path": item["rel_path"],
                    "size_bytes": item["size_bytes"],
                    "sha256": item["sha256"],
                    "status": item["status"],
                    "asset_id": item["asset_id"],
                    "document_id": item["document_id"],
                    "chunks_indexed": item["chunks_indexed"],
                    "error": item["error"],
                    "updated_at": item["updated_at"],
                }
                for item in files
            ],
        }

    def update_ingest_submission_status(self, submission_id: str, status: str) -> None:
        now = time.time()
        with self._connect() as conn:
            row = conn.execute(
                "SELECT root_id FROM knowledge_ingest_submissions WHERE id = ?",
                (submission_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Ingest submission not found: {submission_id}")
            approved_at = now if status == "submitting" else None
            conn.execute(
                """
                UPDATE knowledge_ingest_submissions
                SET status = ?, updated_at = ?, approved_at = COALESCE(approved_at, ?)
                WHERE id = ?
                """,
                (status, now, approved_at, submission_id),
            )
            self._insert_audit(
                conn,
                event_type=f"knowledge.ingest_{status}",
                root_id=row["root_id"],
                metadata={"submission_id": submission_id},
                created_at=now,
            )

    def update_ingest_file(self, file_id: int, **fields: Any) -> None:
        allowed = {"status", "asset_id", "document_id", "chunks_indexed", "error"}
        unknown = set(fields) - allowed
        if unknown:
            raise ValueError(f"Unknown ingest file fields: {sorted(unknown)}")
        if not fields:
            return
        assignments = ", ".join(f"{name} = ?" for name in fields)
        with self._connect() as conn:
            conn.execute(
                f"UPDATE knowledge_ingest_files SET {assignments}, updated_at = ? WHERE id = ?",
                (*fields.values(), time.time(), file_id),
            )

    def list_knowledge_bindings(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT b.root_id FROM knowledge_bindings b
                JOIN approved_roots r ON r.id = b.root_id
                WHERE b.revoked_at IS NULL AND r.revoked_at IS NULL
                ORDER BY b.updated_at DESC
                """
            ).fetchall()
        return [self.get_knowledge_binding(row["root_id"]) for row in rows]

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
