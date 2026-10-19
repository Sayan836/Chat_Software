"""Conversation lifecycle: list, search, create, rename, pin, archive/restore and permanent delete."""

from __future__ import annotations

from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from chat_software.agent.graph_runtime import delete_thread_checkpoints
from chat_software.agent.session_store import SESSION_KINDS, SessionStore
from chat_software.runtime_log import record_runtime_event


class SessionCreateRequest(BaseModel):
    kind: str = "chat"
    title: Optional[str] = Field(default=None, max_length=200)
    meta: dict[str, Any] = Field(default_factory=dict)


class SessionUpdateRequest(BaseModel):
    title: Optional[str] = Field(default=None, max_length=200)
    archived: Optional[bool] = None
    pinned: Optional[bool] = None


def build_router(store: SessionStore) -> APIRouter:
    router = APIRouter()

    @router.get("/sessions")
    def list_sessions(
        archived: str = "active",
        kind: Optional[str] = None,
        q: Optional[str] = None,
        limit: int = 200,
    ) -> dict[str, object]:
        if archived not in {"active", "archived", "all"}:
            raise HTTPException(status_code=400, detail="archived must be active, archived or all")
        if kind and kind not in SESSION_KINDS:
            raise HTTPException(status_code=400, detail=f"Unknown kind: {kind}")
        rows = store.list_sessions(
            include_archived=archived == "all",
            archived_only=archived == "archived",
            kind=kind,
            query=(q or "").strip()[:200] or None,
            limit=limit,
        )
        return {"sessions": [row.to_dict() for row in rows]}

    @router.post("/sessions")
    def create_session(request: SessionCreateRequest) -> dict[str, object]:
        try:
            session_id = store.create_session(kind=request.kind, title=request.title, meta=request.meta)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return store.get_session(session_id).to_dict()

    @router.patch("/sessions/{session_id}")
    def update_session(session_id: str, request: SessionUpdateRequest) -> dict[str, object]:
        try:
            summary = store.update_session(
                session_id, title=request.title, archived=request.archived, pinned=request.pinned
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Session not found") from exc
        return summary.to_dict()

    @router.delete("/sessions/{session_id}")
    def delete_session(session_id: str) -> dict[str, object]:
        if not store.delete_session(session_id):
            raise HTTPException(status_code=404, detail="Session not found")
        checkpoints = delete_thread_checkpoints(session_id)
        record_runtime_event({"type": "session_deleted", "session_id": session_id})
        return {"deleted": True, "session_id": session_id, "checkpoints_removed": checkpoints}

    return router
