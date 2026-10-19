"""Agent team endpoints: CRUD, templates, AI-drafted teams, task runs and team workspace chats."""

from __future__ import annotations

import json
import uuid
from typing import Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from chat_software.agent.session_store import SessionStore
from chat_software.agent.toolbox import strip_thinking
from chat_software.runtime_log import record_runtime_event
from chat_software.skills import KNOWN_TOOLS
from chat_software.teams import TEMPLATES, TeamStore, team_from_dict
from chat_software.teams.runtime import TeamDeps, run_discussion, run_task


class TeamBody(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(default="", max_length=500)
    lead: Optional[str] = None
    members: list[dict[str, Any]] = Field(min_length=1, max_length=8)
    max_turns: int = Field(default=6, ge=1, le=20)
    max_tool_steps: int = Field(default=4, ge=0, le=10)
    documents_root_id: Optional[str] = None


class TemplateRequest(BaseModel):
    template_id: str


class TeamDraftRequest(BaseModel):
    goal: str = Field(min_length=5, max_length=2000)
    provider: Optional[str] = None
    model: Optional[str] = None


class TaskRequest(BaseModel):
    task: str = Field(min_length=1, max_length=16_000)
    provider: Optional[str] = None
    model: Optional[str] = None


class TeamSessionRequest(BaseModel):
    title: Optional[str] = Field(default=None, max_length=200)
    provider: Optional[str] = None
    model: Optional[str] = None


class TeamMessageRequest(BaseModel):
    message: str = Field(min_length=1, max_length=16_000)
    max_turns: Optional[int] = Field(default=None, ge=1, le=20)


DRAFT_PROMPT = (
    "Design a small team of AI agents (2-5 members) for the user's goal. Reply with ONLY JSON: "
    '{"name": "...", "description": "one sentence", "lead": "<member id>", "members": [{"id": '
    '"lowercase-slug", "name": "Title Case", "role": "what they do", "instructions": "2-4 concrete '
    'sentences", "tools": [], "skills": []}]}. Allowed tools: ' + ", ".join(KNOWN_TOOLS) +
    ". Only give a member tools it needs. Allowed skills: {skills}."
)


def build_router(team_store: TeamStore, sessions: SessionStore, deps: TeamDeps, safety: Any) -> APIRouter:
    router = APIRouter()

    def known_skills() -> list[str]:
        return [skill.id for skill in deps.skills.list()]

    def validate(body: dict[str, Any], team_id: Optional[str] = None):
        try:
            team = team_from_dict(body, team_id=team_id, known_skills=known_skills())
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if team.documents_root_id:
            try:
                safety.get_knowledge_binding(team.documents_root_id)
            except PermissionError as exc:
                raise HTTPException(status_code=403, detail=str(exc)) from exc
        return team

    def get_team(team_id: str):
        try:
            return team_store.get(team_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Team not found") from exc

    def team_deps(provider: Optional[str], model: Optional[str]) -> TeamDeps:
        return TeamDeps(client_factory=deps.client_factory, skills=deps.skills, sandbox=deps.sandbox,
                        knowledge=deps.knowledge, default_provider=provider, default_model=model)

    @router.get("/teams/templates")
    def templates() -> dict[str, object]:
        return {"templates": [{"id": key, **value} for key, value in TEMPLATES.items()]}

    @router.get("/teams")
    def list_teams() -> dict[str, object]:
        return {"teams": [team.to_dict() for team in team_store.list()]}

    @router.post("/teams")
    def create_team(body: TeamBody) -> dict[str, object]:
        return team_store.save(validate(body.model_dump())).to_dict()

    @router.post("/teams/from-template")
    def from_template(request: TemplateRequest) -> dict[str, object]:
        if request.template_id not in TEMPLATES:
            raise HTTPException(status_code=404, detail="Template not found")
        return team_store.save(validate(json.loads(json.dumps(TEMPLATES[request.template_id])))).to_dict()

    @router.post("/teams/draft")
    def draft_team(request: TeamDraftRequest) -> dict[str, object]:
        client = deps.client_factory(provider=request.provider, requested_model=request.model)
        raw = strip_thinking(client.complete([
            {"role": "system", "content": DRAFT_PROMPT.replace("{skills}", ", ".join(known_skills()))},
            {"role": "user", "content": request.goal},
        ]))
        from chat_software.teams.runtime import _json_object

        draft = _json_object(raw)
        for member in draft.get("members") or []:
            if isinstance(member, dict):
                member["tools"] = [t for t in member.get("tools") or [] if t in KNOWN_TOOLS]
                member["skills"] = [s for s in member.get("skills") or [] if s in known_skills()]
        try:
            team = team_from_dict(draft, known_skills=known_skills())
        except ValueError as exc:
            raise HTTPException(status_code=502, detail=f"The model's team draft was not usable: {exc}") from exc
        return {"draft": team.to_dict(), "saved": False}

    @router.get("/teams/{team_id}")
    def read_team(team_id: str) -> dict[str, object]:
        return get_team(team_id).to_dict()

    @router.put("/teams/{team_id}")
    def update_team(team_id: str, body: TeamBody) -> dict[str, object]:
        get_team(team_id)
        return team_store.save(validate(body.model_dump(), team_id=team_id)).to_dict()

    @router.delete("/teams/{team_id}")
    def delete_team(team_id: str) -> dict[str, object]:
        if not team_store.delete(team_id):
            raise HTTPException(status_code=404, detail="Team not found")
        return {"deleted": True, "team_id": team_id}

    @router.post("/teams/{team_id}/run/stream")
    def run_team_task(team_id: str, request: TaskRequest) -> StreamingResponse:
        team = get_team(team_id)
        session_id = sessions.create_session(kind="team", title=f"{team.name}: {request.task[:60]}",
                                             meta={"team_id": team.id, "mode": "task"})
        run_deps = team_deps(request.provider, request.model)

        def events():
            sessions.append_message(session_id, "user", request.task, metadata={"team_id": team.id})
            yield _line({"type": "session", "session_id": session_id, "team_id": team.id})
            try:
                for event in run_task(team, request.task, run_deps):
                    _persist(sessions, session_id, event)
                    yield _line({**event, "session_id": session_id})
            except Exception as exc:
                yield _line({"type": "team_failed", "session_id": session_id,
                             "error": f"{exc.__class__.__name__}: {str(exc)[:300]}"})
            record_runtime_event({"type": "team_task_finished", "team_id": team.id, "session_id": session_id})

        return StreamingResponse(events(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache"})

    @router.post("/teams/{team_id}/sessions")
    def open_team_session(team_id: str, request: TeamSessionRequest) -> dict[str, object]:
        team = get_team(team_id)
        session_id = sessions.create_session(
            kind="team", title=request.title or f"{team.name} workspace",
            meta={"team_id": team.id, "mode": "discussion", "provider": request.provider,
                  "model": request.model, "sandbox_workspace_id": deps.sandbox.create_workspace()
                  if deps.sandbox is not None else None},
        )
        return sessions.get_session(session_id).to_dict()

    @router.post("/teams/sessions/{session_id}/messages/stream")
    def team_message(session_id: str, request: TeamMessageRequest) -> StreamingResponse:
        try:
            summary = sessions.get_session(session_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Session not found") from exc
        if summary.kind != "team" or not summary.meta.get("team_id"):
            raise HTTPException(status_code=400, detail="Not a team workspace session")
        if summary.archived_at is not None:
            raise HTTPException(status_code=409, detail="Restore the session before continuing it")
        team = get_team(summary.meta["team_id"])
        history = [
            {"speaker": (m.metadata or {}).get("agent") or ("User" if m.role == "user" else "Assistant"),
             "content": m.content}
            for m in sessions.recent_messages(session_id, limit=60)
            if m.role in {"user", "assistant"}
        ][-40:]
        run_deps = team_deps(summary.meta.get("provider"), summary.meta.get("model"))
        workspace = summary.meta.get("sandbox_workspace_id")

        def events():
            sessions.append_message(session_id, "user", request.message, metadata={"team_id": team.id})
            yield _line({"type": "session", "session_id": session_id, "team_id": team.id})
            try:
                for event in run_discussion(team, history, request.message, run_deps,
                                            max_turns=request.max_turns, sandbox_workspace_id=workspace):
                    _persist(sessions, session_id, event)
                    yield _line({**event, "session_id": session_id})
            except Exception as exc:
                yield _line({"type": "team_failed", "session_id": session_id,
                             "error": f"{exc.__class__.__name__}: {str(exc)[:300]}"})

        return StreamingResponse(events(), media_type="application/x-ndjson", headers={"Cache-Control": "no-cache"})

    return router


def _persist(sessions: SessionStore, session_id: str, event: dict[str, Any]) -> None:
    if event.get("type") == "agent_message":
        sessions.append_message(session_id, "assistant", event["content"], metadata={
            "agent": event["agent"], "member_id": event["member_id"], "final": bool(event.get("final")),
            "tool_steps": event.get("tool_steps", []), "sources": event.get("sources", []),
            "provider": event.get("provider"), "model": event.get("model"),
        })
    elif event.get("type") in {"plan", "moderator"}:
        sessions.append_message(session_id, "event", json.dumps({k: v for k, v in event.items() if k != "type"}),
                                metadata={"event": event["type"], "agent": event.get("agent")})


def _line(event: dict[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False, default=str) + "\n"
