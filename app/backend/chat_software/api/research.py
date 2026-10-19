"""Deep research endpoint: streams plan/search/read/notes/report events and saves a research session."""

from __future__ import annotations

import json
import uuid
from typing import Any, Callable, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from chat_software.agent.session_store import SessionStore
from chat_software.research.engine import DEPTHS, ResearchConfig, ResearchError, run_research
from chat_software.research.web import configured_provider
from chat_software.runtime_log import record_runtime_event


class ResearchRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    depth: str = "standard"
    use_web: bool = True
    documents_root_id: Optional[str] = None
    session_id: Optional[str] = None
    provider: Optional[str] = None
    model: Optional[str] = None


def build_router(
    store: SessionStore,
    client_factory: Callable[..., Any],
    knowledge: Any,
    safety: Any,
) -> APIRouter:
    router = APIRouter()

    @router.get("/research/status")
    def status() -> dict[str, object]:
        return {"web_search_provider": configured_provider(), "depths": DEPTHS}

    @router.post("/research/stream")
    def research_stream(request: ResearchRequest) -> StreamingResponse:
        if request.depth not in DEPTHS:
            raise HTTPException(status_code=400, detail="depth must be quick, standard or deep")
        if request.documents_root_id:
            try:
                safety.get_knowledge_binding(request.documents_root_id)
            except PermissionError as exc:
                raise HTTPException(status_code=403, detail=str(exc)) from exc
        if request.use_web and configured_provider() is None:
            raise HTTPException(status_code=400, detail="Configure a web search provider (Settings) or turn off web.")
        if not request.use_web and not request.documents_root_id:
            raise HTTPException(status_code=400, detail="Choose web research, a document workspace, or both.")
        client = client_factory(provider=request.provider, requested_model=request.model)
        session_id = store.create_session(
            request.session_id, kind="research", title=request.question[:80],
            meta={"depth": request.depth, "web": request.use_web, "documents_root_id": request.documents_root_id},
        )
        request_id = str(uuid.uuid4())
        config = ResearchConfig(question=request.question, depth=request.depth, use_web=request.use_web,
                                documents_root_id=request.documents_root_id)

        def events():
            store.append_message(session_id, "user", request.question, metadata={"research": True})
            steps: list[dict[str, Any]] = []
            yield _line({"type": "session", "session_id": session_id, "request_id": request_id,
                         "provider": client.settings.provider, "model": client.settings.model})
            try:
                for event in run_research(config, client, knowledge=knowledge):
                    if event["type"] == "report":
                        store.append_message(session_id, "assistant", event["report"], metadata={
                            "research": {"sources": event["sources"], "plan": event["plan"],
                                         "notes": event["notes"], "steps": steps,
                                         "provider": client.settings.provider, "model": client.settings.model},
                        })
                        record_runtime_event({"type": "research_completed", "request_id": request_id,
                                              "session_id": session_id, "sources": len(event["sources"])})
                    else:
                        steps.append({k: v for k, v in event.items() if k != "notes"})
                    yield _line({**event, "session_id": session_id})
            except (ResearchError, Exception) as exc:  # stream the failure instead of a broken response
                message = str(exc) if isinstance(exc, ResearchError) else f"Research failed: {exc.__class__.__name__}: {str(exc)[:300]}"
                store.append_message(session_id, "assistant", message, metadata={"research": {"error": True, "steps": steps}})
                yield _line({"type": "research_failed", "session_id": session_id, "error": message})

        return StreamingResponse(events(), media_type="application/x-ndjson",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    return router


def _line(event: dict[str, Any]) -> str:
    return json.dumps(event, ensure_ascii=False, default=str) + "\n"
