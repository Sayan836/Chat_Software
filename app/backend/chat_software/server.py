"""HTTP sidecar service for the desktop shell."""

import json
import os
import uuid
from pathlib import Path
from typing import Optional

from chat_software.agent import ReadOnlyAgent, SessionStore
from chat_software.knowledge import (
    DEFAULT_CREDENTIAL_REF,
    DEFAULT_INGEST_CREDENTIAL_REF,
    IngestionService,
    SubmissionStateError,
    KnowledgeConnector,
    KnowledgeError,
    KnowledgeForbidden,
    KnowledgeNotConfigured,
    KnowledgeNotFound,
    KnowledgeUnauthorized,
)
from chat_software.organize import apply_organize_plan, build_organize_plan, undo_organize_plan
from chat_software.paths import state_dir
from chat_software.provider.config import MODEL_REGISTRY, ProviderSettings, get_provider_settings
from chat_software.provider.factory import get_chat_client
from chat_software.provider import ollama
from chat_software import secrets_store
from chat_software.runtime_log import record_runtime_event
from chat_software.safety_store import ApprovedRoot, SafetyStore
from chat_software.tools.file_ops import glob_files, grep_files, list_dir, read_file, stat_path
from chat_software.tools.file_ops_write import copy_path, edit_file, make_dir, move_path, write_file
from chat_software.tools.git_tools import git_diff, git_status
from chat_software.tools.scope import ScopedPathResolver


DEFAULT_CORS_ORIGINS = (
    "http://127.0.0.1:5173",
    "http://localhost:5173",
    "tauri://localhost",
    "http://tauri.localhost",
)


def session_store() -> SessionStore:
    return SessionStore(_state_db_path("sessions.sqlite3"))


def safety_store() -> SafetyStore:
    return SafetyStore(_state_db_path("safety.sqlite3"))


def _state_db_path(filename: str) -> str | Path:
    directory = state_dir()
    if str(directory) == ":memory:":
        return ":memory:"
    return directory / filename


def cors_origins() -> list[str]:
    configured = os.getenv("CHAT_SOFTWARE_CORS_ORIGINS", "").strip()
    if not configured:
        return list(DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in configured.split(",") if origin.strip()]


def model_error_detail(settings: ProviderSettings, error: Exception) -> str:
    original = str(error)
    if settings.provider == "ollama":
        return (
            f"Model call failed for Ollama model {settings.model!r} at "
            f"{settings.base_url or 'http://localhost:11434'}. Make sure Ollama is running, "
            "then use the Models panel to refresh installed models or install the selected model. "
            f"Original error: {original}"
        )
    if settings.provider == "openai":
        key_state = "set" if secrets_store.get("OPENAI_API_KEY") else "not set"
        return (
            f"Model call failed for OpenAI model {settings.model!r}. "
            f"OPENAI_API_KEY is {key_state}. Original error: {original}"
        )
    if settings.provider == "anthropic":
        key_state = "set" if secrets_store.get("ANTHROPIC_API_KEY") else "not set"
        return (
            f"Model call failed for Claude model {settings.model!r}. "
            f"ANTHROPIC_API_KEY is {key_state}. Original error: {original}"
        )
    return f"Model call failed for {settings.provider}/{settings.model}. Original error: {original}"


def create_app() -> object:
    try:
        from fastapi import FastAPI, HTTPException, Response
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import StreamingResponse
        from pydantic import BaseModel, Field
    except ImportError as exc:
        raise RuntimeError("Install FastAPI and Pydantic to run the sidecar service.") from exc

    class ChatRequest(BaseModel):
        message: str = Field(min_length=1)
        session_id: Optional[str] = None
        roots: Optional[list[str]] = None
        working_path: str = "."
        provider: Optional[str] = None
        model: Optional[str] = None
        knowledge_root_id: Optional[str] = Field(
            default=None,
            description="Documents mode: answer from this root's bound retrieval workspace.",
        )
        skill_ids: list[str] = Field(default_factory=list, max_length=8)
        tools: list[str] = Field(
            default_factory=list,
            description="Tools to allow for this turn (web_search, fetch_url, run_code).",
        )

    class RootApprovalRequest(BaseModel):
        path: str = Field(min_length=1)

    class ListDirRequest(BaseModel):
        path: str = "."
        roots: list[str] = Field(min_length=1)
        max_entries: int = Field(default=200, ge=1, le=1000)

    class StatPathRequest(BaseModel):
        path: str = Field(min_length=1)
        roots: list[str] = Field(min_length=1)

    class ReadFileRequest(BaseModel):
        path: str = Field(min_length=1)
        roots: list[str] = Field(min_length=1)
        max_bytes: int = Field(default=200_000, ge=1, le=1_000_000)

    class MakeDirRequest(BaseModel):
        path: str = Field(min_length=1)
        roots: list[str] = Field(min_length=1)

    class WriteFileRequest(BaseModel):
        path: str = Field(min_length=1)
        roots: list[str] = Field(min_length=1)
        content: str = ""
        overwrite: bool = False

    class CopyPathRequest(BaseModel):
        source: str = Field(min_length=1)
        destination: str = Field(min_length=1)
        roots: list[str] = Field(min_length=1)

    class MovePathRequest(BaseModel):
        source: str = Field(min_length=1)
        destination: str = Field(min_length=1)
        roots: list[str] = Field(min_length=1)

    class EditFileRequest(BaseModel):
        path: str = Field(min_length=1)
        roots: list[str] = Field(min_length=1)
        old_text: str = Field(min_length=1)
        new_text: str = ""
        count: int = Field(default=1, ge=0, le=1000)

    class GlobFilesRequest(BaseModel):
        pattern: str = Field(min_length=1)
        root: str = "."
        roots: list[str] = Field(min_length=1)
        max_results: int = Field(default=200, ge=1, le=1000)

    class GrepFilesRequest(BaseModel):
        query: str = Field(min_length=1)
        root: str = "."
        include: str = "*"
        roots: list[str] = Field(min_length=1)
        max_results: int = Field(default=50, ge=1, le=500)
        case_sensitive: bool = False

    class GitCommandRequest(BaseModel):
        repo_path: str = "."
        roots: list[str] = Field(min_length=1)
        staged: bool = False
        stat_only: bool = True

    class OllamaModelRequest(BaseModel):
        model: str = Field(min_length=1)

    class OrganizePlanRequest(BaseModel):
        root_id: Optional[str] = None
        root: Optional[str] = None
        path: str = "."
        strategy: str = "by_type"
        file_filter: str = "all"
        prompt: Optional[str] = None
        max_files: int = Field(default=1000, ge=1, le=5000)

    class OrganizePlanIdRequest(BaseModel):
        plan_id: str = Field(min_length=1)

    class KnowledgeBindRequest(BaseModel):
        root_id: str = Field(min_length=1)
        workspace_id: str = Field(min_length=1, max_length=36)
        credential_ref: str = Field(default=DEFAULT_CREDENTIAL_REF, max_length=120)

    class IngestEnableRequest(BaseModel):
        root_id: str = Field(min_length=1)
        credential_ref: str = Field(default=DEFAULT_INGEST_CREDENTIAL_REF, max_length=120)

    class IngestPreviewRequest(BaseModel):
        root_id: str = Field(min_length=1)
        paths: list[str] = Field(min_length=1, max_length=50)

    class KnowledgeSearchRequest(BaseModel):
        root_id: str = Field(min_length=1)
        query: str = Field(min_length=1, max_length=16000)
        top_k: int = Field(default=5, ge=1, le=30)

    class KnowledgeSourceRequest(BaseModel):
        root_id: str = Field(min_length=1)
        document_id: str = Field(min_length=1, max_length=128)
        page: int = Field(default=1, ge=1, le=10000)

    app = FastAPI(title="Chat Software Agent Service", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins(),
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        allow_credentials=False,
    )
    store = session_store()
    safety = safety_store()
    knowledge = KnowledgeConnector(safety)
    app.state.knowledge = knowledge
    ingestion = IngestionService(safety)
    app.state.ingestion = ingestion

    from chat_software.api import models as models_api

    models_router = models_api.build_router()
    app.state.pulls = models_router.pull_manager
    app.include_router(models_router)

    from chat_software.api import sessions as sessions_api

    app.include_router(sessions_api.build_router(store))

    from chat_software.api import skills as skills_api
    from chat_software.skills import SkillLibrary, skills_system_prompt

    skill_library = SkillLibrary()
    app.state.skills = skill_library

    def make_chat_client(provider: Optional[str] = None, requested_model: Optional[str] = None):
        return get_chat_client(
            provider=provider,
            requested_model=requested_model,
            offline_echo=_truthy(os.getenv("CHAT_SOFTWARE_OFFLINE_ECHO")),
        )

    app.state.make_chat_client = make_chat_client
    app.include_router(skills_api.build_router(skill_library, make_chat_client))

    from chat_software.api import sandbox as sandbox_api
    from chat_software.sandbox import Sandbox

    sandbox = Sandbox()
    app.state.sandbox = sandbox
    app.include_router(sandbox_api.build_router(sandbox, safety))

    from chat_software.api import research as research_api

    app.include_router(research_api.build_router(store, make_chat_client, knowledge, safety))

    from chat_software.api import teams as teams_api
    from chat_software.teams import TeamStore
    from chat_software.teams.runtime import TeamDeps

    team_store = TeamStore(_state_db_path("teams.sqlite3"))
    app.state.teams = team_store
    app.include_router(teams_api.build_router(
        team_store,
        store,
        TeamDeps(client_factory=make_chat_client, skills=skill_library, sandbox=sandbox, knowledge=knowledge),
        safety,
    ))

    def root_payload(root: ApprovedRoot) -> dict[str, object]:
        return {
            "root_id": root.id,
            "path": str(root.path),
            "name": root.name,
            "created_at": root.created_at,
            "updated_at": root.updated_at,
            "revoked_at": root.revoked_at,
        }

    def authorized_root_paths(roots: Optional[list[str]], required: bool = True) -> list[str]:
        if not roots:
            if required:
                raise PermissionError("At least one server-approved root is required.")
            return []
        try:
            return [str(root.path) for root in safety.authorize_roots(roots)]
        except Exception as exc:
            raise PermissionError(str(exc)) from exc

    def authorized_resolver(roots: list[str]) -> ScopedPathResolver:
        return ScopedPathResolver.from_roots(authorized_root_paths(roots))

    def emit_runtime(
        event: dict[str, object],
        sink: Optional[object] = None,
    ) -> dict[str, object]:
        payload = record_runtime_event(event)
        if callable(sink):
            sink(payload)
        return payload

    def run_logged_tool(
        name: str,
        arguments: dict[str, object],
        operation: object,
    ) -> dict[str, object]:
        operation_id = str(uuid.uuid4())
        emit_runtime(
            {
                "type": "tool_started",
                "source": "api",
                "tool_id": operation_id,
                "name": name,
                "arguments": arguments,
            }
        )
        try:
            if not callable(operation):
                raise TypeError("Tool operation is not callable.")
            result = operation()
        except Exception as exc:
            emit_runtime(
                {
                    "type": "tool_failed",
                    "source": "api",
                    "tool_id": operation_id,
                    "name": name,
                    "arguments": arguments,
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                }
            )
            raise
        emit_runtime(
            {
                "type": "tool_completed",
                "source": "api",
                "tool_id": operation_id,
                "name": name,
                "arguments": arguments,
                "result": result,
            }
        )
        return result

    def approved_root_for_request(root_id: Optional[str], root_path: Optional[str]) -> ApprovedRoot:
        if root_id:
            return safety.get_approved_root(root_id)
        if root_path:
            return safety.authorize_roots([root_path])[0]
        raise PermissionError("Provide root_id or a server-approved root path.")

    def ollama_installed_models() -> dict[str, object]:
        try:
            body = ollama.list_models()
            models = ollama.sort_installed_models(body.get("models", []))
            names = [ollama.model_name(model) for model in models]
            return {
                "base_url": body.get("base_url") or ollama.native_base_url(),
                "models": models,
                "model_names": names,
                "recommended_model": ollama.select_installed_model(None, models),
                "error": None,
            }
        except Exception as exc:
            return {
                "base_url": ollama.native_base_url(),
                "models": [],
                "model_names": [],
                "recommended_model": None,
                "error": str(exc),
            }

    def resolved_chat_settings(request: ChatRequest) -> ProviderSettings:
        settings = get_provider_settings(
            provider=request.provider,
            requested_model=request.model,
        )
        if settings.provider != "ollama" or request.model:
            return settings
        installed = ollama_installed_models()
        selected = ollama.select_installed_model(settings.model, installed["models"])
        if not selected or selected == settings.model:
            return settings
        return get_provider_settings(provider="ollama", requested_model=selected)

    def chat_response_payload(
        result: object,
        settings: ProviderSettings,
    ) -> dict[str, object]:
        return {
            "session_id": result.session_id,
            "provider": settings.provider,
            "model": settings.model,
            "response": result.response,
            "tool_calls": [
                {
                    "name": record.name,
                    "command": record.command,
                    "status": record.status,
                    "arguments": record.arguments,
                    "result": record.result,
                }
                for record in result.tool_calls
            ],
        }

    def execute_chat(
        request: ChatRequest,
        settings: ProviderSettings,
        roots: list[str],
        request_id: str,
        sink: Optional[object] = None,
    ) -> dict[str, object]:
        def agent_runtime(event: dict[str, object]) -> None:
            emit_runtime({"request_id": request_id, **event}, sink)

        emit_runtime(
            {
                "type": "chat_started",
                "request_id": request_id,
                "provider": settings.provider,
                "model": settings.model,
                "session_id": request.session_id,
                "message": request.message,
                "mode": "documents" if request.knowledge_root_id else "chat",
            },
            sink,
        )
        chat_client = get_chat_client(
            requested_model=settings.model,
            provider=settings.provider,
            offline_echo=_truthy(os.getenv("CHAT_SOFTWARE_OFFLINE_ECHO")),
        )
        agent = ReadOnlyAgent(
            roots=roots,
            session_store=store,
            safety_store=safety,
            chat_client=chat_client,
            offline_echo=_truthy(os.getenv("CHAT_SOFTWARE_OFFLINE_ECHO")),
            working_path=request.working_path,
            runtime_callback=agent_runtime,
            knowledge=knowledge,
            knowledge_root_id=request.knowledge_root_id,
        )
        apply_skills(agent, request)
        result = agent.handle(message=request.message, session_id=request.session_id)
        data = chat_response_payload(result, settings)
        emit_runtime(
            {
                "type": "chat_completed",
                "request_id": request_id,
                "provider": settings.provider,
                "model": settings.model,
                "session_id": result.session_id,
                "tool_count": len(result.tool_calls),
                "data": data,
            },
            sink,
        )
        return data

    def stream_execute_chat(
        request: ChatRequest,
        settings: ProviderSettings,
        roots: list[str],
        request_id: str,
    ) -> object:
        started = emit_runtime(
            {
                "type": "chat_started",
                "request_id": request_id,
                "provider": settings.provider,
                "model": settings.model,
                "session_id": request.session_id,
                "message": request.message,
                "mode": "documents" if request.knowledge_root_id else "chat",
            }
        )
        yield started
        chat_client = get_chat_client(
            requested_model=settings.model,
            provider=settings.provider,
            offline_echo=_truthy(os.getenv("CHAT_SOFTWARE_OFFLINE_ECHO")),
        )
        agent = ReadOnlyAgent(
            roots=roots,
            session_store=store,
            safety_store=safety,
            chat_client=chat_client,
            offline_echo=_truthy(os.getenv("CHAT_SOFTWARE_OFFLINE_ECHO")),
            working_path=request.working_path,
            runtime_callback=None,
            knowledge=knowledge,
            knowledge_root_id=request.knowledge_root_id,
        )
        apply_skills(agent, request)
        for event in agent.handle_stream(message=request.message, session_id=request.session_id):
            if event.get("type") == "__agent_result":
                result = event["result"]
                data = chat_response_payload(result, settings)
                yield emit_runtime(
                    {
                        "type": "chat_completed",
                        "request_id": request_id,
                        "provider": settings.provider,
                        "model": settings.model,
                        "session_id": result.session_id,
                        "tool_count": len(result.tool_calls),
                        "data": data,
                    }
                )
                return
            yield emit_runtime({"request_id": request_id, **event})

    def chat_failure(
        request_id: str,
        settings: ProviderSettings,
        exc: Exception,
    ) -> tuple[int, dict[str, object]]:
        if isinstance(exc, (FileNotFoundError, NotADirectoryError, PermissionError, ValueError)):
            status_code = 400
            detail = str(exc)
        elif isinstance(exc, KnowledgeError):
            status_code = knowledge_http_error(exc).status_code
            detail = str(exc)
        else:
            status_code = 502
            detail = model_error_detail(settings, exc)
        return status_code, {
            "type": "chat_failed",
            "request_id": request_id,
            "provider": settings.provider,
            "model": settings.model,
            "status_code": status_code,
            "error": detail,
            "error_type": exc.__class__.__name__,
        }

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        return Response(status_code=204)

    @app.get("/health")
    def health() -> dict[str, object]:
        settings = get_provider_settings()
        active_model = settings.model
        if settings.provider == "ollama":
            installed = ollama_installed_models()
            active_model = (
                ollama.select_installed_model(settings.model, installed["models"])
                or settings.model
            )
        return {
            "ok": True,
            "provider": settings.provider,
            "model": active_model,
            "configured_model": settings.model,
            "chat_mode": "offline_echo" if _truthy(os.getenv("CHAT_SOFTWARE_OFFLINE_ECHO")) else "model",
            "state_dir": str(state_dir()),
        }

    @app.get("/providers")
    def providers() -> dict[str, object]:
        settings = get_provider_settings()
        active_model = settings.model
        if settings.provider == "ollama":
            installed = ollama_installed_models()
            active_model = (
                ollama.select_installed_model(settings.model, installed["models"])
                or settings.model
            )
        return {
            "active": {
                "provider": settings.provider,
                "model": active_model,
                "configured_model": settings.model,
                "base_url": settings.base_url,
            },
            "models": [capability.__dict__ for capability in MODEL_REGISTRY.values()],
        }

    @app.post("/roots/approve")
    def approve_root(request: RootApprovalRequest) -> dict[str, object]:
        try:
            root = safety.approve_root(request.path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except NotADirectoryError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"approved": True, **root_payload(root)}

    @app.get("/roots")
    def approved_roots() -> dict[str, object]:
        return {"roots": [root_payload(root) for root in safety.list_approved_roots()]}

    @app.post("/roots/{root_id}/revoke")
    def revoke_root(root_id: str) -> dict[str, object]:
        try:
            root = safety.revoke_root(root_id)
        except PermissionError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"revoked": True, **root_payload(root)}

    def knowledge_http_error(exc: Exception) -> HTTPException:
        if isinstance(exc, (PermissionError, KnowledgeForbidden)):
            return HTTPException(status_code=403, detail=str(exc))
        if isinstance(exc, (KnowledgeNotFound, KeyError)):
            return HTTPException(status_code=404, detail=str(exc).strip("'\""))
        if isinstance(exc, KnowledgeNotConfigured):
            return HTTPException(status_code=503, detail=str(exc))
        if isinstance(exc, KnowledgeUnauthorized):
            return HTTPException(
                status_code=502,
                detail="The retrieval backend rejected this app's API key. Check the sidecar configuration.",
            )
        if isinstance(exc, KnowledgeError):
            status = exc.status if exc.status in {413, 415, 422} else 503
            return HTTPException(status_code=status, detail=str(exc))
        if isinstance(exc, SubmissionStateError):
            return HTTPException(status_code=409, detail=str(exc))
        if isinstance(exc, ValueError):
            return HTTPException(status_code=400, detail=str(exc))
        return HTTPException(status_code=500, detail=f"Knowledge request failed: {exc.__class__.__name__}")

    @app.get("/knowledge/status")
    def knowledge_status() -> dict[str, object]:
        return knowledge.status()

    @app.get("/knowledge/bindings")
    def knowledge_bindings() -> dict[str, object]:
        return {"bindings": knowledge.bindings()}

    @app.post("/knowledge/bindings")
    def knowledge_bind(request: KnowledgeBindRequest) -> dict[str, object]:
        try:
            binding = knowledge.bind(request.root_id, request.workspace_id, request.credential_ref)
            return {"bound": True, **binding}
        except Exception as exc:
            raise knowledge_http_error(exc) from exc

    @app.delete("/knowledge/bindings/{root_id}")
    def knowledge_unbind(root_id: str) -> dict[str, object]:
        try:
            knowledge.unbind(root_id)
        except Exception as exc:
            raise knowledge_http_error(exc) from exc
        return {"unbound": True, "root_id": root_id}

    @app.post("/knowledge/ingest/enable")
    def knowledge_ingest_enable(request: IngestEnableRequest) -> dict[str, object]:
        try:
            return {"enabled": True, **ingestion.enable(request.root_id, request.credential_ref)}
        except Exception as exc:
            raise knowledge_http_error(exc) from exc

    @app.delete("/knowledge/ingest/enable/{root_id}")
    def knowledge_ingest_disable(root_id: str) -> dict[str, object]:
        try:
            ingestion.disable(root_id)
        except Exception as exc:
            raise knowledge_http_error(exc) from exc
        return {"enabled": False, "root_id": root_id}

    @app.post("/knowledge/ingest/preview")
    def knowledge_ingest_preview(request: IngestPreviewRequest) -> dict[str, object]:
        try:
            return ingestion.preview(request.root_id, request.paths)
        except Exception as exc:
            raise knowledge_http_error(exc) from exc

    @app.post("/knowledge/ingest/{submission_id}/approve")
    def knowledge_ingest_approve(submission_id: str) -> dict[str, object]:
        try:
            return ingestion.approve(submission_id)
        except Exception as exc:
            raise knowledge_http_error(exc) from exc

    @app.get("/knowledge/ingest/{submission_id}")
    def knowledge_ingest_status(submission_id: str) -> dict[str, object]:
        try:
            return ingestion.refresh(submission_id)
        except Exception as exc:
            raise knowledge_http_error(exc) from exc

    @app.delete("/knowledge/ingest/{submission_id}")
    def knowledge_ingest_discard(submission_id: str) -> dict[str, object]:
        try:
            ingestion.discard(submission_id)
        except Exception as exc:
            raise knowledge_http_error(exc) from exc
        return {"discarded": True, "submission_id": submission_id}

    @app.post("/knowledge/search")
    def knowledge_search(request: KnowledgeSearchRequest) -> dict[str, object]:
        try:
            result = knowledge.search(request.root_id, request.query, top_k=request.top_k)
        except Exception as exc:
            raise knowledge_http_error(exc) from exc
        return {"root_id": request.root_id, **result.to_dict()}

    @app.post("/knowledge/source")
    def knowledge_source(request: KnowledgeSourceRequest) -> dict[str, object]:
        try:
            page = knowledge.open_source(request.root_id, request.document_id, request.page)
        except Exception as exc:
            raise knowledge_http_error(exc) from exc
        return {"root_id": request.root_id, **page.to_dict()}

    @app.post("/files/list")
    def files_list(request: ListDirRequest) -> dict[str, object]:
        try:
            roots = authorized_root_paths(request.roots)
            return run_logged_tool(
                "list_dir",
                {"path": request.path, "max_entries": request.max_entries, "roots": roots},
                lambda: list_dir(
                    path=request.path,
                    roots=roots,
                    max_entries=request.max_entries,
                ),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/files/stat")
    def files_stat(request: StatPathRequest) -> dict[str, object]:
        try:
            roots = authorized_root_paths(request.roots)
            return run_logged_tool(
                "stat_path",
                {"path": request.path, "roots": roots},
                lambda: stat_path(path=request.path, roots=roots),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/files/read")
    def files_read(request: ReadFileRequest) -> dict[str, object]:
        try:
            roots = authorized_root_paths(request.roots)
            return run_logged_tool(
                "read_file",
                {"path": request.path, "max_bytes": request.max_bytes, "roots": roots},
                lambda: read_file(
                    path=request.path,
                    roots=roots,
                    max_bytes=request.max_bytes,
                ),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/files/mkdir")
    def files_mkdir(request: MakeDirRequest) -> dict[str, object]:
        try:
            resolver = authorized_resolver(request.roots)
            return run_logged_tool(
                "make_dir",
                {"path": request.path, "roots": list(resolver.roots)},
                lambda: make_dir(path=request.path, resolver=resolver),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/files/write")
    def files_write(request: WriteFileRequest) -> dict[str, object]:
        try:
            resolver = authorized_resolver(request.roots)
            return run_logged_tool(
                "write_file",
                {
                    "path": request.path,
                    "content": request.content,
                    "overwrite": request.overwrite,
                    "roots": list(resolver.roots),
                },
                lambda: write_file(
                    path=request.path,
                    content=request.content,
                    resolver=resolver,
                    overwrite=request.overwrite,
                ),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/files/copy")
    def files_copy(request: CopyPathRequest) -> dict[str, object]:
        try:
            resolver = authorized_resolver(request.roots)
            return run_logged_tool(
                "copy_path",
                {
                    "source": request.source,
                    "destination": request.destination,
                    "roots": list(resolver.roots),
                },
                lambda: copy_path(
                    source=request.source,
                    destination=request.destination,
                    resolver=resolver,
                ),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/files/move")
    def files_move(request: MovePathRequest) -> dict[str, object]:
        try:
            resolver = authorized_resolver(request.roots)
            return run_logged_tool(
                "move_path",
                {
                    "source": request.source,
                    "destination": request.destination,
                    "roots": list(resolver.roots),
                },
                lambda: move_path(
                    source=request.source,
                    destination=request.destination,
                    resolver=resolver,
                ),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/files/edit")
    def files_edit(request: EditFileRequest) -> dict[str, object]:
        try:
            resolver = authorized_resolver(request.roots)
            return run_logged_tool(
                "edit_file",
                {
                    "path": request.path,
                    "old_text": request.old_text,
                    "new_text": request.new_text,
                    "count": request.count,
                    "roots": list(resolver.roots),
                },
                lambda: edit_file(
                    path=request.path,
                    old_text=request.old_text,
                    new_text=request.new_text,
                    resolver=resolver,
                    count=request.count,
                ),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/files/glob")
    def files_glob(request: GlobFilesRequest) -> dict[str, object]:
        try:
            roots = authorized_root_paths(request.roots)
            return run_logged_tool(
                "glob_files",
                {
                    "pattern": request.pattern,
                    "root": request.root,
                    "max_results": request.max_results,
                    "roots": roots,
                },
                lambda: glob_files(
                    pattern=request.pattern,
                    root=request.root,
                    roots=roots,
                    max_results=request.max_results,
                ),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/files/grep")
    def files_grep(request: GrepFilesRequest) -> dict[str, object]:
        try:
            roots = authorized_root_paths(request.roots)
            return run_logged_tool(
                "grep_files",
                {
                    "query": request.query,
                    "root": request.root,
                    "include": request.include,
                    "max_results": request.max_results,
                    "case_sensitive": request.case_sensitive,
                    "roots": roots,
                },
                lambda: grep_files(
                    query=request.query,
                    root=request.root,
                    include=request.include,
                    roots=roots,
                    max_results=request.max_results,
                    case_sensitive=request.case_sensitive,
                ),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/commands/git/status")
    def command_git_status(request: GitCommandRequest) -> dict[str, object]:
        try:
            roots = authorized_root_paths(request.roots)
            return run_logged_tool(
                "git_status",
                {"repo_path": request.repo_path, "roots": roots},
                lambda: git_status(repo_path=request.repo_path, roots=roots),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/commands/git/diff")
    def command_git_diff(request: GitCommandRequest) -> dict[str, object]:
        try:
            roots = authorized_root_paths(request.roots)
            return run_logged_tool(
                "git_diff",
                {
                    "repo_path": request.repo_path,
                    "staged": request.staged,
                    "stat_only": request.stat_only,
                    "roots": roots,
                },
                lambda: git_diff(
                    repo_path=request.repo_path,
                    roots=roots,
                    staged=request.staged,
                    stat_only=request.stat_only,
                ),
            )
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/organize/plan")
    def organize_plan(request: OrganizePlanRequest) -> dict[str, object]:
        try:
            root = approved_root_for_request(request.root_id, request.root)
            plan = build_organize_plan(
                root=root.path,
                strategy=request.strategy,
                file_filter=request.file_filter,
                path=request.path,
                prompt=request.prompt,
                max_files=request.max_files,
            )
            plan_id = safety.create_organize_plan(
                root_id=root.id,
                strategy=str(plan.get("strategy") or request.strategy),
                plan=plan,
                prompt=request.prompt,
            )
            return safety.get_organize_plan(plan_id)
        except PermissionError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/organize/plans/{plan_id}")
    def organize_get_plan(plan_id: str) -> dict[str, object]:
        try:
            return safety.get_organize_plan(plan_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.post("/organize/apply")
    def organize_apply(request: OrganizePlanIdRequest) -> dict[str, object]:
        try:
            return apply_organize_plan(safety, request.plan_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/organize/undo")
    def organize_undo(request: OrganizePlanIdRequest) -> dict[str, object]:
        try:
            return undo_organize_plan(safety, request.plan_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/model-options")
    def model_options() -> dict[str, object]:
        settings = get_provider_settings()
        installed_ollama = ollama_installed_models()
        active_model = settings.model
        if settings.provider == "ollama":
            active_model = (
                ollama.select_installed_model(settings.model, installed_ollama["models"])
                or settings.model
            )
        return {
            "active": {
                "provider": settings.provider,
                "model": active_model,
                "configured_model": settings.model,
                "base_url": settings.base_url,
            },
            "providers": [
                {
                    "provider": "ollama",
                    "label": "Ollama",
                    "api_available": installed_ollama["error"] is None,
                    "base_url": get_provider_settings(provider="ollama").base_url,
                    "native_base_url": installed_ollama["base_url"],
                    "default_model": (
                        ollama.select_installed_model(
                            get_provider_settings(provider="ollama").model,
                            installed_ollama["models"],
                        )
                        or get_provider_settings(provider="ollama").model
                    ),
                    "configured_model": get_provider_settings(provider="ollama").model,
                    "installed_count": len(installed_ollama["model_names"]),
                    "model_error": installed_ollama["error"],
                },
                {
                    "provider": "openai",
                    "label": "OpenAI",
                    "api_available": bool(secrets_store.get("OPENAI_API_KEY")),
                    "base_url": secrets_store.get("OPENAI_BASE_URL"),
                    "default_model": get_provider_settings(provider="openai").model,
                },
                {
                    "provider": "anthropic",
                    "label": "Claude",
                    "api_available": bool(secrets_store.get("ANTHROPIC_API_KEY")),
                    "base_url": secrets_store.get("ANTHROPIC_BASE_URL"),
                    "default_model": get_provider_settings(provider="anthropic").model,
                },
                {
                    "provider": "compatible",
                    "label": "OpenAI-compatible",
                    "api_available": bool(secrets_store.get("COMPATIBLE_BASE_URL")),
                    "base_url": secrets_store.get("COMPATIBLE_BASE_URL"),
                    "default_model": get_provider_settings(provider="compatible").model,
                },
            ],
            "registered_models": [
                capability.__dict__ for capability in MODEL_REGISTRY.values()
            ],
            "ollama_installed": installed_ollama,
            "ollama_catalog": ollama.catalog_models(),
        }

    @app.get("/ollama/models")
    def ollama_models() -> dict[str, object]:
        try:
            body = ollama.list_models()
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        return body

    @app.get("/ollama/catalog")
    def ollama_catalog() -> dict[str, object]:
        return {"models": ollama.catalog_models()}

    @app.post("/ollama/models/pull")
    def ollama_pull(request: OllamaModelRequest) -> dict[str, object]:
        try:
            return ollama.pull_model(request.model)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.delete("/ollama/models")
    def ollama_delete(request: OllamaModelRequest) -> dict[str, object]:
        try:
            return ollama.delete_model(request.model)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    def apply_skills(agent: ReadOnlyAgent, request: ChatRequest) -> None:
        from chat_software.agent.toolbox import ToolContext

        selected = skill_library.resolve(request.skill_ids) if request.skill_ids else []
        tool_names = [t for t in request.tools if t in {"web_search", "fetch_url", "run_code"}]
        for skill in selected:
            tool_names += [t for t in skill.tools if t != "search_documents"]
        if selected:
            agent.system_addendum = skills_system_prompt(selected)
            agent._response_metadata_base = {"skills": [skill.id for skill in selected]}
        if tool_names:
            agent.tool_names = list(dict.fromkeys(tool_names))
            agent.tool_context = ToolContext(sandbox=sandbox, web_enabled=True)

    def check_skills(request: ChatRequest) -> None:
        try:
            skill_library.resolve(request.skill_ids)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc).strip("'")) from exc

    def check_documents_mode(request: ChatRequest) -> None:
        if not request.knowledge_root_id:
            return
        try:
            safety.get_knowledge_binding(request.knowledge_root_id)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc

    @app.post("/chat")
    def chat(request: ChatRequest) -> dict[str, object]:
        settings = resolved_chat_settings(request)
        request_id = str(uuid.uuid4())
        try:
            roots = authorized_root_paths(request.roots, required=False)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        check_documents_mode(request)
        check_skills(request)
        try:
            return execute_chat(request, settings, roots, request_id)
        except Exception as exc:
            status_code, event = chat_failure(request_id, settings, exc)
            emit_runtime(event)
            raise HTTPException(status_code=status_code, detail=event["error"]) from exc

    @app.post("/chat/stream")
    def chat_stream(request: ChatRequest) -> StreamingResponse:
        settings = resolved_chat_settings(request)
        try:
            roots = authorized_root_paths(request.roots, required=False)
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        check_documents_mode(request)
        check_skills(request)

        request_id = str(uuid.uuid4())

        def event_stream() -> object:
            try:
                events = stream_execute_chat(request, settings, roots, request_id)
                for event in events:
                    yield json.dumps(event, default=str, ensure_ascii=False) + "\n"
            except Exception as exc:
                _status_code, event = chat_failure(request_id, settings, exc)
                failed = emit_runtime(event)
                yield json.dumps(failed, default=str, ensure_ascii=False) + "\n"

        return StreamingResponse(
            event_stream(),
            media_type="application/x-ndjson",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "X-Request-ID": request_id,
            },
        )

    @app.get("/sessions/{session_id}")
    def get_session(session_id: str) -> dict[str, object]:
        try:
            summary: Optional[dict[str, object]] = store.get_session(session_id).to_dict()
        except KeyError:
            summary = None
        return {
            "session_id": session_id,
            "session": summary,
            "messages": [
                {
                    "role": message.role,
                    "content": message.content,
                    "metadata": message.metadata,
                    "created_at": message.created_at,
                }
                for message in store.get_messages(session_id)
            ],
        }

    return app


app = create_app()


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Install uvicorn to run the sidecar service.") from exc

    host = os.getenv("CHAT_SOFTWARE_HOST", "127.0.0.1")
    port = int(os.getenv("CHAT_SOFTWARE_PORT", "8765"))
    # Dev mode: auto-restart on code changes and stream verbose logs, the way
    # a web dev server behaves. Enable with CHAT_SOFTWARE_RELOAD=1.
    reload = _truthy(os.getenv("CHAT_SOFTWARE_RELOAD"))
    log_level = os.getenv("CHAT_SOFTWARE_LOG_LEVEL", "debug" if reload else "info")
    uvicorn.run(
        "chat_software.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level=log_level,
        access_log=True,
    )


if __name__ == "__main__":
    main()
