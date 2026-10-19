"""Sandbox endpoints: status, explicit image pull, runs, workspaces and file import from approved roots."""

from __future__ import annotations

import base64
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from chat_software.runtime_log import record_runtime_event
from chat_software.safety_store import SafetyStore
from chat_software.sandbox import Sandbox, SandboxUnavailable
from chat_software.tools.scope import ScopedPathResolver


class RunRequest(BaseModel):
    code: str = Field(min_length=1, max_length=200_000)
    language: str = "python"
    timeout: int = Field(default=30, ge=1, le=120)
    workspace_id: Optional[str] = None
    files: dict[str, str] = Field(default_factory=dict)


class FileRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    content: Optional[str] = None
    content_base64: Optional[str] = None


class ImportRequest(BaseModel):
    root_id: str = Field(min_length=1)
    path: str = Field(min_length=1)
    name: Optional[str] = None


def build_router(sandbox: Sandbox, safety: SafetyStore) -> APIRouter:
    router = APIRouter()

    @router.get("/sandbox/status")
    def status() -> dict[str, object]:
        return sandbox.status()

    @router.post("/sandbox/image/pull")
    def pull_image() -> dict[str, object]:
        try:
            return sandbox.pull_image()
        except SandboxUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @router.post("/sandbox/run")
    def run(request: RunRequest) -> dict[str, object]:
        try:
            result = sandbox.run(request.code, language=request.language, timeout=request.timeout,
                                 workspace_id=request.workspace_id, files=request.files)
        except SandboxUnavailable as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc).strip("'")) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        record_runtime_event({"type": "sandbox_run", "backend": result.backend, "language": result.language,
                              "code_chars": len(request.code), "exit_code": result.exit_code,
                              "timed_out": result.timed_out, "duration_s": round(result.duration_s, 3)})
        return result.to_dict()

    @router.post("/sandbox/workspaces")
    def create_workspace() -> dict[str, object]:
        return {"workspace_id": sandbox.create_workspace(), "files": []}

    @router.get("/sandbox/workspaces/{workspace_id}")
    def list_workspace(workspace_id: str) -> dict[str, object]:
        try:
            return {"workspace_id": workspace_id, "files": sandbox.list_files(workspace_id)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Sandbox workspace not found") from exc

    @router.delete("/sandbox/workspaces/{workspace_id}")
    def delete_workspace(workspace_id: str) -> dict[str, object]:
        try:
            sandbox.delete_workspace(workspace_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Sandbox workspace not found") from exc
        return {"deleted": True, "workspace_id": workspace_id}

    @router.post("/sandbox/workspaces/{workspace_id}/files")
    def add_file(workspace_id: str, request: FileRequest) -> dict[str, object]:
        if (request.content is None) == (request.content_base64 is None):
            raise HTTPException(status_code=400, detail="Send exactly one of content or content_base64.")
        try:
            data = (request.content.encode() if request.content is not None
                    else base64.b64decode(request.content_base64 or "", validate=True))
            return sandbox.write_file(workspace_id, request.name, data)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Sandbox workspace not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/sandbox/workspaces/{workspace_id}/import")
    def import_file(workspace_id: str, request: ImportRequest) -> dict[str, object]:
        """Copy one file from an approved root into the workspace (permission checked now)."""

        try:
            root = safety.get_approved_root(request.root_id)
            source = ScopedPathResolver.from_roots([root.path]).resolve(request.path)
        except PermissionError as exc:
            raise HTTPException(status_code=403, detail=str(exc)) from exc
        if not source.is_file():
            raise HTTPException(status_code=404, detail="File not found in the approved root")
        if source.stat().st_size > 50 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Files are limited to 50 MB.")
        try:
            result = sandbox.write_file(workspace_id, request.name or source.name, source.read_bytes())
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Sandbox workspace not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        safety.record_audit("sandbox.file_imported", root_id=root.id, path=str(source),
                            metadata={"workspace_id": workspace_id, "size_bytes": result["size_bytes"]})
        return result

    return router
