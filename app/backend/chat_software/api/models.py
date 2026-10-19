"""Provider keys/settings, live cloud model lists, Hugging Face GGUF search and background pulls."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from chat_software import secrets_store
from chat_software.provider.catalog import (
    PullManager,
    gguf_quantizations,
    list_remote_models,
    search_gguf_models,
)
from chat_software.provider.config import PROVIDER_LABELS


class SettingRequest(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    value: Optional[str] = Field(default=None, max_length=4096)


class PullRequest(BaseModel):
    model: str = Field(min_length=1, max_length=300)


def build_router(pulls: Optional[PullManager] = None) -> APIRouter:
    router = APIRouter()
    manager = pulls or PullManager()
    router.pull_manager = manager  # type: ignore[attr-defined]

    @router.get("/settings/providers")
    def provider_settings() -> dict[str, object]:
        return {"providers": PROVIDER_LABELS, "values": secrets_store.describe()}

    @router.put("/settings/values")
    def set_setting(request: SettingRequest) -> dict[str, object]:
        try:
            secrets_store.set_value(request.name, request.value)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc).strip("'")) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"name": request.name, **secrets_store.describe()[request.name]}

    @router.get("/providers/{provider}/models")
    def remote_models(provider: str) -> dict[str, object]:
        if provider not in {"openai", "anthropic", "compatible"}:
            raise HTTPException(status_code=404, detail="Use /ollama/models for local models.")
        try:
            models = list_remote_models(provider)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:  # SDK auth/network errors: report type and status, never the key
            status = getattr(exc, "status_code", None)
            detail = f"{provider} rejected the request ({exc.__class__.__name__}"
            detail += f", HTTP {status})" if status else ")"
            raise HTTPException(status_code=502, detail=detail) from exc
        return {"provider": provider, "verified": True, "models": models}

    @router.get("/models/hf/search")
    def hf_search(q: str, limit: int = 20) -> dict[str, object]:
        if not q.strip():
            raise HTTPException(status_code=400, detail="Enter a search term.")
        try:
            return {"query": q, "results": search_gguf_models(q, limit)}
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @router.get("/models/hf/quantizations")
    def hf_quantizations(repo: str) -> dict[str, object]:
        try:
            return {"repo": repo, "quantizations": gguf_quantizations(repo)}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @router.post("/ollama/pulls")
    def start_pull(request: PullRequest) -> dict[str, object]:
        try:
            return manager.start(request.model).to_dict()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.get("/ollama/pulls")
    def list_pulls() -> dict[str, object]:
        return {"pulls": [job.to_dict() for job in manager.list()]}

    @router.get("/ollama/pulls/{job_id}")
    def pull_status(job_id: str) -> dict[str, object]:
        try:
            return manager.get(job_id).to_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Pull job not found") from exc

    return router
