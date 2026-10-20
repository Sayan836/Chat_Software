"""AirLLM endpoints: runtime, worker lifecycle, model catalog, HF checkpoint checks and downloads."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from chat_software.provider.airllm import AirLLMError, AirLLMManager


class RegisterRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=63)
    path: str = Field(min_length=1, max_length=1024)
    context_tokens: int = Field(default=4096, ge=512, le=131072)


class DownloadRequest(BaseModel):
    repo: str = Field(min_length=3, max_length=200)
    model_id: Optional[str] = Field(default=None, max_length=63)
    context_tokens: int = Field(default=4096, ge=512, le=131072)


def build_router(manager: AirLLMManager) -> APIRouter:
    router = APIRouter()

    def fail(exc: Exception) -> HTTPException:
        if isinstance(exc, KeyError):
            return HTTPException(status_code=404, detail=str(exc).strip("'"))
        if isinstance(exc, ValueError):
            return HTTPException(status_code=400, detail=str(exc))
        if isinstance(exc, AirLLMError):
            return HTTPException(status_code=503, detail=str(exc))
        return HTTPException(status_code=502, detail=f"{exc.__class__.__name__}: {str(exc)[:300]}")

    @router.get("/airllm/status")
    def status(refresh: bool = False) -> dict[str, object]:
        return {"runtime": manager.runtime_status(refresh=refresh), "worker": manager.status(),
                "models": manager.list_models(), "downloads": [j.to_dict() for j in manager.downloads()],
                "note": "AirLLM streams one layer at a time, so it can run models larger than memory. "
                        "Small models answer in seconds; large ones can take minutes per reply. "
                        "Ollama (GGUF) is faster for models that fit in RAM. The Metal backend only "
                        "supports Llama-family models without tied embeddings."}

    @router.post("/airllm/runtime/setup")
    def setup_runtime() -> dict[str, object]:
        return manager.setup_runtime()

    @router.post("/airllm/worker/start")
    def start() -> dict[str, object]:
        try:
            return manager.start()
        except Exception as exc:
            raise fail(exc) from exc

    @router.post("/airllm/worker/stop")
    def stop() -> dict[str, object]:
        return manager.stop()

    @router.post("/airllm/worker/cancel")
    def cancel() -> dict[str, object]:
        return manager.cancel()

    @router.get("/airllm/models/check")
    def check(repo: str) -> dict[str, object]:
        try:
            return manager.check_repo(repo)
        except Exception as exc:
            raise fail(exc) from exc

    @router.post("/airllm/models/register")
    def register(request: RegisterRequest) -> dict[str, object]:
        try:
            return manager.register(request.model_id, request.path, context_tokens=request.context_tokens)
        except Exception as exc:
            raise fail(exc) from exc

    @router.post("/airllm/models/download")
    def download(request: DownloadRequest) -> dict[str, object]:
        try:
            return manager.start_download(request.repo, request.model_id, request.context_tokens).to_dict()
        except Exception as exc:
            raise fail(exc) from exc

    @router.get("/airllm/downloads/{job_id}")
    def download_status(job_id: str) -> dict[str, object]:
        try:
            return manager.download(job_id).to_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Download not found") from exc

    @router.delete("/airllm/models/{model_id}")
    def remove(model_id: str, delete_files: bool = False) -> dict[str, object]:
        try:
            result = manager.remove(model_id, delete_files=delete_files)
        except Exception as exc:
            raise fail(exc) from exc
        if manager.running():
            manager.stop()  # restarts with the new catalog on next use
        return result

    return router
