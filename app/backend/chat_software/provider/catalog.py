"""Model discovery and installation beyond the curated Ollama list.

* Cloud providers: list the models an API key can actually use (verifies the key too).
* Hugging Face: search public GGUF repositories and list their quantizations. Ollama pulls these
  directly as ``hf.co/<owner>/<repo>:<QUANT>``, so open models from the internet use the same
  local runtime as curated Ollama models.
* Pulls run in the background with progress, instead of blocking one HTTP request for minutes.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from chat_software import secrets_store

from .config import get_provider_settings
from .factory import get_openai_compatible_client


HF_API = "https://huggingface.co/api"
_MODEL_NAME = re.compile(r"^(?:hf\.co/|huggingface\.co/)?[A-Za-z0-9][A-Za-z0-9._\-/]{0,200}(?::[A-Za-z0-9._\-]{1,80})?$")
_QUANT = re.compile(r"(?i)(?:^|[-_.])((?:IQ|Q)\d[A-Z0-9_]*|BF16|F16|F32)(?:\.gguf)?$")


def validate_model_name(name: str) -> str:
    cleaned = (name or "").strip()
    if not _MODEL_NAME.match(cleaned) or ".." in cleaned:
        raise ValueError(f"Not a valid model name: {name!r}")
    return cleaned


def list_remote_models(provider: str) -> list[dict[str, Any]]:
    """Return models available to the configured key. Raises on a missing or rejected key."""

    settings = get_provider_settings(provider=provider)
    if provider == "anthropic":
        if not settings.api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set.")
        from anthropic import Anthropic

        kwargs: dict[str, Any] = {"api_key": settings.api_key}
        if settings.base_url:
            kwargs["base_url"] = settings.base_url
        page = Anthropic(**kwargs).models.list(limit=100)
        return [
            {"id": item.id, "label": getattr(item, "display_name", None) or item.id}
            for item in page.data
        ]
    if provider in {"openai", "compatible"}:
        if provider == "openai" and not settings.api_key:
            raise ValueError("OPENAI_API_KEY is not set.")
        if provider == "compatible" and not settings.base_url:
            raise ValueError("COMPATIBLE_BASE_URL is not set.")
        items = get_openai_compatible_client(settings).models.list()
        models = [{"id": item.id, "label": item.id} for item in items.data]
        if provider == "openai":
            chat_like = ("gpt", "o1", "o3", "o4", "chatgpt")
            models = [m for m in models if m["id"].startswith(chat_like)]
        return sorted(models, key=lambda m: m["id"])
    raise ValueError(f"Provider {provider!r} has no remote model listing.")


def _hf_get(path: str, timeout: float = 20) -> Any:
    headers = {"Accept": "application/json", "User-Agent": "chat-software/0.1"}
    token = secrets_store.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(f"{HF_API}{path}", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Hugging Face returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Hugging Face: {exc.reason}") from exc


def search_gguf_models(query: str, limit: int = 20) -> list[dict[str, Any]]:
    params = urllib.parse.urlencode(
        {"search": query.strip()[:100], "filter": "gguf", "sort": "downloads", "limit": max(1, min(limit, 50))}
    )
    rows = _hf_get(f"/models?{params}")
    return [
        {
            "repo": row.get("id") or row.get("modelId"),
            "downloads": row.get("downloads"),
            "likes": row.get("likes"),
            "gated": bool(row.get("gated")),
            "updated": row.get("lastModified"),
        }
        for row in rows
        if isinstance(row, dict) and (row.get("id") or row.get("modelId"))
    ]


def gguf_quantizations(repo: str) -> list[dict[str, Any]]:
    """List GGUF files in a repo with the Ollama tag that pulls each quantization."""

    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._\-]*/[A-Za-z0-9][A-Za-z0-9._\-]*$", repo or ""):
        raise ValueError("Repository must look like owner/name.")
    files = _hf_get(f"/models/{repo}/tree/main?recursive=true")
    out: dict[str, dict[str, Any]] = {}
    for item in files if isinstance(files, list) else []:
        path = str(item.get("path") or "")
        if not path.lower().endswith(".gguf") or "mmproj" in path.lower():
            continue
        stem = path.rsplit("/", 1)[-1][:-5]
        stem = re.sub(r"-\d{5}-of-\d{5}$", "", stem)  # split files
        match = _QUANT.search(stem)
        if not match:
            continue
        quant = match.group(1).upper()
        entry = out.setdefault(quant, {"quant": quant, "size_bytes": 0, "files": 0,
                                       "ollama_model": f"hf.co/{repo}:{quant}"})
        entry["size_bytes"] += int(item.get("size") or (item.get("lfs") or {}).get("size") or 0)
        entry["files"] += 1
    return sorted(out.values(), key=lambda e: e["size_bytes"])


@dataclass
class PullJob:
    id: str
    model: str
    status: str = "queued"
    completed: int = 0
    total: int = 0
    message: str = ""
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        percent = round(100 * self.completed / self.total, 1) if self.total else None
        return {
            "id": self.id,
            "model": self.model,
            "status": self.status,
            "completed": self.completed,
            "total": self.total,
            "percent": percent,
            "message": self.message,
            "error": self.error,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
        }


StreamOpener = Callable[[str], Any]


class PullManager:
    """Runs ``ollama pull`` (streaming API) in background threads and keeps progress."""

    def __init__(self, opener: Optional[StreamOpener] = None) -> None:
        self._jobs: dict[str, PullJob] = {}
        self._lock = threading.Lock()
        self._opener = opener or _open_ollama_pull_stream

    def start(self, model: str) -> PullJob:
        model = validate_model_name(model)
        with self._lock:
            for job in self._jobs.values():
                if job.model == model and job.status in {"queued", "running"}:
                    return job
            job = PullJob(id=str(uuid.uuid4()), model=model)
            self._jobs[job.id] = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def get(self, job_id: str) -> PullJob:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"Pull job not found: {job_id}")
            return self._jobs[job_id]

    def list(self) -> list[PullJob]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.started_at, reverse=True)

    def _run(self, job: PullJob) -> None:
        job.status = "running"
        try:
            for event in self._opener(job.model):
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                job.message = str(event.get("status") or job.message)
                if event.get("total"):
                    job.total = int(event["total"])
                    job.completed = int(event.get("completed") or 0)
            if job.message != "success":
                raise RuntimeError(f"Pull ended without success (last status: {job.message or 'none'})")
            job.status = "completed"
        except Exception as exc:  # report every failure on the job
            job.status = "failed"
            job.error = str(exc)[:500]
        finally:
            job.finished_at = time.time()


def _open_ollama_pull_stream(model: str):
    from .ollama import native_base_url

    request = urllib.request.Request(
        f"{native_base_url()}/api/pull",
        data=json.dumps({"model": model, "stream": True}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        response = urllib.request.urlopen(request, timeout=3600)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Ollama returned HTTP {exc.code}: {exc.read().decode(errors='replace')[:300]}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Ollama at {native_base_url()}: {exc.reason}") from exc
    with response:
        for line in response:
            line = line.strip()
            if line:
                yield json.loads(line)
