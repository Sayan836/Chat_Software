"""AirLLM provider: run Hugging Face transformers checkpoints layer-by-layer in an isolated worker.

AirLLM loads one transformer layer at a time, so models larger than RAM/VRAM can run, at a large
speed cost (minutes per reply on CPU). It needs full HF checkpoints (safetensors), not GGUF files:
GGUF models go to Ollama; transformers checkpoints go here.

Pieces:
* runtime  an isolated Python with AirLLM (``AIRLLM_PYTHON`` setting, or ``<home>/venv`` created by
           ``integrations/airllm/setup_runtime.py``). The app venv never imports AirLLM or torch.
* catalog  ``<home>/catalog.json``: downloaded or registered checkpoints (id, path, repo, revision).
* worker   ``integrations/airllm/worker.py`` started on a loopback port with the catalog; one
           resident model, serialized generation, cancel and time limits.
"""

from __future__ import annotations

import atexit
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import httpx

from chat_software import secrets_store
from chat_software.paths import state_dir

from .config import ProviderSettings


APP_DIR = Path(__file__).resolve().parents[3]
INTEGRATION_DIR = APP_DIR / "integrations" / "airllm"
CHECKPOINT_FILES = ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin",
                    "pytorch_model.bin.index.json")
SPECIAL_ARCHITECTURES = {"ChatGLMModel", "QWenLMHeadModel", "BaichuanForCausalLM", "BaiChuanForCausalLM",
                         "InternLMForCausalLM"}
_MODEL_ID = re.compile(r"^[a-z0-9][a-z0-9._-]{0,62}$")
_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
DEFAULT_MAX_TOKENS = 512
GENERATION_TIMEOUT = 1800


class AirLLMError(RuntimeError):
    pass


def model_id_for(repo: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", repo.split("/")[-1].lower()).strip("-")[:63] or f"model-{uuid.uuid4().hex[:6]}"


def architecture_supported(architectures: Sequence[str]) -> tuple[bool, str]:
    arch = architectures[0] if architectures else ""
    if arch in SPECIAL_ARCHITECTURES:
        return True, f"{arch} (AirLLM special-cased)"
    if arch.endswith("ForCausalLM"):
        return True, f"{arch} (generic layer streaming)"
    if not arch:
        return False, "config.json declares no architecture"
    return False, f"{arch} is not a causal language model AirLLM can stream"


@dataclass
class DownloadJob:
    id: str
    repo: str
    model_id: str
    expected_bytes: int
    status: str = "queued"
    downloaded_bytes: int = 0
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        percent = round(100 * min(self.downloaded_bytes, self.expected_bytes) / self.expected_bytes, 1) \
            if self.expected_bytes else None
        return {"id": self.id, "repo": self.repo, "model_id": self.model_id, "status": self.status,
                "downloaded_bytes": self.downloaded_bytes, "expected_bytes": self.expected_bytes,
                "percent": percent, "error": self.error, "started_at": self.started_at,
                "finished_at": self.finished_at}


class AirLLMManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._process: Optional[subprocess.Popen] = None
        self._port: Optional[int] = None
        self._log_handle = None
        self._jobs: dict[str, DownloadJob] = {}
        self._runtime_check: Optional[tuple[str, dict[str, Any]]] = None
        atexit.register(self.stop)

    # ---- locations and runtime -------------------------------------------------------------
    def home(self) -> Path:
        configured = secrets_store.get("AIRLLM_HOME")
        if configured:
            return Path(configured).expanduser()
        directory = state_dir()
        base = Path(".chat_software_state") if str(directory) == ":memory:" else directory
        return base / "airllm"

    def python_path(self) -> Path:
        configured = secrets_store.get("AIRLLM_PYTHON")
        return Path(configured).expanduser() if configured else self.home() / "venv" / "bin" / "python"

    def backend(self) -> str:
        value = (secrets_store.get("AIRLLM_BACKEND") or "cpu").lower()
        return value if value in {"cpu", "metal", "cuda"} else "cpu"

    def runtime_status(self, refresh: bool = False) -> dict[str, Any]:
        python = self.python_path()
        if not python.exists():
            return {"python": str(python), "ready": False,
                    "reason": "No AirLLM runtime yet. Set up the runtime or point AIRLLM_PYTHON at an existing one."}
        key = str(python)
        if not refresh and self._runtime_check and self._runtime_check[0] == key:
            return self._runtime_check[1]
        probe = subprocess.run(
            [str(python), "-c", "import airllm, torch, transformers, fastapi, uvicorn; "
             "print(torch.__version__, transformers.__version__)"],
            capture_output=True, text=True, timeout=120, env={**os.environ, "AIRLLM_BACKEND": "cpu"},
        )
        status = ({"python": key, "ready": True, "versions": probe.stdout.strip().splitlines()[-1:]}
                  if probe.returncode == 0 else
                  {"python": key, "ready": False, "reason": (probe.stderr or probe.stdout).strip()[-400:]})
        self._runtime_check = (key, status)
        return status

    def setup_runtime(self) -> dict[str, Any]:
        """Create the isolated runtime (clone + venv + pip). Explicit user action; slow."""

        home = self.home()
        home.mkdir(parents=True, exist_ok=True)
        result = subprocess.run([sys.executable, str(INTEGRATION_DIR / "setup_runtime.py"), "--home", str(home)],
                                capture_output=True, text=True, timeout=3600)
        self._runtime_check = None
        return {"ok": result.returncode == 0, "python": str(home / "venv" / "bin" / "python"),
                "output": (result.stdout + result.stderr)[-3000:]}

    # ---- catalog ---------------------------------------------------------------------------
    def _catalog_path(self) -> Path:
        return self.home() / "catalog.json"

    def list_models(self) -> list[dict[str, Any]]:
        path = self._catalog_path()
        if not path.exists():
            return []
        try:
            models = json.loads(path.read_text()).get("models", [])
        except (OSError, ValueError):
            return []
        for model in models:
            model["present"] = _checkpoint_present(Path(model["path"]))
        return models

    def _save_models(self, models: list[dict[str, Any]]) -> None:
        path = self._catalog_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        clean = [{k: v for k, v in m.items() if k != "present"} for m in models]
        path.write_text(json.dumps({"models": clean}, indent=2) + "\n")

    def register(self, model_id: str, path: str, *, repo: Optional[str] = None, revision: Optional[str] = None,
                 context_tokens: int = 4096) -> dict[str, Any]:
        if not _MODEL_ID.match(model_id or ""):
            raise ValueError("Model id must be lowercase letters, digits, dots, dashes or underscores.")
        directory = Path(path).expanduser().resolve()
        config_file = directory / "config.json"
        if not config_file.exists() or not _checkpoint_present(directory):
            raise ValueError("That folder is not a transformers checkpoint (needs config.json and safetensors/bin weights).")
        try:
            architectures = json.loads(config_file.read_text()).get("architectures") or []
        except ValueError as exc:
            raise ValueError("config.json is not valid JSON.") from exc
        supported, note = architecture_supported(architectures)
        if not supported:
            raise ValueError(note)
        if not 512 <= int(context_tokens) <= 131072:
            raise ValueError("context_tokens must be between 512 and 131072.")
        with self._lock:
            models = [m for m in self.list_models() if m["id"] != model_id]
            entry = {"id": model_id, "path": str(directory), "repository": repo, "revision": revision,
                     "architecture": architectures[0] if architectures else None,
                     "context_tokens": int(context_tokens), "added_at": time.time()}
            models.append(entry)
            self._save_models(models)
        return {**entry, "present": True, "support": note}

    def remove(self, model_id: str, delete_files: bool = False) -> dict[str, Any]:
        with self._lock:
            models = self.list_models()
            entry = next((m for m in models if m["id"] == model_id), None)
            if entry is None:
                raise KeyError(f"AirLLM model not found: {model_id}")
            self._save_models([m for m in models if m["id"] != model_id])
            deleted = False
            target = Path(entry["path"])
            managed = (self.home() / "models").resolve()
            if delete_files and managed in target.resolve().parents:
                shutil.rmtree(target, ignore_errors=True)
                deleted = True
        return {"removed": model_id, "files_deleted": deleted,
                "note": None if deleted or not delete_files else "Files outside the managed folder were kept."}

    # ---- Hugging Face checkpoints ----------------------------------------------------------
    def check_repo(self, repo: str) -> dict[str, Any]:
        from .catalog import _hf_get

        if not _REPO.match(repo or ""):
            raise ValueError("Repository must look like owner/name.")
        info = _hf_get(f"/models/{repo}?blobs=true")
        architectures = ((info.get("config") or {}).get("architectures") or [])
        supported, note = architecture_supported(architectures)
        siblings = info.get("siblings") or []
        weights = [s for s in siblings if str(s.get("rfilename", "")).endswith(".safetensors")]
        size = sum(int(s.get("size") or 0) for s in weights)
        if not weights:
            supported, note = False, "No safetensors weights in this repository (GGUF-only repos belong to Ollama)."
        free = shutil.disk_usage(self.home().parent if self.home().parent.exists() else Path.home()).free
        return {
            "repo": repo, "revision": info.get("sha"), "architectures": architectures, "supported": supported,
            "support": note, "gated": bool(info.get("gated")), "weights_bytes": size,
            "disk_needed_bytes": size * 2 + 50 * 1024 * 1024,  # AirLLM also writes a per-layer split copy
            "free_bytes": free, "suggested_id": model_id_for(repo),
            "context_tokens": int(((info.get("config") or {}).get("max_position_embeddings") or 4096)),
        }

    def start_download(self, repo: str, model_id: Optional[str] = None, context_tokens: int = 4096) -> DownloadJob:
        check = self.check_repo(repo)
        if not check["supported"]:
            raise ValueError(check["support"])
        if check["gated"] and not secrets_store.get("HF_TOKEN"):
            raise ValueError("This model is gated: accept its licence on Hugging Face and add an HF token in Settings.")
        if check["disk_needed_bytes"] > check["free_bytes"]:
            raise ValueError("Not enough free disk space for the weights plus AirLLM's layer-split copy.")
        model_id = model_id or check["suggested_id"]
        if not _MODEL_ID.match(model_id):
            raise ValueError("Invalid model id.")
        with self._lock:
            for job in self._jobs.values():
                if job.repo == repo and job.status in {"queued", "running"}:
                    return job
            job = DownloadJob(id=str(uuid.uuid4()), repo=repo, model_id=model_id,
                              expected_bytes=check["weights_bytes"])
            self._jobs[job.id] = job
        context = min(int(context_tokens), check["context_tokens"]) if context_tokens else check["context_tokens"]
        threading.Thread(target=self._download, args=(job, check["revision"], context), daemon=True).start()
        return job

    def _download(self, job: DownloadJob, revision: Optional[str], context_tokens: int) -> None:
        target = self.home() / "models" / job.model_id
        job.status = "running"
        done = threading.Event()

        def watch() -> None:
            while not done.wait(1.0):
                job.downloaded_bytes = _dir_size(target)

        threading.Thread(target=watch, daemon=True).start()
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(
                job.repo, revision=revision, local_dir=str(target),
                token=secrets_store.get("HF_TOKEN") or False,
                allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.tiktoken", "tokenizer*"],
            )
            job.downloaded_bytes = _dir_size(target)
            self.register(job.model_id, str(target), repo=job.repo, revision=revision,
                          context_tokens=max(512, min(context_tokens, 131072)))
            job.status = "completed"
        except Exception as exc:
            job.status = "failed"
            job.error = f"{exc.__class__.__name__}: {str(exc)[:400]}"
        finally:
            done.set()
            job.finished_at = time.time()

    def downloads(self) -> list[DownloadJob]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.started_at, reverse=True)

    def download(self, job_id: str) -> DownloadJob:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(job_id)
            return self._jobs[job_id]

    # ---- worker lifecycle ------------------------------------------------------------------
    @property
    def base_url(self) -> Optional[str]:
        return f"http://127.0.0.1:{self._port}" if self._port and self.running() else None

    def running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> dict[str, Any]:
        with self._lock:
            if self.running():
                return self.status()
            runtime = self.runtime_status()
            if not runtime.get("ready"):
                raise AirLLMError(runtime.get("reason") or "AirLLM runtime is not ready.")
            models = [m for m in self.list_models() if m.get("present")]
            if not models:
                raise AirLLMError("No AirLLM models yet. Download a Hugging Face checkpoint or register one.")
            home = self.home()
            home.mkdir(parents=True, exist_ok=True)
            config = {"backend": self.backend(), "cpu_threads": max(1, (os.cpu_count() or 4) // 2),
                      "max_output_tokens": DEFAULT_MAX_TOKENS, "generation_timeout_seconds": GENERATION_TIMEOUT,
                      "prefetching": True,
                      "models": [{k: m[k] for k in ("id", "path", "context_tokens")} for m in models]}
            config_path = home / "worker-config.json"
            config_path.write_text(json.dumps(config, indent=2))
            port = _free_port()
            self._log_handle = open(home / "worker.log", "ab")
            env = {**os.environ, "AIRLLM_BACKEND": config["backend"], "HF_HUB_OFFLINE": "1",
                   "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1", "TOKENIZERS_PARALLELISM": "false"}
            for secret in secrets_store.SECRET_NAMES:  # the worker never needs provider or search keys
                env.pop(secret, None)
            self._process = subprocess.Popen(
                [str(self.python_path()), str(INTEGRATION_DIR / "worker.py"), "--config", str(config_path),
                 "--port", str(port)],
                stdout=self._log_handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env,
                cwd=str(home),
            )
            self._port = port
            deadline = time.time() + 90
            while time.time() < deadline:
                if self._process.poll() is not None:
                    raise AirLLMError(f"AirLLM worker exited early; see {home / 'worker.log'}")
                try:
                    if httpx.get(f"http://127.0.0.1:{port}/health", timeout=2).status_code == 200:
                        return self.status()
                except httpx.HTTPError:
                    time.sleep(0.5)
            self.stop()
            raise AirLLMError("AirLLM worker did not start within 90 seconds.")

    def ensure_running(self, model_id: Optional[str] = None) -> str:
        with self._lock:
            if self.running() and model_id:
                served = self.status().get("health", {}).get("models") or []
                if model_id not in served:
                    self.stop()  # catalog changed since start
            if not self.running():
                self.start()
            return self.base_url or ""

    def stop(self) -> dict[str, Any]:
        with self._lock:
            process, self._process, self._port = self._process, None, None
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None
        return {"running": False}

    def cancel(self) -> dict[str, Any]:
        if not self.base_url:
            return {"running": False}
        return httpx.post(f"{self.base_url}/cancel", timeout=5).json()

    def status(self) -> dict[str, Any]:
        out: dict[str, Any] = {"running": self.running(), "port": self._port, "backend": self.backend(),
                               "home": str(self.home())}
        if self.running():
            try:
                out["health"] = httpx.get(f"{self.base_url}/health", timeout=3).json()
            except httpx.HTTPError as exc:
                out["health_error"] = exc.__class__.__name__
        return out


@dataclass
class AirLLMChatClient:
    """Chat client for the AirLLM worker (OpenAI-compatible wire format, long timeouts)."""

    settings: ProviderSettings
    manager: "AirLLMManager"

    def complete(self, messages, tools=None) -> str:
        base = self.manager.ensure_running(self.settings.model)
        payload = {"model": self.settings.model,
                   "messages": [{"role": m["role"], "content": m.get("content", "")} for m in messages
                                if m.get("role") in {"system", "user", "assistant"}],
                   "temperature": self.settings.temperature,
                   "max_tokens": self.settings.max_tokens or DEFAULT_MAX_TOKENS}
        try:
            response = httpx.post(f"{base}/v1/chat/completions", json=payload, timeout=GENERATION_TIMEOUT + 60)
        except httpx.HTTPError as exc:
            raise AirLLMError(f"AirLLM worker request failed ({exc.__class__.__name__})") from exc
        if response.status_code >= 400:
            detail = response.json().get("detail") if response.headers.get("content-type", "").startswith(
                "application/json") else response.text[:300]
            raise AirLLMError(f"AirLLM: {detail}")
        return response.json()["choices"][0]["message"]["content"]


def _checkpoint_present(directory: Path) -> bool:
    if not directory.is_dir() or not (directory / "config.json").exists():
        return False
    return any((directory / name).exists() for name in CHECKPOINT_FILES) or any(
        directory.glob("model-*.safetensors"))


def _dir_size(directory: Path) -> int:
    if not directory.exists():
        return 0
    return sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


MANAGER = AirLLMManager()
