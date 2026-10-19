"""Code execution sandbox for chat skills, research and agent teams.

Backends, best first (``CHAT_SOFTWARE_SANDBOX_BACKEND`` can force one):

``docker``   ``--network none``, read-only root, all capabilities dropped, no-new-privileges,
             non-root user, memory/CPU/PID limits; only the workspace directory is writable.
``seatbelt`` macOS ``sandbox-exec``: no network, writes only inside the workspace, reads of
             ``/Users`` and ``/Volumes`` denied except the workspace and the interpreter.
``process``  a plain subprocess with CPU/file-size limits and a timeout. NOT isolated; used only
             when ``CHAT_SOFTWARE_SANDBOX_ALLOW_UNISOLATED=1`` and always labelled as such.

Every run has a wall-clock timeout, killed process group, and capped output. Workspaces are
directories under ``<state_dir>/sandbox``: a fresh one per run unless a persistent workspace is
named, so multi-step analyses can keep files between runs.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from chat_software.paths import state_dir


LANGUAGES = {
    "python": {"file": "main.py", "host": [sys.executable], "docker": ["python3"]},
    "bash": {"file": "main.sh", "host": ["/bin/bash"], "docker": ["bash"]},
    "javascript": {"file": "main.js", "host": ["node"], "docker": ["node"]},
}
DEFAULT_IMAGE = "python:3.12-slim"
MAX_TIMEOUT = 120
MAX_OUTPUT = 64_000
MAX_FILE_PREVIEW = 4_000
_WORKSPACE_ID = re.compile(r"^[a-f0-9-]{36}$")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9._-][A-Za-z0-9._ /-]{0,200}$")


class SandboxUnavailable(RuntimeError):
    pass


@dataclass
class RunResult:
    backend: str
    isolated: bool
    language: str
    exit_code: Optional[int]
    stdout: str
    stderr: str
    timed_out: bool
    duration_s: float
    workspace_id: Optional[str]
    files: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "isolated": self.isolated,
            "language": self.language,
            "exit_code": self.exit_code,
            "ok": self.exit_code == 0 and not self.timed_out,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "timed_out": self.timed_out,
            "duration_s": round(self.duration_s, 3),
            "workspace_id": self.workspace_id,
            "files": self.files,
        }


def _root() -> Path:
    directory = state_dir()
    if str(directory) == ":memory:":
        return Path(tempfile.gettempdir()) / "chat_software_sandbox"
    return directory / "sandbox"


def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


class Sandbox:
    def __init__(self, image: Optional[str] = None) -> None:
        self.image = image or os.getenv("CHAT_SOFTWARE_SANDBOX_IMAGE", DEFAULT_IMAGE)

    # ---- capability detection -------------------------------------------------------------
    def docker_state(self) -> dict[str, Any]:
        docker = shutil.which("docker")
        if not docker:
            return {"available": False, "reason": "docker is not installed"}
        try:
            info = subprocess.run([docker, "info", "--format", "{{.ServerVersion}}"],
                                  capture_output=True, text=True, timeout=10)
        except (OSError, subprocess.TimeoutExpired):
            return {"available": False, "reason": "docker daemon did not respond"}
        if info.returncode != 0:
            return {"available": False, "reason": "docker daemon is not running"}
        image = subprocess.run([docker, "image", "inspect", self.image], capture_output=True, timeout=10)
        return {"available": True, "version": info.stdout.strip(), "image": self.image,
                "image_present": image.returncode == 0}

    def seatbelt_available(self) -> bool:
        return platform.system() == "Darwin" and Path("/usr/bin/sandbox-exec").exists()

    def status(self) -> dict[str, Any]:
        docker = self.docker_state()
        backends = []
        if docker.get("available") and docker.get("image_present"):
            backends.append("docker")
        if self.seatbelt_available():
            backends.append("seatbelt")
        if _truthy(os.getenv("CHAT_SOFTWARE_SANDBOX_ALLOW_UNISOLATED")):
            backends.append("process")
        try:
            active = self.select_backend()
        except SandboxUnavailable as exc:
            active, reason = None, str(exc)
        else:
            reason = None
        return {"active": active, "backends": backends, "docker": docker, "reason": reason,
                "languages": sorted(LANGUAGES), "isolated": active in {"docker", "seatbelt"}}

    def select_backend(self) -> str:
        forced = os.getenv("CHAT_SOFTWARE_SANDBOX_BACKEND", "").strip().lower()
        allow_process = _truthy(os.getenv("CHAT_SOFTWARE_SANDBOX_ALLOW_UNISOLATED"))
        if forced:
            if forced == "process" and not allow_process:
                raise SandboxUnavailable("The unisolated process backend is disabled.")
            if forced not in {"docker", "seatbelt", "process"}:
                raise SandboxUnavailable(f"Unknown sandbox backend {forced!r}.")
            return forced
        docker = self.docker_state()
        if docker.get("available") and docker.get("image_present"):
            return "docker"
        if self.seatbelt_available():
            return "seatbelt"
        if allow_process:
            return "process"
        hint = (f" Pull the {self.image} image to enable Docker." if docker.get("available") else "")
        raise SandboxUnavailable("No isolated sandbox backend is available." + hint)

    def pull_image(self) -> dict[str, Any]:
        docker = shutil.which("docker")
        if not docker:
            raise SandboxUnavailable("docker is not installed")
        result = subprocess.run([docker, "pull", self.image], capture_output=True, text=True, timeout=1800)
        return {"image": self.image, "ok": result.returncode == 0,
                "output": (result.stdout + result.stderr)[-2000:]}

    # ---- workspaces ------------------------------------------------------------------------
    def create_workspace(self) -> str:
        workspace_id = str(uuid.uuid4())
        (_root() / workspace_id).mkdir(parents=True, exist_ok=False)
        return workspace_id

    def workspace_path(self, workspace_id: str) -> Path:
        if not _WORKSPACE_ID.match(workspace_id or ""):
            raise KeyError("Invalid workspace id")
        path = (_root() / workspace_id).resolve()
        if not path.is_dir():
            raise KeyError(f"Sandbox workspace not found: {workspace_id}")
        return path

    def delete_workspace(self, workspace_id: str) -> None:
        shutil.rmtree(self.workspace_path(workspace_id))

    def write_file(self, workspace_id: str, name: str, data: bytes) -> dict[str, Any]:
        base = self.workspace_path(workspace_id)
        target = _safe_join(base, name)
        if len(data) > 50 * 1024 * 1024:
            raise ValueError("Files are limited to 50 MB.")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return {"name": target.relative_to(base.resolve()).as_posix(), "size_bytes": len(data)}

    def list_files(self, workspace_id: str) -> list[dict[str, Any]]:
        return _list_files(self.workspace_path(workspace_id))

    # ---- execution -------------------------------------------------------------------------
    def run(
        self,
        code: str,
        *,
        language: str = "python",
        timeout: int = 30,
        workspace_id: Optional[str] = None,
        files: Optional[dict[str, str]] = None,
    ) -> RunResult:
        if language not in LANGUAGES:
            raise ValueError(f"Unsupported language {language!r}; use one of {', '.join(sorted(LANGUAGES))}.")
        if not code.strip():
            raise ValueError("No code to run.")
        timeout = max(1, min(int(timeout), MAX_TIMEOUT))
        backend = self.select_backend()
        persistent = workspace_id is not None
        if persistent:
            workdir = self.workspace_path(workspace_id)
        else:
            workdir = _root() / f"run-{uuid.uuid4()}"
            workdir.mkdir(parents=True)
            workdir = workdir.resolve()
        try:
            for name, content in (files or {}).items():
                target = _safe_join(workdir, name)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
            spec = LANGUAGES[language]
            script = workdir / spec["file"]
            script.write_text(code)
            before = {f["name"]: f["size_bytes"] for f in _list_files(workdir)}
            if backend == "docker":
                command = self._docker_command(workdir, spec, timeout)
                env = None
            else:
                interpreter = shutil.which(spec["host"][0]) or spec["host"][0]
                command = [interpreter, *spec["host"][1:], spec["file"]]
                if backend == "seatbelt":
                    command = ["/usr/bin/sandbox-exec", "-p", _seatbelt_profile(workdir, interpreter), *command]
                env = _minimal_env(workdir)
            started = time.monotonic()
            stdout, stderr, code_, timed_out = _execute(command, workdir, env, timeout,
                                                        host_limits=backend != "docker")
            duration = time.monotonic() - started
            produced = [f for f in _list_files(workdir)
                        if f["name"] != spec["file"] and before.get(f["name"]) != f["size_bytes"]]
            return RunResult(backend=backend, isolated=backend != "process", language=language,
                             exit_code=code_, stdout=stdout, stderr=stderr, timed_out=timed_out,
                             duration_s=duration, workspace_id=workspace_id, files=produced)
        finally:
            if not persistent:
                shutil.rmtree(workdir, ignore_errors=True)

    def _docker_command(self, workdir: Path, spec: dict[str, Any], timeout: int) -> list[str]:
        return [
            shutil.which("docker") or "docker", "run", "--rm", "--network", "none",
            "--memory", os.getenv("CHAT_SOFTWARE_SANDBOX_MEMORY", "512m"), "--cpus", "1",
            "--pids-limit", "128", "--read-only", "--tmpfs", "/tmp:rw,size=64m",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--user", "65534:65534", "-e", "HOME=/tmp", "-e", "PYTHONDONTWRITEBYTECODE=1",
            "-v", f"{workdir}:/work:rw", "-w", "/work",
            "--stop-timeout", "1", self.image, "timeout", "-s", "KILL", str(timeout),
            *spec["docker"], spec["file"],
        ]


def _safe_join(base: Path, name: str) -> Path:
    if not _SAFE_NAME.match(name or "") or ".." in Path(name).parts or name.startswith("/"):
        raise ValueError(f"Unsafe file name: {name!r}")
    base = base.resolve()
    target = (base / name).resolve()
    if base not in target.parents:
        raise ValueError(f"Unsafe file name: {name!r}")
    return target


def _list_files(base: Path) -> list[dict[str, Any]]:
    out = []
    for path in sorted(base.rglob("*")):
        if path.is_file() and not path.is_symlink():
            item: dict[str, Any] = {"name": path.relative_to(base).as_posix(), "size_bytes": path.stat().st_size}
            if path.suffix.lower() in {".txt", ".csv", ".json", ".md", ".py", ".log", ".html", ".tsv"}:
                item["preview"] = path.read_text(errors="replace")[:MAX_FILE_PREVIEW]
            out.append(item)
        if len(out) >= 200:
            break
    return out


def _minimal_env(workdir: Path) -> dict[str, str]:
    return {
        "PATH": "/usr/bin:/bin:/usr/sbin:/sbin:" + os.path.dirname(sys.executable),
        "HOME": str(workdir),
        "TMPDIR": str(workdir),
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "MPLBACKEND": "Agg",
    }


def _seatbelt_profile(workdir: Path, interpreter: str) -> str:
    work = str(workdir.resolve())
    allowed_reads = {work}
    for candidate in (_interpreter_root(interpreter), sys.prefix, sys.base_prefix):
        if candidate and not _is_system_path(candidate) and candidate != "/":
            allowed_reads.add(str(Path(candidate).resolve()))
    reads = " ".join(f'(subpath "{_sb(p)}")' for p in sorted(allowed_reads))
    return f"""(version 1)
(allow default)
(deny network*)
(deny file-write*)
(allow file-write* (subpath "{_sb(work)}") (literal "/dev/null") (literal "/dev/tty") (regex #"^/dev/fd/"))
(deny file-read* (subpath "/Users") (subpath "/Volumes") (subpath "/private/tmp") (subpath "/private/var/folders"))
(allow file-read-metadata)
(allow file-read* {reads})
(deny process-exec (literal "/usr/bin/sudo") (literal "/usr/bin/osascript"))
"""


def _interpreter_root(interpreter: str) -> Optional[str]:
    real = Path(os.path.realpath(interpreter))
    if _is_system_path(str(real)):
        return None  # system interpreters live in readable system directories already
    return str(real.parents[1]) if len(real.parents) > 1 else None


def _is_system_path(path: str) -> bool:
    resolved = os.path.realpath(path)
    return resolved == "/" or any(
        resolved == root or resolved.startswith(root + "/")
        for root in ("/bin", "/sbin", "/usr", "/System", "/Library/Apple", "/private/etc")
    )


def _sb(path: str) -> str:
    return str(path).replace("\\", "\\\\").replace('"', '\\"')


def _limits() -> None:  # runs in the child before exec (host backends only)
    import resource

    os.setsid()
    for limit, value in ((resource.RLIMIT_CPU, MAX_TIMEOUT), (resource.RLIMIT_FSIZE, 50 * 1024 * 1024),
                         (resource.RLIMIT_CORE, 0)):
        try:
            resource.setrlimit(limit, (value, value))
        except (ValueError, OSError):
            pass


def _execute(command, workdir, env, timeout, *, host_limits: bool):
    process = subprocess.Popen(
        command, cwd=str(workdir), env=env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        preexec_fn=_limits if host_limits else os.setsid,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout + (5 if not host_limits else 0))
    except subprocess.TimeoutExpired:
        timed_out = True
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
    code = process.returncode
    if not host_limits and code == 137:  # coreutils `timeout -s KILL` inside the container
        timed_out = True
    return _cap(stdout), _cap(stderr), code, timed_out


def _cap(data: bytes) -> str:
    text = (data or b"").decode("utf-8", errors="replace")
    if len(text) > MAX_OUTPUT:
        return text[:MAX_OUTPUT] + f"\n... [{len(text) - MAX_OUTPUT} more characters truncated]"
    return text
