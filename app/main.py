"""FastAPI entry point for the browser-testable Chat Software UI."""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_DIR / "backend"
WEB_DIR = PROJECT_DIR / "web"

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from chat_software.server import create_app as create_api_app


def _web_file(name: str) -> Path:
    path = WEB_DIR / name
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Missing UI asset: {name}")
    return path


def create_app() -> FastAPI:
    app = create_api_app()
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(_web_file("index.html"))

    @app.get("/ui", include_in_schema=False)
    def ui() -> FileResponse:
        return FileResponse(_web_file("index.html"))

    return app


app = create_app()


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


def _can_bind(host: str, port: int) -> bool:
    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    try:
        with socket.socket(family, socket.SOCK_STREAM) as sock:
            sock.bind((host, port))
    except OSError:
        return False
    return True


def _select_port(host: str, requested_port: int) -> int:
    if _can_bind(host, requested_port):
        return requested_port

    if _truthy(os.getenv("CHAT_SOFTWARE_STRICT_PORT")):
        raise RuntimeError(f"{host}:{requested_port} is already in use.")

    for candidate in range(requested_port + 1, requested_port + 51):
        if _can_bind(host, candidate):
            print(
                f"{host}:{requested_port} is already in use; "
                f"starting on http://{host}:{candidate}/ instead."
            )
            return candidate

    raise RuntimeError(f"No free port found from {requested_port} to {requested_port + 50}.")


def main() -> None:
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("Install uvicorn to run the FastAPI UI server.") from exc

    host = os.getenv("CHAT_SOFTWARE_HOST", "127.0.0.1")
    requested_port = int(os.getenv("CHAT_SOFTWARE_PORT", "8765"))
    port = _select_port(host, requested_port)
    log_level = os.getenv("CHAT_SOFTWARE_LOG_LEVEL", "info")
    uvicorn.run(app, host=host, port=port, log_level=log_level, access_log=True)


if __name__ == "__main__":
    main()
