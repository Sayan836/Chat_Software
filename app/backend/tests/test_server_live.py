"""Live-process integration test: boot the sidecar as a real application.

Unlike test_api_smoke.py (which calls the app in-process via TestClient),
this spawns the actual server the way the Tauri shell will:

    python -m chat_software.server

then talks to it over real HTTP on a real port. This is the "test the whole
backend as a built application" layer -- it catches startup failures, port
binding, config/env loading, and process-level crashes.

Run:
    PYTHONPATH=backend python -m pytest backend/tests/test_server_live.py -v

Uses only the standard library (urllib), so it needs no extra deps beyond
what the server itself requires (fastapi + uvicorn). Skips cleanly if those
are missing.
"""

from __future__ import annotations

import contextlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path


def _free_port() -> int:
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        try:
            s.bind(("127.0.0.1", 0))
        except PermissionError as exc:
            raise unittest.SkipTest(f"local socket bind is blocked in this environment: {exc}")
        return s.getsockname()[1]


def _server_deps_available() -> bool:
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401

        return True
    except Exception:
        return False


def _get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as resp:
        return json.loads(resp.read().decode())


def _post(url: str, payload: dict) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.loads(resp.read().decode())


@unittest.skipUnless(_server_deps_available(), "fastapi/uvicorn not installed")
class LiveServerTests(unittest.TestCase):
    proc: subprocess.Popen
    base: str

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        port = _free_port()
        cls.base = f"http://127.0.0.1:{port}"

        repo_backend = Path(__file__).resolve().parents[1]  # .../backend
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_backend)
        env["CHAT_SOFTWARE_HOST"] = "127.0.0.1"
        env["CHAT_SOFTWARE_PORT"] = str(port)
        env["CHAT_SOFTWARE_STATE_DIR"] = cls._tmp.name
        env["CHAT_SOFTWARE_OFFLINE_ECHO"] = "1"

        cls.proc = subprocess.Popen(
            [sys.executable, "-m", "chat_software.server"],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        # Poll /health until the app is ready (or fail with the server log).
        deadline = time.time() + 20
        last_err: Exception | None = None
        while time.time() < deadline:
            if cls.proc.poll() is not None:
                out = cls.proc.stdout.read() if cls.proc.stdout else ""
                raise AssertionError(f"server exited early:\n{out}")
            try:
                _get(f"{cls.base}/health")
                return  # ready
            except (urllib.error.URLError, ConnectionError) as exc:
                last_err = exc
                time.sleep(0.3)
        cls._stop()
        raise AssertionError(f"server did not become healthy in time: {last_err}")

    @classmethod
    def _stop(cls) -> None:
        if getattr(cls, "proc", None) and cls.proc.poll() is None:
            cls.proc.terminate()
            try:
                cls.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls.proc.kill()
                cls.proc.wait(timeout=5)
        if getattr(cls, "proc", None) and cls.proc.stdout:
            cls.proc.stdout.close()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._stop()
        cls._tmp.cleanup()

    def test_health_over_real_http(self) -> None:
        body = _get(f"{self.base}/health")
        self.assertTrue(body["ok"])

    def test_chat_over_real_http(self) -> None:
        body = _post(f"{self.base}/chat", {"message": "hello"})
        self.assertTrue(body["session_id"])
        self.assertIn("response", body)


if __name__ == "__main__":
    unittest.main()
