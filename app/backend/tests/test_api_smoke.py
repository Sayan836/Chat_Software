"""End-to-end smoke tests for the HTTP sidecar.

These exercise the real FastAPI endpoints in-process using Starlette's
TestClient -- no running server, no network, no Ollama/cloud keys needed.

Run:
    PYTHONPATH=backend python -m pytest backend/tests/test_api_smoke.py -v

Skips automatically if FastAPI/httpx are not installed, so it never breaks
the stdlib-only test suite.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


try:
    from fastapi.testclient import TestClient  # noqa: F401

    _HAS_FASTAPI = True
except Exception:  # pragma: no cover - depends on optional deps
    _HAS_FASTAPI = False


@unittest.skipUnless(_HAS_FASTAPI, "FastAPI/httpx not installed")
class ApiSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        # Isolate the SQLite session store in a temp dir per test run.
        self._tmp = tempfile.TemporaryDirectory()
        os.environ["CHAT_SOFTWARE_STATE_DIR"] = self._tmp.name
        os.environ["CHAT_SOFTWARE_OFFLINE_ECHO"] = "1"
        from fastapi.testclient import TestClient
        from chat_software.server import create_app

        self.client = TestClient(create_app())

    def tearDown(self) -> None:
        os.environ.pop("CHAT_SOFTWARE_OFFLINE_ECHO", None)
        self._tmp.cleanup()

    def test_health_ok(self) -> None:
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["ok"])
        self.assertIn("provider", body)
        self.assertIn("model", body)

    def test_favicon_is_quiet_no_content(self) -> None:
        resp = self.client.get("/favicon.ico")
        self.assertEqual(resp.status_code, 204)

    def test_chat_cors_preflight_allows_local_frontend(self) -> None:
        resp = self.client.options(
            "/chat",
            headers={
                "Origin": "http://127.0.0.1:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["access-control-allow-origin"], "http://127.0.0.1:5173")

    def test_providers_lists_models(self) -> None:
        resp = self.client.get("/providers")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("active", body)
        self.assertIsInstance(body["models"], list)
        self.assertGreater(len(body["models"]), 0)

    def test_model_options_lists_available_provider_choices(self) -> None:
        with patch(
            "chat_software.server.ollama.list_models",
            return_value={
                "base_url": "http://localhost:11434",
                "models": [{"name": "llama3.2:3b"}],
            },
        ):
            resp = self.client.get("/model-options")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        providers = {item["provider"]: item for item in body["providers"]}
        self.assertIn("ollama", providers)
        self.assertIn("openai", providers)
        self.assertIn("anthropic", providers)
        self.assertEqual(providers["ollama"]["installed_count"], 1)
        self.assertEqual(body["ollama_installed"]["model_names"], ["llama3.2:3b"])
        self.assertEqual(body["ollama_installed"]["recommended_model"], "llama3.2:3b")
        self.assertEqual(body["active"]["model"], "llama3.2:3b")
        self.assertIn("qwen3:8b", body["ollama_catalog"])

    def test_chat_uses_newest_installed_ollama_model_when_configured_model_is_missing(self) -> None:
        with patch(
            "chat_software.server.ollama.list_models",
            return_value={
                "base_url": "http://localhost:11434",
                "models": [
                    {"name": "older:latest", "modified_at": "2026-01-01T00:00:00Z"},
                    {"name": "newer:latest", "modified_at": "2026-07-01T00:00:00Z"},
                ],
            },
        ):
            resp = self.client.post(
                "/chat",
                json={"message": "hello", "provider": "ollama"},
            )

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["model"], "newer:latest")
        self.assertIn("ollama/newer:latest", resp.json()["response"])

    def test_chat_uses_explicitly_selected_installed_ollama_model(self) -> None:
        with patch(
            "chat_software.server.ollama.list_models",
            return_value={
                "base_url": "http://localhost:11434",
                "models": [
                    {"name": "newer:latest", "modified_at": "2026-07-01T00:00:00Z"},
                    {"name": "chosen:latest", "modified_at": "2026-01-01T00:00:00Z"},
                ],
            },
        ):
            with self.client.stream(
                "POST",
                "/chat/stream",
                json={
                    "message": "hello",
                    "provider": "ollama",
                    "model": "chosen:latest",
                },
            ) as resp:
                events = [json.loads(line) for line in resp.iter_lines() if line]

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(events[0]["model"], "chosen:latest")
        self.assertEqual(events[-1]["data"]["model"], "chosen:latest")
        self.assertIn("ollama/chosen:latest", events[-1]["data"]["response"])

    def test_chat_stream_file_ops_route_falls_back_to_chat_response(self) -> None:
        root = Path(self._tmp.name) / "approved"
        root.mkdir()
        approved = self.client.post("/roots/approve", json={"path": str(root)})
        self.assertEqual(approved.status_code, 200)

        with self.client.stream(
            "POST",
            "/chat/stream",
            json={
                "message": "explain folder organization",
                "roots": [approved.json()["path"]],
                "provider": "openai",
                "model": "gpt-test",
            },
        ) as resp:
            events = [json.loads(line) for line in resp.iter_lines() if line]

        self.assertEqual(resp.status_code, 200)
        completed = events[-1]["data"]
        self.assertNotEqual(completed["response"], "__CHAT_FALLBACK__")
        self.assertIn("openai/gpt-test", completed["response"])

    def test_chat_accepts_provider_and_model_override(self) -> None:
        resp = self.client.post(
            "/chat",
            json={"message": "hello", "provider": "openai", "model": "gpt-test"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertIn("openai/gpt-test", resp.json()["response"])

    def test_chat_model_error_guidance_does_not_suggest_smoke_mode(self) -> None:
        os.environ.pop("CHAT_SOFTWARE_OFFLINE_ECHO", None)
        with patch(
            "chat_software.server.get_chat_client",
            side_effect=RuntimeError("404 page not found"),
        ):
            resp = self.client.post(
                "/chat",
                json={"message": "hello", "provider": "ollama", "model": "qwen3:8b"},
            )

        self.assertEqual(resp.status_code, 502)
        detail = resp.json()["detail"]
        self.assertIn("Ollama", detail)
        self.assertIn("http://localhost:11434/v1", detail)
        self.assertNotIn("CHAT_SOFTWARE_OFFLINE_ECHO", detail)

    def test_chat_file_tool_errors_are_not_reported_as_model_failures(self) -> None:
        root = Path(self._tmp.name)
        approved = self.client.post("/roots/approve", json={"path": str(root)})
        self.assertEqual(approved.status_code, 200)

        resp = self.client.post(
            "/chat",
            json={"message": "read file missing.txt", "roots": [approved.json()["path"]]},
        )

        self.assertEqual(resp.status_code, 400)
        detail = resp.json()["detail"]
        self.assertIn("missing.txt", detail)
        self.assertNotIn("Model call failed", detail)

    def test_ollama_model_management_endpoints_are_exposed(self) -> None:
        with (
            patch(
                "chat_software.server.ollama.list_models",
                return_value={"base_url": "http://localhost:11434", "models": [{"name": "qwen3:8b"}]},
            ),
            patch(
                "chat_software.server.ollama.pull_model",
                return_value={"model": "llama3.2:3b", "result": {"status": "success"}},
            ) as pull_model,
            patch(
                "chat_software.server.ollama.delete_model",
                return_value={"model": "qwen3:8b", "result": {}},
            ) as delete_model,
        ):
            listed = self.client.get("/ollama/models")
            pulled = self.client.post("/ollama/models/pull", json={"model": "llama3.2:3b"})
            deleted = self.client.request("DELETE", "/ollama/models", json={"model": "qwen3:8b"})

        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json()["models"][0]["name"], "qwen3:8b")
        self.assertEqual(pulled.status_code, 200)
        self.assertEqual(deleted.status_code, 200)
        pull_model.assert_called_once_with("llama3.2:3b")
        delete_model.assert_called_once_with("qwen3:8b")

    def test_root_approval_and_file_command_endpoints(self) -> None:
        root = Path(self._tmp.name)
        (root / "notes.md").write_text("Acme renewal is due.\n", encoding="utf-8")

        approved = self.client.post("/roots/approve", json={"path": str(root)})
        self.assertEqual(approved.status_code, 200)
        approved_root = approved.json()["path"]
        approved_root_id = approved.json()["root_id"]
        self.assertTrue(approved_root_id)

        roots = self.client.get("/roots")
        self.assertEqual(roots.status_code, 200)
        self.assertEqual(roots.json()["roots"][0]["root_id"], approved_root_id)

        listed = self.client.post("/files/list", json={"path": ".", "roots": [approved_root]})
        self.assertEqual(listed.status_code, 200)
        self.assertEqual(listed.json()["entries"][0]["name"], "notes.md")

        read = self.client.post("/files/read", json={"path": "notes.md", "roots": [approved_root]})
        self.assertEqual(read.status_code, 200)
        self.assertIn("Acme", read.json()["content"])

        made = self.client.post("/files/mkdir", json={"path": "Generated", "roots": [approved_root]})
        self.assertEqual(made.status_code, 200)
        self.assertTrue((root / "Generated").is_dir())

        written = self.client.post(
            "/files/write",
            json={"path": "Generated/notes.txt", "content": "Beta\n", "roots": [approved_root]},
        )
        self.assertEqual(written.status_code, 200)
        self.assertTrue(written.json()["created"])

        copied = self.client.post(
            "/files/copy",
            json={"source": "Generated/notes.txt", "destination": "Generated/copy.txt", "roots": [approved_root]},
        )
        self.assertEqual(copied.status_code, 200)
        self.assertEqual((root / "Generated" / "copy.txt").read_text(encoding="utf-8"), "Beta\n")

        edited = self.client.post(
            "/files/edit",
            json={
                "path": "Generated/copy.txt",
                "old_text": "Beta",
                "new_text": "Gamma",
                "roots": [approved_root],
            },
        )
        self.assertEqual(edited.status_code, 200)
        self.assertEqual(edited.json()["replacements"], 1)

        moved = self.client.post(
            "/files/move",
            json={"source": "Generated/copy.txt", "destination": "Archive/copy.txt", "roots": [approved_root]},
        )
        self.assertEqual(moved.status_code, 200)
        self.assertFalse((root / "Generated" / "copy.txt").exists())
        self.assertEqual((root / "Archive" / "copy.txt").read_text(encoding="utf-8"), "Gamma\n")

        searched = self.client.post(
            "/files/grep",
            json={"query": "Acme", "root": ".", "roots": [approved_root]},
        )
        self.assertEqual(searched.status_code, 200)
        self.assertEqual(len(searched.json()["matches"]), 1)

        globbed = self.client.post(
            "/files/glob",
            json={"pattern": "*.md", "root": ".", "roots": [approved_root]},
        )
        self.assertEqual(globbed.status_code, 200)
        self.assertEqual(len(globbed.json()["matches"]), 1)

    def test_file_endpoints_reject_out_of_scope_paths(self) -> None:
        root = Path(self._tmp.name)
        approved = self.client.post("/roots/approve", json={"path": str(root)})
        self.assertEqual(approved.status_code, 200)
        approved_root = approved.json()["path"]

        unapproved = root / "unapproved"
        unapproved.mkdir()
        unapproved_resp = self.client.post(
            "/files/list",
            json={"path": ".", "roots": [str(unapproved)]},
        )
        self.assertEqual(unapproved_resp.status_code, 400)
        self.assertIn("not approved", unapproved_resp.json()["detail"])

        resp = self.client.post("/files/read", json={"path": "/etc/hosts", "roots": [approved_root]})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("outside approved roots", resp.json()["detail"])

        write_resp = self.client.post(
            "/files/write",
            json={"path": "/etc/chat-software-test.txt", "content": "nope", "roots": [approved_root]},
        )
        self.assertEqual(write_resp.status_code, 400)
        self.assertIn("outside approved roots", write_resp.json()["detail"])

    def test_organize_endpoints_preview_apply_and_undo(self) -> None:
        root = Path(self._tmp.name) / "approved"
        root.mkdir()
        (root / "report.txt").write_text("draft\n", encoding="utf-8")

        approved = self.client.post("/roots/approve", json={"path": str(root)})
        self.assertEqual(approved.status_code, 200)
        approved_root_id = approved.json()["root_id"]

        planned = self.client.post(
            "/organize/plan",
            json={"root_id": approved_root_id, "strategy": "by_type"},
        )
        self.assertEqual(planned.status_code, 200)
        plan = planned.json()
        self.assertEqual(plan["status"], "planned")
        self.assertEqual(plan["summary"]["moves"], 1)
        self.assertTrue((root / "report.txt").exists())
        self.assertFalse((root / "Documents" / "report.txt").exists())

        applied = self.client.post("/organize/apply", json={"plan_id": plan["plan_id"]})
        self.assertEqual(applied.status_code, 200)
        self.assertEqual(applied.json()["status"], "applied")
        self.assertFalse((root / "report.txt").exists())
        self.assertTrue((root / "Documents" / "report.txt").exists())

        undone = self.client.post("/organize/undo", json={"plan_id": plan["plan_id"]})
        self.assertEqual(undone.status_code, 200)
        self.assertEqual(undone.json()["status"], "undone")
        self.assertTrue((root / "report.txt").exists())
        self.assertFalse((root / "Documents").exists())

    def test_chat_organize_intent_creates_preview_only(self) -> None:
        root = Path(self._tmp.name) / "approved"
        root.mkdir()
        (root / "report.txt").write_text("draft\n", encoding="utf-8")

        approved = self.client.post("/roots/approve", json={"path": str(root)})
        self.assertEqual(approved.status_code, 200)
        approved_root = approved.json()["path"]

        resp = self.client.post(
            "/chat",
            json={"message": "organize my files by type", "roots": [approved_root]},
        )

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["tool_calls"][0]["name"], "organize_plan")
        self.assertIn("command", body["tool_calls"][0])
        self.assertIn("Nothing has been moved yet.", body["response"])
        self.assertIn("Reply yes/approve", body["response"])
        self.assertIn("Tools run:", body["response"])
        self.assertTrue((root / "report.txt").exists())
        self.assertFalse((root / "Documents" / "report.txt").exists())

        approved_apply = self.client.post(
            "/chat",
            json={
                "message": "yes",
                "session_id": body["session_id"],
                "roots": [approved_root],
            },
        )

        self.assertEqual(approved_apply.status_code, 200)
        apply_body = approved_apply.json()
        self.assertEqual(apply_body["tool_calls"][0]["name"], "organize_apply")
        self.assertTrue((root / "Documents" / "report.txt").exists())

    def test_chat_uses_selected_working_path_for_file_intents(self) -> None:
        root = Path(self._tmp.name) / "approved"
        selected = root / "selected"
        selected.mkdir(parents=True)
        (root / "root-only.txt").write_text("root\n", encoding="utf-8")
        (selected / "inside.txt").write_text("inside\n", encoding="utf-8")

        approved = self.client.post("/roots/approve", json={"path": str(root)})
        self.assertEqual(approved.status_code, 200)
        approved_root = approved.json()["path"]

        resp = self.client.post(
            "/chat",
            json={
                "message": "check selected directory path",
                "roots": [approved_root],
                "working_path": "selected",
            },
        )

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["tool_calls"][0]["name"], "list_dir")
        self.assertEqual(body["tool_calls"][0]["arguments"]["path"], "selected")
        self.assertIn("inside.txt", body["response"])
        self.assertNotIn("root-only.txt", body["response"])

    def test_git_command_endpoints_are_exposed(self) -> None:
        approved = self.client.post("/roots/approve", json={"path": self._tmp.name})
        self.assertEqual(approved.status_code, 200)
        approved_root = approved.json()["path"]
        with (
            patch(
                "chat_software.server.git_status",
                return_value={"returncode": 0, "stdout": "## main\n", "stderr": ""},
            ) as status,
            patch(
                "chat_software.server.git_diff",
                return_value={"returncode": 0, "stdout": "file.py | 1 +\n", "stderr": ""},
            ) as diff,
        ):
            status_resp = self.client.post(
                "/commands/git/status",
                json={"repo_path": ".", "roots": [approved_root]},
            )
            diff_resp = self.client.post(
                "/commands/git/diff",
                json={"repo_path": ".", "roots": [approved_root], "stat_only": True},
            )

        self.assertEqual(status_resp.status_code, 200)
        self.assertEqual(diff_resp.status_code, 200)
        status.assert_called_once()
        diff.assert_called_once()

    def test_chat_roundtrip_and_session_persistence(self) -> None:
        resp = self.client.post("/chat", json={"message": "hello"})
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        session_id = body["session_id"]
        self.assertTrue(session_id)
        self.assertIn("response", body)
        self.assertIsInstance(body["tool_calls"], list)

        # The message should now be retrievable from the session store.
        history = self.client.get(f"/sessions/{session_id}")
        self.assertEqual(history.status_code, 200)
        messages = history.json()["messages"]
        self.assertTrue(any(m["role"] == "user" for m in messages))

    def test_chat_stream_reports_live_tool_events_and_final_response(self) -> None:
        root = Path(self._tmp.name) / "stream-root"
        root.mkdir()
        approved = self.client.post("/roots/approve", json={"path": str(root)})
        self.assertEqual(approved.status_code, 200)

        with self.client.stream(
            "POST",
            "/chat/stream",
            json={
                "message": "create a folder called RuntimeLogs",
                "roots": [approved.json()["path"]],
            },
        ) as resp:
            events = [json.loads(line) for line in resp.iter_lines() if line]

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(
            [event["type"] for event in events],
            ["chat_started", "chat_completed"],
        )
        self.assertFalse((root / "RuntimeLogs").exists())
        self.assertEqual(events[-1]["data"]["tool_calls"][0]["name"], "prepare_write_plan")
        self.assertEqual(events[-1]["data"]["tool_calls"][0]["status"], "pending_approval")
        self.assertIn("Pending approval", events[-1]["data"]["response"])

        session_id = events[-1]["data"]["session_id"]
        with self.client.stream(
            "POST",
            "/chat/stream",
            json={
                "message": "yes",
                "session_id": session_id,
                "roots": [approved.json()["path"]],
            },
        ) as approve_resp:
            approve_events = [json.loads(line) for line in approve_resp.iter_lines() if line]

        self.assertEqual(approve_resp.status_code, 200)
        self.assertEqual(
            [event["type"] for event in approve_events],
            ["chat_started", "tool_started", "tool_completed", "chat_completed"],
        )
        self.assertEqual(approve_events[1]["name"], "make_dir")
        self.assertEqual(approve_events[2]["status"], "completed")
        self.assertTrue((root / "RuntimeLogs").is_dir())
        self.assertIn("Created", approve_events[-1]["data"]["response"])

    def test_chat_stream_resolves_in_it_to_selected_working_path(self) -> None:
        root = Path(self._tmp.name) / "count-root"
        selected = root / "selected"
        selected.mkdir(parents=True)
        (selected / "one.pdf").write_text("pdf\n", encoding="utf-8")
        approved = self.client.post("/roots/approve", json={"path": str(root)})
        self.assertEqual(approved.status_code, 200)

        with self.client.stream(
            "POST",
            "/chat/stream",
            json={
                "message": "check the number of pdf files in it",
                "roots": [approved.json()["path"]],
                "working_path": "selected",
            },
        ) as resp:
            events = [json.loads(line) for line in resp.iter_lines() if line]

        self.assertEqual(resp.status_code, 200)
        self.assertEqual(
            [event["type"] for event in events],
            ["chat_started", "tool_started", "tool_completed", "chat_completed"],
        )
        self.assertEqual(events[1]["arguments"]["root"], "selected")
        self.assertIn("Found 1 PDF file(s)", events[-1]["data"]["response"])

    def test_chat_rejects_empty_message(self) -> None:
        resp = self.client.post("/chat", json={"message": ""})
        self.assertEqual(resp.status_code, 422)  # pydantic validation


if __name__ == "__main__":
    unittest.main()
