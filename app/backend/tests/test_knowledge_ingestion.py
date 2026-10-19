"""M3 selected-file ingestion: explicit preview/approve, fingerprints, scope and key separation."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

from chat_software.knowledge import (
    IngestClient,
    IngestionService,
    KnowledgeClient,
    KnowledgeConnector,
    KnowledgeForbidden,
    KnowledgeNotConfigured,
    SubmissionStateError,
)
from chat_software.safety_store import SafetyStore

from test_knowledge_connector import BASE_URL, KEY_A, FakeRetrievalLab


INGEST_KEY_A = "hik_fixture_ingest_key_for_ws_a"
INGEST_KEY_B = "hik_fixture_ingest_key_for_ws_b"


class FakeIngestLab(FakeRetrievalLab):
    """Adds the Lab's /api/ingest/* contract: one workspace per hik_ key, own-status only."""

    def __init__(self) -> None:
        super().__init__()
        self.ingest_keys = {INGEST_KEY_A: "ws-a", INGEST_KEY_B: "ws-b"}
        self.assets: dict[str, dict] = {}
        self.uploads: list[tuple[str, bytes, str]] = []

    def handle(self, request: httpx.Request) -> httpx.Response:
        if not request.url.path.startswith("/api/ingest/"):
            return super().handle(request)
        self.requests.append(request)
        auth = request.headers.get("authorization", "")
        key = auth[7:] if auth.startswith("Bearer ") else ""
        if key not in self.ingest_keys:
            return httpx.Response(401, json={"detail": "Invalid ingest key"})
        workspace = self.ingest_keys[key]
        path = request.url.path
        if path == "/api/ingest/whoami":
            return httpx.Response(
                200,
                json={
                    "workspace_id": workspace,
                    "workspace_name": self.workspaces[workspace],
                    "max_upload_bytes": 1000,
                    "extensions": [".csv", ".pdf", ".txt"],
                    "cancellation_supported": False,
                },
            )
        if path == "/api/ingest/files":
            body = request.read()
            filename = body.split(b'filename="', 1)[1].split(b'"', 1)[0].decode()
            source_ref = body.split(b'name="source_ref"\r\n\r\n', 1)[1].split(b"\r\n", 1)[0].decode()
            asset_id = f"asset-{len(self.assets) + 1}"
            self.assets[asset_id] = {"key": key, "workspace": workspace, "status": "processing"}
            self.uploads.append((filename, body, source_ref))
            return httpx.Response(
                202,
                json={"asset_id": asset_id, "document_id": f"doc-{asset_id}", "workspace_id": workspace,
                      "status": "queued"},
            )
        if path.startswith("/api/ingest/jobs/"):
            asset = self.assets.get(path.rsplit("/", 1)[-1])
            if asset is None or asset["key"] != key:
                return httpx.Response(404, json={"detail": "not found"})
            asset["status"] = "completed"
            return httpx.Response(200, json={"status": "completed", "chunks_indexed": 3, "error": None})
        return httpx.Response(404, json={"detail": "Not Found"})


class IngestionServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env = patch.dict(
            os.environ,
            {
                "CHAT_SOFTWARE_KNOWLEDGE_API_KEY": KEY_A,
                "CHAT_SOFTWARE_KNOWLEDGE_INGEST_KEY": INGEST_KEY_A,
                "CHAT_SOFTWARE_KNOWLEDGE_OTHER_INGEST_KEY": INGEST_KEY_B,
            },
        )
        self._env.start()
        self._tmp = tempfile.TemporaryDirectory()
        tmp = Path(self._tmp.name)
        self.root_dir = tmp / "policies"
        self.root_dir.mkdir()
        (self.root_dir / "ledger.csv").write_text("invoice,amount\nINV-1,10\n")
        (self.root_dir / "notes.txt").write_text("Invoices are retained for seven years.")
        (self.root_dir / "deck.pptx").write_bytes(b"PK\x03\x04deck")
        (self.root_dir / "big.txt").write_text("x" * 2000)
        (tmp / "outside.txt").write_text("not in the root")
        self.safety = SafetyStore(tmp / "safety.sqlite3")
        self.root = self.safety.approve_root(self.root_dir)
        self.lab = FakeIngestLab()
        transport = self.lab.transport()
        KnowledgeConnector(
            self.safety,
            base_url=BASE_URL,
            client_factory=lambda url, key: KnowledgeClient(url, key, transport=transport),
        ).bind(self.root.id, "ws-a")
        self.service = IngestionService(
            self.safety,
            base_url=BASE_URL,
            client_factory=lambda url, key: IngestClient(url, key, transport=transport),
        )

    def tearDown(self) -> None:
        self._env.stop()
        self._tmp.cleanup()

    def test_preview_sends_nothing_and_approve_uploads_only_listed_files(self) -> None:
        self.service.enable(self.root.id)
        plan = self.service.preview(
            self.root.id,
            ["ledger.csv", "notes.txt", "deck.pptx", "big.txt", "../outside.txt", "missing.txt"],
        )
        self.assertEqual(self.lab.uploads, [])
        self.assertEqual([f["rel_path"] for f in plan["files"]], ["ledger.csv", "notes.txt"])
        reasons = {item["path"]: item["reason"] for item in plan["rejected"]}
        self.assertIn("unsupported type .pptx", reasons["deck.pptx"])
        self.assertIn("larger than the backend limit", reasons["big.txt"])
        self.assertEqual(reasons["../outside.txt"], "outside the approved root")
        self.assertEqual(reasons["missing.txt"], "file not found")
        self.assertEqual(plan["destination"]["workspace_id"], "ws-a")
        self.assertEqual(plan["destination"]["location"], "this computer")
        self.assertEqual(plan["status"], "awaiting_approval")

        done = self.service.approve(plan["submission_id"])
        self.assertEqual([u[0] for u in self.lab.uploads], ["ledger.csv", "notes.txt"])
        self.assertEqual([u[2] for u in self.lab.uploads], ["ledger.csv", "notes.txt"])
        self.assertEqual({f["status"] for f in done["files"]}, {"completed"})
        self.assertEqual(done["status"], "finished")
        self.assertEqual(done["files"][0]["chunks_indexed"], 3)

    def test_file_changed_after_preview_is_not_uploaded(self) -> None:
        self.service.enable(self.root.id)
        plan = self.service.preview(self.root.id, ["notes.txt"])
        (self.root_dir / "notes.txt").write_text("Edited after the user approved the preview.")
        done = self.service.approve(plan["submission_id"])
        self.assertEqual(self.lab.uploads, [])
        self.assertEqual(done["files"][0]["status"], "changed")

    def test_preview_can_only_be_approved_once(self) -> None:
        self.service.enable(self.root.id)
        plan = self.service.preview(self.root.id, ["notes.txt"])
        self.service.approve(plan["submission_id"])
        with self.assertRaises(SubmissionStateError):
            self.service.approve(plan["submission_id"])
        self.assertEqual(len(self.lab.uploads), 1)

    def test_ingest_key_for_another_workspace_is_refused(self) -> None:
        with self.assertRaises(KnowledgeForbidden):
            self.service.enable(self.root.id, "CHAT_SOFTWARE_KNOWLEDGE_OTHER_INGEST_KEY")
        with self.assertRaises(PermissionError):
            self.service.preview(self.root.id, ["notes.txt"])

    def test_key_swapped_after_preview_is_caught_before_upload(self) -> None:
        self.service.enable(self.root.id)
        plan = self.service.preview(self.root.id, ["notes.txt"])
        with patch.dict(os.environ, {"CHAT_SOFTWARE_KNOWLEDGE_INGEST_KEY": INGEST_KEY_B}):
            with self.assertRaises(KnowledgeForbidden):
                self.service.approve(plan["submission_id"])
        self.assertEqual(self.lab.uploads, [])

    def test_revoked_root_blocks_approve_and_status(self) -> None:
        self.service.enable(self.root.id)
        plan = self.service.preview(self.root.id, ["notes.txt"])
        self.safety.revoke_root(self.root.id)
        with self.assertRaises(PermissionError):
            self.service.approve(plan["submission_id"])
        self.assertEqual(self.lab.uploads, [])

    def test_ingest_requires_explicit_enable_and_a_hik_key(self) -> None:
        with self.assertRaises(PermissionError):
            self.service.preview(self.root.id, ["notes.txt"])
        with patch.dict(os.environ, {"CHAT_SOFTWARE_KNOWLEDGE_INGEST_KEY": KEY_A}):
            with self.assertRaises(KnowledgeNotConfigured):
                self.service.enable(self.root.id)  # read-only hrk_ key is not an ingest key
        with self.assertRaises(KnowledgeNotConfigured):
            self.service.enable(self.root.id, "OPENAI_API_KEY")

    def test_discard_closes_a_pending_preview(self) -> None:
        self.service.enable(self.root.id)
        plan = self.service.preview(self.root.id, ["notes.txt"])
        self.service.discard(plan["submission_id"])
        with self.assertRaises(SubmissionStateError):
            self.service.approve(plan["submission_id"])

    def test_audit_trail_records_fingerprints_not_contents(self) -> None:
        self.service.enable(self.root.id)
        plan = self.service.preview(self.root.id, ["notes.txt"])
        self.service.approve(plan["submission_id"])
        with self.safety._connect() as conn:
            rows = conn.execute(
                "SELECT event_type, path, metadata_json FROM audit_log WHERE event_type LIKE 'knowledge.ingest%'"
            ).fetchall()
        events = [row["event_type"] for row in rows]
        self.assertIn("knowledge.ingest_file_submitted", events)
        dump = json.dumps([dict(row) for row in rows])
        self.assertIn(plan["files"][0]["sha256"], dump)
        self.assertNotIn("seven years", dump)


class IngestionApiTests(unittest.TestCase):
    def setUp(self) -> None:
        from fastapi.testclient import TestClient

        from chat_software.server import create_app

        self._tmp = tempfile.TemporaryDirectory()
        self._env = patch.dict(
            os.environ,
            {
                "CHAT_SOFTWARE_STATE_DIR": self._tmp.name,
                "CHAT_SOFTWARE_KNOWLEDGE_URL": BASE_URL,
                "CHAT_SOFTWARE_KNOWLEDGE_API_KEY": KEY_A,
                "CHAT_SOFTWARE_KNOWLEDGE_INGEST_KEY": INGEST_KEY_A,
            },
        )
        self._env.start()
        root_dir = Path(self._tmp.name) / "docs"
        root_dir.mkdir()
        (root_dir / "notes.txt").write_text("Invoices are retained for seven years.")
        app = create_app()
        self.lab = FakeIngestLab()
        transport = self.lab.transport()
        app.state.knowledge._client_factory = lambda url, key: KnowledgeClient(url, key, transport=transport)
        app.state.ingestion._client_factory = lambda url, key: IngestClient(url, key, transport=transport)
        self.client = TestClient(app)
        self.root_id = self.client.post("/roots/approve", json={"path": str(root_dir)}).json()["root_id"]
        self.client.post("/knowledge/bindings", json={"root_id": self.root_id, "workspace_id": "ws-a"})

    def tearDown(self) -> None:
        self._env.stop()
        self._tmp.cleanup()

    def test_enable_preview_approve_status_flow(self) -> None:
        enabled = self.client.post("/knowledge/ingest/enable", json={"root_id": self.root_id})
        self.assertEqual(enabled.status_code, 200, enabled.text)
        self.assertNotIn(INGEST_KEY_A, enabled.text)
        bindings = self.client.get("/knowledge/bindings").json()["bindings"]
        self.assertTrue(bindings[0]["ingest_enabled"])
        self.assertNotIn(INGEST_KEY_A, json.dumps(bindings))

        plan = self.client.post(
            "/knowledge/ingest/preview", json={"root_id": self.root_id, "paths": ["notes.txt"]}
        ).json()
        self.assertEqual(self.lab.uploads, [])
        approved = self.client.post(f"/knowledge/ingest/{plan['submission_id']}/approve")
        self.assertEqual(approved.status_code, 200, approved.text)
        self.assertEqual(approved.json()["files"][0]["status"], "completed")
        again = self.client.post(f"/knowledge/ingest/{plan['submission_id']}/approve")
        self.assertEqual(again.status_code, 409)
        status = self.client.get(f"/knowledge/ingest/{plan['submission_id']}")
        self.assertEqual(status.json()["status"], "finished")
        self.assertEqual(self.client.get("/knowledge/ingest/unknown").status_code, 404)

    def test_preview_without_enable_is_forbidden(self) -> None:
        resp = self.client.post(
            "/knowledge/ingest/preview", json={"root_id": self.root_id, "paths": ["notes.txt"]}
        )
        self.assertEqual(resp.status_code, 403)


if __name__ == "__main__":
    unittest.main()
