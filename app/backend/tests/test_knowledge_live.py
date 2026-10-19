"""Live M1 contract check against a real Hybrid Retrieval Lab process.

Skipped unless CHAT_SOFTWARE_RETRIEVAL_LAB_ROOT points at a Retrieval Lab checkout
with its own ``.venv``. The fixture launcher (scripts/retrieval_lab_fixture.py) runs
the Lab's real app, auth, scoped keys and Chroma/BM25 engine with hash embeddings
over a synthetic corpus in a temp directory.

    CHAT_SOFTWARE_RETRIEVAL_LAB_ROOT=/path/to/SemanticSearchPlatform \\
        python -m pytest backend/tests/test_knowledge_live.py -v
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx


LAB_ROOT = os.getenv("CHAT_SOFTWARE_RETRIEVAL_LAB_ROOT", "").strip()
APP_DIR = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@unittest.skipUnless(LAB_ROOT, "Set CHAT_SOFTWARE_RETRIEVAL_LAB_ROOT to run the live contract check")
class LiveRetrievalLabContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        lab = Path(LAB_ROOT)
        python = lab / ".venv" / "bin" / "python"
        if not python.exists():
            raise unittest.SkipTest(f"No Retrieval Lab interpreter at {python}")
        cls._tmp = tempfile.TemporaryDirectory()
        config_path = Path(cls._tmp.name) / "fixture.json"
        port = _free_port()
        cls._log = open(Path(cls._tmp.name) / "fixture.log", "w")
        cls._proc = subprocess.Popen(
            [
                str(python),
                str(APP_DIR / "scripts" / "retrieval_lab_fixture.py"),
                "--lab-root",
                str(lab),
                "--port",
                str(port),
                "--config-out",
                str(config_path),
            ],
            stdout=cls._log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.time() + 120
        while time.time() < deadline:
            if cls._proc.poll() is not None:
                raise RuntimeError(f"Fixture exited early; see {cls._log.name}")
            if config_path.exists() and config_path.stat().st_size:
                try:
                    httpx.get(f"http://127.0.0.1:{port}/api/tools/list_workspaces", timeout=2)
                    break
                except httpx.HTTPError:
                    pass
            time.sleep(0.5)
        else:
            cls.tearDownClass()
            raise RuntimeError("Retrieval Lab fixture did not start in time")
        cls.config = json.loads(config_path.read_text())

    @classmethod
    def tearDownClass(cls) -> None:
        proc = getattr(cls, "_proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
        if getattr(cls, "_log", None):
            cls._log.close()
        if getattr(cls, "_tmp", None):
            cls._tmp.cleanup()

    def setUp(self) -> None:
        from fastapi.testclient import TestClient

        from chat_software.server import create_app

        self._state = tempfile.TemporaryDirectory()
        policies = self.config["workspaces"]["policies"]
        self._env = patch.dict(
            os.environ,
            {
                "CHAT_SOFTWARE_STATE_DIR": self._state.name,
                "CHAT_SOFTWARE_KNOWLEDGE_URL": self.config["base_url"],
                "CHAT_SOFTWARE_KNOWLEDGE_API_KEY": policies["api_key"],
                "CHAT_SOFTWARE_KNOWLEDGE_INGEST_KEY": policies["ingest_key"],
                "CHAT_SOFTWARE_KNOWLEDGE_OTHER_INGEST_KEY": self.config["workspaces"]["other"][
                    "ingest_key"
                ],
            },
        )
        self._env.start()
        self.root_dir = Path(self._state.name) / "policies"
        self.root_dir.mkdir()
        self.client = TestClient(create_app())
        self.root_id = self.client.post("/roots/approve", json={"path": str(self.root_dir)}).json()[
            "root_id"
        ]

    def tearDown(self) -> None:
        self._env.stop()
        self._state.cleanup()

    def _bind(self, workspace_id: str) -> httpx.Response:
        return self.client.post(
            "/knowledge/bindings", json={"root_id": self.root_id, "workspace_id": workspace_id}
        )

    def test_prepared_corpus_search_and_source_page(self) -> None:
        policies = self.config["workspaces"]["policies"]
        self.assertEqual(self._bind(policies["id"]).status_code, 200)
        resp = self.client.post(
            "/knowledge/search",
            json={"root_id": self.root_id, "query": "How long are supplier invoices retained?"},
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        body = resp.json()
        self.assertEqual(body["content_trust"], "untrusted_document_data")
        top = body["evidence"][0]
        self.assertEqual(top["filename"], "retention-policy.txt")
        self.assertIn("seven years", top["text"])
        self.assertTrue(top["signals"])
        self.assertNotIn(policies["api_key"], resp.text)
        names = {item["filename"] for item in body["evidence"]}
        self.assertNotIn("other-team-notes.txt", names)

        page = self.client.post(
            "/knowledge/source", json={"root_id": self.root_id, "document_id": top["document_id"]}
        )
        self.assertEqual(page.status_code, 200, page.text)
        self.assertIn("seven years", page.json()["text"])

    def test_wrong_corpus_is_denied(self) -> None:
        policies = self.config["workspaces"]["policies"]
        other = self.config["workspaces"]["other"]
        self.assertEqual(self._bind(other["id"]).status_code, 403)
        self.assertEqual(self._bind(policies["id"]).status_code, 200)
        foreign = self.client.post(
            "/knowledge/source",
            json={
                "root_id": self.root_id,
                "document_id": other["documents"]["other-team-notes.txt"],
            },
        )
        self.assertEqual(foreign.status_code, 403)
        merger = self.client.post(
            "/knowledge/search", json={"root_id": self.root_id, "query": "merger codename BLUE HERON"}
        ).json()
        self.assertNotIn("BLUE HERON", json.dumps(merger["evidence"]))
        self.assertNotIn(
            other["documents"]["other-team-notes.txt"],
            {item["document_id"] for item in merger["evidence"]},
        )

    def test_revoked_root_is_denied(self) -> None:
        policies = self.config["workspaces"]["policies"]
        self.assertEqual(self._bind(policies["id"]).status_code, 200)
        self.assertEqual(self.client.post(f"/roots/{self.root_id}/revoke").status_code, 200)
        for path, payload in (
            ("/knowledge/search", {"root_id": self.root_id, "query": "invoices"}),
            (
                "/knowledge/source",
                {"root_id": self.root_id, "document_id": policies["documents"]["retention-policy.txt"]},
            ),
        ):
            self.assertEqual(self.client.post(path, json=payload).status_code, 403)

    def _documents_chat(self, message: str, reply: str) -> tuple[dict, list]:
        from test_document_agent import ScriptedChatClient

        model = ScriptedChatClient(reply)
        with patch("chat_software.server.get_chat_client", lambda **kwargs: model):
            resp = self.client.post(
                "/chat", json={"message": message, "knowledge_root_id": self.root_id}
            )
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json(), model.calls

    def test_m2_sourced_answer_opens_its_source(self) -> None:
        policies = self.config["workspaces"]["policies"]
        self.assertEqual(self._bind(policies["id"]).status_code, 200)
        body, calls = self._documents_chat(
            "How long are supplier invoices retained?",
            "Supplier invoices are retained for seven years [1].",
        )
        self.assertEqual(len(calls), 1)
        context = calls[0][0]["content"]
        self.assertIn("seven years", context)
        self.assertNotIn("BLUE HERON", context)
        self.assertIn("[1] retention-policy.txt", body["response"])
        documents = self.client.get(f"/sessions/{body['session_id']}").json()["messages"][-1][
            "metadata"
        ]["documents"]
        cited = documents["citations"][0]["document_id"]
        self.assertEqual(cited, policies["documents"]["retention-policy.txt"])
        source = self.client.post(
            "/knowledge/source", json={"root_id": self.root_id, "document_id": cited}
        )
        self.assertEqual(source.status_code, 200)
        self.assertIn("seven years", source.json()["text"])

    def test_m2_missing_evidence_refuses_without_model_call(self) -> None:
        policies = self.config["workspaces"]["policies"]
        self.assertEqual(self._bind(policies["id"]).status_code, 200)
        body, calls = self._documents_chat(
            "What is the confidential merger codename?", "BLUE HERON [1]."
        )
        self.assertEqual(calls, [])
        self.assertTrue(
            body["response"].startswith(
                "I don't have enough information in the provided documents"
            )
        )
        self.assertNotIn("BLUE HERON", body["response"])


    def test_m3_selected_file_ingestion_covers_rows_past_the_old_cap(self) -> None:
        policies = self.config["workspaces"]["policies"]
        self.assertEqual(self._bind(policies["id"]).status_code, 200)
        other_key = self.client.post(
            "/knowledge/ingest/enable",
            json={"root_id": self.root_id, "credential_ref": "CHAT_SOFTWARE_KNOWLEDGE_OTHER_INGEST_KEY"},
        )
        self.assertEqual(other_key.status_code, 403, "an ingest key for another workspace is refused")
        enabled = self.client.post("/knowledge/ingest/enable", json={"root_id": self.root_id})
        self.assertEqual(enabled.status_code, 200, enabled.text)
        self.assertIn(".csv", enabled.json()["limits"]["extensions"])

        rows = ["ref,total,vendor"] + [
            f"REF-{i:05d},{i}.00,{'Zephyr Logistics' if i == 250 else 'Acme Hardware'}"
            for i in range(1, 301)
        ]
        (self.root_dir / "vendor-ledger.csv").write_text("\n".join(rows) + "\n")
        (self.root_dir / "private-not-selected.txt").write_text("Not selected; must never be uploaded.")

        plan = self.client.post(
            "/knowledge/ingest/preview",
            json={"root_id": self.root_id, "paths": ["vendor-ledger.csv"]},
        ).json()
        self.assertEqual([f["rel_path"] for f in plan["files"]], ["vendor-ledger.csv"])
        self.assertEqual(plan["destination"]["workspace_id"], policies["id"])
        done = self.client.post(f"/knowledge/ingest/{plan['submission_id']}/approve")
        self.assertEqual(done.status_code, 200, done.text)
        item = done.json()["files"][0]
        self.assertEqual(item["status"], "completed", item)
        self.assertGreater(item["chunks_indexed"], 1)

        search = self.client.post(
            "/knowledge/search",
            json={"root_id": self.root_id, "query": "Which vendor is on REF-00250 Zephyr Logistics?"},
        ).json()
        top = search["evidence"][0]
        self.assertEqual(top["filename"], "vendor-ledger.csv")
        self.assertIn("Row 250: REF-00250, 250.00, Zephyr Logistics", top["text"])
        self.assertLessEqual(top["row_range"][0], 250)
        self.assertGreaterEqual(top["row_range"][1], 250)
        hits = json.dumps(search["evidence"])
        self.assertNotIn("Not selected", hits)

        body, calls = self._documents_chat(
            "Which vendor is on REF-00250?", "REF-00250 is from Zephyr Logistics [1]."
        )
        self.assertEqual(len(calls), 1)
        self.assertIn("[1] vendor-ledger.csv, rows", body["response"])


if __name__ == "__main__":
    unittest.main()
