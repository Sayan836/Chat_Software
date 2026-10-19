"""Smoke tests for the FastAPI-served browser UI."""

from __future__ import annotations

import os
import socket
import tempfile
import unittest


try:
    from fastapi.testclient import TestClient  # noqa: F401

    _HAS_FASTAPI = True
except Exception:  # pragma: no cover - depends on optional deps
    _HAS_FASTAPI = False


@unittest.skipUnless(_HAS_FASTAPI, "FastAPI/httpx not installed")
class MainStaticUiTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        os.environ["CHAT_SOFTWARE_STATE_DIR"] = self._tmp.name
        from fastapi.testclient import TestClient
        from main import create_app

        self.client = TestClient(create_app())

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_root_serves_browser_ui(self) -> None:
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("text/html", resp.headers["content-type"])
        self.assertIn("Chat Software Console", resp.text)
        self.assertIn("/static/app.js", resp.text)
        self.assertIn("organize-preview", resp.text)
        self.assertIn("organize-filter", resp.text)
        self.assertIn('value="by_filename"', resp.text)
        self.assertIn('value="by_topic"', resp.text)
        self.assertIn("command-console", resp.text)
        self.assertIn("Command History", resp.text)
        self.assertIn("command-target", resp.text)
        self.assertIn("command-content", resp.text)
        self.assertIn('<select id="model-input"', resp.text)

    def test_static_assets_are_served(self) -> None:
        js_resp = self.client.get("/static/app.js")
        css_resp = self.client.get("/static/styles.css")

        self.assertEqual(js_resp.status_code, 200)
        self.assertEqual(css_resp.status_code, 200)
        self.assertIn("sendMessage", js_resp.text)
        self.assertIn("previewOrganizePlan", js_resp.text)
        self.assertIn("file_filter", js_resp.text)
        self.assertIn("organizeFilter", js_resp.text)
        self.assertIn("toolCommand", js_resp.text)
        self.assertIn("appendCommandConsole", js_resp.text)
        self.assertIn("renderCommandHistoryFromMessages", js_resp.text)
        self.assertIn("working_path", js_resp.text)
        self.assertIn("ollamaInstalled.addEventListener", js_resp.text)
        self.assertIn("setModelOptions", js_resp.text)
        self.assertIn("/files/write", js_resp.text)
        self.assertIn("/files/move", js_resp.text)
        self.assertIn(".shell", css_resp.text)
        self.assertIn(".organize-table", css_resp.text)
        self.assertIn(".tool-command", css_resp.text)
        self.assertIn(".command-console", css_resp.text)
        self.assertIn(".command-index", css_resp.text)

    def test_documents_mode_controls_are_served(self) -> None:
        html = self.client.get("/").text
        js = self.client.get("/static/app.js").text
        css = self.client.get("/static/styles.css").text
        for element_id in ("knowledge-binding", "documents-mode", "bind-knowledge", "source-viewer"):
            self.assertIn(f'id="{element_id}"', html)
        self.assertIn("knowledge_root_id", js)
        self.assertIn("/knowledge/source", js)
        self.assertIn("renderCitations", js)
        self.assertIn(".citation-chip", css)
        for element_id in ("enable-ingest", "ingest-paths", "ingest-preview", "ingest-approve"):
            self.assertIn(f'id="{element_id}"', html)
        self.assertIn("/knowledge/ingest/preview", js)
        self.assertIn("row_range", js)

    def test_api_routes_still_work_from_main_app(self) -> None:
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(resp.json()["ok"])

    def test_select_port_skips_busy_requested_port(self) -> None:
        import main

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("127.0.0.1", 0))
            except PermissionError as exc:
                self.skipTest(f"local socket bind is blocked in this environment: {exc}")
            busy_port = sock.getsockname()[1]

            selected = main._select_port("127.0.0.1", busy_port)

        self.assertNotEqual(selected, busy_port)
        self.assertGreater(selected, busy_port)


if __name__ == "__main__":
    unittest.main()
