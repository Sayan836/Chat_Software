"""Provider keys, compatible endpoints, Hugging Face GGUF discovery and background Ollama pulls."""

from __future__ import annotations

import os
import stat
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from chat_software import secrets_store
from chat_software.provider import catalog
from chat_software.provider.config import get_provider_settings


class SecretsStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._env = patch.dict(os.environ, {"CHAT_SOFTWARE_STATE_DIR": self._tmp.name}, clear=False)
        self._env.start()
        for name in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "COMPATIBLE_BASE_URL"):
            os.environ.pop(name, None)

    def tearDown(self) -> None:
        self._env.stop()
        self._tmp.cleanup()

    def test_keys_are_owner_only_and_never_described(self) -> None:
        secrets_store.set_value("OPENAI_API_KEY", "sk-test-secret-value")
        path = Path(self._tmp.name) / "secrets.json"
        if os.name != "nt":  # Windows has no POSIX owner-only mode bits; stat reports 0o666
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        described = secrets_store.describe()
        self.assertTrue(described["OPENAI_API_KEY"]["set"])
        self.assertEqual(described["OPENAI_API_KEY"]["source"], "app")
        self.assertNotIn("sk-test-secret-value", repr(described))
        self.assertEqual(get_provider_settings(provider="openai").api_key, "sk-test-secret-value")
        secrets_store.set_value("OPENAI_API_KEY", None)
        self.assertFalse(secrets_store.describe()["OPENAI_API_KEY"]["set"])

    def test_environment_is_the_fallback(self) -> None:
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "from-env"}):
            self.assertEqual(secrets_store.source("ANTHROPIC_API_KEY"), "environment")
            secrets_store.set_value("ANTHROPIC_API_KEY", "from-app")
            self.assertEqual(get_provider_settings(provider="anthropic").api_key, "from-app")

    def test_unknown_names_and_bad_urls_are_rejected(self) -> None:
        with self.assertRaises(KeyError):
            secrets_store.set_value("PATH", "/tmp")
        with self.assertRaises(ValueError):
            secrets_store.set_value("COMPATIBLE_BASE_URL", "file:///etc/passwd")

    def test_compatible_provider_uses_stored_endpoint(self) -> None:
        secrets_store.set_value("COMPATIBLE_BASE_URL", "https://openrouter.ai/api/v1")
        secrets_store.set_value("COMPATIBLE_MODEL", "meta-llama/llama-3.3-70b-instruct")
        settings = get_provider_settings(provider="compatible")
        self.assertEqual(settings.base_url, "https://openrouter.ai/api/v1")
        self.assertEqual(settings.model, "meta-llama/llama-3.3-70b-instruct")


class CatalogTests(unittest.TestCase):
    def test_model_names_are_validated(self) -> None:
        for good in ("qwen3:8b", "hf.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF:Q4_K_M", "library/llama3.2"):
            self.assertEqual(catalog.validate_model_name(good), good)
        for bad in ("", "a b", "../../etc", "x;rm -rf /", "hf.co/a/../b"):
            with self.assertRaises(ValueError):
                catalog.validate_model_name(bad)

    def test_gguf_quantizations_group_split_files_and_skip_projectors(self) -> None:
        tree = [
            {"path": "model-Q4_K_M.gguf", "size": 400},
            {"path": "model-Q8_0-00001-of-00002.gguf", "size": 500},
            {"path": "model-Q8_0-00002-of-00002.gguf", "size": 500},
            {"path": "mmproj-model-f16.gguf", "size": 99},
            {"path": "README.md", "size": 1},
        ]
        with patch.object(catalog, "_hf_get", return_value=tree):
            quants = catalog.gguf_quantizations("owner/model-GGUF")
        self.assertEqual([(q["quant"], q["size_bytes"], q["files"]) for q in quants],
                         [("Q4_K_M", 400, 1), ("Q8_0", 1000, 2)])
        self.assertEqual(quants[0]["ollama_model"], "hf.co/owner/model-GGUF:Q4_K_M")
        with self.assertRaises(ValueError):
            catalog.gguf_quantizations("not-a-repo")

    def test_pull_manager_reports_progress_and_failure(self) -> None:
        def ok_stream(model):
            yield {"status": "pulling manifest"}
            yield {"status": "downloading", "total": 100, "completed": 40}
            yield {"status": "downloading", "total": 100, "completed": 100}
            yield {"status": "success"}

        def bad_stream(model):
            yield {"status": "pulling manifest"}
            yield {"error": "pull model manifest: file does not exist"}

        manager = catalog.PullManager(opener=ok_stream)
        job = manager.start("qwen3:8b")
        _wait(job)
        self.assertEqual(job.to_dict()["status"], "completed")
        self.assertEqual(job.to_dict()["percent"], 100.0)

        failed = catalog.PullManager(opener=bad_stream).start("hf.co/x/y:Q4_K_M")
        _wait(failed)
        self.assertEqual(failed.status, "failed")
        self.assertIn("does not exist", failed.error)


class ModelsApiTests(unittest.TestCase):
    def setUp(self) -> None:
        from fastapi.testclient import TestClient

        from chat_software.server import create_app

        self._tmp = tempfile.TemporaryDirectory()
        self._env = patch.dict(os.environ, {"CHAT_SOFTWARE_STATE_DIR": self._tmp.name})
        self._env.start()
        os.environ.pop("OPENAI_API_KEY", None)
        self.client = TestClient(create_app())

    def tearDown(self) -> None:
        self._env.stop()
        self._tmp.cleanup()

    def test_setting_a_key_never_echoes_it(self) -> None:
        resp = self.client.put("/settings/values", json={"name": "OPENAI_API_KEY", "value": "sk-live-abc"})
        self.assertEqual(resp.status_code, 200)
        self.assertNotIn("sk-live-abc", resp.text)
        listing = self.client.get("/settings/providers").text
        self.assertNotIn("sk-live-abc", listing)
        self.assertEqual(self.client.put("/settings/values", json={"name": "HOME", "value": "x"}).status_code, 404)

    def test_remote_model_errors_do_not_leak_the_key(self) -> None:
        self.client.put("/settings/values", json={"name": "OPENAI_API_KEY", "value": "sk-live-abc"})

        class Rejected(Exception):
            status_code = 401

        with patch("chat_software.api.models.list_remote_models", side_effect=Rejected("bad key sk-live-abc")):
            resp = self.client.get("/providers/openai/models")
        self.assertEqual(resp.status_code, 502)
        self.assertNotIn("sk-live-abc", resp.text)
        self.assertIn("HTTP 401", resp.text)
        missing = self.client.get("/providers/anthropic/models")
        self.assertEqual(missing.status_code, 400)

    def test_pull_endpoints(self) -> None:
        manager = self.client.app.state.pulls
        manager._opener = lambda model: iter([{"status": "success"}])
        job = self.client.post("/ollama/pulls", json={"model": "llama3.2:3b"}).json()
        _wait(manager.get(job["id"]))
        self.assertEqual(self.client.get(f"/ollama/pulls/{job['id']}").json()["status"], "completed")
        self.assertEqual(self.client.post("/ollama/pulls", json={"model": "bad name"}).status_code, 400)
        self.assertEqual(self.client.get("/ollama/pulls/nope").status_code, 404)


def _wait(job, seconds: float = 5.0) -> None:
    deadline = time.time() + seconds
    while job.status in {"queued", "running"} and time.time() < deadline:
        time.sleep(0.01)


if __name__ == "__main__":
    unittest.main()
