"""AirLLM: catalog, HF checkpoint checks/downloads, worker HTTP contract and provider routing (no torch)."""

from __future__ import annotations

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chat_software.provider import airllm


WORKER_PATH = Path(__file__).resolve().parents[2] / "integrations" / "airllm" / "worker.py"


def make_checkpoint(directory: Path, architecture: str = "LlamaForCausalLM") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps({"architectures": [architecture]}))
    (directory / "model.safetensors").write_bytes(b"\0" * 16)
    return directory


class AirLLMCatalogTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self._env = patch.dict(os.environ, {"CHAT_SOFTWARE_STATE_DIR": str(self.tmp / "state")})
        self._env.start()
        for name in ("AIRLLM_HOME", "AIRLLM_PYTHON", "AIRLLM_BACKEND", "HF_TOKEN"):
            os.environ.pop(name, None)
        self.manager = airllm.AirLLMManager()

    def tearDown(self) -> None:
        self.manager.stop()
        self._env.stop()
        self._tmp.cleanup()

    def test_architecture_support(self) -> None:
        self.assertTrue(airllm.architecture_supported(["Qwen2ForCausalLM"])[0])
        self.assertTrue(airllm.architecture_supported(["ChatGLMModel"])[0])
        self.assertFalse(airllm.architecture_supported(["BertModel"])[0])
        self.assertFalse(airllm.architecture_supported([])[0])

    def test_register_validate_and_remove(self) -> None:
        external = make_checkpoint(self.tmp / "external" / "tiny")
        entry = self.manager.register("tiny", str(external), context_tokens=2048)
        self.assertEqual((entry["architecture"], entry["present"]), ("LlamaForCausalLM", True))
        self.assertEqual([m["id"] for m in self.manager.list_models()], ["tiny"])
        with self.assertRaises(ValueError):
            self.manager.register("bad", str(self.tmp / "nowhere"))
        with self.assertRaises(ValueError):
            self.manager.register("bert", str(make_checkpoint(self.tmp / "bert", "BertModel")))
        with self.assertRaises(ValueError):
            self.manager.register("Bad Id", str(external))
        result = self.manager.remove("tiny", delete_files=True)
        self.assertFalse(result["files_deleted"], "files outside the managed folder are never deleted")
        self.assertTrue(external.exists())
        with self.assertRaises(KeyError):
            self.manager.remove("tiny")

    def test_check_repo_sizes_and_rejects_gguf_only(self) -> None:
        info = {"sha": "abc", "gated": False,
                "config": {"architectures": ["Qwen2ForCausalLM"], "max_position_embeddings": 32768},
                "siblings": [{"rfilename": "model-00001-of-00002.safetensors", "size": 600},
                             {"rfilename": "model-00002-of-00002.safetensors", "size": 400},
                             {"rfilename": "config.json", "size": 1}]}
        with patch("chat_software.provider.catalog._hf_get", return_value=info):
            check = self.manager.check_repo("Qwen/Qwen2.5-7B-Instruct")
        self.assertTrue(check["supported"])
        self.assertEqual((check["weights_bytes"], check["revision"], check["context_tokens"]), (1000, "abc", 32768))
        self.assertGreaterEqual(check["disk_needed_bytes"], 2000)
        gguf = {"sha": "x", "config": {"architectures": ["LlamaForCausalLM"]},
                "siblings": [{"rfilename": "m-Q4_K_M.gguf", "size": 5}]}
        with patch("chat_software.provider.catalog._hf_get", return_value=gguf):
            self.assertFalse(self.manager.check_repo("someone/model-GGUF")["supported"])
        with self.assertRaises(ValueError):
            self.manager.check_repo("not a repo")

    def test_download_registers_checkpoint_and_respects_gating(self) -> None:
        check = {"supported": True, "support": "ok", "gated": False, "weights_bytes": 16,
                 "disk_needed_bytes": 32, "free_bytes": 10**12, "suggested_id": "tiny-chat",
                 "revision": "rev1", "context_tokens": 4096}

        def fake_snapshot(repo, revision, local_dir, token, allow_patterns):
            make_checkpoint(Path(local_dir))

        with patch.object(self.manager, "check_repo", return_value=check), \
             patch("huggingface_hub.snapshot_download", side_effect=fake_snapshot):
            job = self.manager.start_download("owner/tiny-chat")
            for _ in range(200):
                if job.status in {"completed", "failed"}:
                    break
                import time
                time.sleep(0.01)
        self.assertEqual(job.status, "completed", job.error)
        model = self.manager.list_models()[0]
        self.assertEqual((model["id"], model["repository"], model["revision"]), ("tiny-chat", "owner/tiny-chat", "rev1"))
        self.assertTrue(self.manager.remove("tiny-chat", delete_files=True)["files_deleted"])
        with patch.object(self.manager, "check_repo", return_value={**check, "gated": True}):
            with self.assertRaises(ValueError):
                self.manager.start_download("owner/gated")
        with patch.object(self.manager, "check_repo", return_value={**check, "free_bytes": 1}):
            with self.assertRaises(ValueError):
                self.manager.start_download("owner/huge")

    def test_worker_start_refuses_without_runtime_or_models(self) -> None:
        with self.assertRaises(airllm.AirLLMError):
            self.manager.start()
        self.assertFalse(self.manager.runtime_status()["ready"])


class FakeEngine:
    def __init__(self) -> None:
        import threading

        self.backend = "cpu"
        self.models = {"tiny": {"id": "tiny"}}
        self.loaded_id = None
        self.lock = threading.Lock()
        self.cancel_event = threading.Event()
        self.progress = {"phase": "idle"}
        self.last_error = None
        self.requests = []

    def complete(self, body):
        from fastapi import HTTPException

        if body.model not in self.models:
            raise HTTPException(404, "Model is not in the local AirLLM catalog")
        self.requests.append(body)
        return {"choices": [{"message": {"role": "assistant", "content": "hi from fake airllm"}}],
                "engine": {"name": "airllm", "backend": "cpu"}}


class WorkerContractTests(unittest.TestCase):
    def setUp(self) -> None:
        from fastapi.testclient import TestClient

        spec = importlib.util.spec_from_file_location("airllm_worker", WORKER_PATH)
        module = importlib.util.module_from_spec(spec)
        import sys

        sys.modules["airllm_worker"] = module  # pydantic resolves postponed annotations via sys.modules
        spec.loader.exec_module(module)
        self.engine = FakeEngine()
        self.client = TestClient(module.create_app({}, engine=self.engine))

    def test_openai_shape_long_history_and_errors(self) -> None:
        messages = [{"role": "user", "content": f"m{i}"} for i in range(40)]
        resp = self.client.post("/v1/chat/completions", json={
            "model": "tiny", "messages": messages, "temperature": 0.2, "max_tokens": 64,
            "top_p": 1, "n": 1})  # extra OpenAI fields are ignored
        self.assertEqual(resp.status_code, 200, resp.text)
        self.assertEqual(resp.json()["engine"]["name"], "airllm")
        self.assertEqual(self.client.post("/v1/chat/completions", json={
            "model": "nope", "messages": [{"role": "user", "content": "x"}]}).status_code, 404)
        self.assertEqual(self.client.get("/v1/models").json()["data"][0]["id"], "tiny")
        self.assertEqual(self.client.get("/health").json()["engine"], "airllm")
        self.assertEqual(self.client.post("/cancel").json()["status"], "cancellation_requested")
        self.assertTrue(self.engine.cancel_event.is_set())


class ProviderRoutingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._env = patch.dict(os.environ, {"CHAT_SOFTWARE_STATE_DIR": self._tmp.name,
                                            "CHAT_SOFTWARE_OFFLINE_ECHO": "0"})
        self._env.start()
        for name in ("AIRLLM_HOME", "AIRLLM_PYTHON"):
            os.environ.pop(name, None)

    def tearDown(self) -> None:
        airllm.MANAGER.stop()
        self._env.stop()
        self._tmp.cleanup()

    def test_chat_routes_to_the_airllm_worker(self) -> None:
        from fastapi.testclient import TestClient

        from chat_software.provider.factory import get_chat_client
        from chat_software.server import create_app

        with self.assertRaises(ValueError):
            get_chat_client(provider="airllm")
        airllm.MANAGER.register("tiny", str(make_checkpoint(Path(self._tmp.name) / "ckpt")))
        client = get_chat_client(provider="airllm")
        self.assertIsInstance(client, airllm.AirLLMChatClient)
        self.assertEqual(client.settings.model, "tiny")

        sent = {}

        class Response:
            status_code = 200
            headers = {"content-type": "application/json"}

            def json(self):
                return {"choices": [{"message": {"content": "local airllm reply"}}]}

        def fake_post(url, json, timeout):
            sent.update(url=url, body=json, timeout=timeout)
            return Response()

        app = TestClient(create_app())
        with patch.object(airllm.MANAGER, "ensure_running", return_value="http://127.0.0.1:9"), \
             patch("chat_software.provider.airllm.httpx.post", side_effect=fake_post):
            body = app.post("/chat", json={"message": "hello", "provider": "airllm", "model": "tiny"}).json()
        self.assertEqual(body["response"], "local airllm reply")
        self.assertEqual(sent["url"], "http://127.0.0.1:9/v1/chat/completions")
        self.assertEqual(sent["body"]["model"], "tiny")
        self.assertGreater(sent["timeout"], 600)
        options = app.get("/model-options").json()
        airllm_entry = next(p for p in options["providers"] if p["provider"] == "airllm")
        self.assertEqual(airllm_entry["models"], ["tiny"])
        status = app.get("/airllm/status").json()
        self.assertEqual(status["models"][0]["id"], "tiny")
        self.assertIn("layer", status["note"])


if __name__ == "__main__":
    unittest.main()
