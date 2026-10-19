import os
import unittest
from pathlib import Path
from unittest.mock import patch

from chat_software.provider.config import (
    get_provider_settings,
    load_project_env,
    normalize_ollama_native_base_url,
    normalize_ollama_openai_base_url,
    resolve_model,
)


class ProviderConfigTests(unittest.TestCase):
    def test_defaults_to_ollama(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = get_provider_settings()
        self.assertEqual(settings.provider, "ollama")
        self.assertEqual(settings.model, "qwen3:8b")
        self.assertEqual(settings.base_url, "http://localhost:11434/v1")

    def test_normalizes_ollama_openai_compatible_url_for_chat(self):
        self.assertEqual(
            normalize_ollama_openai_base_url("http://localhost:11434"),
            "http://localhost:11434/v1",
        )
        self.assertEqual(
            normalize_ollama_native_base_url("http://localhost:11434/v1"),
            "http://localhost:11434",
        )
        with patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://localhost:11434/v1"}, clear=True):
            settings = get_provider_settings(provider="ollama")
        self.assertEqual(settings.base_url, "http://localhost:11434/v1")

    def test_resolves_openai_model(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "openai", "OPENAI_MODEL": "gpt-test"}, clear=True):
            self.assertEqual(resolve_model(), "gpt-test")

    def test_provider_override_does_not_depend_on_active_env_provider(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "ollama"}, clear=True):
            settings = get_provider_settings(provider="anthropic", requested_model="claude-test")
        self.assertEqual(settings.provider, "anthropic")
        self.assertEqual(settings.model, "claude-test")

    def test_rejects_unknown_provider(self):
        with patch.dict(os.environ, {"LLM_PROVIDER": "mystery"}, clear=True):
            with self.assertRaises(ValueError):
                get_provider_settings()

    def test_load_project_env_skips_unreadable_env_files(self):
        with patch("chat_software.provider.config.Path.exists", side_effect=PermissionError):
            load_project_env(start=Path("/restricted"))


if __name__ == "__main__":
    unittest.main()
