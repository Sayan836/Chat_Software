import unittest

from chat_software.provider.ollama import select_installed_model, sort_installed_models


class OllamaModelSelectionTests(unittest.TestCase):
    def test_sorts_latest_installed_model_first(self):
        models = [
            {"name": "old:latest", "modified_at": "2026-01-01T00:00:00Z"},
            {"name": "new:latest", "modified_at": "2026-07-01T00:00:00Z"},
        ]

        sorted_models = sort_installed_models(models)

        self.assertEqual([item["name"] for item in sorted_models], ["new:latest", "old:latest"])
        self.assertEqual(select_installed_model(None, models), "new:latest")

    def test_keeps_installed_preference_and_matches_implicit_latest_tag(self):
        models = [
            {"name": "llama3.1:latest", "modified_at": "2026-01-01T00:00:00Z"},
            {"name": "qwen2.5:7b", "modified_at": "2026-07-01T00:00:00Z"},
        ]

        self.assertEqual(select_installed_model("qwen2.5:7b", models), "qwen2.5:7b")
        self.assertEqual(select_installed_model("llama3.1", models), "llama3.1:latest")


if __name__ == "__main__":
    unittest.main()
