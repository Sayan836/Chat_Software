import unittest
from unittest.mock import Mock, patch

from chat_software.provider.config import ProviderSettings
from chat_software.provider.factory import (
    AnthropicChatClient,
    OpenAICompatibleChatClient,
    get_chat_client,
)


class ProviderFactoryTests(unittest.TestCase):
    def test_ollama_and_openai_use_openai_compatible_client(self):
        with patch.dict(
            "os.environ",
            {"LLM_PROVIDER": "ollama", "OLLAMA_MODEL": "qwen3:8b"},
            clear=True,
        ):
            ollama_client = get_chat_client()
        self.assertIsInstance(ollama_client, OpenAICompatibleChatClient)

        with patch.dict(
            "os.environ",
            {"LLM_PROVIDER": "openai", "OPENAI_MODEL": "gpt-test"},
            clear=True,
        ):
            openai_client = get_chat_client()
        self.assertIsInstance(openai_client, OpenAICompatibleChatClient)

    def test_anthropic_uses_anthropic_client(self):
        with patch.dict(
            "os.environ",
            {"LLM_PROVIDER": "anthropic", "ANTHROPIC_MODEL": "claude-test"},
            clear=True,
        ):
            client = get_chat_client()
        self.assertIsInstance(client, AnthropicChatClient)

    def test_openai_compatible_complete_calls_chat_completions(self):
        settings = ProviderSettings(
            provider="ollama",
            model="qwen3:8b",
            api_key="ollama",
            base_url="http://localhost:11434/v1",
        )
        response = Mock()
        response.choices = [Mock(message=Mock(content="hello from sdk"))]
        sdk_client = Mock()
        sdk_client.chat.completions.create.return_value = response

        with patch(
            "chat_software.provider.factory.get_openai_compatible_client",
            return_value=sdk_client,
        ):
            result = OpenAICompatibleChatClient(settings).complete(
                [{"role": "user", "content": "hello"}]
            )

        self.assertEqual(result, "hello from sdk")
        sdk_client.chat.completions.create.assert_called_once()
        kwargs = sdk_client.chat.completions.create.call_args.kwargs
        self.assertEqual(kwargs["model"], "qwen3:8b")
        self.assertEqual(kwargs["messages"], [{"role": "user", "content": "hello"}])


if __name__ == "__main__":
    unittest.main()
