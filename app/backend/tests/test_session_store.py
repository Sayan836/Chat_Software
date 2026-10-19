import unittest

from chat_software.agent import SessionStore


class SessionStoreTests(unittest.TestCase):
    def test_memory_store_persists_messages_for_lifetime_of_store(self):
        store = SessionStore(":memory:")
        session_id = store.create_session("memory-session")

        store.append_message(session_id, "user", "hello", metadata={"source": "test"})
        store.append_message(session_id, "assistant", "hi")

        messages = store.get_messages(session_id)
        self.assertEqual([message.role for message in messages], ["user", "assistant"])
        self.assertEqual(messages[0].metadata, {"source": "test"})


if __name__ == "__main__":
    unittest.main()
