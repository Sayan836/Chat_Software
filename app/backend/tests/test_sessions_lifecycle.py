"""Sessions: list/search, rename, pin, archive/restore, delete, migration, and chat memory."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chat_software.agent.session_store import SessionStore


class SessionStoreLifecycleTests(unittest.TestCase):
    def test_list_rename_pin_archive_delete(self) -> None:
        store = SessionStore(":memory:")
        a = store.create_session()
        store.append_message(a, "user", "Plan the quarterly budget review for finance")
        b = store.create_session(kind="research", title="Battery chemistry")
        store.append_message(b, "user", "Compare LFP and NMC cells")

        titles = [s.title for s in store.list_sessions()]
        self.assertIn("Plan the quarterly budget review for finance", titles)
        self.assertEqual([s.id for s in store.list_sessions(kind="research")], [b])
        self.assertEqual([s.id for s in store.list_sessions(query="LFP")], [b])

        store.update_session(a, title="Budget review", pinned=True)
        self.assertEqual(store.list_sessions()[0].title, "Budget review")

        store.update_session(b, archived=True)
        self.assertEqual([s.id for s in store.list_sessions()], [a])
        self.assertEqual([s.id for s in store.list_sessions(archived_only=True)], [b])
        store.update_session(b, archived=False)
        self.assertEqual(len(store.list_sessions()), 2)

        self.assertTrue(store.delete_session(b))
        self.assertEqual(store.get_messages(b), [])
        with self.assertRaises(KeyError):
            store.get_session(b)
        self.assertFalse(store.delete_session(b))

    def test_existing_database_is_migrated_in_place(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sessions.sqlite3"
            conn = sqlite3.connect(path)
            conn.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, created_at REAL NOT NULL, updated_at REAL NOT NULL)")
            conn.execute("""CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL,
                role TEXT NOT NULL, content TEXT NOT NULL, metadata_json TEXT NOT NULL, created_at REAL NOT NULL)""")
            conn.execute("INSERT INTO sessions VALUES ('old', 1, 1)")
            conn.execute("INSERT INTO messages (session_id, role, content, metadata_json, created_at) VALUES ('old','user','hi','{}',1)")
            conn.commit()
            conn.close()
            store = SessionStore(path)
            summary = store.get_session("old")
            self.assertEqual((summary.kind, summary.title, summary.message_count), ("chat", "hi", 1))


class RecordingClient:
    def __init__(self) -> None:
        from chat_software.provider.config import get_provider_settings

        self.settings = get_provider_settings(provider="ollama", requested_model="memory-test")
        self.calls: list[list[dict]] = []

    def complete(self, messages, tools=None) -> str:
        self.calls.append([dict(m) for m in messages])
        return f"reply {len(self.calls)}"


class SessionApiTests(unittest.TestCase):
    def setUp(self) -> None:
        from fastapi.testclient import TestClient

        from chat_software.server import create_app

        self._tmp = tempfile.TemporaryDirectory()
        self._env = patch.dict(os.environ, {"CHAT_SOFTWARE_STATE_DIR": self._tmp.name,
                                            "CHAT_SOFTWARE_OFFLINE_ECHO": "0"})
        self._env.start()
        self.model = RecordingClient()
        self._patch = patch("chat_software.server.get_chat_client", lambda **kw: self.model)
        self._patch.start()
        self.client = TestClient(create_app())

    def tearDown(self) -> None:
        self._patch.stop()
        self._env.stop()
        self._tmp.cleanup()

    def test_chat_remembers_earlier_turns_in_the_same_session(self) -> None:
        first = self.client.post("/chat", json={"message": "My name is Priya."}).json()
        self.client.post("/chat", json={"message": "What is my name?", "session_id": first["session_id"]})
        sent = self.model.calls[-1]
        self.assertEqual(sent[-1], {"role": "user", "content": "What is my name?"})
        self.assertIn({"role": "user", "content": "My name is Priya."}, sent)
        self.assertIn({"role": "assistant", "content": "reply 1"}, sent)
        other = self.client.post("/chat", json={"message": "Fresh start"}).json()
        self.assertNotEqual(other["session_id"], first["session_id"])
        self.assertNotIn({"role": "user", "content": "My name is Priya."}, self.model.calls[-1])

    def test_session_endpoints(self) -> None:
        sid = self.client.post("/chat", json={"message": "Draft a cover letter"}).json()["session_id"]
        listed = self.client.get("/sessions").json()["sessions"]
        self.assertEqual(listed[0]["title"], "Draft a cover letter")
        renamed = self.client.patch(f"/sessions/{sid}", json={"title": "Cover letter", "pinned": True}).json()
        self.assertEqual((renamed["title"], renamed["pinned"]), ("Cover letter", True))
        self.client.patch(f"/sessions/{sid}", json={"archived": True})
        self.assertEqual(self.client.get("/sessions").json()["sessions"], [])
        self.assertEqual(len(self.client.get("/sessions?archived=archived").json()["sessions"]), 1)
        self.assertEqual(self.client.get(f"/sessions/{sid}").json()["session"]["archived"], True)
        self.assertEqual(self.client.delete(f"/sessions/{sid}").status_code, 200)
        self.assertEqual(self.client.get(f"/sessions/{sid}").json()["messages"], [])
        self.assertEqual(self.client.delete(f"/sessions/{sid}").status_code, 404)
        self.assertEqual(self.client.patch("/sessions/nope", json={"title": "x"}).status_code, 404)
        created = self.client.post("/sessions", json={"kind": "team", "title": "Launch team"}).json()
        self.assertEqual(created["kind"], "team")
        self.assertEqual(self.client.post("/sessions", json={"kind": "bogus"}).status_code, 400)

    def test_delete_removes_graph_checkpoints(self) -> None:
        from chat_software.agent.graph_runtime import delete_thread_checkpoints, graph_checkpoint_path

        path = graph_checkpoint_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE checkpoints (thread_id TEXT, checkpoint BLOB)")
        conn.execute("INSERT INTO checkpoints VALUES ('t1', x'00'), ('t2', x'00')")
        conn.commit()
        conn.close()
        self.assertEqual(delete_thread_checkpoints("t1"), 1)
        conn = sqlite3.connect(path)
        self.assertEqual(conn.execute("SELECT thread_id FROM checkpoints").fetchall(), [("t2",)])
        conn.close()


if __name__ == "__main__":
    unittest.main()
