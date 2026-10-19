import tempfile
import unittest
from pathlib import Path

from chat_software.safety_store import SafetyStore


class SafetyStoreTests(unittest.TestCase):
    def test_initializes_phase_two_safety_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SafetyStore(Path(tmp) / "safety.sqlite3")

            self.assertTrue(
                {
                    "approved_roots",
                    "organize_plans",
                    "organize_actions",
                    "operation_journal",
                    "audit_log",
                }.issubset(store.table_names())
            )

    def test_approve_root_is_persistent_and_authoritative(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "approved"
            root.mkdir()
            unapproved = Path(tmp) / "other"
            unapproved.mkdir()

            store = SafetyStore(Path(tmp) / "safety.sqlite3")
            approved = store.approve_root(root)
            reapproved = store.approve_root(root)

            self.assertEqual(approved.id, reapproved.id)
            self.assertEqual(store.authorize_roots([root])[0].id, approved.id)
            with self.assertRaises(PermissionError):
                store.authorize_roots([unapproved])

    def test_create_organize_plan_records_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "approved"
            root.mkdir()
            store = SafetyStore(Path(tmp) / "safety.sqlite3")
            approved = store.approve_root(root)

            plan_id = store.create_organize_plan(
                root_id=approved.id,
                strategy="by_type",
                plan={
                    "actions": [
                        {"op": "mkdir", "path": "Documents"},
                        {"op": "move", "from": "a.txt", "to": "Documents/a.txt"},
                    ]
                },
            )

            self.assertTrue(plan_id)
            stored = store.get_organize_plan(plan_id)
            actions = store.get_organize_actions(plan_id)

            self.assertEqual(stored["status"], "planned")
            self.assertEqual(stored["root_id"], approved.id)
            self.assertEqual(actions[1]["action"]["to"], "Documents/a.txt")

            store.update_action_status(actions[1]["id"], "running")
            updated = store.get_organize_actions(plan_id)
            self.assertEqual(updated[1]["status"], "running")

            journal_id = store.start_journal_entry(
                batch_id=plan_id,
                plan_id=plan_id,
                action_id=actions[1]["id"],
                op="move",
                forward=actions[1]["action"],
                reverse={"op": "move", "from": "Documents/a.txt", "to": "a.txt"},
            )
            store.complete_journal_entry(journal_id)
            entries = store.get_journal_entries(plan_id)
            self.assertEqual(entries[0]["status"], "completed")
            self.assertEqual(entries[0]["reverse"]["to"], "a.txt")


if __name__ == "__main__":
    unittest.main()
