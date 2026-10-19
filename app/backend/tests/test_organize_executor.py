import tempfile
import unittest
from pathlib import Path

from chat_software.organize import apply_organize_plan, build_organize_plan, undo_organize_plan
from chat_software.safety_store import SafetyStore


class OrganizeExecutorTests(unittest.TestCase):
    def test_apply_and_undo_organize_plan_are_journaled(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            root = workspace / "approved"
            root.mkdir()
            (root / "report.txt").write_text("draft\n", encoding="utf-8")
            (root / "photo.png").write_text("pixels\n", encoding="utf-8")

            store = SafetyStore(workspace / "safety.sqlite3")
            approved = store.approve_root(root)
            plan = build_organize_plan(root=root, strategy="by_type")
            plan_id = store.create_organize_plan(
                root_id=approved.id,
                strategy="by_type",
                plan=plan,
                prompt="organize files by type",
            )

            applied = apply_organize_plan(store, plan_id)

            self.assertEqual(applied["status"], "applied")
            self.assertFalse((root / "report.txt").exists())
            self.assertFalse((root / "photo.png").exists())
            self.assertEqual((root / "Documents" / "report.txt").read_text(encoding="utf-8"), "draft\n")
            self.assertEqual((root / "Images" / "photo.png").read_text(encoding="utf-8"), "pixels\n")
            self.assertEqual(store.get_organize_plan(plan_id)["status"], "applied")
            self.assertTrue(store.get_journal_entries(plan_id))

            undone = undo_organize_plan(store, plan_id)

            self.assertEqual(undone["status"], "undone")
            self.assertEqual((root / "report.txt").read_text(encoding="utf-8"), "draft\n")
            self.assertEqual((root / "photo.png").read_text(encoding="utf-8"), "pixels\n")
            self.assertFalse((root / "Documents").exists())
            self.assertFalse((root / "Images").exists())
            self.assertEqual(store.get_organize_plan(plan_id)["status"], "undone")
            self.assertTrue(
                all(entry["status"] == "undone" for entry in store.get_journal_entries(plan_id))
            )

    def test_undo_requires_applied_or_partial_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            root = workspace / "approved"
            root.mkdir()
            (root / "report.txt").write_text("draft\n", encoding="utf-8")

            store = SafetyStore(workspace / "safety.sqlite3")
            approved = store.approve_root(root)
            plan_id = store.create_organize_plan(
                root_id=approved.id,
                strategy="by_type",
                plan=build_organize_plan(root=root, strategy="by_type"),
            )

            with self.assertRaises(RuntimeError):
                undo_organize_plan(store, plan_id)

    def test_undo_removes_all_nested_folders_created_by_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            root = workspace / "approved"
            root.mkdir()
            (root / "party_001.jpg").write_text("jpg\n", encoding="utf-8")

            store = SafetyStore(workspace / "safety.sqlite3")
            approved = store.approve_root(root)
            plan = build_organize_plan(root=root, strategy="by_filename", file_filter="image")
            plan_id = store.create_organize_plan(
                root_id=approved.id,
                strategy="by_filename",
                plan=plan,
                prompt="organize images by filename",
            )

            apply_organize_plan(store, plan_id)
            self.assertTrue((root / "Images" / "party" / "party_001.jpg").exists())

            undo_organize_plan(store, plan_id)

            self.assertTrue((root / "party_001.jpg").exists())
            self.assertFalse((root / "Images" / "party").exists())
            self.assertFalse((root / "Images").exists())


if __name__ == "__main__":
    unittest.main()
