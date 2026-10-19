import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

from chat_software.organize import build_organize_plan
from chat_software.tools.file_ops_write import APP_TRASH_DIR


def _actions(plan, op):
    return [action for action in plan["actions"] if action["op"] == op]


class OrganizePlannerTests(unittest.TestCase):
    def test_by_type_plan_is_read_only_and_skips_ignored_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Images").mkdir()
            (root / "Images" / "photo.png").write_text("existing\n", encoding="utf-8")
            (root / "photo.png").write_text("incoming\n", encoding="utf-8")
            (root / "report.txt").write_text("draft\n", encoding="utf-8")
            (root / ".git").mkdir()
            (root / ".git" / "ignored.txt").write_text("ignore\n", encoding="utf-8")
            (root / APP_TRASH_DIR).mkdir()
            (root / APP_TRASH_DIR / "old.txt").write_text("ignore\n", encoding="utf-8")

            plan = build_organize_plan(root=root, strategy="by_type")

            moves = {action["from"]: action["to"] for action in _actions(plan, "move")}
            mkdirs = {action["path"] for action in _actions(plan, "mkdir")}
            skipped = {item["path"] for item in plan["skipped"]}

            self.assertEqual(plan["strategy"], "by_type")
            self.assertEqual(moves["report.txt"], "Documents/report.txt")
            self.assertEqual(moves["photo.png"], "Images/photo (1).png")
            self.assertEqual(mkdirs, {"Documents"})
            self.assertIn("Images/photo.png", skipped)
            self.assertNotIn(".git/ignored.txt", moves)
            self.assertNotIn(f"{APP_TRASH_DIR}/old.txt", moves)
            self.assertTrue(any("collision:" in warning for warning in plan["warnings"]))
            self.assertTrue((root / "photo.png").exists())
            self.assertFalse((root / "Documents").exists())

    def test_by_date_plan_uses_modified_month_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "trip.mov"
            source.write_text("video\n", encoding="utf-8")
            timestamp = datetime(2024, 2, 15, 12, 0).timestamp()
            os.utime(source, (timestamp, timestamp))

            plan = build_organize_plan(root=root, strategy="by_date")

            moves = _actions(plan, "move")
            self.assertEqual(len(moves), 1)
            self.assertEqual(moves[0]["to"], "2024/02-February/trip.mov")
            self.assertEqual(
                [action["path"] for action in _actions(plan, "mkdir")],
                ["2024", "2024/02-February"],
            )

    def test_by_filename_plan_can_filter_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "party_001.jpg").write_text("jpg\n", encoding="utf-8")
            (root / "party_002.png").write_text("png\n", encoding="utf-8")
            (root / "report.txt").write_text("draft\n", encoding="utf-8")
            (root / ".DS_Store").write_text("metadata\n", encoding="utf-8")

            plan = build_organize_plan(root=root, strategy="by_filename", file_filter="image")

            moves = {action["from"]: action["to"] for action in _actions(plan, "move")}
            mkdirs = {action["path"] for action in _actions(plan, "mkdir")}

            self.assertEqual(plan["strategy"], "by_filename")
            self.assertEqual(plan["file_filter"], "image")
            self.assertEqual(
                moves,
                {
                    "party_001.jpg": "Images/party/party_001.jpg",
                    "party_002.png": "Images/party/party_002.png",
                },
            )
            self.assertEqual(mkdirs, {"Images", "Images/party"})
            self.assertNotIn("report.txt", moves)
            self.assertNotIn(".DS_Store", moves)

    def test_by_topic_groups_files_by_safe_semantic_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "invoice_2024.pdf").write_text("pdf\n", encoding="utf-8")
            (root / "beach_photo.jpg").write_text("jpg\n", encoding="utf-8")
            (root / "app.py").write_text("print('hello')\n", encoding="utf-8")

            plan = build_organize_plan(root=root, strategy="by_topic")
            moves = {action["from"]: action["to"] for action in _actions(plan, "move")}

            self.assertEqual(plan["strategy"], "by_topic")
            self.assertEqual(moves["invoice_2024.pdf"], "Finance/invoice_2024.pdf")
            self.assertEqual(moves["beach_photo.jpg"], "Photos/beach_photo.jpg")
            self.assertEqual(moves["app.py"], "Code/app.py")

    def test_semantic_graph_compiles_and_assigns_topics(self):
        try:
            from chat_software.organize.semantic.graph import build_semantic_planner_graph
        except ImportError:
            self.skipTest("langgraph is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "invoice_2024.pdf").write_text("pdf\n", encoding="utf-8")

            graph = build_semantic_planner_graph()
            state = graph.invoke({"root": str(root), "files": ["invoice_2024.pdf"]})

            self.assertEqual(state["labels"][0]["label"], "Finance")
            self.assertEqual(state["taxonomy"][0]["name"], "Finance")
            self.assertEqual(state["assignments"][0]["folder_index"], 1)

    def test_by_filename_uses_safe_folder_for_windows_reserved_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "CON.txt").write_text("reserved\n", encoding="utf-8")

            plan = build_organize_plan(root=root, strategy="by_filename")
            moves = {action["from"]: action["to"] for action in _actions(plan, "move")}

            self.assertEqual(moves["CON.txt"], "By Name/unnamed/CON.txt")

    def test_nested_parent_dirs_are_all_journaled_top_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "party_001.jpg").write_text("jpg\n", encoding="utf-8")

            plan = build_organize_plan(root=root, strategy="by_filename", file_filter="image")

            self.assertEqual(
                [action["path"] for action in _actions(plan, "mkdir")],
                ["Images", "Images/party"],
            )

    def test_rejects_unknown_strategy(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                build_organize_plan(root=tmp, strategy="magic")

    def test_skips_move_when_target_parent_is_a_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Documents").write_text("not a folder\n", encoding="utf-8")
            (root / "report.txt").write_text("draft\n", encoding="utf-8")

            plan = build_organize_plan(root=root, strategy="by_type")

            moves = {action["from"]: action["to"] for action in _actions(plan, "move")}
            self.assertEqual(moves, {"Documents": "Others/Documents"})
            self.assertEqual({action["path"] for action in _actions(plan, "mkdir")}, {"Others"})
            self.assertEqual(
                [item["path"] for item in plan["skipped"] if item["path"] == "report.txt"],
                ["report.txt"],
            )
            self.assertIn("target parent is not a folder: Documents", plan["warnings"])


if __name__ == "__main__":
    unittest.main()
