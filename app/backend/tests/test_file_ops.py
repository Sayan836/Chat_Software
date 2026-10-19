import tempfile
import unittest
from pathlib import Path

from chat_software.tools.file_ops import glob_files, grep_files, list_dir, read_file


class FileOpsTests(unittest.TestCase):
    def test_read_only_file_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "notes.md").write_text("Acme contract renewal is due.\n", encoding="utf-8")
            (root / "other.txt").write_text("No match here.\n", encoding="utf-8")

            listed = list_dir(roots=[root])
            self.assertEqual(len(listed["entries"]), 2)

            content = read_file("notes.md", roots=[root])
            self.assertIn("Acme", content["content"])

            matches = grep_files("Acme", roots=[root])
            self.assertEqual(len(matches["matches"]), 1)
            self.assertIn("notes.md", matches["matches"][0]["path"])

            globbed = glob_files("*.md", roots=[root])
            self.assertEqual(len(globbed["matches"]), 1)

    def test_rejects_out_of_scope_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(PermissionError):
                read_file("/etc/hosts", roots=[root])


if __name__ == "__main__":
    unittest.main()
