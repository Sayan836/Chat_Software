import errno
import inspect
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chat_software.tools import file_ops_write
from chat_software.tools.file_ops_write import (
    APP_TRASH_DIR,
    copy_path,
    edit_file,
    make_dir,
    move_path,
    quarantine_path,
    safe_move,
    write_file,
)
from chat_software.tools.scope import ScopedPathResolver


class FileOpsWriteTests(unittest.TestCase):
    def test_make_dir_and_safe_move_stay_inside_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resolver = ScopedPathResolver.from_roots([root])
            source = root / "notes.txt"
            source.write_text("hello\n", encoding="utf-8")

            made = make_dir("Documents", resolver)
            result = safe_move("notes.txt", "Documents/notes.txt", resolver, batch_id="batch-1")

            self.assertTrue(Path(made["path"]).is_dir())
            self.assertFalse(source.exists())
            self.assertEqual((root / "Documents" / "notes.txt").read_text(encoding="utf-8"), "hello\n")
            self.assertEqual(result["method"], "rename")

    def test_safe_move_refuses_overwrite_and_out_of_scope_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resolver = ScopedPathResolver.from_roots([root])
            (root / "source.txt").write_text("source\n", encoding="utf-8")
            (root / "dest.txt").write_text("dest\n", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                safe_move("source.txt", "dest.txt", resolver, batch_id="batch-1")

            with self.assertRaises(PermissionError):
                safe_move("source.txt", "/etc/source.txt", resolver, batch_id="batch-1")

    def test_write_copy_edit_and_move_path_helpers(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resolver = ScopedPathResolver.from_roots([root])

            written = write_file("drafts/notes.txt", "hello\n", resolver)
            copied = copy_path("drafts/notes.txt", "copies/notes.txt", resolver)
            edited = edit_file("copies/notes.txt", "hello", "hi", resolver)
            make_dir("archive", resolver)
            moved = move_path("copies/notes.txt", "archive", resolver, batch_id="batch-1")

            self.assertTrue(written["created"])
            self.assertEqual(copied["kind"], "file")
            self.assertEqual(edited["replacements"], 1)
            self.assertEqual((root / "copies" / "notes.txt").exists(), False)
            self.assertEqual((root / "archive" / "notes.txt").read_text(encoding="utf-8"), "hi\n")
            self.assertEqual(moved["method"], "rename")

    def test_quarantine_path_is_inside_approved_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resolver = ScopedPathResolver.from_roots([root])
            target = root / "notes.txt"
            target.write_text("hello\n", encoding="utf-8")

            quarantined = quarantine_path(target, resolver, batch_id="batch 1", kind="files")

            self.assertIn(APP_TRASH_DIR, quarantined.parts)
            self.assertTrue(quarantined.is_relative_to(root))

    def test_exdev_file_move_copies_destination_and_quarantines_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            resolver = ScopedPathResolver.from_roots([root])
            source = root / "source.txt"
            dest = root / "archive" / "source.txt"
            source.write_text("source\n", encoding="utf-8")
            real_rename = os.rename
            first_move = {"raised": False}

            def fake_rename(src, dst):
                if Path(src) == source and Path(dst) == dest and not first_move["raised"]:
                    first_move["raised"] = True
                    error = OSError("cross-device link")
                    error.errno = errno.EXDEV
                    raise error
                return real_rename(src, dst)

            with patch("chat_software.tools.file_ops_write.os.rename", side_effect=fake_rename):
                result = safe_move(source, dest, resolver, batch_id="batch-1")

            self.assertEqual(result["method"], "copy_then_quarantine_source")
            self.assertEqual(dest.read_text(encoding="utf-8"), "source\n")
            self.assertFalse(source.exists())
            self.assertTrue(Path(result["source_quarantine"]).exists())
            self.assertIn(APP_TRASH_DIR, Path(result["source_quarantine"]).parts)

    def test_write_layer_has_no_permanent_user_delete_calls(self):
        source = inspect.getsource(file_ops_write)

        self.assertNotIn("os.remove", source)
        self.assertNotIn("os.unlink", source)
        self.assertNotIn("Path.unlink", source)
        self.assertNotIn("shutil.rmtree", source)


if __name__ == "__main__":
    unittest.main()
