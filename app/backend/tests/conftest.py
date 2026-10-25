"""Test-session setup shared by every test module."""

from __future__ import annotations

import gc
import sys
import tempfile


if sys.platform == "win32":
    # Windows refuses to delete a file that still has an open handle. The stores open SQLite
    # connections whose release depends on garbage collection (and langgraph checkpointers sit
    # in reference cycles), so a temp dir can still hold safety/sessions/graph_checkpoints
    # .sqlite3 open at cleanup time (WinError 32). Collect first, and never fail a test just
    # because a scratch file could not be removed.
    _TemporaryDirectory = tempfile.TemporaryDirectory

    class _WindowsTemporaryDirectory(_TemporaryDirectory):
        def __init__(self, *args, **kwargs) -> None:
            kwargs.setdefault("ignore_cleanup_errors", True)
            super().__init__(*args, **kwargs)

        def cleanup(self) -> None:
            gc.collect()
            super().cleanup()

    tempfile.TemporaryDirectory = _WindowsTemporaryDirectory
