"""Shared filesystem locations for app-local runtime state."""

from __future__ import annotations

import os
from pathlib import Path


def state_dir() -> Path:
    return Path(os.getenv("CHAT_SOFTWARE_STATE_DIR", ".chat_software_state")).expanduser()


def ensure_state_dir() -> Path:
    directory = state_dir()
    if str(directory) != ":memory:":
        directory.mkdir(parents=True, exist_ok=True)
    return directory
