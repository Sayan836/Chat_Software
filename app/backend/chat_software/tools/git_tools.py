"""Read-only git helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Iterable, Optional

from .scope import ScopedPathResolver


def _run_git(
    args: list[str],
    repo_path: str | Path = ".",
    roots: Optional[Iterable[str | Path]] = None,
) -> dict[str, object]:
    resolver = ScopedPathResolver.from_roots(roots)
    cwd = resolver.resolve(repo_path)
    if cwd.is_file():
        cwd = cwd.parent
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    return {
        "cwd": str(cwd),
        "command": ["git", *args],
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def git_status(
    repo_path: str | Path = ".",
    roots: Optional[Iterable[str | Path]] = None,
) -> dict[str, object]:
    return _run_git(["status", "--short", "--branch"], repo_path=repo_path, roots=roots)


def git_diff(
    repo_path: str | Path = ".",
    roots: Optional[Iterable[str | Path]] = None,
    staged: bool = False,
    stat_only: bool = True,
) -> dict[str, object]:
    args = ["diff"]
    if staged:
        args.append("--staged")
    if stat_only:
        args.append("--stat")
    return _run_git(args, repo_path=repo_path, roots=roots)
