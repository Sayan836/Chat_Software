"""Read-only file operations for Phase 1."""

from __future__ import annotations

import fnmatch
import os
import re
from pathlib import Path
from typing import Iterable, Optional

from .scope import ScopedPathResolver


TEXT_EXTENSIONS = {
    ".csv",
    ".json",
    ".log",
    ".md",
    ".py",
    ".rs",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".yaml",
    ".yml",
}

DEFAULT_IGNORED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "dist",
    "node_modules",
    "target",
}


def _resolver(roots: Optional[Iterable[str | Path]]) -> ScopedPathResolver:
    return ScopedPathResolver.from_roots(roots)


def _entry_dict(path: Path, resolver: ScopedPathResolver) -> dict[str, object]:
    stat = path.stat()
    return {
        "name": path.name,
        "path": str(path),
        "relative_path": str(path.relative_to(resolver.roots[0]))
        if path.is_relative_to(resolver.roots[0])
        else str(path),
        "is_dir": path.is_dir(),
        "size": stat.st_size,
        "modified": stat.st_mtime,
    }


def _is_ignored(path: Path) -> bool:
    return any(part in DEFAULT_IGNORED_DIRS or part.endswith(".egg-info") for part in path.parts)


def list_dir(
    path: str | Path = ".",
    roots: Optional[Iterable[str | Path]] = None,
    max_entries: int = 200,
) -> dict[str, object]:
    resolver = _resolver(roots)
    directory = resolver.resolve(path)
    if not directory.is_dir():
        raise NotADirectoryError(str(directory))

    entries = []
    for child in sorted(directory.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
        if not resolver.is_allowed(child):
            continue
        entries.append(_entry_dict(child, resolver))
        if len(entries) >= max_entries:
            break
    return {"path": str(directory), "entries": entries, "truncated": len(entries) >= max_entries}


def stat_path(path: str | Path, roots: Optional[Iterable[str | Path]] = None) -> dict[str, object]:
    resolver = _resolver(roots)
    target = resolver.resolve(path)
    return _entry_dict(target, resolver)


def read_file(
    path: str | Path,
    roots: Optional[Iterable[str | Path]] = None,
    max_bytes: int = 200_000,
) -> dict[str, object]:
    resolver = _resolver(roots)
    target = resolver.resolve(path)
    if not target.is_file():
        raise FileNotFoundError(str(target))
    data = target.read_bytes()[:max_bytes]
    text = data.decode("utf-8", errors="replace")
    file_size = target.stat().st_size
    return {
        "path": str(target),
        "content": text,
        "bytes_read": len(data),
        "truncated": file_size > len(data),
    }


def glob_files(
    pattern: str,
    root: str | Path = ".",
    roots: Optional[Iterable[str | Path]] = None,
    max_results: int = 200,
) -> dict[str, object]:
    resolver = _resolver(roots)
    start = resolver.resolve(root)
    if not start.exists():
        raise FileNotFoundError(str(start))

    results: list[dict[str, object]] = []
    iterator = start.rglob(pattern) if start.is_dir() else [start]
    for path in iterator:
        if not resolver.is_allowed(path) or _is_ignored(path):
            continue
        results.append(_entry_dict(path, resolver))
        if len(results) >= max_results:
            break
    return {"root": str(start), "pattern": pattern, "matches": results, "truncated": len(results) >= max_results}


def _looks_textual(path: Path) -> bool:
    if path.suffix.lower() in TEXT_EXTENSIONS:
        return True
    try:
        sample = path.read_bytes()[:2048]
    except OSError:
        return False
    return b"\0" not in sample


def grep_files(
    query: str,
    root: str | Path = ".",
    include: str = "*",
    roots: Optional[Iterable[str | Path]] = None,
    max_results: int = 50,
    max_file_bytes: int = 500_000,
    case_sensitive: bool = False,
) -> dict[str, object]:
    resolver = _resolver(roots)
    start = resolver.resolve(root)
    if not start.exists():
        raise FileNotFoundError(str(start))

    flags = 0 if case_sensitive else re.IGNORECASE
    pattern = re.compile(re.escape(query), flags)
    candidates = [start] if start.is_file() else start.rglob("*")
    matches: list[dict[str, object]] = []

    for path in candidates:
        if len(matches) >= max_results:
            break
        if not path.is_file() or not resolver.is_allowed(path) or _is_ignored(path):
            continue
        if not fnmatch.fnmatch(path.name, include):
            continue
        if path.stat().st_size > max_file_bytes or not _looks_textual(path):
            continue

        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue

        for line_number, line in enumerate(lines, start=1):
            found = pattern.search(line) or pattern.search(path.name)
            if not found:
                continue
            snippet = line.strip()
            matches.append(
                {
                    "path": str(path),
                    "line": line_number,
                    "snippet": snippet[:500],
                }
            )
            if len(matches) >= max_results:
                break

    return {"root": str(start), "query": query, "matches": matches, "truncated": len(matches) >= max_results}


def summarize_matches(matches: list[dict[str, object]], query: str) -> str:
    if not matches:
        return f"No readable files matched {query!r} inside the approved roots."
    files = sorted({str(match["path"]) for match in matches})
    lines = [
        f"Found {len(matches)} matching line(s) across {len(files)} file(s) for {query!r}.",
        "Most relevant files:",
    ]
    for file_path in files[:8]:
        lines.append(f"- {file_path}")
    lines.append("Representative snippets:")
    for match in matches[:8]:
        lines.append(f"- {match['path']}:{match['line']}: {match['snippet']}")
    return os.linesep.join(lines)
