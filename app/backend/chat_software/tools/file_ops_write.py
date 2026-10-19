"""Scoped write helpers for Phase 2 filesystem mutations.

These helpers form the deterministic write boundary for approved file roots.
Callers are expected to authorize roots before constructing the resolver.
"""

from __future__ import annotations

import errno
import os
import re
import shutil
import uuid
from pathlib import Path

from .scope import ScopedPathResolver


APP_TRASH_DIR = ".chat_software_trash"


def make_dir(path: str | Path, resolver: ScopedPathResolver) -> dict[str, object]:
    target = resolver.resolve(path)
    existed = target.exists()
    target.mkdir(parents=True, exist_ok=True)
    return {"op": "mkdir", "path": str(target), "created": not existed}


def remove_empty_dir(path: str | Path, resolver: ScopedPathResolver) -> dict[str, object]:
    target = resolver.resolve(path)
    if not target.exists():
        return {"op": "rmdir", "path": str(target), "removed": False, "reason": "missing"}
    if not target.is_dir():
        raise NotADirectoryError(str(target))
    target.rmdir()
    return {"op": "rmdir", "path": str(target), "removed": True}


def write_file(
    path: str | Path,
    content: str,
    resolver: ScopedPathResolver,
    *,
    overwrite: bool = False,
    encoding: str = "utf-8",
) -> dict[str, object]:
    target = resolver.resolve(path)
    existed = target.exists()
    if existed and target.is_dir():
        raise IsADirectoryError(str(target))
    if existed and not overwrite:
        raise FileExistsError(f"Destination already exists: {target}")

    resolver.root_for(target.parent)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding=encoding)
    return {
        "op": "write",
        "path": str(target),
        "created": not existed,
        "overwritten": existed,
        "bytes_written": len(content.encode(encoding)),
    }


def edit_file(
    path: str | Path,
    old_text: str,
    new_text: str,
    resolver: ScopedPathResolver,
    *,
    count: int = 1,
    encoding: str = "utf-8",
) -> dict[str, object]:
    if not old_text:
        raise ValueError("old_text must be non-empty.")

    target = resolver.resolve(path)
    if not target.is_file():
        raise FileNotFoundError(str(target))

    original = target.read_text(encoding=encoding)
    available = original.count(old_text)
    if available == 0:
        raise ValueError(f"Text to replace was not found in: {target}")

    limit = count if count > 0 else available
    replacements = min(available, limit)
    updated = original.replace(old_text, new_text, replacements)
    target.write_text(updated, encoding=encoding)
    return {
        "op": "edit",
        "path": str(target),
        "replacements": replacements,
        "bytes_written": len(updated.encode(encoding)),
    }


def copy_path(
    source: str | Path,
    destination: str | Path,
    resolver: ScopedPathResolver,
) -> dict[str, object]:
    src = resolver.resolve(source)
    if not src.exists():
        raise FileNotFoundError(str(src))

    dest = _destination_path(src, destination, resolver)
    if dest.exists():
        raise FileExistsError(f"Destination already exists: {dest}")

    resolver.root_for(dest.parent)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if _is_relative_to(dest, src):
            raise ValueError("Cannot copy a directory into itself.")
        shutil.copytree(src, dest)
        kind = "directory"
    else:
        shutil.copy2(src, dest)
        kind = "file"

    return {
        "op": "copy",
        "from": str(src),
        "to": str(dest),
        "kind": kind,
    }


def move_path(
    source: str | Path,
    destination: str | Path,
    resolver: ScopedPathResolver,
    *,
    batch_id: str | None = None,
) -> dict[str, object]:
    src = resolver.resolve(source)
    dest = _destination_path(src, destination, resolver)
    return safe_move(src, dest, resolver, batch_id=batch_id)


def safe_move(
    source: str | Path,
    destination: str | Path,
    resolver: ScopedPathResolver,
    *,
    batch_id: str | None = None,
) -> dict[str, object]:
    """Move a file without overwriting and without permanently deleting user data.

    Same-filesystem moves use `os.rename`. Cross-device file moves copy to the
    destination and quarantine the original source inside the approved root.
    Cross-device directory moves fail closed until the journaled executor can
    restore complex directory trees safely.
    """

    selected_batch_id = batch_id or str(uuid.uuid4())
    src = resolver.resolve(source)
    dest = resolver.resolve(destination)
    source_path = Path(source).expanduser()
    destination_path = Path(destination).expanduser()
    rename_src = source_path if source_path.is_absolute() else src
    rename_dest = destination_path if destination_path.is_absolute() else dest

    if not src.exists():
        raise FileNotFoundError(str(src))
    if dest.exists():
        raise FileExistsError(f"Destination already exists: {dest}")

    resolver.root_for(dest.parent)
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        os.rename(rename_src, rename_dest)
        return {
            "op": "move",
            "from": str(src),
            "to": str(dest),
            "method": "rename",
            "source_quarantine": None,
        }
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise

    if src.is_dir():
        raise OSError("Cross-device directory moves are not supported yet.")

    temp = _unique_path(dest.parent / f".{dest.name}.{uuid.uuid4().hex}.tmp")
    resolver.resolve(temp)
    try:
        shutil.copy2(src, temp)
        if dest.exists():
            raise FileExistsError(f"Destination already exists: {dest}")
        os.replace(temp, dest)
    except Exception:
        if temp.exists():
            _move_to_quarantine(
                temp,
                resolver,
                batch_id=selected_batch_id,
                kind="temporary_files",
            )
        raise

    source_quarantine = _move_to_quarantine(
        src,
        resolver,
        batch_id=selected_batch_id,
        kind="cross_device_sources",
    )
    return {
        "op": "move",
        "from": str(src),
        "to": str(dest),
        "method": "copy_then_quarantine_source",
        "source_quarantine": str(source_quarantine),
    }


def quarantine_path(
    path: str | Path,
    resolver: ScopedPathResolver,
    *,
    batch_id: str,
    kind: str = "files",
) -> Path:
    resolved = resolver.resolve(path)
    root = resolver.root_for(resolved)
    relative = resolved.relative_to(root.resolve())
    candidate = root / APP_TRASH_DIR / _safe_segment(batch_id) / _safe_segment(kind) / relative
    return _unique_path(candidate)


def _move_to_quarantine(
    path: Path,
    resolver: ScopedPathResolver,
    *,
    batch_id: str,
    kind: str,
) -> Path:
    target = quarantine_path(path, resolver, batch_id=batch_id, kind=kind)
    target.parent.mkdir(parents=True, exist_ok=True)
    os.rename(path, target)
    return target


def _safe_segment(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "batch"


def _unique_path(candidate: Path) -> Path:
    if not candidate.exists():
        return candidate
    parent = candidate.parent
    stem = candidate.stem
    suffix = candidate.suffix
    for index in range(1, 10_000):
        replacement = parent / f"{stem} ({index}){suffix}"
        if not replacement.exists():
            return replacement
    raise FileExistsError(f"Could not find an available path for: {candidate}")


def _destination_path(source: Path, destination: str | Path, resolver: ScopedPathResolver) -> Path:
    dest = resolver.resolve(destination)
    if dest.exists() and dest.is_dir():
        dest = resolver.resolve(dest / source.name)
    return dest


def _is_relative_to(candidate: Path, parent: Path) -> bool:
    try:
        candidate.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False
