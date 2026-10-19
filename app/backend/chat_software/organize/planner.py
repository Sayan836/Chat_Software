"""Deterministic organization planners.

The planner is read-only: it inspects files, computes relative destination paths,
and returns a structured plan. It never mutates disk.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
import re
from typing import Iterable

from chat_software.tools.file_ops import DEFAULT_IGNORED_DIRS
from chat_software.tools.file_ops_write import APP_TRASH_DIR
from chat_software.tools.scope import ScopedPathResolver


SUPPORTED_STRATEGIES = {"by_type", "by_date", "by_filename", "by_topic"}
STRATEGY_ALIASES = {
    "by_name": "by_filename",
    "filename": "by_filename",
    "name": "by_filename",
    "topic": "by_topic",
    "semantic": "by_topic",
    "by_semantic": "by_topic",
}
SUPPORTED_FILE_FILTERS = {"all", "image", "video", "audio", "document"}
DEFAULT_MAX_FILES = 1000

EXTRA_IGNORED_DIRS = {
    APP_TRASH_DIR,
    ".chat_software_state",
}

IGNORED_FILENAMES = {".DS_Store"}

TYPE_FOLDERS = {
    "Images": {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
        ".heic",
        ".svg",
    },
    "Videos": {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"},
    "Audio": {".mp3", ".wav", ".aac", ".flac", ".m4a", ".ogg"},
    "Archives": {".zip", ".tar", ".gz", ".tgz", ".bz2", ".7z", ".rar"},
    "PDFs": {".pdf"},
    "Documents": {".doc", ".docx", ".rtf", ".txt", ".md", ".odt"},
    "Spreadsheets": {".xls", ".xlsx", ".csv", ".tsv", ".ods"},
    "Presentations": {".ppt", ".pptx", ".key", ".odp"},
    "Code": {
        ".py",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".rs",
        ".go",
        ".java",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".cs",
        ".php",
        ".rb",
        ".swift",
        ".kt",
        ".sh",
        ".ps1",
        ".html",
        ".css",
        ".json",
        ".yaml",
        ".yml",
        ".toml",
        ".xml",
        ".sql",
    },
    "Apps": {".app", ".exe", ".msi", ".dmg", ".pkg", ".deb", ".rpm"},
}

FILE_FILTER_EXTENSIONS = {
    "image": TYPE_FOLDERS["Images"],
    "video": TYPE_FOLDERS["Videos"],
    "audio": TYPE_FOLDERS["Audio"],
    "document": TYPE_FOLDERS["Documents"] | TYPE_FOLDERS["PDFs"],
}

FILE_FILTER_FOLDERS = {
    "image": "Images",
    "video": "Videos",
    "audio": "Audio",
    "document": "Documents",
}

GENERIC_FILENAME_TOKENS = {
    "dsc",
    "dscf",
    "img",
    "image",
    "photo",
    "pic",
    "picture",
    "screenshot",
    "scan",
}

WINDOWS_RESERVED_NAMES = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{index}" for index in range(1, 10)),
    *(f"LPT{index}" for index in range(1, 10)),
}


def build_organize_plan(
    *,
    root: str | Path,
    strategy: str = "by_type",
    path: str | Path = ".",
    prompt: str | None = None,
    max_files: int = DEFAULT_MAX_FILES,
    file_filter: str = "all",
) -> dict[str, object]:
    strategy = _normalize_strategy(strategy)
    file_filter = _normalize_file_filter(file_filter)
    if strategy not in SUPPORTED_STRATEGIES:
        supported = ", ".join(sorted(SUPPORTED_STRATEGIES))
        raise ValueError(f"Unsupported organization strategy: {strategy}. Use one of: {supported}.")
    if file_filter not in SUPPORTED_FILE_FILTERS:
        supported = ", ".join(sorted(SUPPORTED_FILE_FILTERS))
        raise ValueError(f"Unsupported organization file filter: {file_filter}. Use one of: {supported}.")

    root_path = Path(root).expanduser().resolve()
    resolver = ScopedPathResolver.from_roots([root_path])
    start = resolver.resolve(path)
    if not start.exists():
        raise FileNotFoundError(str(start))

    files, truncated = _collect_files(
        start,
        root_path,
        max_files=max_files,
        file_filter=file_filter,
    )
    used_targets: set[str] = set()
    mkdirs: dict[str, dict[str, object]] = {}
    moves: list[dict[str, object]] = []
    skipped: list[dict[str, object]] = []
    warnings: list[str] = []

    for source in files:
        relative_source = _relative(source, root_path)
        target_relative = _target_for(
            source,
            root_path,
            strategy,
            file_filter=file_filter,
            prompt=prompt,
        )
        if relative_source == target_relative:
            skipped.append(
                {
                    "path": relative_source,
                    "reason": "already in target location",
                }
            )
            continue

        unique_target, collision_warning = _unique_target(
            root_path=root_path,
            source=source,
            target_relative=target_relative,
            used_targets=used_targets,
        )
        if collision_warning:
            warnings.append(collision_warning)

        should_skip = False
        for target_parent in _target_parent_paths(PurePosixPath(unique_target)):
            target_parent_path = root_path / Path(target_parent)
            if target_parent_path.exists() and not target_parent_path.is_dir():
                warning = f"target parent is not a folder: {target_parent}"
                warnings.append(warning)
                skipped.append(
                    {
                        "path": relative_source,
                        "reason": warning,
                    }
                )
                should_skip = True
                break
            if not target_parent_path.exists():
                mkdirs[target_parent] = {
                    "op": "mkdir",
                    "path": target_parent,
                    "reason": "target folder for organization",
                }
        if should_skip:
            continue

        moves.append(
            {
                "op": "move",
                "from": relative_source,
                "to": unique_target,
                "reason": _reason_for(source, root_path, strategy, prompt=prompt),
            }
        )

    actions = [mkdirs[key] for key in _sorted_parent_paths(mkdirs)] + moves
    return {
        "root": str(root_path),
        "strategy": strategy,
        "file_filter": file_filter,
        "path": _relative(start, root_path) if start != root_path else ".",
        "prompt": prompt,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "actions": actions,
        "summary": {
            "moves": len(moves),
            "new_dirs": len(mkdirs),
            "skipped": len(skipped),
            "files_seen": len(files),
            "truncated": truncated,
        },
        "warnings": warnings,
        "skipped": skipped,
    }


def _collect_files(
    start: Path,
    root: Path,
    *,
    max_files: int,
    file_filter: str,
) -> tuple[list[Path], bool]:
    candidates: Iterable[Path]
    if start.is_file():
        candidates = [start]
    elif start.is_dir():
        candidates = sorted(start.rglob("*"), key=lambda item: _relative(item, root).casefold())
    else:
        return [], False

    files: list[Path] = []
    truncated = False
    for candidate in candidates:
        if not candidate.is_file():
            continue
        if _is_ignored(candidate, root):
            continue
        if not _matches_file_filter(candidate, file_filter):
            continue
        if len(files) >= max_files:
            truncated = True
            break
        files.append(candidate)
    return files, truncated


def _is_ignored(path: Path, root: Path) -> bool:
    try:
        relative_parts = path.relative_to(root).parts
    except ValueError:
        return True
    ignored = DEFAULT_IGNORED_DIRS | EXTRA_IGNORED_DIRS
    return (
        path.name in IGNORED_FILENAMES
        or any(part in ignored or part.endswith(".egg-info") for part in relative_parts)
    )


def _matches_file_filter(source: Path, file_filter: str) -> bool:
    if file_filter == "all":
        return True
    return source.suffix.lower() in FILE_FILTER_EXTENSIONS.get(file_filter, set())


def _target_for(
    source: Path,
    root: Path,
    strategy: str,
    *,
    file_filter: str,
    prompt: str | None = None,
) -> str:
    if strategy == "by_date":
        modified = datetime.fromtimestamp(source.stat().st_mtime)
        folder = PurePosixPath(str(modified.year)) / modified.strftime("%m-%B")
        return str(folder / source.name)
    if strategy == "by_filename":
        base_folder = FILE_FILTER_FOLDERS.get(file_filter, "By Name")
        folder = PurePosixPath(base_folder) / _filename_group(source)
        return str(folder / source.name)
    if strategy == "by_topic":
        from .semantic.compiler import topic_folder_for_source

        folder = topic_folder_for_source(source, root=root, prompt=prompt, file_filter=file_filter)
        return str(PurePosixPath(folder) / source.name)
    folder_name = _type_folder(source.suffix.lower())
    return str(PurePosixPath(folder_name) / source.name)


def _type_folder(extension: str) -> str:
    for folder, extensions in TYPE_FOLDERS.items():
        if extension in extensions:
            return folder
    return "Others"


def _reason_for(source: Path, root: Path, strategy: str, *, prompt: str | None = None) -> str:
    if strategy == "by_date":
        modified = datetime.fromtimestamp(source.stat().st_mtime)
        return f"modified in {modified.strftime('%B %Y')}"
    if strategy == "by_filename":
        return f"filename group {_filename_group(source)!r}"
    if strategy == "by_topic":
        from .semantic.compiler import topic_label_for_source

        return f"topic group {topic_label_for_source(source, root=root, prompt=prompt)!r}"
    return f"{source.suffix.lower() or 'no-extension'} file"


def _filename_group(source: Path) -> str:
    stem = source.stem.strip()
    parts = [part for part in re.split(r"[\s._-]+", stem) if part]
    group = stem or "unnamed"
    for part in parts:
        lowered = part.lower()
        if lowered in GENERIC_FILENAME_TOKENS or part.isdigit():
            continue
        group = part
        break
    return _safe_folder_name(group)


def _safe_folder_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._ -]+", "-", value).strip(" .-_")
    cleaned = cleaned.rstrip(" .")
    root_name = cleaned.split(".", 1)[0].upper()
    if root_name in WINDOWS_RESERVED_NAMES:
        return "unnamed"
    return cleaned[:80] or "unnamed"


def _target_parent_paths(target: PurePosixPath) -> list[str]:
    parent = target.parent
    if str(parent) in {"", "."}:
        return []
    paths = [parent]
    paths.extend(candidate for candidate in parent.parents if str(candidate) not in {"", "."})
    return [str(candidate) for candidate in reversed(paths)]


def _sorted_parent_paths(mkdirs: dict[str, dict[str, object]]) -> list[str]:
    return sorted(
        mkdirs,
        key=lambda value: (len(PurePosixPath(value).parts), value.casefold()),
    )


def _normalize_strategy(strategy: str) -> str:
    selected = str(strategy or "by_type").strip().lower()
    return STRATEGY_ALIASES.get(selected, selected)


def _normalize_file_filter(file_filter: str) -> str:
    selected = str(file_filter or "all").strip().lower()
    aliases = {
        "images": "image",
        "photo": "image",
        "photos": "image",
        "picture": "image",
        "pictures": "image",
        "videos": "video",
        "movies": "video",
        "audios": "audio",
        "music": "audio",
        "docs": "document",
        "documents": "document",
        "pdf": "document",
        "pdfs": "document",
    }
    return aliases.get(selected, selected)


def _unique_target(
    *,
    root_path: Path,
    source: Path,
    target_relative: str,
    used_targets: set[str],
) -> tuple[str, str | None]:
    candidate = PurePosixPath(target_relative)
    original = candidate
    index = 1
    warning: str | None = None
    while True:
        absolute_target = root_path / Path(str(candidate))
        key = str(candidate).casefold()
        target_exists = absolute_target.exists() and absolute_target.resolve() != source.resolve()
        if key not in used_targets and not target_exists:
            used_targets.add(key)
            if candidate != original:
                warning = f"collision: {original} already exists; using {candidate}"
            return str(candidate), warning
        candidate = candidate.with_name(f"{candidate.stem} ({index}){candidate.suffix}")
        index += 1


def _relative(path: Path, root: Path) -> str:
    return PurePosixPath(path.relative_to(root)).as_posix()
