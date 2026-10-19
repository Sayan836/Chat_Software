"""Deterministic scan feature extraction for semantic file planning."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Iterable


TEXT_SNIPPET_EXTENSIONS = {
    ".csv",
    ".json",
    ".log",
    ".md",
    ".rtf",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}


@dataclass(frozen=True)
class FileFeature:
    id: int
    path: str
    name_tokens: list[str]
    ext: str
    size_bucket: str
    age_bucket: str
    parent: str
    snippet: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def featurize_files(
    files: Iterable[Path],
    *,
    root: str | Path,
    provider: str = "",
    snippet_bytes: int = 512,
) -> list[FileFeature]:
    root_path = Path(root).expanduser().resolve()
    features: list[FileFeature] = []
    for index, source in enumerate(files, start=1):
        relative = _relative(source, root_path)
        features.append(
            FileFeature(
                id=index,
                path=relative,
                name_tokens=_name_tokens(source.stem),
                ext=source.suffix.lower(),
                size_bucket=_size_bucket(source.stat().st_size),
                age_bucket=_age_bucket(source.stat().st_mtime),
                parent=_parent_hint(source, root_path),
                snippet=_snippet(source, relative=relative, provider=provider, max_bytes=snippet_bytes),
            )
        )
    return features


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _name_tokens(stem: str) -> list[str]:
    tokens = [
        token.lower()
        for token in re.split(r"[\s._-]+", stem)
        if token and not token.isdigit()
    ]
    return tokens[:12]


def _size_bucket(size: int) -> str:
    if size < 16_384:
        return "tiny"
    if size < 1_000_000:
        return "small"
    if size < 25_000_000:
        return "medium"
    if size < 250_000_000:
        return "large"
    return "huge"


def _age_bucket(mtime: float) -> str:
    now = datetime.now(timezone.utc).timestamp()
    age_days = max(0, int((now - mtime) / 86_400))
    if age_days <= 7:
        return "this-week"
    if age_days <= 31:
        return "this-month"
    if age_days <= 93:
        return "this-quarter"
    if age_days <= 366:
        return "this-year"
    return "older"


def _parent_hint(source: Path, root: Path) -> str:
    parent = source.parent.resolve()
    if parent == root:
        return "."
    parts = parent.relative_to(root).parts[:3]
    return "/".join(parts) or "."


def _snippet(source: Path, *, relative: str, provider: str, max_bytes: int) -> str | None:
    if provider.lower() != "ollama":
        return None
    if source.suffix.lower() not in TEXT_SNIPPET_EXTENSIONS:
        return None
    try:
        data = source.read_bytes()[: max(0, max_bytes)]
    except OSError:
        return None
    text = data.decode("utf-8", errors="replace")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]+", " ", text).strip()
    if not text:
        return None
    return f"<untrusted-file-snippet path={relative!r}>{text}</untrusted-file-snippet>"
