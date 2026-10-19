"""Scoped path validation for file tools."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


def default_roots() -> list[Path]:
    configured = os.getenv("CHAT_SOFTWARE_ALLOWED_ROOTS", "").strip()
    if configured:
        return [Path(part).expanduser().resolve() for part in configured.split(os.pathsep) if part]
    return [Path.cwd().resolve()]


@dataclass(frozen=True)
class ScopedPathResolver:
    roots: tuple[Path, ...]

    @classmethod
    def from_roots(cls, roots: Optional[Iterable[str | Path]] = None) -> "ScopedPathResolver":
        selected = list(default_roots() if roots is None else roots)
        absolute = tuple(Path(root).expanduser().absolute() for root in selected)
        if not absolute:
            raise ValueError("At least one scoped root is required.")
        return cls(roots=absolute)

    def resolve(self, path: str | Path = ".") -> Path:
        requested = Path(path).expanduser()
        if not requested.is_absolute():
            requested = self.roots[0] / requested
        resolved = requested.resolve()
        if not self.is_allowed(resolved):
            roots = ", ".join(str(root) for root in self.roots)
            raise PermissionError(f"Path is outside approved roots: {resolved}. Roots: {roots}")
        return resolved

    def is_allowed(self, path: Path) -> bool:
        candidate = path.resolve()
        for root in self.roots:
            try:
                candidate.relative_to(root.resolve())
                return True
            except ValueError:
                continue
        return False

    def root_for(self, path: str | Path) -> Path:
        candidate = Path(path).expanduser()
        if not candidate.is_absolute():
            candidate = self.roots[0] / candidate
        resolved = candidate.resolve()
        for root in self.roots:
            try:
                resolved.relative_to(root.resolve())
                return root
            except ValueError:
                continue
        roots = ", ".join(str(root) for root in self.roots)
        raise PermissionError(f"Path is outside approved roots: {resolved}. Roots: {roots}")
