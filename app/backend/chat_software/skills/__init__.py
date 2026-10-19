"""Skills: reusable instructions (plus an optional tool allowlist) applied to chats, research and agents.

A skill is a Markdown file with a small front matter block::

    ---
    name: Data analysis
    description: Analyse CSV/Excel data with Python in the sandbox and explain the results.
    tools: run_code
    ---
    Instructions in Markdown...

Built-in skills ship in ``skills/builtin`` and are read-only (duplicate one to customise it).
User skills live in ``<state_dir>/skills/<id>.md`` and can be edited in the app or any editor.
Skill text is user-authored configuration, applied as extra system instructions; it never grants a
tool that the caller has not also enabled for the run.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from chat_software.paths import state_dir


BUILTIN_DIR = Path(__file__).resolve().parent / "builtin"
KNOWN_TOOLS = ("web_search", "fetch_url", "search_documents", "run_code")
MAX_INSTRUCTIONS = 20_000
_ID = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}$")


@dataclass(frozen=True)
class Skill:
    id: str
    name: str
    description: str
    instructions: str
    tools: tuple[str, ...] = ()
    builtin: bool = False
    updated_at: Optional[float] = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self, include_instructions: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tools": list(self.tools),
            "tags": list(self.tags),
            "builtin": self.builtin,
            "editable": not self.builtin,
            "updated_at": self.updated_at,
        }
        if include_instructions:
            data["instructions"] = self.instructions
        return data

    def to_markdown(self) -> str:
        lines = ["---", f"name: {self.name}", f"description: {self.description}"]
        if self.tools:
            lines.append(f"tools: {', '.join(self.tools)}")
        if self.tags:
            lines.append(f"tags: {', '.join(self.tags)}")
        lines += ["---", self.instructions.strip(), ""]
        return "\n".join(lines)


def slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")[:63]
    return slug if len(slug) >= 2 else f"skill-{int(time.time())}"


def parse_skill(skill_id: str, text: str, *, builtin: bool, updated_at: Optional[float] = None) -> Skill:
    meta: dict[str, str] = {}
    body = text
    if text.startswith("---"):
        parts = text.split("\n---", 1)
        if len(parts) == 2:
            for line in parts[0].splitlines()[1:]:
                if ":" in line:
                    key, value = line.split(":", 1)
                    meta[key.strip().lower()] = value.strip()
            body = parts[1].lstrip("-").lstrip("\n")
    tools = tuple(t for t in _split(meta.get("tools")) if t in KNOWN_TOOLS)
    return Skill(
        id=skill_id,
        name=meta.get("name") or skill_id,
        description=meta.get("description", ""),
        instructions=body.strip(),
        tools=tools,
        builtin=builtin,
        updated_at=updated_at,
        tags=tuple(_split(meta.get("tags"))),
    )


def _split(value: Optional[str]) -> list[str]:
    return [part.strip() for part in (value or "").split(",") if part.strip()]


class SkillLibrary:
    def __init__(self, user_dir: Optional[Path] = None, builtin_dir: Path = BUILTIN_DIR) -> None:
        self._user_dir = user_dir
        self._builtin_dir = builtin_dir

    @property
    def user_dir(self) -> Optional[Path]:
        if self._user_dir is not None:
            return self._user_dir
        directory = state_dir()
        return None if str(directory) == ":memory:" else directory / "skills"

    def list(self) -> list[Skill]:
        skills = {s.id: s for s in self._read_dir(self._builtin_dir, builtin=True)}
        for skill in self._read_dir(self.user_dir, builtin=False):
            skills[skill.id] = skill  # a user skill with the same id overrides the built-in
        return sorted(skills.values(), key=lambda s: (s.builtin, s.name.lower()))

    def get(self, skill_id: str) -> Skill:
        for skill in self.list():
            if skill.id == skill_id:
                return skill
        raise KeyError(f"Skill not found: {skill_id}")

    def resolve(self, skill_ids: Iterable[str]) -> list[Skill]:
        return [self.get(skill_id) for skill_id in dict.fromkeys(skill_ids)]

    def save(
        self,
        *,
        name: str,
        description: str,
        instructions: str,
        tools: Iterable[str] = (),
        tags: Iterable[str] = (),
        skill_id: Optional[str] = None,
    ) -> Skill:
        directory = self.user_dir
        if directory is None:
            raise RuntimeError("User skills need a persistent state directory.")
        name = " ".join((name or "").split())[:80]
        description = " ".join((description or "").split())[:300]
        if not name or not description or not (instructions or "").strip():
            raise ValueError("A skill needs a name, a description and instructions.")
        if len(instructions) > MAX_INSTRUCTIONS:
            raise ValueError(f"Instructions are limited to {MAX_INSTRUCTIONS} characters.")
        unknown = [t for t in tools if t not in KNOWN_TOOLS]
        if unknown:
            raise ValueError(f"Unknown tools: {', '.join(unknown)}. Allowed: {', '.join(KNOWN_TOOLS)}")
        selected_id = skill_id or slugify(name)
        if not _ID.match(selected_id):
            raise ValueError("Skill id must be lowercase letters, digits and dashes.")
        skill = Skill(
            id=selected_id,
            name=name,
            description=description,
            instructions=instructions.strip(),
            tools=tuple(dict.fromkeys(tools)),
            tags=tuple(t[:40] for t in tags if t.strip())[:10],
        )
        directory.mkdir(parents=True, exist_ok=True)
        (directory / f"{selected_id}.md").write_text(skill.to_markdown())
        return self.get(selected_id)

    def delete(self, skill_id: str) -> None:
        directory = self.user_dir
        path = directory / f"{skill_id}.md" if directory is not None and _ID.match(skill_id) else None
        if path is None or not path.exists():
            if any(s.id == skill_id for s in self._read_dir(self._builtin_dir, builtin=True)):
                raise PermissionError("Built-in skills cannot be deleted; duplicate one to customise it.")
            raise KeyError(f"Skill not found: {skill_id}")
        path.unlink()

    def suggest(self, text: str, limit: int = 3) -> list[Skill]:
        words = {w for w in re.findall(r"[a-z0-9]{3,}", (text or "").lower())}
        scored = []
        for skill in self.list():
            haystack = f"{skill.name} {skill.description} {' '.join(skill.tags)}".lower()
            score = sum(1 for word in words if word in haystack)
            if score:
                scored.append((score, skill))
        scored.sort(key=lambda pair: (-pair[0], pair[1].name))
        return [skill for _score, skill in scored[:limit]]

    @staticmethod
    def _read_dir(directory: Optional[Path], *, builtin: bool) -> list[Skill]:
        if directory is None or not directory.is_dir():
            return []
        skills = []
        for path in sorted(directory.glob("*.md")):
            if not _ID.match(path.stem):
                continue
            try:
                skills.append(parse_skill(path.stem, path.read_text(), builtin=builtin,
                                          updated_at=path.stat().st_mtime))
            except (OSError, UnicodeDecodeError):
                continue
        return skills


def skills_system_prompt(skills: Iterable[Skill]) -> str:
    blocks = [
        f"## Skill: {skill.name}\n{skill.description}\n\n{skill.instructions}" for skill in skills
    ]
    if not blocks:
        return ""
    return "Apply the following skills the user selected for this conversation.\n\n" + "\n\n".join(blocks)
