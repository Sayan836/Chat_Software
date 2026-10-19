"""Agent teams: definitions, templates and persistence.

A team is a small group of role-playing agents (1-8) with a lead. Each member has its own
instructions, optional provider/model, skills and an explicit tool allowlist. Members get no file
write or shell access on the host: their only execution tool is the isolated sandbox, and tools
are limited to what the team definition grants. Runs are bounded by turn and tool-step budgets.
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional

from chat_software.skills import KNOWN_TOOLS


MAX_MEMBERS = 8
_MEMBER_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,40}$")


@dataclass
class Member:
    id: str
    name: str
    role: str
    instructions: str = ""
    provider: Optional[str] = None
    model: Optional[str] = None
    skills: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)


@dataclass
class Team:
    id: str
    name: str
    description: str
    members: list[Member]
    lead: str
    max_turns: int = 6
    max_tool_steps: int = 4
    documents_root_id: Optional[str] = None
    created_at: float = 0.0
    updated_at: float = 0.0

    def member(self, member_id: str) -> Member:
        for member in self.members:
            if member.id == member_id:
                return member
        raise KeyError(f"No member {member_id!r} in team {self.name!r}")

    @property
    def lead_member(self) -> Member:
        return self.member(self.lead)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


def member_id_for(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40]
    return slug or f"agent-{uuid.uuid4().hex[:6]}"


def team_from_dict(data: dict[str, Any], *, team_id: Optional[str] = None,
                   known_skills: Optional[Iterable[str]] = None) -> Team:
    name = " ".join(str(data.get("name") or "").split())[:80]
    if not name:
        raise ValueError("A team needs a name.")
    raw_members = data.get("members") or []
    if not isinstance(raw_members, list) or not 1 <= len(raw_members) <= MAX_MEMBERS:
        raise ValueError(f"A team needs between 1 and {MAX_MEMBERS} members.")
    skills_allowed = set(known_skills) if known_skills is not None else None
    members: list[Member] = []
    seen: set[str] = set()
    for raw in raw_members:
        if not isinstance(raw, dict):
            raise ValueError("Each member must be an object.")
        member_name = " ".join(str(raw.get("name") or "").split())[:60]
        if not member_name:
            raise ValueError("Each member needs a name.")
        member_id = str(raw.get("id") or member_id_for(member_name))
        if not _MEMBER_ID.match(member_id) or member_id in seen:
            raise ValueError(f"Member ids must be unique slugs (problem with {member_name!r}).")
        seen.add(member_id)
        tools = [t for t in raw.get("tools") or [] if t]
        unknown_tools = [t for t in tools if t not in KNOWN_TOOLS]
        if unknown_tools:
            raise ValueError(f"Unknown tools for {member_name}: {', '.join(unknown_tools)}")
        skills = [s for s in raw.get("skills") or [] if s]
        if skills_allowed is not None:
            missing = [s for s in skills if s not in skills_allowed]
            if missing:
                raise ValueError(f"Unknown skills for {member_name}: {', '.join(missing)}")
        provider = raw.get("provider") or None
        if provider and provider not in {"ollama", "openai", "anthropic", "compatible"}:
            raise ValueError(f"Unknown provider for {member_name}: {provider}")
        members.append(Member(
            id=member_id,
            name=member_name,
            role=" ".join(str(raw.get("role") or "Team member").split())[:120],
            instructions=str(raw.get("instructions") or "")[:6000],
            provider=provider,
            model=(str(raw.get("model")).strip()[:200] or None) if raw.get("model") else None,
            skills=list(dict.fromkeys(skills)),
            tools=list(dict.fromkeys(tools)),
        ))
    lead = str(data.get("lead") or members[0].id)
    if lead not in seen:
        raise ValueError("The lead must be one of the members.")
    max_turns = int(data.get("max_turns") or 6)
    max_tool_steps = int(data.get("max_tool_steps") or 4)
    if not 1 <= max_turns <= 20 or not 0 <= max_tool_steps <= 10:
        raise ValueError("max_turns must be 1-20 and max_tool_steps 0-10.")
    return Team(
        id=team_id or str(data.get("id") or uuid.uuid4()),
        name=name,
        description=" ".join(str(data.get("description") or "").split())[:500],
        members=members,
        lead=lead,
        max_turns=max_turns,
        max_tool_steps=max_tool_steps,
        documents_root_id=data.get("documents_root_id") or None,
    )


TEMPLATES: dict[str, dict[str, Any]] = {
    "research-squad": {
        "name": "Research squad",
        "description": "Investigates a question on the web, checks the numbers and challenges weak claims.",
        "lead": "lead-researcher",
        "members": [
            {"id": "lead-researcher", "name": "Lead Researcher", "role": "Plans the investigation and writes the final answer",
             "instructions": "Break the problem down, delegate, and only state conclusions supported by teammates' sources."},
            {"id": "web-researcher", "name": "Web Researcher", "role": "Finds and reads primary sources",
             "skills": ["web-researcher"], "tools": ["web_search", "fetch_url"]},
            {"id": "analyst", "name": "Analyst", "role": "Checks figures and runs calculations",
             "tools": ["run_code"], "instructions": "Recompute any number that matters with run_code."},
            {"id": "critic", "name": "Critic", "role": "Challenges assumptions and finds gaps",
             "instructions": "Look for missing evidence, bias and alternative explanations. Be specific."},
        ],
    },
    "software-team": {
        "name": "Software team",
        "description": "Designs, implements and reviews small pieces of code, testing them in the sandbox.",
        "lead": "tech-lead",
        "members": [
            {"id": "tech-lead", "name": "Tech Lead", "role": "Clarifies requirements, designs and integrates",
             "instructions": "Agree on a small design first, then delegate implementation and review."},
            {"id": "engineer", "name": "Engineer", "role": "Writes and tests code",
             "skills": ["write-tests"], "tools": ["run_code"]},
            {"id": "reviewer", "name": "Reviewer", "role": "Reviews code for bugs and security issues",
             "skills": ["code-review"], "tools": ["run_code"]},
        ],
    },
    "content-team": {
        "name": "Content team",
        "description": "Drafts, fact-checks and edits articles, emails and documents.",
        "lead": "editor",
        "members": [
            {"id": "editor", "name": "Editor", "role": "Owns structure, tone and the final draft"},
            {"id": "writer", "name": "Writer", "role": "Writes clear first drafts"},
            {"id": "fact-checker", "name": "Fact Checker", "role": "Verifies claims against sources",
             "tools": ["web_search", "fetch_url"]},
        ],
    },
    "data-team": {
        "name": "Data team",
        "description": "Explores datasets, computes results in the sandbox and explains them.",
        "lead": "lead-analyst",
        "members": [
            {"id": "lead-analyst", "name": "Lead Analyst", "role": "Frames the question and reports findings"},
            {"id": "data-engineer", "name": "Data Engineer", "role": "Cleans data and runs the analysis",
             "skills": ["data-analysis"], "tools": ["run_code"]},
            {"id": "reviewer", "name": "Reviewer", "role": "Checks methods and conclusions"},
        ],
    },
}


class TeamStore:
    def __init__(self, db_path: str | Path) -> None:
        self._memory: Optional[sqlite3.Connection] = None
        if str(db_path) == ":memory:":
            self._memory = sqlite3.connect(":memory:", check_same_thread=False)
            self._path = None
        else:
            self._path = Path(db_path)
            self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS teams (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, config_json TEXT NOT NULL,
                    created_at REAL NOT NULL, updated_at REAL NOT NULL)"""
            )

    def _connect(self) -> sqlite3.Connection:
        return self._memory if self._memory is not None else sqlite3.connect(self._path)

    def save(self, team: Team) -> Team:
        now = time.time()
        with self._connect() as conn:
            row = conn.execute("SELECT created_at FROM teams WHERE id = ?", (team.id,)).fetchone()
            team.created_at = row[0] if row else now
            team.updated_at = now
            conn.execute(
                "INSERT OR REPLACE INTO teams (id, name, config_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                (team.id, team.name, json.dumps(team.to_dict()), team.created_at, team.updated_at),
            )
        return team

    def get(self, team_id: str) -> Team:
        with self._connect() as conn:
            row = conn.execute("SELECT config_json FROM teams WHERE id = ?", (team_id,)).fetchone()
        if row is None:
            raise KeyError(f"Team not found: {team_id}")
        data = json.loads(row[0])
        team = team_from_dict(data, team_id=team_id)
        team.created_at, team.updated_at = data.get("created_at", 0.0), data.get("updated_at", 0.0)
        return team

    def list(self) -> list[Team]:
        with self._connect() as conn:
            ids = [r[0] for r in conn.execute("SELECT id FROM teams ORDER BY updated_at DESC").fetchall()]
        return [self.get(team_id) for team_id in ids]

    def delete(self, team_id: str) -> bool:
        with self._connect() as conn:
            return bool(conn.execute("DELETE FROM teams WHERE id = ?", (team_id,)).rowcount)
