"""Skill library: list, read, create, edit, duplicate, delete, suggest, and AI-drafted skills."""

from __future__ import annotations

import json
import re
from typing import Callable, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from chat_software.skills import KNOWN_TOOLS, SkillLibrary


class SkillRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    description: str = Field(min_length=1, max_length=300)
    instructions: str = Field(min_length=1, max_length=20_000)
    tools: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class SkillDraftRequest(BaseModel):
    goal: str = Field(min_length=5, max_length=2000)
    provider: Optional[str] = None
    model: Optional[str] = None


DRAFT_PROMPT = (
    "You write reusable 'skills' for an AI assistant. A skill is a short set of instructions the "
    "assistant follows for one kind of task. Reply with ONLY a JSON object with keys: name (max 6 "
    "words), description (one sentence saying when to use it), instructions (numbered Markdown steps, "
    "concise, concrete, no fluff), tools (a list chosen only from: " + ", ".join(KNOWN_TOOLS) + "; empty "
    "if none are needed), tags (up to 5 lowercase words)."
)


def build_router(library: SkillLibrary, client_factory: Callable[..., object]) -> APIRouter:
    router = APIRouter()

    def _error(exc: Exception) -> HTTPException:
        if isinstance(exc, KeyError):
            return HTTPException(status_code=404, detail="Skill not found")
        if isinstance(exc, PermissionError):
            return HTTPException(status_code=403, detail=str(exc))
        return HTTPException(status_code=400, detail=str(exc))

    @router.get("/skills")
    def list_skills() -> dict[str, object]:
        return {"skills": [s.to_dict(include_instructions=False) for s in library.list()],
                "tools": list(KNOWN_TOOLS)}

    @router.get("/skills/suggest")
    def suggest(q: str) -> dict[str, object]:
        return {"skills": [s.to_dict(include_instructions=False) for s in library.suggest(q)]}

    @router.get("/skills/{skill_id}")
    def get_skill(skill_id: str) -> dict[str, object]:
        try:
            return library.get(skill_id).to_dict()
        except Exception as exc:
            raise _error(exc) from exc

    @router.post("/skills")
    def create_skill(request: SkillRequest) -> dict[str, object]:
        try:
            existing = {s.id for s in library.list()}
            from chat_software.skills import slugify

            skill_id = slugify(request.name)
            base, n = skill_id, 2
            while skill_id in existing:
                skill_id = f"{base[:58]}-{n}"
                n += 1
            return library.save(skill_id=skill_id, **request.model_dump()).to_dict()
        except Exception as exc:
            raise _error(exc) from exc

    @router.put("/skills/{skill_id}")
    def update_skill(skill_id: str, request: SkillRequest) -> dict[str, object]:
        try:
            current = library.get(skill_id)
            if current.builtin:
                raise PermissionError("Built-in skills are read-only; duplicate it to customise.")
            return library.save(skill_id=skill_id, **request.model_dump()).to_dict()
        except Exception as exc:
            raise _error(exc) from exc

    @router.post("/skills/{skill_id}/duplicate")
    def duplicate_skill(skill_id: str) -> dict[str, object]:
        try:
            source = library.get(skill_id)
            names = {s.name for s in library.list()}
            name = f"{source.name} (copy)"
            while name in names:
                name += " +"
            return create_skill(SkillRequest(name=name, description=source.description,
                                             instructions=source.instructions,
                                             tools=list(source.tools), tags=list(source.tags)))
        except HTTPException:
            raise
        except Exception as exc:
            raise _error(exc) from exc

    @router.delete("/skills/{skill_id}")
    def delete_skill(skill_id: str) -> dict[str, object]:
        try:
            library.delete(skill_id)
        except Exception as exc:
            raise _error(exc) from exc
        return {"deleted": True, "id": skill_id}

    @router.post("/skills/draft")
    def draft_skill(request: SkillDraftRequest) -> dict[str, object]:
        """Ask the selected model to draft a skill. The draft is returned for review, not saved."""

        client = client_factory(provider=request.provider, requested_model=request.model)
        raw = client.complete([
            {"role": "system", "content": DRAFT_PROMPT},
            {"role": "user", "content": request.goal},
        ])
        draft = _parse_json_object(raw)
        if not draft:
            raise HTTPException(status_code=502, detail="The model did not return a usable skill draft.")
        tools = [t for t in draft.get("tools") or [] if t in KNOWN_TOOLS]
        instructions = draft.get("instructions")
        if isinstance(instructions, list):
            instructions = "\n".join(f"{i}. {step}" for i, step in enumerate(instructions, 1))
        return {
            "draft": {
                "name": str(draft.get("name") or "")[:80],
                "description": str(draft.get("description") or "")[:300],
                "instructions": str(instructions or "")[:20_000],
                "tools": tools,
                "tags": [str(t)[:40] for t in (draft.get("tags") or [])][:5],
            },
            "saved": False,
        }

    return router


def _parse_json_object(text: str) -> dict:
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else cleaned[cleaned.find("{"): cleaned.rfind("}") + 1]
    try:
        value = json.loads(candidate)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}
