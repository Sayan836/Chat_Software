"""Running agent teams: delegated tasks and moderated multi-agent discussion.

Task mode    The lead writes a plan assigning subtasks to members; each member works (with its
             granted tools); the lead integrates the results into one answer.
Discussion   A team workspace chat. A user message addressed with @Name goes to those members;
             otherwise the lead moderates: after each contribution it picks who speaks next (or
             ``done``) until the turn budget runs out. Members see the shared transcript, so they
             respond to and build on each other. The lead closes multi-speaker rounds with a summary.

Everything yields events (``agent_started``, tool events, ``agent_message``, ...) so the UI can show
who is speaking and what tools they used.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterator, Optional

from chat_software.agent.toolbox import ToolContext, available_tools, run_tool_loop, strip_thinking
from chat_software.skills import SkillLibrary, skills_system_prompt

from . import Member, Team


ClientFactory = Callable[..., Any]
TRANSCRIPT_CHARS = 16_000


@dataclass
class TeamDeps:
    client_factory: ClientFactory
    skills: SkillLibrary
    sandbox: Any = None
    knowledge: Any = None
    default_provider: Optional[str] = None
    default_model: Optional[str] = None


class _EventBuffer:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def __call__(self, event: dict[str, Any]) -> None:
        self.items.append(event)

    def drain(self) -> list[dict[str, Any]]:
        items, self.items = self.items, []
        return items


def member_system_prompt(team: Team, member: Member, deps: TeamDeps, *, is_lead: bool) -> str:
    teammates = ", ".join(f"{m.name} ({m.role})" for m in team.members if m.id != member.id) or "none"
    parts = [
        f"You are {member.name}, {member.role}, in the agent team \"{team.name}\".",
        f"Team purpose: {team.description}" if team.description else "",
        f"Teammates: {teammates}.",
        "You are the team lead: you coordinate and make the final call." if is_lead else "",
        "Speak only as yourself. Address teammates as @Name when you need something from them. "
        "Be concise and concrete; do not repeat what others already said. Never claim to have run "
        "code or read a source unless a tool result in this conversation shows it.",
        member.instructions,
    ]
    if member.skills:
        parts.append(skills_system_prompt(deps.skills.resolve(member.skills)))
    return "\n\n".join(part for part in parts if part)


def _client(deps: TeamDeps, member: Member) -> Any:
    return deps.client_factory(provider=member.provider or deps.default_provider,
                               requested_model=member.model or (None if member.provider else deps.default_model))


def _transcript(lines: list[dict[str, str]]) -> str:
    text = "\n\n".join(f"[{line['speaker']}]: {line['content']}" for line in lines)
    return text[-TRANSCRIPT_CHARS:]


def run_member(
    team: Team,
    member: Member,
    deps: TeamDeps,
    prompt: str,
    *,
    sandbox_workspace_id: Optional[str] = None,
) -> Iterator[dict[str, Any]]:
    """Run one member turn; yields tool events then ``agent_message`` with the reply."""

    buffer = _EventBuffer()
    ctx = ToolContext(sandbox=deps.sandbox, knowledge=deps.knowledge, documents_root_id=team.documents_root_id,
                      sandbox_workspace_id=sandbox_workspace_id, web_enabled=True, on_event=buffer,
                      actor=member.name)
    tools = available_tools(member.tools, ctx) if team.max_tool_steps else []
    client = _client(deps, member)
    yield {"type": "agent_started", "member_id": member.id, "agent": member.name, "role": member.role,
           "provider": client.settings.provider, "model": client.settings.model,
           "tools": [tool.name for tool in tools]}
    messages = [
        {"role": "system", "content": member_system_prompt(team, member, deps, is_lead=member.id == team.lead)},
        {"role": "user", "content": prompt},
    ]
    result = run_tool_loop(client, messages, tools, ctx, max_steps=team.max_tool_steps)
    yield from buffer.drain()
    yield {"type": "agent_message", "member_id": member.id, "agent": member.name, "content": result.text,
           "tool_steps": [{"tool": s["tool"], "status": s["status"]} for s in result.steps],
           "sources": list(ctx.sources), "provider": client.settings.provider, "model": client.settings.model}


def run_task(team: Team, task: str, deps: TeamDeps, *, sandbox_workspace_id: Optional[str] = None
             ) -> Iterator[dict[str, Any]]:
    lead = team.lead_member
    others = [m for m in team.members if m.id != lead.id]
    yield {"type": "team_started", "mode": "task", "team": team.name,
           "members": [{"id": m.id, "name": m.name, "role": m.role} for m in team.members]}
    assignments: list[dict[str, str]] = []
    if others:
        roster = "\n".join(f"- {m.id}: {m.name}, {m.role}; tools: {', '.join(m.tools) or 'none'}" for m in others)
        planning_prompt = (
            f"Task from the user:\n{task}\n\nYour teammates:\n{roster}\n\n"
            "Assign each useful teammate one concrete subtask (skip anyone not needed). Reply with ONLY "
            'JSON: {"plan": "one sentence", "assignments": [{"member": "<id>", "task": "..."}]}'
        )
        plan_text = ""
        for event in run_member(team, lead, deps, planning_prompt, sandbox_workspace_id=sandbox_workspace_id):
            if event["type"] == "agent_message":
                plan_text = event["content"]
            else:
                yield event
        data = _json_object(plan_text)
        valid = {m.id for m in others}
        for item in data.get("assignments") or []:
            if isinstance(item, dict) and item.get("member") in valid and item.get("task"):
                assignments.append({"member": item["member"], "task": str(item["task"])[:2000]})
        if not assignments:
            assignments = [{"member": m.id, "task": task} for m in others]
        yield {"type": "plan", "agent": lead.name, "plan": str(data.get("plan") or "")[:500],
               "assignments": [{**a, "agent": team.member(a["member"]).name} for a in assignments]}
    transcript: list[dict[str, str]] = [{"speaker": "User", "content": task}]
    for assignment in assignments:
        member = team.member(assignment["member"])
        prompt = (f"Overall task:\n{task}\n\nWork so far:\n{_transcript(transcript[1:]) or '(none)'}\n\n"
                  f"Your assignment from {lead.name}: {assignment['task']}")
        for event in run_member(team, member, deps, prompt, sandbox_workspace_id=sandbox_workspace_id):
            if event["type"] == "agent_message":
                transcript.append({"speaker": member.name, "content": event["content"]})
            yield event
    final_prompt = (
        f"Task from the user:\n{task}\n\nTeam contributions:\n{_transcript(transcript[1:]) or '(none)'}\n\n"
        "Write the final answer for the user. Integrate the contributions, resolve disagreements, "
        "keep any [n] citations, and state remaining uncertainties briefly."
        if others else task
    )
    final = ""
    for event in run_member(team, lead, deps, final_prompt, sandbox_workspace_id=sandbox_workspace_id):
        if event["type"] == "agent_message":
            final = event["content"]
            event = {**event, "final": True}
        yield event
    yield {"type": "team_completed", "mode": "task", "final": final}


_MENTION = re.compile(r"@([A-Za-z][\w-]*(?:\s[A-Z][\w-]*)?)")


def mentioned_members(team: Team, message: str) -> list[Member]:
    found: list[Member] = []
    lowered = message.lower()
    for member in team.members:
        handles = {member.name.lower(), member.id.lower(), member.name.split()[0].lower(),
                   member.name.lower().replace(" ", "")}
        if any(f"@{handle}" in lowered for handle in handles):
            found.append(member)
    return found


def run_discussion(
    team: Team,
    history: list[dict[str, str]],
    user_message: str,
    deps: TeamDeps,
    *,
    max_turns: Optional[int] = None,
    sandbox_workspace_id: Optional[str] = None,
) -> Iterator[dict[str, Any]]:
    """One user message in a team workspace chat. ``history`` holds prior {speaker, content} lines."""

    budget = max(1, min(int(max_turns or team.max_turns), 20))
    transcript = list(history) + [{"speaker": "User", "content": user_message}]
    lead = team.lead_member
    yield {"type": "team_started", "mode": "discussion", "team": team.name, "max_turns": budget}
    spoken: list[str] = []

    def turn(member: Member, instruction: str):
        prompt = (f"Team conversation so far:\n{_transcript(transcript)}\n\n"
                  f"It is your turn, {member.name}. {instruction}")
        for event in run_member(team, member, deps, prompt, sandbox_workspace_id=sandbox_workspace_id):
            if event["type"] == "agent_message":
                transcript.append({"speaker": member.name, "content": event["content"]})
                spoken.append(member.id)
            yield event

    addressed = mentioned_members(team, user_message)
    if addressed:
        for member in addressed[:budget]:
            yield from turn(member, "The user addressed you directly; reply to them (and to teammates if useful).")
    else:
        roster = "\n".join(f"- {m.id}: {m.name}, {m.role}" for m in team.members)
        for _ in range(budget):
            decision_prompt = (
                f"Team conversation so far:\n{_transcript(transcript)}\n\nTeam members:\n{roster}\n\n"
                "As moderator, decide who should speak next to best help the user, or whether the "
                "discussion has answered them. Reply with ONLY JSON: "
                '{"next": "<member id or done>", "reason": "short"}'
            )
            client = _client(deps, lead)
            raw = strip_thinking(client.complete([
                {"role": "system", "content": member_system_prompt(team, lead, deps, is_lead=True)},
                {"role": "user", "content": decision_prompt},
            ]))
            choice = _json_object(raw)
            next_id = str(choice.get("next") or "").strip()
            if next_id == "done" and spoken:
                yield {"type": "moderator", "agent": lead.name, "decision": "done", "reason": str(choice.get("reason") or "")[:200]}
                break
            valid = {m.id for m in team.members}
            if next_id not in valid:
                # Fallback: let the next member who has not spoken yet take the turn, lead first.
                order = [lead] + [m for m in team.members if m.id != lead.id]
                next_id = next((m.id for m in order if m.id not in spoken), lead.id)
            yield {"type": "moderator", "agent": lead.name, "decision": next_id,
                   "reason": str(choice.get("reason") or "")[:200]}
            yield from turn(team.member(next_id),
                            "Contribute your expertise. Respond to teammates' points where relevant.")
    if len(set(spoken)) > 1 and spoken[-1] != lead.id:
        yield from turn(lead, "Wrap up for the user: the answer or decision, and any open points. Be brief.")
    yield {"type": "team_completed", "mode": "discussion", "speakers": spoken}


def _json_object(text: str) -> dict[str, Any]:
    cleaned = strip_thinking(text)
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else cleaned[cleaned.find("{"): cleaned.rfind("}") + 1]
    try:
        value = json.loads(candidate)
    except ValueError:
        return {}
    return value if isinstance(value, dict) else {}
