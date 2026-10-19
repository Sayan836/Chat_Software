"""Local-first agent adapter.

The public adapter preserves the original response/session contract while Phase
3 moves orchestration into a LangGraph root graph when that dependency is
available.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from time import perf_counter
from typing import Any, Callable, Iterable, Iterator, Optional

from chat_software.organize import apply_organize_plan, build_organize_plan
from chat_software.provider.factory import ChatClient, get_chat_client
from chat_software.safety_store import SafetyStore
from chat_software.tools.file_ops import (
    glob_files,
    grep_files,
    list_dir,
    read_file,
    stat_path,
    summarize_matches,
)
from chat_software.tools.file_ops_write import copy_path, edit_file, make_dir, move_path, write_file
from chat_software.tools.git_tools import git_diff, git_status
from chat_software.tools.scope import ScopedPathResolver

from .session_store import SessionStore


LLM_PLANNED_TOOL_ALLOWLIST = {
    "list_dir",
    "stat_path",
    "read_file",
    "glob_files",
    "grep_files",
    "git_status",
    "git_diff",
    "organize_plan",
}

CHAT_FALLBACK_RESPONSE = "__CHAT_FALLBACK__"
PENDING_OPERATION_KEY = "pending_operation"
PENDING_OPERATION_RESOLVED_KEY = "pending_operation_resolved"
WRITE_TOOL_NAMES = {"make_dir", "write_file", "copy_path", "move_path", "edit_file"}
PENDING_DECISIONS = {"approve", "decline", "revise", "clarify"}
ROOT_REQUIRED_PENDING_TYPE = "root_required"


def _extension_patterns(*extensions: str) -> tuple[str, ...]:
    patterns: list[str] = []
    for extension in extensions:
        suffix = extension.lstrip(".")
        patterns.append(f"*.{suffix.lower()}")
        upper = suffix.upper()
        if upper != suffix.lower():
            patterns.append(f"*.{upper}")
    return tuple(patterns)


FILE_COUNT_PATTERN_GROUPS = {
    "image": _extension_patterns(
        "png",
        "jpg",
        "jpeg",
        "gif",
        "bmp",
        "tif",
        "tiff",
        "webp",
        "heic",
        "svg",
    ),
    "photo": _extension_patterns("png", "jpg", "jpeg", "heic", "webp"),
    "picture": _extension_patterns("png", "jpg", "jpeg", "gif", "bmp", "webp", "heic"),
    "video": _extension_patterns("mp4", "mov", "avi", "mkv", "webm", "m4v"),
    "audio": _extension_patterns("mp3", "wav", "aac", "flac", "m4a", "ogg"),
    "archive": _extension_patterns("zip", "tar", "gz", "tgz", "bz2", "7z", "rar"),
    "document": _extension_patterns("doc", "docx", "rtf", "txt", "md", "odt", "pdf"),
    "spreadsheet": _extension_patterns("xls", "xlsx", "csv", "tsv", "ods"),
    "presentation": _extension_patterns("ppt", "pptx", "key", "odp"),
}

FILE_COUNT_LABELS = {
    "image": "IMAGE",
    "photo": "PHOTO",
    "picture": "PICTURE",
    "video": "VIDEO",
    "audio": "AUDIO",
    "archive": "ARCHIVE",
    "document": "DOCUMENT",
    "spreadsheet": "SPREADSHEET",
    "presentation": "PRESENTATION",
}

FILE_COUNT_IGNORED_TYPES = {"all", "total", "many", "number", "matching"}
FILE_COUNT_EXTENSION_TYPES = {
    "pdf",
    "png",
    "jpg",
    "jpeg",
    "gif",
    "svg",
    "webp",
    "heic",
    "mp4",
    "mov",
    "mp3",
    "wav",
    "zip",
    "doc",
    "docx",
    "txt",
    "md",
    "xls",
    "xlsx",
    "csv",
    "ppt",
    "pptx",
}


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict[str, object]
    result: dict[str, object]
    command: str = ""
    status: str = "completed"


@dataclass
class AgentResult:
    session_id: str
    response: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)


class ReadOnlyAgent:
    def __init__(
        self,
        roots: Optional[Iterable[str | Path]],
        session_store: SessionStore,
        safety_store: Optional[SafetyStore] = None,
        chat_client: Optional[ChatClient] = None,
        offline_echo: bool = False,
        working_path: str | Path = ".",
        runtime_callback: Optional[Callable[[dict[str, object]], None]] = None,
    ) -> None:
        self.roots = [str(Path(root).expanduser()) for root in roots] if roots is not None else []
        self.session_store = session_store
        self.safety_store = safety_store
        self.working_path = self._normalize_working_path(working_path)
        self.enable_llm_planner = chat_client is not None
        self.chat_client = chat_client or get_chat_client(offline_echo=offline_echo)
        self.runtime_callback = runtime_callback
        self._response_metadata: dict[str, object] = {}
        self._root_graph: Any | None = None

    def handle(self, message: str, session_id: Optional[str] = None) -> AgentResult:
        active_session = self.session_store.create_session(session_id)
        self._response_metadata = {}
        self.session_store.append_message(active_session, "user", message)

        response, tool_calls = self._handle_with_graph(message, active_session)
        response = self._with_tool_trace(response, tool_calls)
        metadata: dict[str, object] = {
            "tool_names": [record.name for record in tool_calls],
            "tool_calls": [self._tool_call_metadata(record) for record in tool_calls],
        }
        metadata.update(self._response_metadata)
        self.session_store.append_message(
            active_session,
            "assistant",
            response,
            metadata=metadata,
        )
        return AgentResult(session_id=active_session, response=response, tool_calls=tool_calls)

    def handle_stream(
        self,
        message: str,
        session_id: Optional[str] = None,
    ) -> Iterator[dict[str, object]]:
        active_session = self.session_store.create_session(session_id)
        self._response_metadata = {}
        self.session_store.append_message(active_session, "user", message)

        response, tool_calls = yield from self._handle_with_graph_stream(message, active_session)
        response = self._with_tool_trace(response, tool_calls)
        metadata: dict[str, object] = {
            "tool_names": [record.name for record in tool_calls],
            "tool_calls": [self._tool_call_metadata(record) for record in tool_calls],
        }
        metadata.update(self._response_metadata)
        self.session_store.append_message(
            active_session,
            "assistant",
            response,
            metadata=metadata,
        )
        yield {
            "type": "__agent_result",
            "result": AgentResult(
                session_id=active_session,
                response=response,
                tool_calls=tool_calls,
            ),
        }

    def _handle_with_graph(self, message: str, session_id: str) -> tuple[str, list[ToolCallRecord]]:
        try:
            from chat_software.agent.root_graph import build_root_graph
        except ImportError:
            return self._handle_core(message, session_id)

        try:
            if self._root_graph is None:
                self._root_graph = build_root_graph(self)
            graph = self._root_graph
        except (ImportError, ModuleNotFoundError):
            return self._handle_core(message, session_id)
        config = {"configurable": {"thread_id": session_id}}
        state = graph.invoke({"message": message, "session_id": session_id}, config=config)
        response = str(state.get("response") or "")
        tool_calls = self._coerce_tool_call_records(state.get("tool_calls") or [])
        return response, tool_calls

    def _handle_with_graph_stream(
        self,
        message: str,
        session_id: str,
    ) -> Iterator[dict[str, object]]:
        try:
            from chat_software.agent.root_graph import build_root_graph
        except ImportError:
            return self._handle_core(message, session_id)

        try:
            if self._root_graph is None:
                self._root_graph = build_root_graph(self)
            graph = self._root_graph
        except (ImportError, ModuleNotFoundError):
            return self._handle_core(message, session_id)

        final_state: dict[str, object] = {}
        config = {"configurable": {"thread_id": session_id}}
        stream = graph.stream(
            {"message": message, "session_id": session_id, "emit_to_stream": True},
            config=config,
            stream_mode=["custom", "updates", "values"],
            subgraphs=True,
        )
        for item in stream:
            mode, chunk = self._normalize_graph_stream_item(item)
            if mode == "custom":
                if isinstance(chunk, dict):
                    yield chunk
                else:
                    yield {"type": "graph_event", "data": chunk}
            elif mode in {"updates", "values"}:
                self._merge_graph_stream_state(final_state, chunk)

        response = str(final_state.get("response") or "")
        tool_calls = self._coerce_tool_call_records(final_state.get("tool_calls") or [])
        return response, tool_calls

    def _coerce_tool_call_records(self, values: object) -> list[ToolCallRecord]:
        records: list[ToolCallRecord] = []
        if not isinstance(values, list):
            return records
        for value in values:
            if isinstance(value, ToolCallRecord):
                records.append(value)
                continue
            if not isinstance(value, dict):
                continue
            arguments = value.get("arguments") if isinstance(value.get("arguments"), dict) else {}
            result = value.get("result") if isinstance(value.get("result"), dict) else {}
            records.append(
                ToolCallRecord(
                    name=str(value.get("name") or ""),
                    arguments=dict(arguments),
                    result=dict(result),
                    command=str(value.get("command") or ""),
                    status=str(value.get("status") or "completed"),
                )
            )
        return records

    def _normalize_graph_stream_item(self, item: object) -> tuple[str, object]:
        if isinstance(item, tuple):
            if len(item) == 2 and isinstance(item[0], str):
                return item[0], item[1]
            if len(item) == 3 and isinstance(item[1], str):
                return item[1], item[2]
        return "", item

    def _merge_graph_stream_state(self, final_state: dict[str, object], chunk: object) -> None:
        if not isinstance(chunk, dict):
            return
        if any(key in chunk for key in {"response", "tool_calls", "route", "route_reason"}):
            final_state.update(chunk)
            return
        for value in chunk.values():
            if isinstance(value, dict):
                final_state.update(value)

    def _handle_core(self, message: str, session_id: str) -> tuple[str, list[ToolCallRecord]]:
        file_response, file_tool_calls = self._run_file_ops_agent(message, session_id)
        if file_tool_calls or file_response != CHAT_FALLBACK_RESPONSE:
            return file_response, file_tool_calls
        return self._run_chat_agent(message), []

    def _run_file_ops_agent(
        self,
        message: str,
        session_id: Optional[str] = None,
        force_llm_planner: bool = False,
    ) -> tuple[str, list[ToolCallRecord]]:
        pending = self._latest_pending_operation(session_id) if session_id else None
        if pending is not None:
            return self._handle_pending_operation_reply(message, pending, session_id)
        return self._run_file_ops_request(message, force_llm_planner=force_llm_planner)

    def _run_file_ops_request(
        self,
        message: str,
        *,
        force_llm_planner: bool = False,
    ) -> tuple[str, list[ToolCallRecord]]:
        lowered = message.lower()

        if not self.roots and self._requires_scoped_tool(lowered):
            pending = self._pending_root_required_operation(message)
            self._set_pending_operation(pending)
            return self._format_root_required_response(pending), []
        elif self._looks_like_direct_write_request(lowered):
            return self._run_direct_write_request(message, lowered)

        intent_result = self._run_tool_intent(message, lowered)
        if intent_result is not None:
            return intent_result

        should_plan = self._should_use_llm_tool_loop(lowered) or (
            force_llm_planner and self.enable_llm_planner and bool(self.roots)
        )
        planned = self._run_llm_tool_loop(message) if should_plan else None
        if planned is not None:
            return planned
        return CHAT_FALLBACK_RESPONSE, []

    def _run_chat_agent(self, message: str) -> str:
        return self.chat_client.complete(
            [
                {
                    "role": "system",
                    "content": (
                        "You are a local-first file assistant. "
                        "For questions about files, folders, repositories, or commands, "
                        "do not guess. Ask the user to approve a root or use the available "
                        "file tools. For ordinary conversation, answer normally."
                    ),
                },
                {"role": "user", "content": message},
            ]
        )

    def _tool_call(
        self,
        name: str,
        arguments: dict[str, object],
        result: dict[str, object],
        status: str = "completed",
    ) -> ToolCallRecord:
        return ToolCallRecord(
            name=name,
            arguments=arguments,
            result=result,
            command=self._format_tool_command(name, arguments, result),
            status=status,
        )

    def _tool_call_metadata(self, record: ToolCallRecord) -> dict[str, object]:
        return {
            "name": record.name,
            "command": record.command,
            "status": record.status,
            "arguments": self._compact_tool_arguments(record.arguments),
            "result": self._compact_tool_result(record.result),
        }

    def _latest_pending_operation(self, session_id: Optional[str]) -> Optional[dict[str, Any]]:
        if not session_id:
            return None
        for message in reversed(self.session_store.get_messages(session_id)):
            metadata = message.metadata
            if not isinstance(metadata, dict):
                continue
            pending = metadata.get(PENDING_OPERATION_KEY)
            if isinstance(pending, dict):
                return pending
            if isinstance(metadata.get(PENDING_OPERATION_RESOLVED_KEY), dict):
                return None
        return None

    def _has_pending_operation(self, session_id: Optional[str]) -> bool:
        return self._latest_pending_operation(session_id) is not None

    def _set_pending_operation(self, pending: dict[str, Any]) -> None:
        self._response_metadata[PENDING_OPERATION_KEY] = pending

    def _mark_pending_operation_resolved(
        self,
        pending: dict[str, Any],
        decision: str,
    ) -> None:
        self._response_metadata[PENDING_OPERATION_RESOLVED_KEY] = {
            "id": str(pending.get("id") or ""),
            "decision": decision,
        }

    def _handle_pending_operation_reply(
        self,
        message: str,
        pending: dict[str, Any],
        session_id: Optional[str],
    ) -> tuple[str, list[ToolCallRecord]]:
        decision, revision = self._pending_operation_decision(message, pending)
        if decision == "approve":
            if pending.get("type") == ROOT_REQUIRED_PENDING_TYPE and not self.roots:
                return self._format_root_still_required_response(pending), []
            self._mark_pending_operation_resolved(pending, "approved")
            return self._execute_pending_operation(pending)
        if decision == "decline":
            self._mark_pending_operation_resolved(pending, "declined")
            return "Cancelled the pending file operation. Nothing was changed.", []
        if decision == "revise":
            self._mark_pending_operation_resolved(pending, "revised")
            return self._redesign_pending_operation(pending, revision or message)

        return (
            "I still have a pending file operation waiting for approval.\n"
            f"- {self._pending_operation_summary(pending)}\n"
            "Reply yes/approve to execute it, no/cancel to discard it, "
            "or no with changes to redesign it.",
            [],
        )

    def _pending_operation_decision(
        self,
        message: str,
        pending: dict[str, Any],
    ) -> tuple[str | None, str | None]:
        exact = self._exact_pending_operation_decision(message)
        if exact is not None:
            return exact

        llm_decision = self._run_pending_decision_llm(message, pending)
        if llm_decision is not None:
            return llm_decision

        return self._rule_based_pending_operation_decision(message)

    def _exact_pending_operation_decision(self, message: str) -> Optional[tuple[str, str | None]]:
        stripped = message.strip()
        lowered = stripped.lower()
        revision = self._extract_revision_text(stripped)
        if revision == "":
            return "decline", None
        if revision:
            return "revise", revision
        if self._is_approval_message(lowered):
            return "approve", None
        return None

    def _run_pending_decision_llm(
        self,
        message: str,
        pending: dict[str, Any],
    ) -> Optional[tuple[str, str | None]]:
        if not self.enable_llm_planner:
            return None
        try:
            decision_text = self.chat_client.complete(
                [
                    {
                        "role": "system",
                        "content": self._pending_decision_system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": self._pending_decision_user_prompt(message, pending),
                    },
                ]
            )
        except Exception:
            return None
        return self._parse_pending_decision(decision_text)

    def _pending_decision_system_prompt(self) -> str:
        return (
            "You classify a user's reply to a pending local file operation. "
            "Return only JSON, no markdown. Schema: "
            "{\"decision\":\"approve|decline|revise|clarify\",\"revision\":\"\",\"reason\":\"\"}. "
            "approve means the user clearly grants permission to execute the pending operation. "
            "decline means the user cancels or rejects it with no replacement. "
            "revise means the user wants a different plan or provides replacement requirements. "
            "clarify means the reply is not a permission decision. "
            "Never choose approve if the reply contains no, not, don't, cancel, reject, or stop. "
            "If a negative reply also includes new requirements, choose revise and put only the "
            "replacement request in revision."
        )

    def _pending_decision_user_prompt(self, message: str, pending: dict[str, Any]) -> str:
        return json.dumps(
            {
                "pending_operation": self._compact_pending_operation(pending),
                "pending_summary": self._pending_operation_summary(pending),
                "user_reply": message,
            },
            indent=2,
            sort_keys=True,
        )

    def _parse_pending_decision(self, text: str) -> Optional[tuple[str, str | None]]:
        parsed = self._parse_json_object(text)
        if not isinstance(parsed, dict):
            return None
        decision = str(parsed.get("decision") or "").strip().lower()
        if decision not in PENDING_DECISIONS:
            return None
        revision = str(parsed.get("revision") or "").strip()
        if decision == "approve":
            return "approve", None
        if decision == "decline":
            return "decline", None
        if decision == "revise" and revision:
            return "revise", revision
        return None, None

    def _compact_pending_operation(self, pending: dict[str, Any]) -> dict[str, object]:
        compact = dict(pending)
        if "arguments" in compact and isinstance(compact["arguments"], dict):
            compact["arguments"] = self._compact_tool_arguments(dict(compact["arguments"]))
        if "commands" in compact and isinstance(compact["commands"], list):
            commands: list[dict[str, object]] = []
            for command in compact["commands"][:8]:
                if not isinstance(command, dict):
                    continue
                commands.append(
                    {
                        "name": command.get("name"),
                        "arguments": self._compact_tool_arguments(
                            dict(command.get("arguments") or {})
                        ),
                    }
                )
            compact["commands"] = commands
        return compact

    def _rule_based_pending_operation_decision(
        self,
        message: str,
    ) -> tuple[str | None, str | None]:
        stripped = message.strip()
        lowered = stripped.lower()
        revision = self._extract_revision_text(stripped)
        if revision is not None:
            if revision:
                return "revise", revision
            return "decline", None
        if self._is_approval_message(lowered):
            return "approve", None
        if self._is_revision_message(lowered):
            return "revise", stripped
        if self._looks_like_direct_write_request(lowered) or self._looks_like_organize_request(
            lowered
        ):
            return "revise", stripped
        return None, None

    def _extract_revision_text(self, message: str) -> Optional[str]:
        lowered = message.strip().lower()
        decline_match = re.match(
            r"^(?:no|nope|nah|cancel|stop|reject|decline)\b(?P<rest>.*)$",
            lowered,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if not decline_match:
            return None
        rest = message[decline_match.start("rest") :].strip(" \t\r\n,;:-")
        if not rest:
            return ""
        rest_lowered = rest.lower().strip()
        if rest_lowered in {"thanks", "thank you", "please", "not now", "later"}:
            return ""
        if rest_lowered.startswith(("don't", "dont", "do not", "nothing", "leave it")):
            return ""
        return self._clean_revision_text(rest)

    def _clean_revision_text(self, text: str) -> str:
        cleaned = text.strip(" \t\r\n,;:-")
        cleaned = re.sub(
            r"^(?:but|instead|rather|please|can you|could you|would you)\b[\s,;:-]*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        ).strip()
        cleaned = re.sub(r"\s+instead\s*$", "", cleaned, flags=re.IGNORECASE).strip()
        return cleaned

    def _is_approval_message(self, lowered: str) -> bool:
        normalized = re.sub(r"[^a-z0-9 ]+", " ", lowered).strip()
        if not normalized:
            return False
        if re.search(r"\b(?:but|instead|rather|change)\b", normalized):
            return False
        words = normalized.split()
        short_approvals = {
            "yes",
            "y",
            "ok",
            "okay",
            "approve",
            "approved",
            "confirm",
            "confirmed",
            "proceed",
            "execute",
            "apply",
            "run",
        }
        if len(words) <= 3 and words[0] in short_approvals:
            return True
        return normalized in {
            "go ahead",
            "do it",
            "run it",
            "apply it",
            "execute it",
            "looks good",
            "that works",
            "yes please",
            "yes go ahead",
            "ok go ahead",
            "okay go ahead",
        }

    def _is_revision_message(self, lowered: str) -> bool:
        return bool(
            re.match(
                r"^(?:change|revise|redesign|instead|rather|use|make it|update the plan)\b",
                lowered.strip(),
                flags=re.IGNORECASE,
            )
        )

    def _redesign_pending_operation(
        self,
        pending: dict[str, Any],
        revision: str,
    ) -> tuple[str, list[ToolCallRecord]]:
        cleaned = self._clean_revision_text(revision)
        if not cleaned:
            return "Cancelled the pending file operation. Nothing was changed.", []

        if self._looks_like_direct_write_request(
            cleaned.lower()
        ) or self._looks_like_organize_request(cleaned.lower()):
            return self._run_file_ops_request(cleaned)

        if pending.get("type") == "organize_plan":
            return self._run_file_ops_request(f"organize files {cleaned}")

        if pending.get("type") == "write_tool":
            rebuilt = self._rewrite_pending_write_arguments(pending, cleaned)
            if rebuilt is not None:
                tool_name, arguments = rebuilt
                return self._preview_direct_write_request(tool_name, arguments)

        return (
            "I cancelled the previous pending operation, but I need a clearer replacement request. "
            "Please describe the new file operation with the exact path(s).",
            [],
        )

    def _rewrite_pending_write_arguments(
        self,
        pending: dict[str, Any],
        revision: str,
    ) -> Optional[tuple[str, dict[str, object]]]:
        tool_name = str(pending.get("tool_name") or "")
        arguments = dict(pending.get("arguments") or {})
        cleaned = self._clean_argument(
            re.sub(
                r"^(?:use|make it|name it|call it|called|named)\s+",
                "",
                revision,
                flags=re.IGNORECASE,
            )
        )
        if not cleaned:
            return None
        if tool_name == "make_dir":
            return tool_name, {"path": cleaned}
        if tool_name == "write_file" and "/" in cleaned:
            arguments["path"] = cleaned
            return tool_name, arguments
        return None

    def _pending_operation_summary(self, pending: dict[str, Any]) -> str:
        if pending.get("type") == "organize_plan":
            return f"apply organize plan {pending.get('plan_id')}"
        if pending.get("type") == "write_tool":
            return self._write_plan_summary(
                str(pending.get("tool_name") or ""),
                dict(pending.get("arguments") or {}),
            )
        if pending.get("type") == "write_batch":
            commands = pending.get("commands") or []
            return f"run {len(commands) if isinstance(commands, list) else 0} file write command(s)"
        if pending.get("type") == ROOT_REQUIRED_PENDING_TYPE:
            return f"inspect an approved file root for: {pending.get('message')}"
        return str(pending.get("summary") or "pending file operation")

    def _execute_pending_operation(
        self,
        pending: dict[str, Any],
    ) -> tuple[str, list[ToolCallRecord]]:
        if pending.get("type") == ROOT_REQUIRED_PENDING_TYPE:
            original_message = str(pending.get("message") or "").strip()
            if not original_message:
                return self._format_root_still_required_response(pending), []
            return self._run_file_ops_request(original_message)

        if pending.get("type") == "write_tool":
            tool_name = str(pending.get("tool_name") or "")
            if tool_name not in WRITE_TOOL_NAMES:
                raise ValueError(f"Unsupported pending write operation: {tool_name}")
            arguments = dict(pending.get("arguments") or {})
            record = self._execute_tool(tool_name, arguments)
            return self._format_write_result(record), [record]

        if pending.get("type") == "write_batch":
            return self._execute_pending_write_batch(pending)

        if pending.get("type") == "organize_plan":
            plan_id = str(pending.get("plan_id") or "")
            if not plan_id:
                raise ValueError("Pending organize operation is missing a plan_id.")
            record = self._execute_tool("organize_apply", {"plan_id": plan_id})
            return self._format_organize_apply_result(record.result), [record]

        raise ValueError("Unsupported pending file operation.")

    def _execute_pending_write_batch(
        self,
        pending: dict[str, Any],
    ) -> tuple[str, list[ToolCallRecord]]:
        records: list[ToolCallRecord] = []
        commands = pending.get("commands") or []
        if not isinstance(commands, list):
            raise ValueError("Pending write batch is missing commands.")

        for command in commands:
            if not isinstance(command, dict):
                continue
            tool_name = str(command.get("name") or "")
            arguments = dict(command.get("arguments") or {})
            if tool_name not in WRITE_TOOL_NAMES:
                records.append(
                    self._tool_call(
                        tool_name or "unknown_tool",
                        arguments,
                        {"error": f"Unsupported pending write operation: {tool_name}"},
                        status="rejected",
                    )
                )
                break
            try:
                records.append(self._execute_tool(tool_name, arguments))
            except Exception as exc:
                records.append(
                    self._tool_call(
                        tool_name,
                        arguments,
                        {"error": str(exc), "error_type": exc.__class__.__name__},
                        status="failed",
                    )
                )
                break

        return self._format_write_batch_result(records), records

    def _compact_tool_arguments(self, arguments: dict[str, object]) -> dict[str, object]:
        compact: dict[str, object] = {}
        for key, value in arguments.items():
            if (
                key in {"content", "old_text", "new_text"}
                and isinstance(value, str)
                and len(value) > 1000
            ):
                compact[key] = value[:1000] + f"... [{len(value) - 1000} more chars]"
            else:
                compact[key] = value
        return compact

    def _run_tool_intent(
        self,
        message: str,
        lowered: str,
    ) -> Optional[tuple[str, list[ToolCallRecord]]]:
        """Deterministic intent router for filesystem questions.

        This is deliberately ahead of any model-planned tool loop. Obvious file
        questions should always inspect the approved path first, then answer from
        the observed data.
        """
        default_path = self._default_tool_path()

        file_count = self._extract_file_count_request(message, default_path)
        if file_count is not None:
            result, records = self._run_file_count_request(file_count)
            return self._format_file_count(result, str(file_count["label"])), records

        if self._looks_like_organize_request(lowered):
            if self.safety_store is None:
                return "The organize planner is not available in this session.", []
            record = self._execute_tool(
                "organize_plan",
                {
                    "path": self._extract_directory_check_path(message, fallback=default_path),
                    "strategy": self._extract_organize_strategy(lowered),
                    "file_filter": self._extract_organize_file_filter(lowered),
                    "max_files": 1000,
                },
            )
            self._set_pending_operation(self._pending_organize_operation(record.result))
            return self._format_organize_plan(record.result), [record]

        if "git diff" in lowered or ("diff" in lowered and "repo" in lowered):
            record = self._execute_tool("git_diff", {"repo_path": default_path})
            return self._format_command_result(record.result, "Git diff"), [record]

        if "git status" in lowered or ("status" in lowered and "repo" in lowered):
            record = self._execute_tool("git_status", {"repo_path": default_path})
            return self._format_command_result(record.result, "Git status"), [record]

        if self._looks_like_directory_check_request(lowered):
            path = self._extract_directory_check_path(message, fallback=default_path)
            record = self._execute_tool("list_dir", {"path": path})
            return self._format_directory_listing(record.result), [record]

        if self._looks_like_read_request(lowered):
            path = self._extract_path_argument(message, fallback=default_path)
            record = self._execute_tool("read_file", {"path": path})
            return self._format_file_content(record.result), [record]

        if lowered.startswith("stat") or lowered.startswith("info for"):
            path = self._extract_path_argument(message, fallback=default_path)
            record = self._execute_tool("stat_path", {"path": path})
            return self._format_stat(record.result), [record]

        if "glob" in lowered or "match files" in lowered:
            record = self._execute_tool(
                "glob_files",
                {
                    "pattern": self._extract_pattern(message),
                    "root": default_path,
                },
            )
            return self._format_glob(record.result), [record]

        if self._looks_like_list_request(lowered):
            path = self._extract_list_path_argument(message, fallback=default_path)
            record = self._execute_tool("list_dir", {"path": path})
            return self._format_directory_listing(record.result), [record]

        if self._looks_like_project_overview_request(lowered):
            return self._run_project_overview(default_path)

        if self._looks_like_file_search_request(lowered):
            query = self._extract_query(message)
            record = self._execute_tool(
                "grep_files",
                {
                    "query": query,
                    "root": default_path,
                },
            )
            return summarize_matches(record.result["matches"], query=query), [record]

        return None

    def _run_file_count_request(
        self,
        file_count: dict[str, object],
    ) -> tuple[dict[str, object], list[ToolCallRecord]]:
        patterns = [str(pattern) for pattern in file_count.get("patterns", [])]
        if not patterns:
            patterns = [str(file_count.get("pattern") or "*")]
        root = str(file_count.get("root") or self._default_tool_path())

        records: list[ToolCallRecord] = []
        matches: list[dict[str, object]] = []
        seen: set[str] = set()
        truncated = False
        resolved_root = root

        for pattern in patterns:
            record = self._execute_tool(
                "glob_files",
                {
                    "pattern": pattern,
                    "root": root,
                    "max_results": 5000,
                },
            )
            records.append(record)
            result = record.result
            resolved_root = str(result.get("root") or resolved_root)
            truncated = truncated or bool(result.get("truncated"))
            for entry in result.get("matches", []):
                if not isinstance(entry, dict):
                    continue
                key = str(entry.get("path") or entry.get("relative_path") or entry.get("name"))
                if key in seen:
                    continue
                seen.add(key)
                matches.append(entry)

        return (
            {
                "root": resolved_root,
                "pattern": patterns[0] if len(patterns) == 1 else ", ".join(patterns),
                "patterns": patterns,
                "matches": matches,
                "truncated": truncated,
            },
            records,
        )

    def _run_project_overview(self, path: str) -> tuple[str, list[ToolCallRecord]]:
        listing = self._execute_tool("list_dir", {"path": path})
        tool_calls = [listing]
        overview_files: list[dict[str, object]] = []
        for candidate in self._overview_candidate_paths(listing.result):
            read_path = self._join_tool_path(path, candidate)
            result = self._execute_tool(
                "read_file",
                {
                    "path": read_path,
                    "max_bytes": 12_000,
                },
            )
            tool_calls.append(result)
            overview_files.append(result.result)
        return self._format_project_overview(listing.result, overview_files), tool_calls

    def _normalize_working_path(self, working_path: str | Path) -> str:
        value = str(working_path or ".").strip()
        return self._clean_argument(value) or "."

    def _default_tool_path(self) -> str:
        return self.working_path or "."

    def _join_tool_path(self, base: str, child: str) -> str:
        if not base or base == ".":
            return child
        return str(PurePosixPath(base) / child)

    def _should_use_llm_tool_loop(self, lowered: str) -> bool:
        return (
            self.enable_llm_planner
            and bool(self.roots)
            and self._requires_scoped_tool(lowered)
            and not self._looks_like_directory_check_request(lowered)
            and not self._looks_like_direct_write_request(lowered)
            and not self._looks_like_simple_command_request(lowered)
        )

    def _run_llm_tool_loop(self, message: str) -> Optional[tuple[str, list[ToolCallRecord]]]:
        try:
            plan_text = self.chat_client.complete(
                [
                    {
                        "role": "system",
                        "content": self._planner_system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": self._planner_user_prompt(message),
                    },
                ]
            )
            commands = self._parse_tool_plan(plan_text)
        except Exception:
            return None

        if not commands:
            return None

        tool_calls = [self._execute_planned_command(command) for command in commands]
        if not tool_calls:
            return None

        response = self._final_answer_from_tools(message, tool_calls)
        return response, tool_calls

    def _planner_system_prompt(self) -> str:
        return (
            "You are the command planner for a local-first file assistant. "
            "Return only JSON, with no markdown. The schema is: "
            "{\"commands\":[{\"name\":\"tool_name\",\"arguments\":{...}}]}. "
            "Use at most 6 commands. Use only these read-only or preview-only tools: "
            "list_dir(path), stat_path(path), read_file(path,max_bytes), "
            "glob_files(pattern,root,max_results), grep_files(query,root,include,max_results), "
            "git_status(repo_path), git_diff(repo_path,staged,stat_only), "
            "organize_plan(path,strategy,file_filter,max_files). "
            "For organize_plan, strategies are by_type, by_date, by_filename, by_topic; "
            "file_filter is all, image, video, audio, or document. "
            "Return an empty commands array if no filesystem/repo inspection is needed. "
            "Use relative paths inside the approved root. Do not plan apply, undo, delete, write, "
            "shell, network, install, or unapproved filesystem commands. "
            "When the user says this/current/selected folder or path, use the selected_path value. "
            "For broad project questions, first list_dir(selected_path), then read README.md, package.json, "
            "pyproject.toml, or another obvious manifest if present or likely."
        )

    def _planner_user_prompt(self, message: str) -> str:
        roots = "\n".join(f"- {root}" for root in self.roots)
        return f"Approved roots:\n{roots}\n\nSelected path: {self._default_tool_path()}\n\nUser query:\n{message}"

    def _parse_tool_plan(self, text: str) -> list[dict[str, Any]]:
        parsed = self._parse_json_object(text)
        if isinstance(parsed, list):
            commands = parsed
        elif isinstance(parsed, dict):
            commands = parsed.get("commands", [])
        else:
            return []
        if not isinstance(commands, list):
            return []
        return [command for command in commands[:6] if isinstance(command, dict)]

    def _parse_json_object(self, text: str) -> Any:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

        start_candidates = [index for index in [cleaned.find("{"), cleaned.find("[")] if index >= 0]
        if not start_candidates:
            return None
        start = min(start_candidates)
        end = max(cleaned.rfind("}"), cleaned.rfind("]"))
        if end <= start:
            return None
        try:
            return json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError:
            return None

    def _execute_planned_command(self, command: dict[str, Any]) -> ToolCallRecord:
        raw_name = str(command.get("name") or command.get("tool") or "").strip()
        name = self._normalize_tool_name(raw_name)
        arguments = command.get("arguments") or command.get("args") or {}
        if not isinstance(arguments, dict):
            arguments = {}
        if name not in LLM_PLANNED_TOOL_ALLOWLIST:
            result = {
                "error": f"Planner requested disallowed tool: {name or raw_name}",
                "allowed_tools": sorted(LLM_PLANNED_TOOL_ALLOWLIST),
            }
            return self._tool_call(name or raw_name or "unknown_tool", dict(arguments), result, status="rejected")
        try:
            return self._execute_tool(name, arguments)
        except Exception as exc:
            result = {"error": str(exc), "error_type": exc.__class__.__name__}
            return self._tool_call(name or raw_name or "unknown_tool", dict(arguments), result, status="failed")

    def _normalize_tool_name(self, name: str) -> str:
        aliases = {
            "list": "list_dir",
            "ls": "list_dir",
            "files.list": "list_dir",
            "stat": "stat_path",
            "read": "read_file",
            "cat": "read_file",
            "glob": "glob_files",
            "grep": "grep_files",
            "search": "grep_files",
            "git.status": "git_status",
            "git.diff": "git_diff",
            "organize": "organize_plan",
            "mkdir": "make_dir",
            "create_directory": "make_dir",
            "write": "write_file",
            "create_file": "write_file",
            "copy": "copy_path",
            "move": "move_path",
            "rename": "move_path",
            "edit": "edit_file",
        }
        return aliases.get(name, name)

    def _execute_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallRecord:
        tool_id = str(uuid.uuid4())
        started_at = perf_counter()
        self._emit_runtime(
            {
                "type": "tool_started",
                "tool_id": tool_id,
                "name": name,
                "arguments": self._compact_tool_arguments(dict(arguments)),
            }
        )
        try:
            record = self._execute_tool_impl(name, arguments)
        except Exception as exc:
            self._emit_runtime(
                {
                    "type": "tool_failed",
                    "tool_id": tool_id,
                    "name": name,
                    "arguments": self._compact_tool_arguments(dict(arguments)),
                    "error": str(exc),
                    "error_type": exc.__class__.__name__,
                    "elapsed_ms": round((perf_counter() - started_at) * 1000, 2),
                }
            )
            raise
        self._emit_runtime(
            {
                "type": "tool_completed",
                "tool_id": tool_id,
                "name": record.name,
                "command": record.command,
                "status": record.status,
                "arguments": self._compact_tool_arguments(record.arguments),
                "result": self._compact_tool_result(record.result),
                "elapsed_ms": round((perf_counter() - started_at) * 1000, 2),
            }
        )
        return record

    def _emit_runtime(self, event: dict[str, object]) -> None:
        if self.runtime_callback is not None:
            self.runtime_callback(event)

    def _execute_tool_impl(self, name: str, arguments: dict[str, Any]) -> ToolCallRecord:
        if name == "make_dir":
            args = {"path": str(arguments.get("path") or self._default_tool_path())}
            result = make_dir(path=args["path"], resolver=self._write_resolver())
            return self._tool_call(name, args, result)
        if name == "write_file":
            args = {
                "path": str(arguments.get("path") or ""),
                "content": str(arguments.get("content") or ""),
                "overwrite": bool(arguments.get("overwrite", False)),
            }
            if not args["path"]:
                raise ValueError("write_file requires a path.")
            result = write_file(
                path=args["path"],
                content=args["content"],
                resolver=self._write_resolver(),
                overwrite=bool(args["overwrite"]),
            )
            return self._tool_call(name, args, result)
        if name == "copy_path":
            args = {
                "source": str(arguments.get("source") or arguments.get("from") or ""),
                "destination": str(arguments.get("destination") or arguments.get("to") or ""),
            }
            if not args["source"] or not args["destination"]:
                raise ValueError("copy_path requires source and destination.")
            result = copy_path(
                source=args["source"],
                destination=args["destination"],
                resolver=self._write_resolver(),
            )
            return self._tool_call(name, args, result)
        if name == "move_path":
            args = {
                "source": str(arguments.get("source") or arguments.get("from") or ""),
                "destination": str(arguments.get("destination") or arguments.get("to") or ""),
            }
            if not args["source"] or not args["destination"]:
                raise ValueError("move_path requires source and destination.")
            result = move_path(
                source=args["source"],
                destination=args["destination"],
                resolver=self._write_resolver(),
            )
            return self._tool_call(name, args, result)
        if name == "edit_file":
            args = {
                "path": str(arguments.get("path") or ""),
                "old_text": str(arguments.get("old_text") or arguments.get("old") or ""),
                "new_text": str(arguments.get("new_text") or arguments.get("new") or ""),
                "count": self._bounded_int(arguments.get("count"), 1, 0, 1000),
            }
            if not args["path"] or not args["old_text"]:
                raise ValueError("edit_file requires path and old_text.")
            result = edit_file(
                path=args["path"],
                old_text=args["old_text"],
                new_text=args["new_text"],
                resolver=self._write_resolver(),
                count=int(args["count"]),
            )
            return self._tool_call(name, args, result)
        if name == "list_dir":
            args = {
                "path": str(arguments.get("path") or self._default_tool_path()),
                "max_entries": self._bounded_int(arguments.get("max_entries"), 200, 1, 1000),
            }
            result = list_dir(path=args["path"], roots=self.roots, max_entries=int(args["max_entries"]))
            return self._tool_call(name, args, result)
        if name == "stat_path":
            args = {"path": str(arguments.get("path") or self._default_tool_path())}
            return self._tool_call(name, args, stat_path(path=args["path"], roots=self.roots))
        if name == "read_file":
            args = {
                "path": str(arguments.get("path") or self._default_tool_path()),
                "max_bytes": self._bounded_int(arguments.get("max_bytes"), 200_000, 1, 1_000_000),
            }
            result = read_file(path=args["path"], roots=self.roots, max_bytes=int(args["max_bytes"]))
            return self._tool_call(name, args, result)
        if name == "glob_files":
            args = {
                "pattern": str(arguments.get("pattern") or "*"),
                "root": str(arguments.get("root") or self._default_tool_path()),
                "max_results": self._bounded_int(arguments.get("max_results"), 200, 1, 5000),
            }
            result = glob_files(
                pattern=args["pattern"],
                root=args["root"],
                roots=self.roots,
                max_results=int(args["max_results"]),
            )
            return self._tool_call(name, args, result)
        if name == "grep_files":
            query = str(arguments.get("query") or "").strip()
            if not query:
                raise ValueError("grep_files requires a non-empty query.")
            args = {
                "query": query,
                "root": str(arguments.get("root") or self._default_tool_path()),
                "include": str(arguments.get("include") or "*"),
                "max_results": self._bounded_int(arguments.get("max_results"), 50, 1, 500),
            }
            result = grep_files(
                query=args["query"],
                root=args["root"],
                include=args["include"],
                roots=self.roots,
                max_results=int(args["max_results"]),
            )
            return self._tool_call(name, args, result)
        if name == "git_status":
            args = {"repo_path": str(arguments.get("repo_path") or self._default_tool_path())}
            return self._tool_call(name, args, git_status(repo_path=args["repo_path"], roots=self.roots))
        if name == "git_diff":
            args = {
                "repo_path": str(arguments.get("repo_path") or self._default_tool_path()),
                "staged": bool(arguments.get("staged", False)),
                "stat_only": bool(arguments.get("stat_only", True)),
            }
            result = git_diff(
                repo_path=args["repo_path"],
                roots=self.roots,
                staged=bool(args["staged"]),
                stat_only=bool(args["stat_only"]),
            )
            return self._tool_call(name, args, result)
        if name == "organize_plan":
            if self.safety_store is None:
                raise RuntimeError("The organize planner is not available in this session.")
            root = self.safety_store.authorize_roots([self.roots[0]])[0]
            strategy = str(arguments.get("strategy") or "by_type")
            file_filter = str(arguments.get("file_filter") or "all")
            args = {
                "root": str(root.path),
                "path": str(arguments.get("path") or self._default_tool_path()),
                "strategy": strategy,
                "file_filter": file_filter,
                "max_files": self._bounded_int(arguments.get("max_files"), 1000, 1, 5000),
            }
            plan = build_organize_plan(
                root=root.path,
                strategy=strategy,
                file_filter=file_filter,
                path=args["path"],
                prompt="LLM planned organize preview",
                max_files=int(args["max_files"]),
            )
            stored_strategy = str(plan.get("strategy") or strategy)
            plan_id = self.safety_store.create_organize_plan(
                root_id=root.id,
                strategy=stored_strategy,
                plan=plan,
                prompt="LLM planned organize preview",
            )
            return self._tool_call(name, args, self.safety_store.get_organize_plan(plan_id))
        if name == "organize_apply":
            if self.safety_store is None:
                raise RuntimeError("The organize planner is not available in this session.")
            args = {"plan_id": str(arguments.get("plan_id") or "")}
            if not args["plan_id"]:
                raise ValueError("organize_apply requires a plan_id.")
            return self._tool_call(
                name,
                args,
                apply_organize_plan(self.safety_store, args["plan_id"]),
            )
        raise ValueError(f"Unsupported planned tool: {name}")

    def _bounded_int(self, value: object, default: int, minimum: int, maximum: int) -> int:
        try:
            selected = int(value)
        except (TypeError, ValueError):
            selected = default
        return min(max(selected, minimum), maximum)

    def _final_answer_from_tools(self, message: str, tool_calls: list[ToolCallRecord]) -> str:
        fallback = self._fallback_answer_from_tools(tool_calls)
        if any(record.status != "completed" for record in tool_calls):
            return fallback
        try:
            final = self.chat_client.complete(
                [
                    {
                        "role": "system",
                        "content": (
                            "You answer after tools have run. Use only the tool results below. "
                            "If the results do not answer the user, say what was inspected and "
                            "what is still missing. Do not invent files, commands, or outcomes."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "user_query": message,
                                "tool_results": [
                                    {
                                        "name": record.name,
                                        "command": record.command,
                                        "status": record.status,
                                        "result": self._compact_tool_result(record.result),
                                    }
                                    for record in tool_calls
                                ],
                            },
                            indent=2,
                            sort_keys=True,
                        ),
                    },
                ]
            ).strip()
        except Exception:
            return fallback
        if not final or final.startswith("Offline echo mode"):
            return fallback
        return final

    def _compact_tool_result(self, result: dict[str, object]) -> dict[str, object]:
        compact: dict[str, object] = {}
        for key, value in result.items():
            if key == "content" and isinstance(value, str):
                compact[key] = value[:5000]
            elif key in {"stdout", "stderr"} and isinstance(value, str):
                compact[key] = value[:5000]
            elif key in {"entries", "matches", "actions", "applied", "undone"} and isinstance(value, list):
                compact[key] = value[:30]
                if len(value) > 30:
                    compact[f"{key}_truncated_count"] = len(value) - 30
            else:
                compact[key] = value
        return compact

    def _fallback_answer_from_tools(self, tool_calls: list[ToolCallRecord]) -> str:
        if not tool_calls:
            return "No tools were run."
        if any(record.status != "completed" for record in tool_calls):
            lines = ["Some planned tools did not complete:"]
            for record in tool_calls:
                if record.status != "completed":
                    lines.append(f"- {record.status}: {record.command}: {record.result.get('error')}")
            return "\n".join(lines)
        if len(tool_calls) >= 2 and tool_calls[0].name == "list_dir":
            read_results = [record.result for record in tool_calls[1:] if record.name == "read_file"]
            return self._format_project_overview(tool_calls[0].result, read_results)
        last = tool_calls[-1]
        if last.name == "list_dir":
            names = [entry["name"] for entry in last.result.get("entries", [])[:30]]
            return "Files and folders:\n" + "\n".join(f"- {name}" for name in names)
        if last.name == "grep_files":
            return summarize_matches(list(last.result.get("matches", [])), query=str(last.arguments.get("query", "")))
        if last.name == "glob_files":
            return self._format_glob(last.result)
        if last.name == "read_file":
            return self._format_file_content(last.result)
        if last.name == "stat_path":
            return self._format_stat(last.result)
        if last.name in {"git_status", "git_diff"}:
            return self._format_command_result(last.result, last.name)
        if last.name == "organize_plan":
            return self._format_organize_plan(last.result)
        if last.name == "organize_apply":
            return self._format_organize_apply_result(last.result)
        if last.name in {"make_dir", "write_file", "copy_path", "move_path", "edit_file"}:
            return self._format_write_result(last)
        return "Tools completed. Review the command console for details."

    def _write_resolver(self) -> ScopedPathResolver:
        return ScopedPathResolver.from_roots(self.roots)

    def _run_direct_write_request(
        self,
        message: str,
        lowered: str,
    ) -> tuple[str, list[ToolCallRecord]]:
        planned_preview = self._run_llm_write_preview_request(message)
        if planned_preview is not None:
            return planned_preview

        parsed = self._parse_direct_write_request(message, lowered)
        if parsed is None:
            return (
                "I can create folders/files, copy paths, move paths, "
                "and edit text inside approved roots. "
                "Please include the exact source and destination paths.",
                [],
            )
        name, arguments = parsed
        return self._preview_direct_write_request(name, arguments)

    def _run_llm_write_preview_request(
        self,
        message: str,
    ) -> Optional[tuple[str, list[ToolCallRecord]]]:
        if not self.enable_llm_planner:
            return None
        try:
            plan_text = self.chat_client.complete(
                [
                    {
                        "role": "system",
                        "content": self._write_planner_system_prompt(),
                    },
                    {
                        "role": "user",
                        "content": self._write_planner_user_prompt(message),
                    },
                ]
            )
        except Exception:
            return None

        commands = self._parse_write_preview_plan(plan_text)
        if not commands:
            return None
        return self._preview_write_commands(commands, source="llm")

    def _write_planner_system_prompt(self) -> str:
        return (
            "You are the command planner for local file write previews. "
            "Return only JSON, no markdown. Schema: "
            "{\"commands\":[{\"name\":\"tool_name\",\"arguments\":{...}}]}. "
            "Use at most 6 commands. Allowed tool_name values are: "
            "make_dir(path), write_file(path,content,overwrite), "
            "copy_path(source,destination), move_path(source,destination), "
            "edit_file(path,old_text,new_text,count). "
            "Do not plan delete, shell, network, install, apply, undo, or unapproved paths. "
            "Use relative paths inside the approved root unless the user supplied an approved "
            "absolute path. Return an empty commands array if the request is not a file write "
            "operation or lacks exact paths/text. These commands will be previewed for human "
            "approval before execution."
        )

    def _write_planner_user_prompt(self, message: str) -> str:
        return json.dumps(
            {
                "approved_roots": self.roots,
                "selected_path": self._default_tool_path(),
                "user_query": message,
            },
            indent=2,
            sort_keys=True,
        )

    def _parse_write_preview_plan(self, text: str) -> list[dict[str, object]]:
        parsed = self._parse_json_object(text)
        commands = parsed if isinstance(parsed, list) else None
        if isinstance(parsed, dict):
            commands = parsed.get("commands")
        if not isinstance(commands, list):
            return []

        normalized: list[dict[str, object]] = []
        for command in commands[:6]:
            if not isinstance(command, dict):
                continue
            prepared = self._normalize_write_preview_command(command)
            if prepared is not None:
                normalized.append(prepared)
        return normalized

    def _normalize_write_preview_command(
        self,
        command: dict[str, Any],
    ) -> Optional[dict[str, object]]:
        raw_name = str(command.get("name") or command.get("tool") or "").strip()
        name = self._normalize_tool_name(raw_name)
        if name not in WRITE_TOOL_NAMES:
            return None
        arguments = command.get("arguments") or command.get("args") or {}
        if not isinstance(arguments, dict):
            return None
        coerced = self._coerce_write_preview_arguments(name, arguments)
        if coerced is None:
            return None
        return {"name": name, "arguments": coerced}

    def _coerce_write_preview_arguments(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> Optional[dict[str, object]]:
        if name == "make_dir":
            path = str(arguments.get("path") or "").strip()
            return {"path": path} if path else None
        if name == "write_file":
            path = str(arguments.get("path") or "").strip()
            if not path:
                return None
            return {
                "path": path,
                "content": str(arguments.get("content") or ""),
                "overwrite": bool(arguments.get("overwrite", False)),
            }
        if name in {"copy_path", "move_path"}:
            source = str(arguments.get("source") or arguments.get("from") or "").strip()
            destination = str(
                arguments.get("destination") or arguments.get("to") or ""
            ).strip()
            if not source or not destination:
                return None
            return {"source": source, "destination": destination}
        if name == "edit_file":
            path = str(arguments.get("path") or "").strip()
            old_text = str(arguments.get("old_text") or arguments.get("old") or "")
            if not path or not old_text:
                return None
            return {
                "path": path,
                "old_text": old_text,
                "new_text": str(arguments.get("new_text") or arguments.get("new") or ""),
                "count": self._bounded_int(arguments.get("count"), 1, 0, 1000),
            }
        return None

    def _preview_direct_write_request(
        self,
        name: str,
        arguments: dict[str, object],
    ) -> tuple[str, list[ToolCallRecord]]:
        return self._preview_write_commands(
            [{"name": name, "arguments": dict(arguments)}],
            source="deterministic",
        )

    def _preview_write_commands(
        self,
        commands: list[dict[str, object]],
        *,
        source: str,
    ) -> tuple[str, list[ToolCallRecord]]:
        if not commands:
            raise ValueError("A write preview requires at least one command.")
        if len(commands) == 1:
            only = commands[0]
            pending = self._pending_write_operation(
                str(only["name"]),
                dict(only.get("arguments") or {}),
            )
        else:
            pending = self._pending_write_batch_operation(commands)
        self._set_pending_operation(pending)
        record = self._tool_call(
            "prepare_write_plan",
            {
                "source": source,
                "commands": self._compact_write_preview_commands(commands),
            },
            {
                "pending_operation_id": pending["id"],
                "operation": pending["type"],
                "summary": pending["summary"],
                "commands": self._compact_write_preview_commands(commands),
            },
            status="pending_approval",
        )
        return self._format_pending_write_plan(pending), [record]

    def _compact_write_preview_commands(
        self,
        commands: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        compact: list[dict[str, object]] = []
        for command in commands:
            compact.append(
                {
                    "name": command.get("name"),
                    "arguments": self._compact_tool_arguments(
                        dict(command.get("arguments") or {})
                    ),
                }
            )
        return compact

    def _pending_write_operation(
        self,
        name: str,
        arguments: dict[str, object],
    ) -> dict[str, Any]:
        if name not in WRITE_TOOL_NAMES:
            raise ValueError(f"Unsupported write operation: {name}")
        return {
            "id": str(uuid.uuid4()),
            "type": "write_tool",
            "tool_name": name,
            "arguments": dict(arguments),
            "summary": self._write_plan_summary(name, arguments),
        }

    def _pending_write_batch_operation(
        self,
        commands: list[dict[str, object]],
    ) -> dict[str, Any]:
        normalized = [
            {
                "name": str(command.get("name") or ""),
                "arguments": dict(command.get("arguments") or {}),
            }
            for command in commands
        ]
        return {
            "id": str(uuid.uuid4()),
            "type": "write_batch",
            "commands": normalized,
            "summary": f"run {len(normalized)} file write command(s)",
        }

    def _pending_root_required_operation(self, message: str) -> dict[str, Any]:
        return {
            "id": str(uuid.uuid4()),
            "type": ROOT_REQUIRED_PENDING_TYPE,
            "message": message,
            "summary": f"inspect an approved file root for: {message}",
        }

    def _pending_organize_operation(self, plan: dict[str, object]) -> dict[str, Any]:
        return {
            "id": str(uuid.uuid4()),
            "type": "organize_plan",
            "plan_id": str(plan.get("plan_id") or ""),
            "strategy": str(plan.get("strategy") or ""),
            "summary": self._organize_plan_summary(plan),
        }

    def _parse_direct_write_request(
        self,
        message: str,
        lowered: str,
    ) -> tuple[str, dict[str, object]] | None:
        cleaned = self._strip_request_prefix(message)
        folder_path = self._extract_make_dir_path(cleaned)
        if folder_path is not None:
            return "make_dir", {"path": folder_path}

        write_args = self._extract_write_file_args(cleaned)
        if write_args is not None:
            return "write_file", write_args

        move_match = re.search(
            r"^(?:move|rename)\s+(.+?)\s+(?:to|into|as)\s+(.+)$",
            cleaned,
            flags=re.IGNORECASE,
        )
        if move_match:
            return "move_path", {
                "source": self._clean_file_reference(move_match.group(1)),
                "destination": self._clean_argument(move_match.group(2)),
            }

        copy_match = re.search(
            r"^copy\s+(.+?)\s+(?:to|into|as)\s+(.+)$",
            cleaned,
            flags=re.IGNORECASE,
        )
        if copy_match:
            return "copy_path", {
                "source": self._clean_file_reference(copy_match.group(1)),
                "destination": self._clean_argument(copy_match.group(2)),
            }

        replace_match = re.search(
            r"^replace\s+(.+?)\s+with\s+(.+?)\s+in\s+(?:file\s+)?(.+)$",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if replace_match:
            return "edit_file", {
                "old_text": self._clean_argument(replace_match.group(1), preserve_newlines=True),
                "new_text": self._clean_argument(replace_match.group(2), preserve_newlines=True),
                "path": self._clean_argument(replace_match.group(3)),
            }

        edit_match = re.search(
            r"^edit\s+(?:file\s+)?(.+?)\s+(?:replace|replacing)\s+(.+?)\s+with\s+(.+)$",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if edit_match:
            return "edit_file", {
                "path": self._clean_argument(edit_match.group(1)),
                "old_text": self._clean_argument(edit_match.group(2), preserve_newlines=True),
                "new_text": self._clean_argument(edit_match.group(3), preserve_newlines=True),
            }

        return None

    def _strip_request_prefix(self, value: str) -> str:
        return re.sub(
            r"^(?:(?:please|kindly)\s+|(?:could|can|would|will)\s+you\s+)",
            "",
            value.strip(),
            flags=re.IGNORECASE,
        ).strip()

    def _clean_file_reference(self, value: str) -> str:
        cleaned = re.sub(
            r"^(?:the\s+)?(?:file|folder|directory)\s+",
            "",
            value,
            flags=re.IGNORECASE,
        )
        return self._clean_argument(cleaned)

    def _extract_make_dir_path(self, message: str) -> str | None:
        cleaned = message.strip()
        patterns = [
            r"^(?:please\s+)?mkdir\s+(.+)$",
            r"^(?:please\s+)?(?:create|make|add)\s+(?:a\s+|new\s+)?(?:folder|directory|dir)\s+(?:called\s+|named\s+)?(.+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if match:
                return self._clean_argument(match.group(1))
        return None

    def _extract_write_file_args(self, message: str) -> dict[str, object] | None:
        cleaned = message.strip()
        create_match = re.search(
            r"^(?:please\s+)?(?:create|make|add)\s+(?:a\s+|new\s+)?file\s+(.+?)(?:\s+with\s+(?:content|text)\s+(.+))?$",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if create_match:
            return {
                "path": self._clean_argument(create_match.group(1)),
                "content": self._clean_argument(create_match.group(2) or "", preserve_newlines=True),
            }

        write_to_match = re.search(
            r"^(?:please\s+)?write\s+(.+?)\s+to\s+(?:file\s+)?(.+)$",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if write_to_match:
            return {
                "path": self._clean_argument(write_to_match.group(2)),
                "content": self._clean_argument(write_to_match.group(1), preserve_newlines=True),
            }

        write_file_match = re.search(
            r"^(?:please\s+)?write\s+(?:file\s+)?(.+?)(?:\s+with\s+(?:content|text)\s+(.+)|\s*:\s*(.*))?$",
            cleaned,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if write_file_match and "write" in cleaned.lower():
            return {
                "path": self._clean_argument(write_file_match.group(1)),
                "content": self._clean_argument(
                    write_file_match.group(2) if write_file_match.group(2) is not None else write_file_match.group(3) or "",
                    preserve_newlines=True,
                ),
            }

        return None

    def _clean_argument(self, value: str, *, preserve_newlines: bool = False) -> str:
        cleaned = value.strip().rstrip("?").strip()
        if cleaned.lower().startswith("called "):
            cleaned = cleaned[7:].strip()
        if cleaned.lower().startswith("named "):
            cleaned = cleaned[6:].strip()
        if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
            cleaned = cleaned[1:-1]
        if preserve_newlines:
            cleaned = cleaned.replace("\\n", "\n")
        return cleaned

    def _extract_query(self, message: str) -> str:
        cleaned = message.strip()
        patterns = [
            r"about\s+(.+)$",
            r"for\s+(.+)$",
            r"find\s+(.+)$",
            r"search\s+(.+)$",
            r"summarize\s+(.+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if match:
                query = match.group(1).strip(" .?\"'")
                query = re.sub(r"\s+and\s+summari[sz]e.*$", "", query, flags=re.IGNORECASE)
                query = re.sub(r"^files?\s+about\s+", "", query, flags=re.IGNORECASE)
                query = re.sub(r"^files?\s+", "", query, flags=re.IGNORECASE)
                if query:
                    return query
        return cleaned.strip(" .?\"'") or "project"

    def _looks_like_read_request(self, lowered: str) -> bool:
        return (
            lowered.startswith("read ")
            or (lowered.startswith("show ") and not self._looks_like_list_request(lowered))
            or (lowered.startswith("open ") and not self._looks_like_directory_check_request(lowered))
            or lowered.startswith("cat ")
            or "read file" in lowered
            or bool(re.search(r"\bshow\s+file\b", lowered, flags=re.IGNORECASE))
        )

    def _looks_like_list_request(self, lowered: str) -> bool:
        return (
            lowered.startswith("list")
            or "list files" in lowered
            or "show files" in lowered
            or "show the files" in lowered
            or "show folders" in lowered
            or "show the folders" in lowered
            or "what files" in lowered
            or "which files" in lowered
            or "files are in" in lowered
            or "files in selected" in lowered
            or "files from selected" in lowered
            or "what is in this folder" in lowered
            or "what is in this directory" in lowered
            or "what is in selected" in lowered
            or "what is inside" in lowered
        )

    def _looks_like_project_overview_request(self, lowered: str) -> bool:
        if not any(word in lowered for word in ["project", "repo", "repository", "folder", "directory", "path"]):
            return False
        return any(
            phrase in lowered
            for phrase in [
                "what is this",
                "what does this",
                "check this project",
                "check this repo",
                "check this repository",
                "inspect this project",
                "inspect this repo",
                "inspect this repository",
                "overview",
                "summarize this",
                "summarise this",
                "summarize selected",
                "summarise selected",
                "summary of this",
                "data of the files",
                "data in the files",
                "get the data",
                "explain this",
            ]
        )

    def _looks_like_direct_write_request(self, lowered: str) -> bool:
        cleaned = self._strip_request_prefix(lowered)
        return (
            cleaned.startswith(("mkdir ", "copy ", "move ", "rename ", "edit ", "replace "))
            or bool(
                re.search(
                    r"\b(?:create|make|add)\s+(?:a\s+|new\s+)?(?:file|folder|directory|dir)\b",
                    cleaned,
                    flags=re.IGNORECASE,
                )
            )
            or cleaned.startswith("write file ")
            or bool(re.search(r"^write\s+.+\s+to\s+(?:file\s+)?", cleaned, flags=re.IGNORECASE))
        )

    def _looks_like_directory_check_request(self, lowered: str) -> bool:
        cleaned = lowered.strip()
        return bool(
            re.search(
                r"^(?:please\s+)?(?:check|inspect|explore|scan|open|look\s+at)\s+"
                r"(?:the\s+)?(?:(?:this|current|approved|selected)\s+)?"
                r"(?:folder|directory|dir|root|path)\b",
                cleaned,
                flags=re.IGNORECASE,
            )
            or re.search(
                r"^(?:please\s+)?(?:check|inspect|explore|scan|open|look\s+at)\s+.+\s+"
                r"(?:folder|directory|dir)\s*$",
                cleaned,
                flags=re.IGNORECASE,
            )
        )

    def _looks_like_simple_command_request(self, lowered: str) -> bool:
        return (
            self._looks_like_file_count_request(lowered)
            or self._looks_like_list_request(lowered)
            or self._looks_like_read_request(lowered)
            or lowered.startswith("stat")
            or lowered.startswith("info for")
            or "glob" in lowered
            or "match files" in lowered
            or "git diff" in lowered
            or ("diff" in lowered and "repo" in lowered)
            or "git status" in lowered
            or ("status" in lowered and "repo" in lowered)
        )

    def _requires_scoped_tool(self, lowered: str) -> bool:
        return (
            self._looks_like_organize_request(lowered)
            or self._looks_like_direct_write_request(lowered)
            or self._looks_like_file_count_request(lowered)
            or self._looks_like_directory_check_request(lowered)
            or "git diff" in lowered
            or ("diff" in lowered and "repo" in lowered)
            or "git status" in lowered
            or ("status" in lowered and "repo" in lowered)
            or self._looks_like_read_request(lowered)
            or lowered.startswith("stat")
            or lowered.startswith("info for")
            or "glob" in lowered
            or "match files" in lowered
            or self._looks_like_list_request(lowered)
            or self._looks_like_project_overview_request(lowered)
            or self._looks_like_file_search_request(lowered)
        )

    def _looks_like_file_search_request(self, lowered: str) -> bool:
        if not re.search(r"\b(?:find|search|summari[sz]e|about)\b", lowered, flags=re.IGNORECASE):
            return False
        has_search_action = bool(
            re.search(r"\b(?:find|search|summari[sz]e)\b", lowered, flags=re.IGNORECASE)
        )
        has_file_context = bool(
            re.search(
                r"\b(?:file|files|folder|folders|directory|directories|repo|repository|"
                r"project|workspace|path|root|selected|current)\b",
                lowered,
                flags=re.IGNORECASE,
            )
        )
        has_path_or_glob = bool(
            re.search(r"(?:\*\.|/|\\|\.[a-z0-9]{1,8}\b)", lowered, flags=re.IGNORECASE)
        )
        has_location_phrase = bool(
            re.search(r"\b(?:in|inside|under|from)\s+(?:this|the|selected|current|\.|/|~)", lowered)
        )
        return has_search_action and (has_file_context or has_path_or_glob or has_location_phrase)

    def _looks_like_organize_request(self, lowered: str) -> bool:
        return any(
            word in lowered
            for word in ["organize", "organise", "arrange", "distribute", "group", "separate"]
        ) or (
            "sort" in lowered and any(word in lowered for word in ["file", "folder", "directory"])
        )

    def _extract_organize_strategy(self, lowered: str) -> str:
        if any(word in lowered for word in ["topic", "topics", "semantic", "content", "subject"]):
            return "by_topic"
        if any(
            phrase in lowered
            for phrase in [
                "file name",
                "filename",
                "by name",
                "by their name",
                "as per name",
                "as per their name",
                "as per filename",
                "as per their filename",
            ]
        ):
            return "by_filename"
        if any(word in lowered for word in ["date", "month", "year", "time"]):
            return "by_date"
        return "by_type"

    def _extract_organize_file_filter(self, lowered: str) -> str:
        if any(word in lowered for word in ["image", "images", "photo", "photos", "picture", "pictures"]):
            return "image"
        if any(word in lowered for word in ["video", "videos", "movie", "movies"]):
            return "video"
        if any(word in lowered for word in ["audio", "music", "song", "songs"]):
            return "audio"
        if any(word in lowered for word in ["document", "documents", "doc", "docs", "pdf", "pdfs"]):
            return "document"
        return "all"

    def _looks_like_file_count_request(self, lowered: str) -> bool:
        has_count_word = bool(
            re.search(r"\b(?:count|number\s+of|how\s+many|total)\b", lowered, flags=re.IGNORECASE)
        )
        if not has_count_word:
            return False
        return bool(
            re.search(r"\bfiles?\b", lowered, flags=re.IGNORECASE)
            or self._extract_file_count_type_word(lowered) is not None
            or re.search(r"(\*\.[a-z0-9][a-z0-9_-]*)", lowered, flags=re.IGNORECASE)
        )

    def _extract_file_count_request(
        self,
        message: str,
        fallback_path: str,
    ) -> dict[str, object] | None:
        cleaned = message.strip()
        lowered = cleaned.lower()
        if not self._looks_like_file_count_request(lowered):
            return None

        patterns = ["*"]
        label = "matching"
        explicit_pattern = re.search(r"(\*\.[a-z0-9][a-z0-9_-]*)", cleaned, flags=re.IGNORECASE)
        extension = re.search(r"\.([a-z0-9][a-z0-9_-]*)\s+files?\b", cleaned, flags=re.IGNORECASE)
        named_type = re.search(r"\b([a-z0-9][a-z0-9_-]*)\s+files?\b", cleaned, flags=re.IGNORECASE)
        type_word = self._extract_file_count_type_word(cleaned)

        if explicit_pattern:
            patterns = [explicit_pattern.group(1).lower()]
            label = patterns[0]
        elif extension:
            suffix = extension.group(1).lower()
            patterns = [f"*.{suffix}"]
            label = suffix.upper()
        elif named_type or type_word:
            file_type = self._normalize_file_count_type(
                str(named_type.group(1) if named_type else type_word)
            )
            if file_type not in FILE_COUNT_IGNORED_TYPES:
                patterns = list(FILE_COUNT_PATTERN_GROUPS.get(file_type, (f"*.{file_type}",)))
                label = FILE_COUNT_LABELS.get(file_type, file_type.upper())

        root = fallback_path
        path_match = re.search(
            r"\b(?:in|inside|under|from)\s+(.+?)(?:\s+(?:folder|directory|dir|path))?$",
            cleaned,
            flags=re.IGNORECASE,
        )
        if path_match:
            root = self._clean_path_reference(path_match.group(1)) or fallback_path

        return {"pattern": patterns[0], "patterns": patterns, "root": root, "label": label}

    def _extract_file_count_type_word(self, text: str) -> str | None:
        explicit_extension = re.search(r"\.([a-z0-9][a-z0-9_-]*)\b", text, flags=re.IGNORECASE)
        if explicit_extension:
            suffix = explicit_extension.group(1).lower()
            if suffix in FILE_COUNT_EXTENSION_TYPES:
                return suffix
        for word in re.findall(r"\b[a-z0-9][a-z0-9_-]*\b", text, flags=re.IGNORECASE):
            normalized = self._normalize_file_count_type(word)
            if normalized in FILE_COUNT_PATTERN_GROUPS or normalized in FILE_COUNT_EXTENSION_TYPES:
                return word
        return None

    def _normalize_file_count_type(self, word: str) -> str:
        normalized = word.strip().strip(".").lower()
        plural_aliases = {
            "images": "image",
            "photos": "photo",
            "pictures": "picture",
            "videos": "video",
            "audios": "audio",
            "archives": "archive",
            "documents": "document",
            "spreadsheets": "spreadsheet",
            "presentations": "presentation",
        }
        if normalized in plural_aliases:
            return plural_aliases[normalized]
        if normalized.endswith("s") and len(normalized) > 3:
            singular = normalized[:-1]
            if singular in FILE_COUNT_PATTERN_GROUPS or singular in FILE_COUNT_EXTENSION_TYPES:
                return singular
        return normalized

    def _extract_path_argument(self, message: str, fallback: str) -> str:
        cleaned = message.strip()
        patterns = [
            r"(?:read|show|open|cat)\s+(?:file\s+)?(.+)$",
            r"(?:stat|info for|list)\s+(.+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if match:
                value = match.group(1).strip(" .\"'")
                return value or fallback
        return fallback

    def _extract_directory_check_path(self, message: str, fallback: str) -> str:
        cleaned = message.strip()
        patterns = [
            (
                r"^(?:please\s+)?(?:check|inspect|explore|scan|open|look\s+at)\s+"
                r"(?:the\s+)?(?:(?:this|current|approved|selected)\s+)?"
                r"(?:folder|directory|dir|root|path)(?:\s+(.+))?$"
            ),
            (
                r"^(?:please\s+)?(?:check|inspect|explore|scan|open|look\s+at)\s+"
                r"(.+?)\s+(?:folder|directory|dir)$"
            ),
        ]
        for pattern in patterns:
            match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if not match:
                continue
            value = self._clean_path_reference(match.group(1) or "")
            return value or fallback
        return fallback

    def _extract_list_path_argument(self, message: str, fallback: str) -> str:
        cleaned = message.strip()
        patterns = [
            r"^(?:list|show)\s+(?:the\s+)?(?:files|folders|directories|directory|folder|dirs)?\s*(?:in|inside|under|from)\s+(.+)$",
            r"^(?:what|which)\s+files\s+(?:are\s+)?(?:in|inside|under)\s+(.+)$",
            r"^(?:what\s+is\s+in|what\s+is\s+inside)\s+(.+)$",
            r"^(?:list|show)\s+(.+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if not match:
                continue
            value = self._clean_path_reference(match.group(1) or "")
            return value or fallback
        return fallback

    def _clean_path_reference(self, value: str) -> str:
        cleaned = self._clean_argument(value).strip().rstrip("?").strip()
        cleaned = re.sub(r"^(?:at|path|of|for)\s+", "", cleaned, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"^the\s+", "", cleaned, flags=re.IGNORECASE).strip()
        if re.fullmatch(
            r"(?:files|folders|directories|directory|folder|dirs)\s+(?:in|inside|under|from)",
            cleaned,
            flags=re.IGNORECASE,
        ):
            return ""
        if cleaned.lower() in {
            "",
            ".",
            "it",
            "there",
            "here",
            "that",
            "this",
            "current",
            "approved",
            "selected",
            "files",
            "folders",
            "directories",
            "folder",
            "directory",
            "dir",
            "root",
            "path",
            "this folder",
            "this directory",
            "this dir",
            "this root",
            "this path",
            "that folder",
            "that directory",
            "that dir",
            "that root",
            "that path",
            "the folder",
            "the directory",
            "the dir",
            "the root",
            "the path",
            "current folder",
            "current directory",
            "current dir",
            "current root",
            "approved folder",
            "approved directory",
            "approved root",
            "selected folder",
            "selected directory",
            "selected directory path",
            "selected dir",
            "selected folder path",
            "selected root",
            "selected path",
        }:
            return ""
        return cleaned

    def _extract_pattern(self, message: str) -> str:
        cleaned = message.strip()
        patterns = [
            r"glob\s+(.+)$",
            r"match files\s+(.+)$",
            r"files matching\s+(.+)$",
        ]
        for pattern in patterns:
            match = re.search(pattern, cleaned, flags=re.IGNORECASE)
            if match:
                value = match.group(1).strip(" .\"'")
                return value or "*"
        return "*"

    def _format_file_content(self, result: dict[str, object]) -> str:
        suffix = "\n\n[truncated]" if result.get("truncated") else ""
        return f"{result['path']} ({result['bytes_read']} bytes read):\n{result['content']}{suffix}"

    def _format_directory_listing(self, result: dict[str, object]) -> str:
        entries = list(result.get("entries", []))
        if not entries:
            return f"No files or folders were found in {result.get('path')}."
        names = [entry["name"] for entry in entries[:30]]
        lines = [f"Files and folders in {result.get('path')}:"]
        lines.extend(f"- {name}" for name in names)
        if result.get("truncated"):
            lines.append("[truncated]")
        return "\n".join(lines)

    def _format_stat(self, result: dict[str, object]) -> str:
        kind = "directory" if result.get("is_dir") else "file"
        return (
            f"{result['path']}\n"
            f"type: {kind}\n"
            f"size: {result['size']} bytes\n"
            f"modified: {result['modified']}"
        )

    def _format_glob(self, result: dict[str, object]) -> str:
        matches = result.get("matches", [])
        if not matches:
            return f"No files matched {result.get('pattern')!r}."
        lines = [f"Matched {len(matches)} file(s):"]
        for entry in matches[:50]:
            lines.append(f"- {entry['relative_path']}")
        if result.get("truncated"):
            lines.append("[truncated]")
        return "\n".join(lines)

    def _format_file_count(self, result: dict[str, object], label: str) -> str:
        matches = list(result.get("matches", []))
        prefix = "at least " if result.get("truncated") else ""
        lines = [
            (
                f"Found {prefix}{len(matches)} {label} file(s) "
                f"in {result.get('root')}."
            )
        ]
        if result.get("truncated"):
            lines.append("The result set was truncated, so the exact count may be higher.")
        if matches:
            lines.append("First matches:")
            for entry in matches[:10]:
                lines.append(f"- {entry.get('relative_path') or entry.get('name')}")
        return "\n".join(lines)

    def _overview_candidate_paths(self, result: dict[str, object]) -> list[str]:
        priority = [
            "README.md",
            "README.txt",
            "README",
            "package.json",
            "pyproject.toml",
            "Cargo.toml",
            "requirements.txt",
        ]
        names = {str(entry["name"]): entry for entry in result.get("entries", [])}
        return [
            candidate
            for candidate in priority
            if candidate in names and not names[candidate].get("is_dir")
        ][:3]

    def _format_project_overview(
        self,
        result: dict[str, object],
        overview_files: list[dict[str, object]],
    ) -> str:
        entries = list(result.get("entries", []))
        if not entries:
            return f"No files or folders were found in {result.get('path')}."
        directories = [entry["name"] for entry in entries if entry.get("is_dir")]
        files = [entry["name"] for entry in entries if not entry.get("is_dir")]
        lines = [
            f"I inspected {result.get('path')} and can only summarize what is visible from the file tree.",
            f"Top-level folders: {', '.join(directories[:12]) or 'none'}",
            f"Top-level files: {', '.join(files[:12]) or 'none'}",
        ]
        if overview_files:
            lines.append("Grounding files inspected:")
            for file_result in overview_files:
                path = Path(str(file_result["path"])).name
                content = str(file_result.get("content") or "").strip()
                excerpt = " ".join(content.split())[:500]
                lines.append(f"- {path}: {excerpt or '[empty]'}")
        if result.get("truncated"):
            lines.append("[truncated]")
        return "\n".join(lines)

    def _format_command_result(self, result: dict[str, object], label: str) -> str:
        stdout = str(result.get("stdout") or "").strip()
        stderr = str(result.get("stderr") or "").strip()
        returncode = result.get("returncode")
        if stdout:
            return f"{label} returned {returncode}:\n{stdout}"
        if stderr:
            return f"{label} returned {returncode}:\n{stderr}"
        return f"{label} returned {returncode} with no output."

    def _format_root_required_response(self, pending: dict[str, Any]) -> str:
        return (
            "I can do that after a file root is approved and attached to this chat.\n"
            f"Pending request: {pending.get('message')}\n"
            "Approve or select a folder in the file-root panel, then reply yes in this same session."
        )

    def _format_root_still_required_response(self, pending: dict[str, Any]) -> str:
        return (
            "I received your approval, but no approved file root was included with this chat request yet.\n"
            f"Pending request: {pending.get('message')}\n"
            "Please approve or select a folder first, then reply yes again in this same session."
        )

    def _format_organize_plan(self, plan: dict[str, object]) -> str:
        summary = dict(plan.get("summary") or {})
        actions = list(plan.get("actions") or [])
        moves = [action for action in actions if isinstance(action, dict) and action.get("op") == "move"]
        warnings = list(plan.get("warnings") or [])
        file_filter = str(plan.get("file_filter") or "all")
        filter_text = "" if file_filter == "all" else f" for {file_filter} files"
        lines = [
            f"Prepared organize plan {plan['plan_id']} using {plan['strategy']}{filter_text}.",
            (
                f"Preview: {summary.get('moves', 0)} move(s), "
                f"{summary.get('new_dirs', 0)} new folder(s), "
                f"{summary.get('skipped', 0)} skipped."
            ),
            "Nothing has been moved yet.",
        ]
        if warnings:
            lines.append("Warnings:")
            lines.extend(f"- {warning}" for warning in warnings[:5])
        if moves:
            lines.append("First moves:")
            for action in moves[:8]:
                lines.append(f"- {action['from']} -> {action['to']}")
        lines.append(
            "Reply yes/approve to apply this plan, no/cancel to discard it, "
            "or no with changes to redesign it."
        )
        return "\n".join(lines)

    def _organize_plan_summary(self, plan: dict[str, object]) -> str:
        summary = dict(plan.get("summary") or {})
        file_filter = str(plan.get("file_filter") or "all")
        filter_text = "" if file_filter == "all" else f" for {file_filter} files"
        return (
            f"apply organize plan {plan.get('plan_id')}{filter_text} "
            f"({summary.get('moves', 0)} move(s), {summary.get('new_dirs', 0)} new folder(s))"
        )

    def _format_organize_apply_result(self, result: dict[str, object]) -> str:
        applied = list(result.get("applied") or [])
        status = result.get("status") or "completed"
        if result.get("already_applied"):
            return f"Organize plan {result.get('plan_id')} was already applied."
        if result.get("failed"):
            return (
                f"Organize plan {result.get('plan_id')} finished with status {status}. "
                f"Applied {len(applied)} action(s) before the failure: {result.get('failed')}"
            )
        return f"Applied organize plan {result.get('plan_id')}: {len(applied)} action(s) completed."

    def _format_pending_write_plan(self, pending: dict[str, Any]) -> str:
        lines = [
            f"Pending approval: {pending.get('summary')}.",
            "Planned command(s):",
        ]
        for summary in self._pending_write_command_summaries(pending):
            lines.append(f"- {summary}")
        lines.extend(
            [
                "Nothing has been changed yet.",
                (
                    "Reply yes/approve to execute this operation, "
                    "no/cancel to discard it, "
                    "or no with changes to redesign it."
                ),
            ]
        )
        return "\n".join(lines)

    def _pending_write_command_summaries(self, pending: dict[str, Any]) -> list[str]:
        if pending.get("type") == "write_tool":
            return [
                self._write_plan_summary(
                    str(pending.get("tool_name") or ""),
                    dict(pending.get("arguments") or {}),
                )
            ]
        if pending.get("type") == "write_batch":
            summaries: list[str] = []
            for command in list(pending.get("commands") or [])[:8]:
                if not isinstance(command, dict):
                    continue
                summaries.append(
                    self._write_plan_summary(
                        str(command.get("name") or ""),
                        dict(command.get("arguments") or {}),
                    )
                )
            return summaries
        return [str(pending.get("summary") or "file operation")]

    def _format_write_batch_result(self, records: list[ToolCallRecord]) -> str:
        if not records:
            return "No pending file commands were executed."
        if len(records) == 1:
            record = records[0]
            if record.status == "completed":
                return self._format_write_result(record)
            return f"{record.status}: {record.command}: {record.result.get('error')}"

        completed = [record for record in records if record.status == "completed"]
        failed = [record for record in records if record.status != "completed"]
        lines = [f"Executed {len(completed)} of {len(records)} pending file command(s)."]
        for record in records:
            if record.status == "completed":
                lines.append(f"- {self._format_write_result(record)}")
            else:
                lines.append(
                    f"- {record.status}: {record.command}: {record.result.get('error')}"
                )
        if failed:
            lines.append("Stopped after the first failed command.")
        return "\n".join(lines)

    def _write_plan_summary(self, name: str, arguments: dict[str, object]) -> str:
        if name == "make_dir":
            return f"create folder {arguments.get('path')}"
        if name == "write_file":
            content = str(arguments.get("content") or "")
            action = "overwrite" if arguments.get("overwrite") else "create"
            size = len(content.encode("utf-8"))
            return f"{action} file {arguments.get('path')} ({size} byte(s))"
        if name == "copy_path":
            source = arguments.get("source") or arguments.get("from")
            destination = arguments.get("destination") or arguments.get("to")
            return f"copy {source} to {destination}"
        if name == "move_path":
            source = arguments.get("source") or arguments.get("from")
            destination = arguments.get("destination") or arguments.get("to")
            return f"move {source} to {destination}"
        if name == "edit_file":
            count = arguments.get("count", 1)
            return f"edit {arguments.get('path')} ({count} replacement(s))"
        return "run file operation"

    def _format_write_result(self, record: ToolCallRecord) -> str:
        result = record.result
        if record.name == "make_dir":
            action = "Created" if result.get("created") else "Folder already existed"
            return f"{action}: {result.get('path')}"
        if record.name == "write_file":
            action = "Updated" if result.get("overwritten") else "Created"
            return f"{action} file: {result.get('path')} ({result.get('bytes_written')} bytes)"
        if record.name == "copy_path":
            return f"Copied {result.get('kind', 'path')}: {result.get('from')} -> {result.get('to')}"
        if record.name == "move_path":
            return f"Moved: {result.get('from')} -> {result.get('to')}"
        if record.name == "edit_file":
            return f"Edited {result.get('path')}: {result.get('replacements')} replacement(s)"
        return "File operation completed."

    def _format_tool_command(
        self,
        name: str,
        arguments: dict[str, object],
        result: dict[str, object],
    ) -> str:
        if name in {"git_status", "git_diff"} and isinstance(result.get("command"), list):
            cwd = result.get("cwd") or "."
            command = " ".join(str(part) for part in result["command"])
            return f"(cd {cwd} && {command})"
        args = ", ".join(f"{key}={value!r}" for key, value in sorted(arguments.items()))
        roots = f", roots={self.roots!r}" if self.roots and "root" not in arguments else ""
        return f"{name}({args}{roots})"

    def _with_tool_trace(self, response: str, tool_calls: list[ToolCallRecord]) -> str:
        if not tool_calls:
            return response
        lines = ["", "Tools run:"]
        lines.extend(f"- {record.command or record.name}" for record in tool_calls)
        return response.rstrip() + "\n".join(lines)
