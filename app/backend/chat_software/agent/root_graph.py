"""Root LangGraph orchestration for the local-first assistant."""

from __future__ import annotations

from typing import Any, TypedDict

from chat_software.agent.read_only_agent import CHAT_FALLBACK_RESPONSE

from .deps import AgentDeps
from .graph_runtime import build_checkpointer
from .registry import AGENTS, register_default_agents
from .routing import route_message


class RootState(TypedDict, total=False):
    message: str
    session_id: str
    route: str
    route_reason: str
    response: str
    tool_calls: list[Any]
    needs_chat_fallback: bool
    emit_to_stream: bool


def build_root_graph(agent: Any) -> Any:
    from langgraph.graph import END, START, StateGraph
    from langgraph.types import Command

    register_default_agents()
    deps = AgentDeps.from_values(
        chat_client=agent.chat_client,
        settings=agent.chat_client.settings,
        session_store=agent.session_store,
        safety_store=agent.safety_store,
        roots=agent.roots,
        working_path=agent.working_path,
        chat_handler=agent._run_chat_agent,
        file_ops_handler=agent._run_file_ops_agent,
        document_handler=agent._run_document_agent,
    )
    mounted = {name: spec.build(deps) for name, spec in AGENTS.items()}

    def router(state: RootState) -> Command:
        session_id = str(state.get("session_id") or "") or None
        if agent._has_pending_operation(session_id):
            return Command(
                goto="file_ops",
                update={"route": "file_ops", "route_reason": "pending_file_operation"},
            )
        if getattr(agent, "knowledge_root_id", None):
            return Command(
                goto="documents",
                update={"route": "documents", "route_reason": "documents mode selected"},
            )
        decision = route_message(
            str(state.get("message") or ""),
            roots_present=bool(agent.roots),
            enable_llm_planner=bool(agent.enable_llm_planner),
        )
        target = decision.route if decision.route in mounted else "chat"
        return Command(
            goto=target,
            update={"route": decision.route, "route_reason": decision.reason},
        )

    def chat_wrapper(state: RootState) -> RootState:
        previous_callback = agent.runtime_callback
        writer = _runtime_event_writer() if state.get("emit_to_stream") else None
        if writer is not None:
            agent.runtime_callback = writer
        try:
            output = mounted["chat"].invoke({"message": state.get("message", "")})
        finally:
            if writer is not None:
                agent.runtime_callback = previous_callback
        return {
            "response": output.get("response", ""),
            "tool_calls": output.get("tool_calls", []),
            "needs_chat_fallback": False,
        }

    def file_ops_wrapper(state: RootState) -> RootState:
        previous_callback = agent.runtime_callback
        writer = _runtime_event_writer() if state.get("emit_to_stream") else None
        if writer is not None:
            agent.runtime_callback = writer
        try:
            output = mounted["file_ops"].invoke(
                {
                    "message": state.get("message", ""),
                    "session_id": state.get("session_id", ""),
                    "force_llm_planner": state.get("route_reason") == "LLM planner candidate",
                }
            )
        finally:
            if writer is not None:
                agent.runtime_callback = previous_callback
        needs_chat_fallback = output.get("response") == CHAT_FALLBACK_RESPONSE
        return {
            "response": "" if needs_chat_fallback else output.get("response", ""),
            "tool_calls": []
            if needs_chat_fallback
            else [_tool_call_to_state(record) for record in output.get("tool_calls", [])],
            "needs_chat_fallback": needs_chat_fallback,
        }

    def documents_wrapper(state: RootState) -> RootState:
        previous_callback = agent.runtime_callback
        writer = _runtime_event_writer() if state.get("emit_to_stream") else None
        if writer is not None:
            agent.runtime_callback = writer
        try:
            output = mounted["documents"].invoke({"message": state.get("message", "")})
        finally:
            if writer is not None:
                agent.runtime_callback = previous_callback
        return {
            "response": output.get("response", ""),
            "tool_calls": [_tool_call_to_state(record) for record in output.get("tool_calls", [])],
            "needs_chat_fallback": False,
        }

    def after_file_ops(state: RootState) -> str:
        if state.get("needs_chat_fallback"):
            return "chat"
        if _should_fallback_from_empty_tool_result(state):
            return "chat"
        return "respond"

    def respond(state: RootState) -> RootState:
        return {}

    def _should_fallback_from_empty_tool_result(state: RootState) -> bool:
        if state.get("route_reason") != "LLM planner candidate":
            return False
        tool_calls = list(state.get("tool_calls") or [])
        if len(tool_calls) != 1:
            return False
        record = tool_calls[0]
        if isinstance(record, dict):
            name = str(record.get("name") or "")
            result = record.get("result") or {}
        else:
            name = getattr(record, "name", "")
            result = getattr(record, "result", {}) or {}
        if not isinstance(result, dict):
            return False
        if name == "grep_files":
            return not list(result.get("matches") or [])
        if name == "glob_files":
            return not list(result.get("matches") or [])
        return False

    def _tool_call_to_state(record: Any) -> dict[str, object]:
        if isinstance(record, dict):
            return dict(record)
        return {
            "name": str(getattr(record, "name", "")),
            "arguments": dict(getattr(record, "arguments", {}) or {}),
            "result": dict(getattr(record, "result", {}) or {}),
            "command": str(getattr(record, "command", "")),
            "status": str(getattr(record, "status", "completed")),
        }

    def _runtime_event_writer() -> Any | None:
        try:
            from langgraph.config import get_stream_writer

            writer = get_stream_writer()
        except Exception:
            return None
        return writer if callable(writer) else None

    graph = StateGraph(RootState)
    graph.add_node("router", router, destinations=("file_ops", "chat", "documents"))
    graph.add_node("file_ops", file_ops_wrapper)
    graph.add_node("chat", chat_wrapper)
    graph.add_node("documents", documents_wrapper)
    graph.add_node("respond", respond)
    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "file_ops",
        after_file_ops,
        {
            "chat": "chat",
            "respond": "respond",
        },
    )
    graph.add_edge("chat", "respond")
    graph.add_edge("documents", "respond")
    graph.add_edge("respond", END)
    try:
        return graph.compile(checkpointer=build_checkpointer())
    except RuntimeError:
        try:
            from langgraph.checkpoint.memory import InMemorySaver

            return graph.compile(checkpointer=InMemorySaver())
        except ImportError:
            return graph.compile()
