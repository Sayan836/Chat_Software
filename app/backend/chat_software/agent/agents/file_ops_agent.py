"""Private file-ops agent subgraph wrapping the existing deterministic tools."""

from __future__ import annotations

from typing import Any, TypedDict

from chat_software.agent.deps import AgentDeps


class FileOpsAgentState(TypedDict, total=False):
    message: str
    session_id: str
    force_llm_planner: bool
    response: str
    tool_calls: list[Any]


def build_file_ops_agent(deps: AgentDeps) -> Any:
    from langgraph.graph import END, START, StateGraph

    def run_file_ops(state: FileOpsAgentState) -> FileOpsAgentState:
        if deps.file_ops_handler is None:
            raise RuntimeError("file_ops agent requires a file_ops_handler in this phase.")
        response, tool_calls = deps.file_ops_handler(
            str(state.get("message") or ""),
            str(state.get("session_id") or "") or None,
            bool(state.get("force_llm_planner")),
        )
        return {"response": response, "tool_calls": tool_calls}

    graph = StateGraph(FileOpsAgentState)
    graph.add_node("file_ops", run_file_ops)
    graph.add_edge(START, "file_ops")
    graph.add_edge("file_ops", END)
    return graph.compile()
