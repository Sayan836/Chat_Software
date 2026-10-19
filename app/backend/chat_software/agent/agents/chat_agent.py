"""Private chat-agent subgraph for ordinary conversation."""

from __future__ import annotations

from typing import Any, TypedDict

from chat_software.agent.deps import AgentDeps


class ChatAgentState(TypedDict, total=False):
    message: str
    response: str
    tool_calls: list[Any]


def build_chat_agent(deps: AgentDeps) -> Any:
    from langgraph.graph import END, START, StateGraph

    def run_chat(state: ChatAgentState) -> ChatAgentState:
        message = str(state.get("message") or "")
        if deps.chat_handler is not None:
            response = deps.chat_handler(message)
        else:
            response = deps.chat_client.complete([{"role": "user", "content": message}])
        return {"response": response, "tool_calls": []}

    graph = StateGraph(ChatAgentState)
    graph.add_node("chat", run_chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    return graph.compile()
