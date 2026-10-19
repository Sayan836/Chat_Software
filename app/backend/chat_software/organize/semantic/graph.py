"""Private LangGraph subgraph for semantic file planning.

The first production slice uses deterministic labels, but the graph shape is
the planned map/reduce contract: scan -> Send(label batches) -> reduce -> assign.
"""

from __future__ import annotations

import operator
from pathlib import Path
from typing import Annotated, Any, TypedDict

from .compiler import sanitize_folder_name, topic_label_for_source
from .features import featurize_files

try:
    from langgraph.types import Send
except ImportError:  # pragma: no cover - optional dependency guard
    Send = Any  # type: ignore[misc, assignment]


class SemanticPlannerState(TypedDict, total=False):
    root: str
    files: list[str]
    features: list[dict[str, object]]
    labels: Annotated[list[dict[str, object]], operator.add]
    taxonomy: list[dict[str, object]]
    assignments: list[dict[str, object]]
    warnings: list[str]


def build_semantic_planner_graph() -> Any:
    from langgraph.graph import END, START, StateGraph

    def scan(state: SemanticPlannerState) -> SemanticPlannerState:
        root = Path(str(state.get("root") or ".")).expanduser().resolve()
        raw_files = list(state.get("files") or [])
        files = [
            Path(raw) if Path(raw).is_absolute() else root / raw
            for raw in raw_files
        ]
        features = [feature.to_dict() for feature in featurize_files(files, root=root)]
        return {"features": features, "labels": []}

    def route_label_batches(state: SemanticPlannerState) -> str | list[Send]:
        features = list(state.get("features") or [])
        if not features:
            return "reduce_taxonomy"
        return [
            Send("label_batch", {"root": state.get("root", "."), "features": features[index : index + 40]})
            for index in range(0, len(features), 40)
        ]

    def label_batch(state: SemanticPlannerState) -> SemanticPlannerState:
        root = Path(str(state.get("root") or ".")).expanduser().resolve()
        labels: list[dict[str, object]] = []
        for feature in list(state.get("features") or []):
            file_id = int(feature.get("id") or 0)
            path = str(feature.get("path") or "")
            labels.append(
                {
                    "id": file_id,
                    "label": topic_label_for_source(root / path, root=root),
                }
            )
        return {"labels": labels}

    def reduce_taxonomy(state: SemanticPlannerState) -> SemanticPlannerState:
        names = []
        seen: set[str] = set()
        for label in list(state.get("labels") or []):
            name = sanitize_folder_name(str(label.get("label") or "Unsorted"))
            key = name.casefold()
            if key in seen:
                continue
            seen.add(key)
            names.append(name)
        if not names:
            names = ["Unsorted"]
        taxonomy = [
            {
                "index": index,
                "name": name,
                "parent": None,
                "rationale": "Grouped by semantic filename and metadata signals.",
            }
            for index, name in enumerate(names[:24], start=1)
        ]
        return {"taxonomy": taxonomy}

    def assign(state: SemanticPlannerState) -> SemanticPlannerState:
        index_by_name = {
            str(folder.get("name") or "").casefold(): int(folder.get("index") or 0)
            for folder in list(state.get("taxonomy") or [])
        }
        assignments: list[dict[str, object]] = []
        for label in list(state.get("labels") or []):
            folder_index = index_by_name.get(str(label.get("label") or "Unsorted").casefold(), 0)
            assignments.append(
                {
                    "file_id": int(label.get("id") or 0),
                    "folder_index": folder_index,
                }
            )
        return {"assignments": assignments}

    graph = StateGraph(SemanticPlannerState)
    graph.add_node("scan", scan)
    graph.add_node("label_batch", label_batch)
    graph.add_node("reduce_taxonomy", reduce_taxonomy)
    graph.add_node("assign", assign)
    graph.add_edge(START, "scan")
    graph.add_conditional_edges(
        "scan",
        route_label_batches,
        {
            "label_batch": "label_batch",
            "reduce_taxonomy": "reduce_taxonomy",
        },
    )
    graph.add_edge("label_batch", "reduce_taxonomy")
    graph.add_edge("reduce_taxonomy", "assign")
    graph.add_edge("assign", END)
    return graph.compile()
