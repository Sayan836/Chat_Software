import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from chat_software.agent.graph_runtime import (
    GRAPH_CHECKPOINT_DB,
    build_checkpointer,
    graph_checkpoint_path,
)


class GraphRuntimeTests(unittest.TestCase):
    def test_checkpoint_path_uses_state_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"CHAT_SOFTWARE_STATE_DIR": tmp}):
                self.assertEqual(graph_checkpoint_path(), Path(tmp) / GRAPH_CHECKPOINT_DB)

    def test_memory_state_dir_uses_in_memory_saver(self):
        try:
            from langgraph.checkpoint.memory import InMemorySaver
        except ImportError:
            self.skipTest("langgraph is not installed")

        with patch.dict(os.environ, {"CHAT_SOFTWARE_STATE_DIR": ":memory:"}):
            self.assertIsInstance(build_checkpointer(), InMemorySaver)

    def test_command_resume_round_trips_in_memory_graph(self):
        try:
            from langgraph.graph import END, START, StateGraph
            from langgraph.types import Command, interrupt
        except ImportError:
            self.skipTest("langgraph is not installed")

        with patch.dict(os.environ, {"CHAT_SOFTWARE_STATE_DIR": ":memory:"}):
            graph = StateGraph(dict)

            def approve(state):
                approved = interrupt({"question": "continue?"})
                return {"approved": bool(approved), "value": int(state.get("value", 0)) + 1}

            graph.add_node("approve", approve)
            graph.add_edge(START, "approve")
            graph.add_edge("approve", END)
            app = graph.compile(checkpointer=build_checkpointer())
            config = {"configurable": {"thread_id": "resume-test"}}

            interrupted = app.invoke({"value": 0}, config=config)
            self.assertIn("__interrupt__", interrupted)
            self.assertEqual(len(app.get_state(config).interrupts), 1)

            resumed = app.invoke(Command(resume=True), config=config)
            self.assertEqual(resumed["value"], 1)
            self.assertTrue(resumed["approved"])

    def test_sqlite_checkpointer_round_trip_when_installed(self):
        try:
            from langgraph.graph import END, START, StateGraph
            from langgraph.checkpoint.sqlite import SqliteSaver
        except ImportError:
            self.skipTest("langgraph-checkpoint-sqlite is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(os.environ, {"CHAT_SOFTWARE_STATE_DIR": tmp}):
                checkpointer = build_checkpointer()
                self.assertIsInstance(checkpointer, SqliteSaver)

                graph = StateGraph(dict)

                def first(state):
                    return {"value": int(state.get("value", 0)) + 1}

                def second(state):
                    return {"value": int(state.get("value", 0)) + 1}

                graph.add_node("first", first)
                graph.add_node("second", second)
                graph.add_edge(START, "first")
                graph.add_edge("first", "second")
                graph.add_edge("second", END)
                app = graph.compile(checkpointer=checkpointer)

                config = {"configurable": {"thread_id": "graph-runtime-test"}}
                self.assertEqual(app.invoke({"value": 0}, config=config)["value"], 2)
                state = app.get_state(config)
                self.assertEqual(state.values["value"], 2)


if __name__ == "__main__":
    unittest.main()
