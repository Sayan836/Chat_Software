import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from chat_software.agent import ReadOnlyAgent, SessionStore
from chat_software.provider.config import ProviderSettings
from chat_software.safety_store import SafetyStore


@dataclass
class StubChatClient:
    settings: ProviderSettings
    response: str = "stubbed model response"
    calls: int = 0

    def complete(self, messages, tools=None):
        self.calls += 1
        return self.response


@dataclass
class SequenceChatClient:
    settings: ProviderSettings
    responses: list[str]

    def complete(self, messages, tools=None):
        if not self.responses:
            return ""
        return self.responses.pop(0)


class ReadOnlyAgentTests(unittest.TestCase):
    def test_find_and_summarize_flow(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "matter.md").write_text(
                "Acme contract renewal includes a termination clause.\n",
                encoding="utf-8",
            )
            store = SessionStore(root / "sessions.sqlite3")
            agent = ReadOnlyAgent(roots=[root], session_store=store)

            result = agent.handle("find files about Acme and summarize")

            self.assertEqual(result.tool_calls[0].name, "grep_files")
            self.assertIn("Acme", result.response)
            self.assertEqual(len(store.get_messages(result.session_id)), 2)

    def test_read_file_request_uses_read_file_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "notes.md").write_text("Read me from the tool.\n", encoding="utf-8")
            store = SessionStore(root / "sessions.sqlite3")
            agent = ReadOnlyAgent(roots=[root], session_store=store)

            result = agent.handle("read file notes.md")

            self.assertEqual(result.tool_calls[0].name, "read_file")
            self.assertIn("Read me", result.response)

    def test_stat_and_glob_requests_use_file_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "notes.md").write_text("hello\n", encoding="utf-8")
            store = SessionStore(root / "sessions.sqlite3")
            agent = ReadOnlyAgent(roots=[root], session_store=store)

            stat_result = agent.handle("stat notes.md")
            glob_result = agent.handle("glob *.md")

            self.assertEqual(stat_result.tool_calls[0].name, "stat_path")
            self.assertIn("type: file", stat_result.response)
            self.assertEqual(glob_result.tool_calls[0].name, "glob_files")
            self.assertIn("notes.md", glob_result.response)

    def test_plain_chat_uses_configured_chat_client(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            client = StubChatClient(
                settings=ProviderSettings(
                    provider="test",
                    model="stub",
                    api_key=None,
                    base_url=None,
                )
            )
            agent = ReadOnlyAgent(roots=[root], session_store=store, chat_client=client)

            result = agent.handle("hello")

            self.assertEqual(result.response, "stubbed model response")
            self.assertEqual(result.tool_calls, [])

    def test_ordinary_about_find_and_summarize_chat_does_not_search_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "notes.md").write_text("local-only\n", encoding="utf-8")
            store = SessionStore(root / "sessions.sqlite3")
            client = StubChatClient(
                settings=ProviderSettings(
                    provider="test",
                    model="stub",
                    api_key=None,
                    base_url=None,
                )
            )
            agent = ReadOnlyAgent(roots=[root], session_store=store, chat_client=client)

            for message in [
                "tell me about python decorators",
                "what do you know about the weather",
                "summarize the plot of Hamlet",
                "how many people find this hard?",
            ]:
                with self.subTest(message=message):
                    result = agent.handle(message)
                    self.assertEqual(result.response, "stubbed model response")
                    self.assertEqual(result.tool_calls, [])

    def test_graph_file_ops_route_can_fallback_to_chat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            client = StubChatClient(
                settings=ProviderSettings(
                    provider="test",
                    model="planner",
                    api_key=None,
                    base_url=None,
                ),
                response="chat fallback response",
            )
            agent = ReadOnlyAgent(roots=[root], session_store=store, chat_client=client)

            result = agent.handle("explain folder organization")

            self.assertEqual(result.response, "chat fallback response")
            self.assertEqual(result.tool_calls, [])

    def test_llm_candidate_empty_search_result_falls_back_to_chat(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "notes.md").write_text("local-only\n", encoding="utf-8")
            store = SessionStore(root / "sessions.sqlite3")
            client = SequenceChatClient(
                settings=ProviderSettings(
                    provider="test",
                    model="planner",
                    api_key=None,
                    base_url=None,
                ),
                responses=[
                    '{"commands":[{"name":"grep_files","arguments":{"query":"decorators","root":"."}}]}',
                    "No local files mention decorators.",
                    "Decorators are a Python language feature.",
                ],
            )
            agent = ReadOnlyAgent(roots=[root], session_store=store, chat_client=client)

            result = agent.handle("tell me about decorators in this project")

            self.assertEqual(result.response, "Decorators are a Python language feature.")
            self.assertEqual(result.tool_calls, [])

    def test_local_project_question_uses_file_tree_instead_of_model_guess(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("# Real Project\n", encoding="utf-8")
            (root / "src").mkdir()
            store = SessionStore(root / "sessions.sqlite3")
            client = StubChatClient(
                settings=ProviderSettings(
                    provider="test",
                    model="stub",
                    api_key=None,
                    base_url=None,
                ),
                response="wrong model guess",
            )
            agent = ReadOnlyAgent(roots=[root], session_store=store, chat_client=client)

            result = agent.handle("what is this project?")

            self.assertEqual(result.tool_calls[0].name, "list_dir")
            self.assertEqual(result.tool_calls[1].name, "read_file")
            self.assertIn("README.md", result.response)
            self.assertIn("Real Project", result.response)
            self.assertIn("src", result.response)
            self.assertIn("Tools run:", result.response)
            self.assertIn("list_dir", result.tool_calls[0].command)
            self.assertNotIn("wrong model guess", result.response)
            self.assertEqual(client.calls, 0)

    def test_llm_plans_tools_then_final_answer_uses_results(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "README.md").write_text("# Planned Project\n", encoding="utf-8")
            store = SessionStore(root / "sessions.sqlite3")
            client = SequenceChatClient(
                settings=ProviderSettings(
                    provider="test",
                    model="planner",
                    api_key=None,
                    base_url=None,
                ),
                responses=[
                    '{"commands":[{"name":"list_dir","arguments":{"path":"."}},'
                    '{"name":"read_file","arguments":{"path":"README.md"}}]}',
                    "This is the final grounded answer.",
                ],
            )
            agent = ReadOnlyAgent(roots=[root], session_store=store, chat_client=client)

            planned = agent._run_llm_tool_loop("inspect relevant project files")

            self.assertIsNotNone(planned)
            response, tool_calls = planned
            self.assertEqual([call.name for call in tool_calls], ["list_dir", "read_file"])
            self.assertIn("This is the final grounded answer.", response)
            self.assertIn("read_file", tool_calls[1].command)

    def test_selected_working_path_is_used_for_folder_intents(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected = root / "selected"
            selected.mkdir()
            (root / "root-only.txt").write_text("root\n", encoding="utf-8")
            (selected / "inside.txt").write_text("inside\n", encoding="utf-8")
            store = SessionStore(root / "sessions.sqlite3")
            agent = ReadOnlyAgent(
                roots=[root],
                session_store=store,
                working_path="selected",
            )

            result = agent.handle("check selected directory path")

            self.assertEqual(result.tool_calls[0].name, "list_dir")
            self.assertEqual(result.tool_calls[0].arguments["path"], "selected")
            self.assertIn("inside.txt", result.response)
            self.assertNotIn("root-only.txt", result.response)

    def test_number_of_pdf_files_uses_glob_instead_of_path_phrase(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "resume.pdf").write_text("pdf\n", encoding="utf-8")
            (root / "cover.pdf").write_text("pdf\n", encoding="utf-8")
            (root / "notes.txt").write_text("txt\n", encoding="utf-8")
            store = SessionStore(root / "sessions.sqlite3")
            agent = ReadOnlyAgent(roots=[root], session_store=store)

            result = agent.handle("check the number of pdf files in this folder")

            self.assertEqual(result.tool_calls[0].name, "glob_files")
            self.assertEqual(result.tool_calls[0].arguments["pattern"], "*.pdf")
            self.assertEqual(result.tool_calls[0].arguments["root"], ".")
            self.assertIn("Found 2 PDF file(s)", result.response)
            self.assertNotIn("number of pdf files in this", result.tool_calls[0].command)

    def test_number_of_image_files_uses_image_extension_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "photo.jpg").write_text("jpg\n", encoding="utf-8")
            (root / "scan.PNG").write_text("png\n", encoding="utf-8")
            (root / "diagram.svg").write_text("svg\n", encoding="utf-8")
            (root / "notes.txt").write_text("txt\n", encoding="utf-8")
            store = SessionStore(root / "sessions.sqlite3")
            agent = ReadOnlyAgent(roots=[root], session_store=store)

            result = agent.handle("check the number of image files in this folder")

            patterns = [call.arguments["pattern"] for call in result.tool_calls]
            self.assertIn("*.jpg", patterns)
            self.assertIn("*.PNG", patterns)
            self.assertNotIn("*.image", patterns)
            self.assertIn("Found 3 IMAGE file(s)", result.response)

    def test_number_of_images_without_file_word_uses_image_extension_family(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "photo.jpg").write_text("jpg\n", encoding="utf-8")
            (root / "scan.PNG").write_text("png\n", encoding="utf-8")
            (root / "notes.txt").write_text("txt\n", encoding="utf-8")
            store = SessionStore(root / "sessions.sqlite3")
            agent = ReadOnlyAgent(roots=[root], session_store=store)

            result = agent.handle("check the number of images")

            patterns = [call.arguments["pattern"] for call in result.tool_calls]
            self.assertIn("*.jpg", patterns)
            self.assertIn("*.PNG", patterns)
            self.assertIn("Found 2 IMAGE file(s)", result.response)

    def test_no_root_file_count_yes_stays_in_file_ops_until_root_is_attached(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "photo.jpg").write_text("jpg\n", encoding="utf-8")
            (root / "scan.PNG").write_text("png\n", encoding="utf-8")
            store = SessionStore(root / "sessions.sqlite3")
            agent_without_roots = ReadOnlyAgent(
                roots=[],
                session_store=store,
                offline_echo=True,
            )

            pending = agent_without_roots.handle("check the number of images")

            self.assertEqual(pending.tool_calls, [])
            self.assertIn("Pending request", pending.response)
            self.assertIn("approve or select", pending.response.lower())

            still_pending = agent_without_roots.handle("yes", session_id=pending.session_id)

            self.assertEqual(still_pending.tool_calls, [])
            self.assertIn("no approved file root", still_pending.response.lower())
            self.assertNotIn("Offline echo mode", still_pending.response)

            agent_with_roots = ReadOnlyAgent(
                roots=[root],
                session_store=store,
                offline_echo=True,
            )
            counted = agent_with_roots.handle("yes", session_id=pending.session_id)

            self.assertTrue(counted.tool_calls)
            self.assertEqual(counted.tool_calls[0].name, "glob_files")
            self.assertIn("Found 2 IMAGE file(s)", counted.response)

    def test_file_count_pronoun_uses_selected_working_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            selected = root / "selected"
            selected.mkdir()
            (selected / "resume.pdf").write_text("pdf\n", encoding="utf-8")
            (root / "root.pdf").write_text("pdf\n", encoding="utf-8")
            store = SessionStore(root / "sessions.sqlite3")
            agent = ReadOnlyAgent(
                roots=[root],
                session_store=store,
                working_path="selected",
            )

            result = agent.handle("check the number of pdf files in it")

            self.assertEqual(result.tool_calls[0].name, "glob_files")
            self.assertEqual(result.tool_calls[0].arguments["root"], "selected")
            self.assertIn("Found 1 PDF file(s)", result.response)

    def test_llm_planned_write_is_rejected_without_touching_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            client = SequenceChatClient(
                settings=ProviderSettings(
                    provider="test",
                    model="planner",
                    api_key=None,
                    base_url=None,
                ),
                responses=[
                    '{"commands":[{"name":"write_file","arguments":{"path":"bad.txt","content":"nope"}}]}',
                    "I wrote the file.",
                ],
            )
            agent = ReadOnlyAgent(roots=[root], session_store=store, chat_client=client)

            planned = agent._run_llm_tool_loop("inspect relevant project files")

            self.assertIsNotNone(planned)
            response, tool_calls = planned
            self.assertEqual(tool_calls[0].name, "write_file")
            self.assertEqual(tool_calls[0].status, "rejected")
            self.assertIn("disallowed tool", response)
            self.assertFalse((root / "bad.txt").exists())

    def test_file_listing_question_uses_list_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "notes.md").write_text("hello\n", encoding="utf-8")
            store = SessionStore(root / "sessions.sqlite3")
            agent = ReadOnlyAgent(roots=[root], session_store=store)

            result = agent.handle("what files are in this folder?")

            self.assertEqual(result.tool_calls[0].name, "list_dir")
            self.assertIn("notes.md", result.response)
            self.assertIn("Tools run:", result.response)

    def test_check_this_folder_uses_approved_root_without_model_guess(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "notes.md").write_text("hello\n", encoding="utf-8")
            (root / "subdir").mkdir()
            store = SessionStore(root / "sessions.sqlite3")
            client = StubChatClient(
                settings=ProviderSettings(
                    provider="test",
                    model="stub",
                    api_key=None,
                    base_url=None,
                ),
                response="hallucinated folder summary",
            )
            agent = ReadOnlyAgent(roots=[root], session_store=store, chat_client=client)

            folder_result = agent.handle("check this folder")
            directory_result = agent.handle("inspect this directory")
            subdir_result = agent.handle("check folder subdir")

            self.assertEqual(folder_result.tool_calls[0].name, "list_dir")
            self.assertEqual(directory_result.tool_calls[0].name, "list_dir")
            self.assertEqual(subdir_result.tool_calls[0].arguments["path"], "subdir")
            self.assertIn("notes.md", folder_result.response)
            self.assertIn("subdir", folder_result.response)
            self.assertNotIn("hallucinated", folder_result.response)
            self.assertEqual(client.calls, 0)

    def test_show_files_uses_list_tool_instead_of_reading_literal_files_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "notes.md").write_text("hello\n", encoding="utf-8")
            store = SessionStore(root / "sessions.sqlite3")
            agent = ReadOnlyAgent(roots=[root], session_store=store)

            result = agent.handle("show files")

            self.assertEqual(result.tool_calls[0].name, "list_dir")
            self.assertIn("notes.md", result.response)

    def test_show_the_files_in_path_extracts_only_the_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nested = root / "resumes"
            nested.mkdir()
            (nested / "cv.txt").write_text("hello\n", encoding="utf-8")
            store = SessionStore(root / "sessions.sqlite3")
            agent = ReadOnlyAgent(roots=[root], session_store=store)

            result = agent.handle("show the files in resumes")

            self.assertEqual(result.tool_calls[0].name, "list_dir")
            self.assertEqual(result.tool_calls[0].arguments["path"], "resumes")
            self.assertIn("cv.txt", result.response)

    def test_incomplete_show_the_files_in_uses_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "notes.md").write_text("hello\n", encoding="utf-8")
            store = SessionStore(root / "sessions.sqlite3")
            agent = ReadOnlyAgent(roots=[root], session_store=store)

            result = agent.handle("show the files in")

            self.assertEqual(result.tool_calls[0].name, "list_dir")
            self.assertEqual(result.tool_calls[0].arguments["path"], ".")
            self.assertIn("notes.md", result.response)

    def test_explicit_file_mutation_requests_use_scoped_write_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            agent = ReadOnlyAgent(roots=[root], session_store=store)

            made_preview = agent.handle("create a folder called PDFs")
            self.assertEqual(made_preview.tool_calls[0].name, "prepare_write_plan")
            self.assertEqual(made_preview.tool_calls[0].status, "pending_approval")
            self.assertFalse((root / "PDFs").exists())
            made = agent.handle("yes", session_id=made_preview.session_id)

            written_preview = agent.handle(
                "create file PDFs/index.txt with content hello",
                session_id=made.session_id,
            )
            self.assertEqual(written_preview.tool_calls[0].name, "prepare_write_plan")
            written = agent.handle("yes", session_id=made.session_id)

            copied_preview = agent.handle(
                "copy PDFs/index.txt to PDFs/copy.txt",
                session_id=made.session_id,
            )
            self.assertEqual(copied_preview.tool_calls[0].name, "prepare_write_plan")
            copied = agent.handle("yes", session_id=made.session_id)

            edited_preview = agent.handle(
                "replace hello with hi in PDFs/copy.txt",
                session_id=made.session_id,
            )
            self.assertEqual(edited_preview.tool_calls[0].name, "prepare_write_plan")
            edited = agent.handle("yes", session_id=made.session_id)

            moved_preview = agent.handle(
                "move PDFs/copy.txt to moved.txt",
                session_id=made.session_id,
            )
            self.assertEqual(moved_preview.tool_calls[0].name, "prepare_write_plan")
            moved = agent.handle("yes", session_id=made.session_id)

            self.assertEqual(made.tool_calls[0].name, "make_dir")
            self.assertEqual(written.tool_calls[0].name, "write_file")
            self.assertEqual(copied.tool_calls[0].name, "copy_path")
            self.assertEqual(edited.tool_calls[0].name, "edit_file")
            self.assertEqual(moved.tool_calls[0].name, "move_path")
            self.assertEqual((root / "PDFs" / "index.txt").read_text(encoding="utf-8"), "hello")
            self.assertEqual((root / "moved.txt").read_text(encoding="utf-8"), "hi")
            self.assertFalse((root / "PDFs" / "copy.txt").exists())
            assistant_messages = [
                message
                for message in store.get_messages(moved.session_id)
                if message.role == "assistant"
            ]
            self.assertIn("command", assistant_messages[-1].metadata["tool_calls"][0])

    def test_pending_file_mutation_can_be_declined_or_redesigned(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            agent = ReadOnlyAgent(roots=[root], session_store=store)

            declined_preview = agent.handle("create a folder called Temp")
            declined = agent.handle("no", session_id=declined_preview.session_id)

            self.assertEqual(declined.tool_calls, [])
            self.assertIn("Cancelled", declined.response)
            self.assertFalse((root / "Temp").exists())

            original = agent.handle("create a folder called Old", session_id=declined.session_id)
            revised = agent.handle("no, create folder New instead", session_id=original.session_id)

            self.assertEqual(revised.tool_calls[0].name, "prepare_write_plan")
            self.assertFalse((root / "Old").exists())
            self.assertFalse((root / "New").exists())

            approved = agent.handle("yes", session_id=original.session_id)

            self.assertEqual(approved.tool_calls[0].name, "make_dir")
            self.assertFalse((root / "Old").exists())
            self.assertTrue((root / "New").is_dir())

    def test_llm_write_planner_creates_pending_command_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            client = SequenceChatClient(
                settings=ProviderSettings(
                    provider="test",
                    model="planner",
                    api_key=None,
                    base_url=None,
                ),
                responses=[
                    (
                        '{"commands":['
                        '{"name":"make_dir","arguments":{"path":"Reports"}},'
                        '{"name":"write_file","arguments":'
                        '{"path":"Reports/summary.txt","content":"hello"}}'
                        "]}"
                    ),
                ],
            )
            agent = ReadOnlyAgent(roots=[root], session_store=store, chat_client=client)

            preview = agent.handle(
                "create a folder called Reports and create file Reports/summary.txt"
            )

            self.assertEqual(preview.tool_calls[0].name, "prepare_write_plan")
            self.assertIn("Planned command(s):", preview.response)
            self.assertIn("create folder Reports", preview.response)
            self.assertFalse((root / "Reports").exists())

            approved = agent.handle("yes", session_id=preview.session_id)

            self.assertEqual(
                [call.name for call in approved.tool_calls],
                ["make_dir", "write_file"],
            )
            self.assertEqual(
                (root / "Reports" / "summary.txt").read_text(encoding="utf-8"),
                "hello",
            )

    def test_llm_pending_decision_can_redesign_unclear_negative_reply(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = SessionStore(root / "sessions.sqlite3")
            client = SequenceChatClient(
                settings=ProviderSettings(
                    provider="test",
                    model="planner",
                    api_key=None,
                    base_url=None,
                ),
                responses=[
                    '{"commands":[{"name":"make_dir","arguments":{"path":"Old"}}]}',
                    '{"decision":"revise","revision":"create folder Reports"}',
                    '{"commands":[{"name":"make_dir","arguments":{"path":"Reports"}}]}',
                ],
            )
            agent = ReadOnlyAgent(roots=[root], session_store=store, chat_client=client)

            preview = agent.handle("create a folder called Old")
            revised = agent.handle(
                "I do not like that name; use Reports instead",
                session_id=preview.session_id,
            )

            self.assertEqual(revised.tool_calls[0].name, "prepare_write_plan")
            self.assertIn("create folder Reports", revised.response)
            self.assertFalse((root / "Old").exists())
            self.assertFalse((root / "Reports").exists())

            approved = agent.handle("yes", session_id=preview.session_id)

            self.assertEqual(approved.tool_calls[0].name, "make_dir")
            self.assertFalse((root / "Old").exists())
            self.assertTrue((root / "Reports").is_dir())

    def test_polite_rename_and_edit_requests_emit_runtime_events(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "draft.txt").write_text("old text\n", encoding="utf-8")
            store = SessionStore(root / "sessions.sqlite3")
            runtime_events = []
            agent = ReadOnlyAgent(
                roots=[root],
                session_store=store,
                runtime_callback=runtime_events.append,
            )

            rename_preview = agent.handle("Could you rename file draft.txt as final.txt?")
            self.assertEqual(rename_preview.tool_calls[0].name, "prepare_write_plan")
            self.assertFalse((root / "final.txt").exists())
            renamed = agent.handle("yes", session_id=rename_preview.session_id)

            edit_preview = agent.handle(
                "Please edit file final.txt replacing old with new",
                session_id=renamed.session_id,
            )
            self.assertEqual(edit_preview.tool_calls[0].name, "prepare_write_plan")
            edited = agent.handle("yes", session_id=renamed.session_id)

            self.assertEqual(renamed.tool_calls[0].name, "move_path")
            self.assertEqual(edited.tool_calls[0].name, "edit_file")
            self.assertEqual((root / "final.txt").read_text(encoding="utf-8"), "new text\n")
            self.assertEqual(
                [event["type"] for event in runtime_events],
                ["tool_started", "tool_completed", "tool_started", "tool_completed"],
            )

    def test_organize_request_creates_preview_without_moving_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            root = workspace / "approved"
            root.mkdir()
            (root / "report.txt").write_text("draft\n", encoding="utf-8")
            sessions = SessionStore(workspace / "sessions.sqlite3")
            safety = SafetyStore(workspace / "safety.sqlite3")
            safety.approve_root(root)
            agent = ReadOnlyAgent(
                roots=[root],
                session_store=sessions,
                safety_store=safety,
            )

            result = agent.handle("organize these files by type")

            self.assertEqual(result.tool_calls[0].name, "organize_plan")
            self.assertIn("Nothing has been moved yet.", result.response)
            self.assertIn("Reply yes/approve", result.response)
            self.assertTrue((root / "report.txt").exists())
            self.assertFalse((root / "Documents" / "report.txt").exists())
            plan_id = result.tool_calls[0].result["plan_id"]
            self.assertEqual(safety.get_organize_plan(plan_id)["status"], "planned")

            applied = agent.handle("yes", session_id=result.session_id)

            self.assertEqual(applied.tool_calls[0].name, "organize_apply")
            self.assertEqual(safety.get_organize_plan(plan_id)["status"], "applied")
            self.assertFalse((root / "report.txt").exists())
            self.assertTrue((root / "Documents" / "report.txt").exists())

    def test_organize_redesign_can_change_to_filename_image_filter(self):
        with tempfile.TemporaryDirectory() as tmp:
            workspace = Path(tmp)
            root = workspace / "approved"
            root.mkdir()
            (root / "party_001.jpg").write_text("jpg\n", encoding="utf-8")
            (root / "party_002.png").write_text("png\n", encoding="utf-8")
            (root / "report.txt").write_text("draft\n", encoding="utf-8")
            (root / ".DS_Store").write_text("metadata\n", encoding="utf-8")
            sessions = SessionStore(workspace / "sessions.sqlite3")
            safety = SafetyStore(workspace / "safety.sqlite3")
            safety.approve_root(root)
            agent = ReadOnlyAgent(
                roots=[root],
                session_store=sessions,
                safety_store=safety,
            )

            first = agent.handle("arrange all the files in different sub-folders")
            revised = agent.handle(
                "no, distribute the images files as per their filename into different sub-folders",
                session_id=first.session_id,
            )

            self.assertEqual(revised.tool_calls[0].name, "organize_plan")
            plan = revised.tool_calls[0].result
            moves = {action["from"]: action["to"] for action in plan["actions"] if action["op"] == "move"}

            self.assertEqual(plan["strategy"], "by_filename")
            self.assertEqual(plan["file_filter"], "image")
            self.assertIn("using by_filename for image files", revised.response)
            self.assertEqual(
                moves,
                {
                    "party_001.jpg": "Images/party/party_001.jpg",
                    "party_002.png": "Images/party/party_002.png",
                },
            )
            self.assertNotIn(".DS_Store", moves)
            self.assertNotIn("report.txt", moves)
            self.assertTrue((root / "party_001.jpg").exists())
            self.assertFalse((root / "Images" / "party").exists())


if __name__ == "__main__":
    unittest.main()
