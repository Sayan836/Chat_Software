# Phase 3 verification report — LangGraph orchestration audit

**Date:** 2026-08-05
**Method:** full-suite run + 27 executable probes against a clean install of the current tree.
Every finding below has a reproduction that was actually run. Nothing here is inferred from reading.

---

## 0. Verdict

The **orchestration skeleton is sound and the safety machinery is genuinely solid** — propose→commit,
apply/undo, partial-failure recovery and scope enforcement all survived every attack I threw at them.

But Phase 3 is **not** in the state the summary implies. Against the six locked decisions:

| Decision | Status |
|---|---|
| D1 raw `StateGraph`, no `create_agent` | ✅ Done |
| D2 private-state subgraphs + wrapper nodes | ✅ Done, and correct — no reducer duplication |
| D3 apply gate stays HTTP propose→commit | ✅ Done (plus a nice conversational approve/decline layer) |
| D4 `SqliteSaver` checkpointer | ❌ **Broken and unused** — see F1, F2 |
| D5 `get_stream_writer()` replaces `runtime_callback` | ❌ Not started |
| D6 semantic File Planner (`by_topic`) | ❌ Not built — a deterministic `by_filename` shipped instead |

**One blocker: CI is red right now.** Details in F1.

---

## 1. Test suite baseline

Clean install, `python -m pytest backend/tests`:

```
1 failed, 86 passed in 2.50s
```

The 86 passes are real. The single failure is F1.

---

## 2. Findings

### F1 — BLOCKER · `build_checkpointer()` returns a context manager, not a checkpointer

`agent/graph_runtime.py` ends with:

```python
return SqliteSaver.from_conn_string(str(graph_checkpoint_path()))
```

`SqliteSaver.from_conn_string` is a `@classmethod @contextmanager` — it *yields* a saver, it does not
return one. The call returns `contextlib._GeneratorContextManager`.

**Reproduction** — `test_graph_runtime.py::test_sqlite_checkpointer_round_trip_when_installed`:

```
AssertionError: <contextlib._GeneratorContextManager object ...>
  is not an instance of <class 'langgraph.checkpoint.sqlite.SqliteSaver'>
```

**Why it wasn't caught:** the test opens with `except ImportError: self.skipTest(...)`, and
`langgraph-checkpoint-sqlite` is **not installed in `app/.venv`** — only `base`, `memory`, `serde` are
present. Locally the test skips. But `pyproject.toml` now lists it as a hard dependency and CI runs
`pip install -e ".[test]"`, so **CI installs it, the test runs, and it fails** on all six matrix legs.

**Fix** — the context manager must be opened and its lifetime owned by the app:

```python
def build_checkpointer() -> Any:
    if str(state_dir()) == ":memory:":
        from langgraph.checkpoint.memory import InMemorySaver
        return InMemorySaver()
    from langgraph.checkpoint.sqlite import SqliteSaver
    ensure_state_dir()
    conn = sqlite3.connect(str(graph_checkpoint_path()), check_same_thread=False)
    return SqliteSaver(conn)          # construct directly; caller owns `conn`
```

`check_same_thread=False` is required — `/chat/stream` runs the graph on a worker thread.

### F2 — HIGH · The root graph is compiled without a checkpointer, so D4 buys nothing

`root_graph.py` ends `return graph.compile()` — no `checkpointer=`. Verified at runtime:

```
=== P1: is the root graph compiled with a checkpointer? ===
  checkpointer: None
  -> interrupts possible: False
```

So `build_checkpointer()` is exercised **only by its own unit test**. Nothing in the request path
persists graph state. Consequences: no `interrupt()`, no resume, no time-travel, no durability —
Phase 3.4 cannot be built on this as it stands.

This is not urgent to *use*, but it is urgent to be honest about: the checkpointer is currently
decoration. Either wire it (`compile(checkpointer=..., )` plus a stable `thread_id` per session, which
is what `session_id` already is) or delete `graph_runtime.py` until 3.4 needs it. Shipping a broken,
unused component is the worst of the three options.

Related: `agent/langgraph_app.py` is now **dead code** — nothing imports `build_langgraph_app`.

### F3 — HIGH · Routing sends ordinary chat into the filesystem, and it never recovers

`routing.requires_scoped_tool()` ends with a bare substring sweep:

```python
or any(word in lowered for word in ["find", "search", "summarize", "about"])
```

plus `looks_like_file_count_request()` firing on "how many". Measured:

```
=== P3 ===
  'tell me about python decorators'      -> file_ops (deterministic file intent)
  'what do you know about the weather'   -> file_ops (deterministic file intent)
  'summarize the plot of Hamlet'         -> file_ops (deterministic file intent)
  'how many people find this hard?'      -> file_ops (deterministic file intent)
```

The `after_file_ops` conditional edge is meant to catch this, but it only fires on
`CHAT_FALLBACK_RESPONSE`. Both real paths dodge it:

**With an approved root** — the message becomes a grep over the user's files:

```
=== P8 ===
  'tell me about python decorators' -> "No readable files matched 'python decorators'
                                        inside the approved root"   tools=1
  'summarize the plot of Hamlet'    -> "No readable files matched 'the plot of Hamlet'
                                        inside the approved root"   tools=1
```

**With no root** — it's swallowed by the root-required pending branch:

```
=== P4 ===
  'tell me about python decorators' -> 'I can do that after a file root is approved
                                        and attached to this chat.'
```

Either way the user never gets an answer. `grep_files` returned a *successful* result, so nothing
signals fallback.

**Fix, two parts:**
1. Require co-occurrence, not bare presence — `("find"|"search") AND ("file"|"folder"|"in <path>"|a
   glob)`. `looks_like_file_count_request` should likewise require a file/extension word, not just
   "how many".
2. Make the fallback result-driven, not sentinel-driven: if the only tool run was `grep_files`/
   `glob_files` and it returned zero matches, route to `chat` instead of `respond`. That closes the
   class of bug rather than one instance.

### F4 — MEDIUM · Undo does not fully restore the tree: implicit parent directories are orphaned

The planner records a `mkdir` action for the **immediate** parent only (`planner.py:184`), while
`make_dir()` uses `mkdir(parents=True)` (`file_ops_write.py:25`). Intermediate directories are created
but never journaled, so undo's `rmdir` never removes them.

**Reproduction (P18)** — four files, `by_filename`:

```
  mkdir actions in plan: ['By Name/a', 'By Name/b', 'By Name/invoice']
  restored exactly: False
  leftover: ['By Name']
```

Confirmed end-to-end through the HTTP API as well (P25): after `/organize/plan` → `/organize/apply`
→ `/organize/undo`, `By Name/` remains.

Affects every strategy with nesting — so **`by_date` too** (`2026/07-July/`), meaning this is a
pre-existing Phase 2 defect that `by_filename` merely made easy to see. No data loss, but it breaks
the "reversible" promise the undo journal exists to make.

**Fix:** in the planner, walk each target's ancestor chain and emit a `mkdir` action for **every**
missing directory, deepest-last. The executor's existing reverse-order `rmdir` then unwinds correctly,
and `remove_empty_dir` already no-ops safely on non-empty dirs.

### F5 — MEDIUM · `_safe_folder_name` accepts Windows reserved device names

```
=== P14 ===
  '../../etc'  -> 'etc'        ✅ traversal stripped
  'a/b'        -> 'a-b'        ✅ separators stripped
  '..'         -> 'unnamed'    ✅
  'CON'        -> 'CON'        ❌
  'COM1'       -> 'COM1'       ❌
```

And the planner will happily emit them (P15): `mkdir By Name/CON`, `mkdir By Name/nul`. On Windows,
`CON`, `PRN`, `AUX`, `NUL`, `COM1-9`, `LPT1-9` are reserved device names and directory creation fails.
Windows is a v1 target platform and CI runs `windows-latest`, so this is a real cross-platform break —
it just needs a file named `CON.txt` to trigger.

**Fix:** reject the reserved set (case-insensitive, also when followed by an extension) and any
trailing dot or space, mapping to `unnamed` as the existing code already does for `..`.

Minor, same function: non-ASCII is mangled rather than preserved — `réservé 🙂` → `r-serv`. For
clerical/legal users working in non-English languages that's a usability problem worth revisiting.

### F6 — MEDIUM · `by_filename` and `file_filter` are unreachable from the browser UI

Both work in the planner and through `/organize/plan` (P25 verified apply+undo). But
`web/index.html:145` still offers only:

```html
<option value="by_type">By type</option>
<option value="by_date">By date</option>
```

and `organizePayload()` in `web/app.js` never sends `file_filter`. Since the browser UI is the *only*
UI in scope this phase, the feature effectively doesn't exist for users — it's reachable solely by
typing a chat message that hits `_extract_organize_strategy`.

### F7 — MEDIUM · D5 not started; subgraph activity is invisible

Streaming is still `runtime_callback` + thread + `Queue`. No `get_stream_writer`, no `stream_mode`.
The event sequence is correct today (P24: `chat_started, tool_started, tool_completed, chat_completed`),
so nothing is broken — but tool events from inside subgraphs bypass the graph entirely, so as soon as
a second agent is added its work will not appear in the console.

### F8 — MEDIUM · D6 / Phase 3.3 not implemented

No `organize/semantic/` package. `SUPPORTED_STRATEGIES = {by_type, by_date, by_filename}` — no
`by_topic`, no LLM taxonomy, no structured output, no `Send` fan-out. `by_filename` is deterministic
token grouping, which is a reasonable increment and cheap to trust, but it is not the semantic File
Planner the phase was about. Worth stating plainly so the roadmap stays honest.

### F9 — LOW · `SessionStore(":memory:")` is broken while `SafetyStore(":memory:")` works

```
=== P6 ===
  SessionStore(':memory:') BROKEN -> OperationalError: no such table: sessions
  SafetyStore(':memory:')  OK
```

`SafetyStore` keeps a persistent `_memory_conn` for `:memory:` (`safety_store.py:34-39`);
`SessionStore` opens a fresh connection per call, so each one gets an empty database. Compounding it,
`build_checkpointer()` treats `CHAT_SOFTWARE_STATE_DIR=":memory:"` as a special value while
`session_store()`/`safety_store()` join it as a path — set that env var and you get an in-memory
checkpointer plus real files under `./:memory:/`. Pick one convention.

### F10 — LOW · Root graph rebuilt on every message

`_handle_with_graph` calls `build_root_graph(self)` per turn: registry lookup, subgraph builds, and a
`StateGraph.compile()` each time. Measured **1.9 ms/message** (P2) — negligible now, but it also means
a checkpointer attached later would never see a stable graph identity. Build once per `AgentDeps` and
cache.

---

## 3. What held up under attack

Worth recording, because this is the part that matters most and it passed cleanly:

| Probe | Result |
|---|---|
| P15 — adversarial filenames (`../`, `..\`, 200-char, emoji, reserved) | **0** destinations escaped the root |
| P17 — plan targeting a path outside the approved root | `PermissionError`, rejected |
| P26 — unapproved root against `/files/list`, `/organize/plan`, `/chat` | 400 on all three |
| P19 — double-apply, double-undo, re-apply after undo | idempotent; re-apply correctly refused |
| P20 — file deleted between plan and apply | `partial`, journal intact, undo still succeeded |
| P21 — target folder name occupied by a file | warned and skipped, no clobber |
| P5 — empty / whitespace / 8k chars / emoji / null byte | all handled, no crash |
| P11 — 8 concurrent messages on one session | 0 errors |
| D2 wrapper pattern | correct — `RootState` has no reducer fields, so the duplication trap is avoided |
| P9 — conversational approve / decline / revise | works, and is a genuinely good addition beyond the plan |

---

## 4. Recommended order

1. **F1** — fix `build_checkpointer`, unskip the test. CI is red until this lands. *(~15 min)*
2. **F3** — routing precision + result-driven fallback. Biggest user-visible defect. *(~2 h)*
3. **F4** — journal every intermediate `mkdir`. Restores the reversibility guarantee. *(~1 h)*
4. **F5** — reserved-name rejection, with a Windows CI test. *(~30 min)*
5. **F6** — expose `by_filename` + `file_filter` in the browser UI. *(~30 min)*
6. **F2** — decide: wire the checkpointer with `thread_id = session_id`, or remove it until 3.4.
7. **F9, F10** — housekeeping.
8. **F7 (D5), F8 (D6)** — the remaining planned phases.

Items 1–5 are roughly a day and would put the tree in the state the Phase 3 summary describes.

---

## Appendix — reproducing

The 27 probes were run against a clean copy of `app/` with dependencies installed fresh:

```bash
pip install -e ".[test]" langgraph-checkpoint-sqlite
python -m pytest backend/tests -q -rs      # -> 1 failed, 86 passed
```

The failing test is the one that matters; it is currently masked on any machine where
`langgraph-checkpoint-sqlite` is absent, which includes `app/.venv`.
