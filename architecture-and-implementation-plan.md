# Chat_Software — Architecture & Implementation Plan

*A private, local-first conversational agent for file management, clerical work, and development tasks (macOS, Linux, Windows). File management is the headline workflow; general chat, document work, and coding assistance ride on the same safe agent core. Local model by default (Ollama), optional cloud (OpenAI, Claude). Preview + confirm on every destructive action, with undo. Built-in private web search + download subsystem, and role-based onboarding (e.g. "clerical assistant", "developer assistant", "legal assistant") that configures skills and optional offline domain knowledge packs — designed for non-technical and technical users alike.*

Version 0.2 · 2026-07-05 · Finalized refinement grounded in current (mid-2026) sources; see Sources.

---

## 0. The one constraint that shapes everything

You asked to "check the Claude Code repo and copy/reference it" for the agent architecture and MCP. Here's the load-bearing finding, and it changes the plan:

**The Claude Agent SDK gives you Claude Code's agent loop, tools, hooks, permissions, subagents, and MCP integration as a library — but it only runs Claude models** (Anthropic API, Bedrock, or Vertex). It cannot drive Ollama or OpenAI. Since your product is *offline-first on Ollama by default*, the Agent SDK **cannot be your runtime engine**.

So the decision is: **reference its design, don't adopt its engine.** We replicate the patterns that make it good — the gather→act→verify loop, a typed tool registry, pre/post-tool hooks, an explicit permission gate, isolated subagents, resumable sessions, and MCP as the tool-transport standard — on top of a **provider-agnostic loop** we control. This is the honest reading of "take reference from there," and it's the only design consistent with your locked decisions.

Everything below builds on that.

---

## 1. Reference architectures — what we borrow from each

| Source | What it proves / gives us | What we take |
|---|---|---|
| **Claude Agent SDK / Claude Code** | Production agent loop: *gather context → take action → verify → repeat*; 20+ built-in tools (Read/Write/Edit/Glob/Grep/Bash); `PreToolUse`/`PostToolUse` hooks; permission modes; subagents with isolated context; resumable sessions (JSONL); MCP via stdio/HTTP | The *pattern* for loop, tools, hooks, permissions, subagents, sessions — re-implemented provider-agnostically |
| **LangGraph 1.2** | Agent run as a durable, checkpointed graph; `interrupt()` pauses for human approval and resumes exactly where it left off | Our **agent loop engine** + the mechanism for **preview + confirm** and pause/resume sessions |
| **LiteLLM** | One OpenAI-format interface across OpenAI, Anthropic, Ollama, Bedrock, etc.; normalizes tool-calling & structured outputs (incl. Anthropic's format transforms) | Our **provider abstraction layer** — implemented as a new project module, optionally borrowing patterns from any prior `llm_provider.py` code |
| **langchain-mcp-adapters** (`MultiServerMCPClient`) | Converts MCP server tools into LangChain/LangGraph tools; multi-server, stdio + HTTP | Our **MCP tool loader** — file tools and future connectors speak one protocol |
| **AnythingLLM** | Single-user desktop app (Mac/Win/Linux) bundling LLM engine + CPU embedder + LanceDB; drag-drop ingest, chunking, vector store, agents | Reference for **desktop packaging + built-in RAG** with bundled LanceDB |
| **Jan / LibreChat** | Privacy-first desktop chat over local *or* cloud providers; LibreChat was early to full **MCP** support | Reference for **provider-switching UX** and MCP-in-a-chat-app |
| **Local-File-Organizer / AI File Sorter** | 100% on-device organize/rename via small models (Llama 3.2 3B + LLaVA); dry-run/preview, local-or-remote toggle | Reference for the **file-management core** and preview-based, user-controlled changes |

---

## 1b. Local implementation reference map

Keep these local sources in mind when coding starts. They are implementation inputs, not all direct dependencies.

| Local path | Use | Pull from it | Guardrail |
|---|---|---|---|
| `/Users/sayansen/Downloads/edur/utils/llm_provider.py` | **Provider seed** | Existing `get_llm_provider()`, `resolve_llm_model()`, OpenAI-compatible Ollama/OpenAI client construction, `ChatOpenAI` builder | Migrate, don't paste unchanged: v1 defaults to Ollama, removes `utils.settings` coupling, adds Claude through LiteLLM, and adds model capability metadata |
| `/Users/sayansen/Documents/Chat_Software/reference/claude-agent-sdk` | Agent-loop reference | Gather→act→verify loop, hooks, permissions, subagents, session patterns | Claude-only runtime; use patterns, not engine |
| `/Users/sayansen/Documents/Chat_Software/reference/mcp-servers` | MCP/file tools reference | `src/filesystem/` as the closest blueprint for File-Ops MCP tools | Keep our dry-run/permission/audit contract stricter than the reference |
| `/Users/sayansen/Documents/Chat_Software/reference/local-file-organizer` | File-management core reference | On-device organize/rename flow, parsers, prompts, dry-run behavior | Copy only license-compatible snippets with attribution |
| `/Users/sayansen/Documents/Chat_Software/reference/anything-llm` | Desktop RAG reference | LanceDB-backed ingest/chunk/embed/search pipeline, provider bridge, packaging ideas | Reuse architecture selectively; don't inherit server assumptions blindly |
| `/Users/sayansen/Documents/Chat_Software/reference/librechat` | Chat/provider/MCP reference | Multi-provider chat UX, MCP integration, conversations/sessions | Web-app patterns need adaptation to Tauri desktop |
| `/Users/sayansen/Documents/Chat_Software/reference-repos.md` and `/Users/sayansen/Documents/Chat_Software/reference/README.md` | Canonical map | What is cloned, dependency vs reference vs service, license cautions | `reference/` is study-only and should stay out of shipped source |

---

## 2. System architecture (high level)

```
┌───────────────────────────────────────────────────────────────┐
│  DESKTOP SHELL  (Tauri 2.0 — WebView UI + Rust core)           │
│  Chat · files · preview/approve · jobs · settings              │
└───────────────▲───────────────────────────────┬───────────────┘
                │ local IPC (websocket / HTTP)   │
┌───────────────┴───────────────────────────────▼───────────────┐
│  AGENT SERVICE  (Python sidecar, bundled)                      │
│                                                               │
│  ┌─────────────┐   ┌──────────────────┐   ┌────────────────┐  │
│  │ Provider     │   │ Agent Core        │   │ Safety /       │  │
│  │ Layer        │◄─►│ (LangGraph)       │◄─►│ Permission     │  │
│  │ LiteLLM +    │   │ ReAct loop,       │   │ Gate           │  │
│  │ model config │   │ interrupt(),      │   │ preview·confirm │  │
│  │ Ollama/OAI/  │   │ checkpointer,     │   │ ·undo·audit     │  │
│  │ Claude       │   │ subagents         │   │ (hooks)        │  │
│  └─────────────┘   └────────┬─────────┘   └────────────────┘  │
│                             │ tool calls                      │
│                ┌────────────▼──────────────┐                  │
│                │ Tool Layer (MCP + native) │                  │
│                │ file ops · search · RAG   │                  │
│                └────────┬─────────┬────────┘                  │
│         ┌───────────────▼───┐ ┌───▼──────────────────┐        │
│         │ File-Ops MCP srv   │ │ Index/RAG subsystem   │        │
│         │ read/write/move/   │ │ watcher · parser ·    │        │
│         │ rename/delete/glob │ │ OCR · embeddings ·    │        │
│         │ (dry-run aware)    │ │ LanceDB/sqlite-vec    │        │
│         └────────────────────┘ └──────────────────────┘        │
└───────────────────────────────────────────────────────────────┘
        all local · no network unless user enables cloud provider
```

**Why a Python sidecar rather than a pure-Rust agent:** the richest agent ecosystem (LangGraph, LiteLLM, langchain-mcp-adapters, MCP SDK, document parsers, embedding libs) is Python. Tauri can bundle and spawn a Python process as a sidecar and talk to it over local websocket/HTTP. You get Tauri's small footprint and secure file access on the UI side, and Python's AI ecosystem on the engine side. This is now the finalized topology; Electron/Node remains only a fallback if Tauri packaging becomes the blocker.

---

## 3. Provider abstraction layer

Start this new project with a provider module seeded from `/Users/sayansen/Downloads/edur/utils/llm_provider.py`: env/user-settings driven provider selection, an OpenAI-compatible path for Ollama/OpenAI where possible, model resolution via provider-specific env vars, and a normalized chat-client builder. Two gaps to close for the product:

1. **Claude isn't OpenAI-shaped over the same path.** Anthropic's native API differs (message format, tool-call schema, structured-output headers). Two clean ways to add it:
   - **Recommended — route everything through LiteLLM.** One unified OpenAI-format call site for `ollama/*`, `openai/*`, and `anthropic/*`; LiteLLM transforms Anthropic's tool-calling/structured-output format for you. Keeps a single code path and simplifies tool-calling normalization across providers — the hardest part of multi-provider agents.
   - Alternative — add `langchain_anthropic.ChatAnthropic` as a third branch in `build_chat_llm()`. Simpler dependency, but you now maintain per-provider tool-calling quirks yourself.

2. **Capability metadata per model.** The agent must know each model's tool-calling reliability and context limit, because they differ sharply (see §7). Extend the provider config so each model carries `{supports_tools, tool_reliability, context_window, is_local}` — the agent core uses this to decide how much autonomy to allow and when to suggest escalating to a cloud model.

**Resulting shape** (evolution of your file):
```
provider/
  config.py        # model registry: provider, model, capabilities, is_local
  factory.py       # get_client() -> unified LiteLLM/LangChain chat model
  router.py        # picks model per task + hardware; local-first, cloud opt-in
```
Default provider is `ollama` (unlike the seed file's current OpenAI default). The default local model should be selected by a small tool-call reliability bakeoff on target hardware, with `qwen3:8b` / `qwen2.5:7b`-class models as the first candidates and `llama3.1` as a fallback. `openai` and `anthropic` are opt-in via user-supplied keys.

---

## 4. Agent core (LangGraph)

The loop is a *solved problem*; the value is the engineering around it — context management, safety, cost. LangGraph gives us that as a durable graph instead of a fragile `while` loop.

- **Loop:** a ReAct-style graph — `plan → select tool → (gate) → execute → observe → repeat → respond`. Start from LangGraph's prebuilt `create_react_agent`, then customize.
- **Preview + confirm (your locked decision) = LangGraph `interrupt()`.** Every *write* tool (move/rename/delete/write) calls `interrupt()` with a payload describing the proposed change (source → dest, diff, affected files). The graph pauses; the Tauri UI renders an approve/reject panel; on `Command(resume="approve")` it continues, on reject it skips. This is exactly the human-in-the-loop pattern LangGraph is built for, and it maps 1:1 to Claude Code's `PreToolUse` gate.
- **Sessions:** LangGraph checkpointer snapshots every state transition to local SQLite → resumable, forkable sessions (mirrors the Agent SDK's session resume) and survives app restarts.
- **Subagents:** for heavy jobs (e.g., "re-index and reorganize 10k files"), spawn a focused sub-graph with its own context so intermediate tool spam doesn't pollute the main conversation — the Agent SDK subagent pattern.
- **Context management:** compact/summarize when nearing the model's context window; keep offline agent loops short and verifiable because small local models degrade on long context (§7).

---

## 5. Tool layer & MCP

**Design principle: every tool is an MCP tool.** Build file operations as a local **File-Ops MCP server** (stdio) and load it — plus any future connectors — through `MultiServerMCPClient.get_tools()`, which converts them to LangGraph tools. Benefits: the same tools work if you later swap the loop, third-party MCP servers plug in for free, and you match the emerging industry standard.

**Core tool set (v1):**

| Category | Tools | Notes |
|---|---|---|
| Read/nav | `read_file`, `glob`, `grep`, `list_dir`, `stat` | Safe, auto-approved |
| Write (gated) | `write_file`, `edit_file`, `move`, `rename`, `delete`, `mkdir` | Every call is **dry-run first** → `interrupt()` → apply |
| Search | `semantic_search`, `find_similar`, `find_duplicates` | Backed by RAG subsystem (§6) |
| Understand | `summarize_file`, `classify`, `extract_metadata`, `describe_image` (VLM) | Uses local model / LLaVA-class VLM |
| Organize | `propose_organization`, `apply_plan` | Generates a full move/rename plan, previewed as one batch |
| Development | `inspect_repo`, `git_status`, `git_diff`, `run_command`, `run_tests` | Scoped to chosen workspaces; commands are permission-gated and audited |

**Dry-run contract:** write tools accept `dry_run: bool`. In dry-run they return the *would-be* effect (no disk change); the gate shows it; approval re-invokes with `dry_run=false`. This is the AI File Sorter / Local-File-Organizer preview pattern, enforced at the tool boundary so no code path can bypass it.

---

## 6. Indexing & RAG subsystem (the offline "understand your files" engine)

- **Watcher:** filesystem watcher tracks the user's chosen folders; incremental re-index on change.
- **Parsers:** PDF (PyMuPDF), DOCX (python-docx), pptx/xlsx/csv/md/txt; **OCR** via Tesseract for scans/images; VLM (LLaVA-class) for image content.
- **Chunking + embeddings:** local embedding model (sentence-transformers class) — no network.
- **Vector store:** **LanceDB** (embedded, columnar, memory-maps, scales past ~200k vectors) as default; **sqlite-vec** if you prefer a single-file store for smaller libraries. (AnythingLLM ships LanceDB bundled — validated choice.)
- **Metadata store:** SQLite for file metadata, tags, hashes (duplicate detection), and the audit log.

This subsystem powers `semantic_search`, dedupe, and "find the Acme contract from last quarter" — all offline.

---

## 6b. Web search & acquisition subsystem

You want the app to have its *own* strong web search and downloading — not lean on a cloud provider's search. This is very doable and fits the local-first ethos.

- **Private search engine — SearXNG (bundled/local).** SearXNG is a free, open-source **metasearch** engine that aggregates 250+ sources, keeps no user data, exposes a clean **JSON API**, needs **no API keys**, and runs locally on the desktop. It already integrates with LangChain, Ollama/Open WebUI, and has an MCP server. This becomes our `web_search` tool — private by default, and the app's search doesn't depend on OpenAI/Anthropic. *(It requires network at query time; it is "private," not "offline." Search results are live.)*
- **Fetch & read — `web_fetch`.** Retrieve a page and extract clean text/markdown for the agent to reason over (Trafilatura/Readability-class extraction). For JS-heavy pages, an optional headless-browser fetch.
- **Crawl — `site_crawl`** (bounded): follow links within a domain up to a depth/page cap to gather a small corpus (e.g., a statute site), feeding straight into the RAG index (§6).
- **Download manager — `download_file`.** A real, resumable download subsystem (concurrent, checksum-verified, resume-capable) that saves into the user's scoped folders. **Downloads are gated by the same preview + confirm** as file writes — the agent proposes what it will fetch and where it lands; the user approves. Large/bulk downloads (knowledge packs, §6c) run as tracked background jobs with progress in the UI.

All four are exposed as MCP tools, so they sit behind the same permission gate and audit log as everything else. When the app is offline, web tools are cleanly disabled and the agent falls back to local RAG.

## 6c. Role-based onboarding & domain knowledge packs

This is the feature that makes the product usable by non-technical people (clerical, legal, and beyond). On first launch, the app asks **"What's your role?"** and configures itself accordingly — skills, prompts, default folders, and optional domain data. The pattern is proven: role-based AI *persona packs* already exist for coding assistants; we generalize it to professions.

**Three things a role configures:**

1. **A Skill Pack** — a bundle of task templates the agent loads for that role (the Agent SDK / Claude Code `SKILL.md` model: a folder of skills the agent invokes automatically or by name). Examples:
   - *Legal assistant:* draft/redline contracts, summarize a case, build a cite list, organize a matter folder, extract clauses, compare two contracts.
   - *Clerical/admin:* sort & rename incoming documents, extract fields from forms, build indexes, batch-rename scans, file by client/date.
   - *Developer assistant:* inspect a repo, explain code, edit files, run tests, summarize diffs, prepare commits/PR notes.
   - Each pack ships as data (Markdown + prompts), so adding a profession is content, not code.
2. **Defaults & UX tone** — folder conventions, output formats (e.g., .docx for legal), and plain-language explanations suited to non-technical users (no jargon, preview everything, one clear action at a time).
3. **An optional Domain Knowledge Pack** — a curated, downloadable corpus indexed into local RAG so the agent can answer with citations *offline* after the initial download.

**Reality check on "download all the laws at the start" (grounded, important):**
Downloading *everything* is not practical, but curated packs absolutely are. Concrete facts:
- **CourtListener / Free Law Project** offers **bulk US case law** (9M+ decisions from 2,000+ courts) as public-domain data (marked "free of known copyright restrictions"), regenerated quarterly. But it's huge — the opinions text file is very large and the precomputed **embeddings alone are ~2 TB**. You cannot ship "all of it" to a laptop by default.
- **Harvard Caselaw Access Project** (case.law) data is open and also mirrored as the `harvard-lil/cold-cases` dataset on Hugging Face.

**Design implication:** knowledge packs must be **curated and scoped**, chosen at onboarding, downloaded on demand with a clear size estimate and progress, then embedded locally:
- Scope by **jurisdiction + practice area** (e.g., "California landlord-tenant," "US federal tax") — tens/hundreds of MB, not terabytes.
- Prefer **statutes/regulations/high-value case law** over dumping every opinion; let the user add more packs later.
- Ship a **pack catalog** with size, source, license, and freshness date; download via the §6b manager; index into LanceDB.
- Always show provenance/citations and a "this may be outdated / not legal advice" disclaimer — critical for a legal audience.

So the onboarding flow becomes: **pick role → app installs the matching Skill Pack + defaults → offers relevant Knowledge Packs with sizes → downloads & indexes in the background → ready.** Roles ship as declarative manifests (`role.yaml` + skills + pack references), so new professions (accounting, medical-admin, HR, real estate) are added without touching the core.

## 7. Safety, permissions & the small-model reality

Grounded facts that drive the safety design:
- Small local models are **less reliable at tool calling** than cloud frontier models. Ollama classifies tool-calling as *Native (stable, 75%+)* for `qwen2.5:7b`+, `llama3.1`, `llama3.2`, `mistral-nemo`; weaker models fall to *partial/prompt (<50%)*. → **Prefer a known-good tool-calling model as the local default; verify tool-call JSON before executing.**
- **Long-context degradation** is real; keep loops short. → cap tool-call chains, summarize, escalate to cloud for big jobs (opt-in).

Safety mechanisms (all mirror Claude Code patterns):
1. **Permission gate** — read tools auto-approved; write tools always gated by `interrupt()`.
2. **Dry-run + diff preview** for every mutation; **batch preview** for multi-file plans.
3. **Undo/trash** — deletes go to an app-managed trash; moves/renames recorded in a reversible journal (never hard-delete in v1).
4. **PostTool audit hook** — append every action to a local audit log (the SDK's audit-hook example).
5. **Scoped roots** — the agent can only touch user-selected folders (Tauri capability model + server-side path validation). Cross-platform path handling is a first-class concern (Windows drive letters/locked files vs. POSIX).
6. **Command execution gate** — development commands run only in approved workspaces, show the command before execution, log stdout/stderr, and require stronger confirmation for package installs, network access, destructive shell commands, and destructive git operations.

---

## 8. Technology stack (recommended defaults)

| Layer | Choice | Why / alternative |
|---|---|---|
| Desktop shell | **Tauri 2.0** | ~10× smaller, ~5× less RAM than Electron; secure capability model. *Alt: Electron if staying all-JS.* |
| Agent engine | **LangGraph 1.2** | Durable graph, `interrupt()` HITL, checkpointing. *Alt: custom loop (more control, more work).* |
| Provider layer | **LiteLLM** inside a new `provider/` module | Unified OpenAI-format for Ollama/OpenAI/Claude + tool-call normalization. *Alt: LangChain per-provider classes.* |
| Local inference | **Ollama** | Native tool-calling parsers; easy model mgmt. *Alt: llama.cpp / LM Studio / Nexa SDK.* |
| MCP tools | **langchain-mcp-adapters** + **MCP Python SDK** | Load local + remote MCP servers as agent tools. |
| Vector store | **LanceDB** (embedded) | Bundled, scales; *alt: sqlite-vec, Chroma for prototypes.* |
| Metadata/sessions | **SQLite** | Audit, journal, checkpoints, file metadata. |
| Parsing/OCR | PyMuPDF, python-docx, Tesseract, LLaVA-class VLM | All offline. |
| Web search | **SearXNG** (bundled, local) | Private metasearch, JSON API, no keys; the app's own search. |
| Web extract/download | Trafilatura/Readability + resumable download mgr | Clean text extraction; gated, verifiable downloads. |
| Role/knowledge packs | Declarative `role.yaml` + `SKILL.md` + pack catalog | Professions as content; curated offline corpora (e.g. CourtListener/CAP public-domain legal data). |
| Language | **Python** agent + **TS/Rust** shell | Best AI ecosystem on the backend, secure lightweight desktop shell on the frontend. |

---

## 9. Finalized decisions before coding

1. **Shell + backend topology:** Tauri 2.0 shell + bundled Python sidecar is locked for v1.
2. **Provider integration:** LiteLLM is locked as the unified provider layer. Build a fresh `provider/` package for this project, seeded from `/Users/sayansen/Downloads/edur/utils/llm_provider.py`; import prior code only after adapting defaults, settings, Claude support, and model capability metadata.
3. **Loop engine:** LangGraph is locked as the durable agent engine. Claude Agent SDK / Claude Code remains a design reference, not the runtime.
4. **Tool transport:** MCP is locked for file, web, RAG, and development tools. Native helpers are allowed internally, but the agent-facing contract is MCP.
5. **Safety model:** preview + confirm, undo/trash, scoped roots, audit log, and command execution gates are non-negotiable v1 requirements.
6. **Web search deployment:** SearXNG local sidecar is the target. If packaging weight becomes painful, use lazy first-run install, not a hosted search API as the default.
7. **V1 role packs:** ship general file manager, clerical/admin, and developer assistant as core roles. Ship legal assistant as a beta role only if citations, disclaimers, and a scoped knowledge-pack source are ready.
8. **Still to validate, not re-decide:** the default local tool-calling model (`qwen3:8b` / `qwen2.5:7b`-class vs `llama3.1`) via a small benchmark on target hardware; the first legal knowledge-pack scope and license; packaging size for SearXNG + Python sidecars.

These decisions are now stable enough to start implementation. Capture any future reversal as an ADR, not an inline plan change.

---

## 10. Phased implementation plan

**Phase 0 — Foundations (1–2 wks)**
Repo scaffold (Tauri + Python sidecar, local websocket bridge). Create `provider/` by migrating the uploaded `llm_provider.py` seed into Ollama/OpenAI/**Claude** via LiteLLM + a model registry with capability metadata. Smoke test: chat round-trip through each provider. *Exit:* pick a provider in settings, get a reply, fully offline on Ollama.

**Phase 1 — Agent core + read-only tools (2–3 wks)**
LangGraph ReAct loop; File-Ops MCP server with **read/nav tools only** (read, glob, grep, list, stat) plus read-only development tools (`git_status`, `git_diff`) loaded via langchain-mcp-adapters; session checkpointing. *Exit:* "find files about X and summarize them" works end-to-end, no writes.

**Phase 2 — Write tools + preview/confirm (2–3 wks)**
Add gated write tools (move/rename/delete/write) with dry-run contract; add gated development commands (`run_command`, `run_tests`) for approved workspaces; wire `interrupt()` to a Tauri approve/reject + diff UI; undo journal + app trash; audit hook. *Exit:* "organize my Downloads by project" → preview plan → approve → applied → undoable, and "run tests for this repo" executes with logged approval.

**Phase 3 — Indexing & semantic search (2–3 wks)**
Filesystem watcher, parsers, OCR, local embeddings, LanceDB; `semantic_search`, `find_duplicates`, `describe_image`. *Exit:* natural-language search over local files, offline.

**Phase 4 — Web search & acquisition (2 wks)**
Bundle SearXNG as a local sidecar → `web_search`; `web_fetch` + text extraction; bounded `site_crawl`; resumable, gated `download_file` manager with background jobs. All as MCP tools behind the permission gate. *Exit:* the agent can research online and download files itself, privately, with approval.

**Phase 5 — Role-based onboarding + knowledge packs (2–3 wks)**
First-run role picker; declarative role manifests (`role.yaml` + Skill Packs) for general file manager, clerical/admin, developer assistant, and legal beta; pack catalog with sizes/sources/licenses; background download + local indexing of a first domain pack (start with one scoped legal pack only if provenance is solid, e.g. CourtListener/CAP public-domain data); legal disclaimers + citation UX. *Exit:* users get role-specific workflows, and the legal beta can answer cited questions offline after a scoped pack is installed.

**Phase 6 — Cloud opt-in + polish (1–2 wks)**
Bring-your-own-key for OpenAI/Claude; router escalates big/long jobs to cloud on user opt-in; scheduled auto-organization; settings. *Exit:* privacy-default with optional cloud boost.

**Phase 7 — Cross-platform hardening (ongoing)**
Windows CI (GitHub Actions runners) from Phase 0; path/permission/locked-file tests; packaging & signing for macOS/Linux/Windows (incl. bundling the SearXNG + Python sidecars across OSes); manual QA on a Windows box. *Exit:* installable, tested on all three OSes.

**Verification baked in:** each phase ships with tool-call reliability tests on the local default model, a destructive-action safety test (nothing mutates without approval; everything is undoable), and cross-platform path tests.

---

## 11. Key risks

- **Small-model tool-calling flakiness** → verify tool JSON, prefer known-good models, keep loops short, escalate on failure. *(Mitigated, not eliminated.)*
- **Sidecar packaging** (bundling Python across 3 OSes) → pin runtime, test in CI early; consider PyInstaller/embedded Python.
- **Windows file-system edge cases** (locked/in-use files, long paths, drive letters) → dedicated tests from Phase 0.
- **Destructive-action trust** → undo/trash + preview are non-negotiable; never hard-delete in v1.
- **Bundling SearXNG + Python sidecars across 3 OSes** → adds packaging weight; test the multi-sidecar bundle in CI from the start; consider lazy/first-run install of SearXNG.
- **Knowledge-pack size & freshness** → scope packs tightly (jurisdiction + practice area); show size before download; stamp packs with source date; schedule refresh. Never imply "complete" coverage.
- **Legal/professional liability** → mandatory citations + provenance + "not legal advice" disclaimers; the agent surfaces sources, it doesn't opine. Lock with a domain expert before shipping the legal role.
- **Download safety** → verify checksums, sandbox/scan downloads, and gate every download through preview/confirm; treat agent-initiated downloads as untrusted until approved.
- **Development-command safety** → command execution is powerful and must stay scoped, permissioned, logged, and easy to cancel; package installs and destructive git/shell operations require explicit approval.
- **Scope creep from "all-purpose" positioning** → keep the core agent stable and add professions as role packs; do not hard-code every workflow into the engine.

---

## Sources

- [Claude Agent SDK overview — Claude Code Docs](https://code.claude.com/docs/en/agent-sdk/overview)
- [anthropics/claude-agent-sdk-python — GitHub](https://github.com/anthropics/claude-agent-sdk-python)
- [Subagents in the SDK — Claude API Docs](https://platform.claude.com/docs/en/agent-sdk/subagents)
- [Claude Agent SDK: Agent Loops, Tool Calls — Augment Code](https://www.augmentcode.com/guides/claude-agent-sdk-agent-loops-tool-calls)
- [modelcontextprotocol/python-sdk — GitHub](https://github.com/modelcontextprotocol/python-sdk)
- [Build an MCP client — Model Context Protocol](https://modelcontextprotocol.io/docs/develop/build-client)
- [langchain-ai/langchain-mcp-adapters — GitHub](https://github.com/langchain-ai/langchain-mcp-adapters)
- [Model Context Protocol (MCP) — LangChain Docs](https://docs.langchain.com/oss/python/langchain/mcp)
- [Human-in-the-loop — LangChain Docs](https://docs.langchain.com/oss/python/langchain/human-in-the-loop)
- [From ReAct Loop to Production Agent: LangGraph Tutorial — Medium](https://medium.com/@mzeynali01/from-react-loop-to-production-agent-a-hands-on-langgraph-tutorial-ffd2649706ad)
- [LangGraph — LangChain](https://www.langchain.com/langgraph)
- [Anthropic — LiteLLM Docs](https://docs.litellm.ai/docs/providers/anthropic)
- [LiteLLM vs LangChain — TrueFoundry](https://www.truefoundry.com/blog/litellm-vs-langchain)
- [BerriAI/litellm — GitHub](https://github.com/BerriAI/litellm)
- [Tool calling — Ollama Docs](https://docs.ollama.com/capabilities/tool-calling)
- [Structured outputs — Ollama Blog](https://ollama.com/blog/structured-outputs)
- [AnythingLLM — DataCamp guide](https://www.datacamp.com/blog/anythingllm)
- [AnythingLLM Desktop Installation — Docs](https://docs.anythingllm.com/installation-desktop/overview)
- [LLM chat UIs that support MCP — ClickHouse](https://clickhouse.com/blog/llm-chat-mcp-support)
- [Jan / local LLM tools 2026 — Toolchase](https://toolchase.com/blog/best-local-llm-tools/)
- [QiuYannnn/Local-File-Organizer — GitHub](https://github.com/QiuYannnn/Local-File-Organizer)
- [hyperfield/ai-file-sorter — GitHub](https://github.com/hyperfield/ai-file-sorter)
- [Tauri vs Electron 2026 — Rustify](https://rustify.rs/articles/rust-tauri-vs-electron-2026)
- [SearXNG Documentation](https://searxng.org/)
- [SearXNG — Wikipedia](https://en.wikipedia.org/wiki/SearXNG)
- [SearXNG Search — LiteLLM Docs](https://docs.litellm.ai/docs/search/searxng)
- [agent-search: self-hosted search API + MCP bundling SearXNG — GitHub](https://github.com/brcrusoe72/agent-search)
- [Bulk Legal Data — CourtListener / Free Law Project](https://www.courtlistener.com/help/api/bulk-data/)
- [Full CourtListener API included with membership (2026) — Free Law Project](https://free.law/2026/05/07/api-included-in-memberships/)
- [Caselaw Access Project](https://case.law/)
- [harvard-lil/cold-cases dataset — Hugging Face](https://huggingface.co/datasets/harvard-lil/cold-cases)
- [CAP launches API and bulk data — Harvard Law School](https://hls.harvard.edu/today/caselaw-access-project-launches-api-and-bulk-data-service/)
