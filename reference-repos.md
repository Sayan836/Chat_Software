# Reference & Dependency Repositories

*Which repos to clone, which to install, which to bundle — mapped to the architecture, with licenses. Verified July 2026.*

## How to read this

Three kinds of "use," and they're handled differently:

- **DEPENDENCY** — don't clone into your repo. Install via `pip`/`npm`/`cargo`. Listed here so you know what to add.
- **REFERENCE** — clone into a `reference/` folder *outside* your shipped source. Study it, copy small snippets (respect the license/attribution). Not shipped as-is.
- **SERVICE** — bundle and run as a separate process; talk to it over its API. Do **not** copy its source into your app.

**License rule of thumb:** MIT / Apache-2.0 → safe to copy with attribution. **AGPL-3.0 → do NOT copy code into your app** (viral copyleft); only run it as a separate service (SearXNG) or take ideas, not code (Jan).

Suggested layout:
```
Chat_Software/
  app/            # your code
  reference/      # cloned REFERENCE repos (gitignored, study-only)
  services/       # bundled SERVICE (searxng config)
```

---

## A. Agent core & loop  (the "Claude Code pattern" + engine)

| Repo | Use | License | Take from it |
|---|---|---|---|
| `anthropics/claude-agent-sdk-python` | REFERENCE | MIT | The loop (gather→act→verify), tool registry, `PreToolUse`/`PostToolUse` hooks, permission modes, subagents, sessions. *Claude-only — patterns, not engine.* Note Anthropic branding limits. |
| `anthropics/claude-agent-sdk-demos` | REFERENCE | MIT | Worked example agents (research, email assistant). |
| `langchain-ai/langgraph` | DEPENDENCY | MIT | Your actual agent engine — ReAct graph, `interrupt()` for preview/confirm, checkpointer for sessions. |

```bash
git clone https://github.com/anthropics/claude-agent-sdk-python reference/claude-agent-sdk
git clone https://github.com/anthropics/claude-agent-sdk-demos reference/claude-agent-sdk-demos
# langgraph: pip install langgraph
```

## B. Provider abstraction  (seeded by your uploaded llm_provider.py)

| Repo | Use | License | Take from it |
|---|---|---|---|
| `BerriAI/litellm` | DEPENDENCY | MIT | Unified OpenAI-format calls across Ollama / OpenAI / Claude; tool-call + structured-output normalization. |

Local seed file:

| Local path | Use | Take from it |
|---|---|---|
| `/Users/sayansen/Downloads/edur/utils/llm_provider.py` | MIGRATION SEED | Existing OpenAI/Ollama provider selection, model resolution, OpenAI-compatible client creation, and `ChatOpenAI` builder. Update before reuse: default to Ollama, remove project-specific `utils.settings` dependency, route through LiteLLM, add Claude, and add model capability metadata. |

```bash
# pip install litellm
git clone https://github.com/BerriAI/litellm reference/litellm   # optional, to read provider transforms
```

## C. MCP & tool layer

| Repo | Use | License | Take from it |
|---|---|---|---|
| `modelcontextprotocol/python-sdk` | DEPENDENCY | MIT | Build your File-Ops MCP server + clients (stdio/HTTP). |
| `langchain-ai/langchain-mcp-adapters` | DEPENDENCY | MIT | `MultiServerMCPClient.get_tools()` → load MCP tools into LangGraph. |
| `modelcontextprotocol/servers` | REFERENCE | MIT | **Contains a filesystem server** — direct reference for your file-ops tools; plus dozens of connector examples. |

```bash
# pip install mcp langchain-mcp-adapters
git clone https://github.com/modelcontextprotocol/servers reference/mcp-servers
```

## D. File-management core  (organize / rename / preview)

| Repo | Use | License | Take from it |
|---|---|---|---|
| `QiuYannnn/Local-File-Organizer` | REFERENCE / COPY | MIT + Apache-2.0 (dual) | On-device organize/rename logic, document+image parsing, prompt templates, dry-run pattern. Closest match to your core. |
| `hyperfield/ai-file-sorter` | REFERENCE | *verify (repo LICENSE)* | Local-**or**-remote LLM toggle + preview-based, user-controlled changes UX. (C++/Qt — study the flow, not the code.) |

```bash
git clone https://github.com/QiuYannnn/Local-File-Organizer reference/local-file-organizer
git clone https://github.com/hyperfield/ai-file-sorter reference/ai-file-sorter
```

## E. Desktop app + local RAG  (packaging, provider bridge, vector store)

| Repo | Use | License | Take from it |
|---|---|---|---|
| `Mintplex-Labs/anything-llm` | REFERENCE / COPY | MIT | Desktop packaging, bundled **LanceDB** integration, RAG pipeline (ingest→chunk→embed→search), provider bridge (local+cloud). Best all-round reference. |
| `danny-avila/LibreChat` | REFERENCE / COPY | MIT | Multi-provider chat + **MCP** integration in a chat app; conversation/session handling. |
| `open-webui/open-webui` | REFERENCE | Permissive (verify current **branding clause**) | Provider-switching UX, SearXNG + Ollama wiring. |
| `menloresearch/jan` | REFERENCE (ideas only) | **AGPL-3.0** | Privacy-first desktop UX for local/cloud models. **Do not copy code** — patterns only. |

```bash
git clone https://github.com/Mintplex-Labs/anything-llm reference/anything-llm
git clone https://github.com/danny-avila/LibreChat reference/librechat
git clone https://github.com/open-webui/open-webui reference/open-webui
# Jan: browse on GitHub for UX ideas; avoid cloning-to-copy due to AGPL
```

## F. Vector store & indexing

| Repo | Use | License | Take from it |
|---|---|---|---|
| `lancedb/lancedb` | DEPENDENCY | Apache-2.0 | Embedded vector DB (default). |
| `asg017/sqlite-vec` | DEPENDENCY | MIT / Apache-2.0 | Single-file vector store (lighter alternative). |

```bash
# pip install lancedb   (or the sqlite-vec extension)
```

## G. Web search & content extraction

| Repo | Use | License | Take from it |
|---|---|---|---|
| `searxng/searxng` | **SERVICE (bundle+run)** | **AGPL-3.0** | Your private `web_search`. Run as a local sidecar, call its JSON API. **Never copy its source into your app.** |
| `adbar/trafilatura` | DEPENDENCY | Apache-2.0 (v1.8.0+) | Clean main-text/markdown extraction for `web_fetch` / crawl. |
| `brcrusoe72/agent-search` | REFERENCE | *verify* | Example of bundling SearXNG + an MCP search server for agents — good blueprint for wiring. |

```bash
# SearXNG: run via its official Docker image or packaged binary as a sidecar (config in services/)
# pip install trafilatura
git clone https://github.com/brcrusoe72/agent-search reference/agent-search
```

## H. Desktop shell

| Repo | Use | License | Take from it |
|---|---|---|---|
| `tauri-apps/tauri` | DEPENDENCY / REFERENCE | MIT / Apache-2.0 | The shell; see its **sidecar** docs/examples for spawning the Python engine. |

```bash
# scaffold: npm create tauri-app@latest   (don't clone tauri itself)
```

## I. Domain knowledge data (not code — data sources for packs)

| Source | Use | License | Notes |
|---|---|---|---|
| `freelawproject/courtlistener` + [bulk data](https://www.courtlistener.com/help/api/bulk-data/) | DATA | Public domain (data) | US case law, 9M+ decisions. **Scope by jurisdiction/practice area — do not ship all (~TB-scale).** |
| `harvard-lil/cold-cases` (Hugging Face dataset) | DATA | Open (verify per-record) | Harvard Caselaw Access Project data, ready for indexing. |

---

## The short answer: what to actually clone now

For Phase 0–2 you only need these cloned as **reference**:
```bash
mkdir -p reference && cd reference
git clone https://github.com/anthropics/claude-agent-sdk-python claude-agent-sdk
git clone https://github.com/modelcontextprotocol/servers mcp-servers
git clone https://github.com/QiuYannnn/Local-File-Organizer local-file-organizer
git clone https://github.com/Mintplex-Labs/anything-llm anything-llm
git clone https://github.com/danny-avila/LibreChat librechat
```
Everything else (LangGraph, LiteLLM, MCP SDK, langchain-mcp-adapters, LanceDB, trafilatura) is a `pip install`, and SearXNG is a bundled service. Add `reference/` to `.gitignore`.

## License cautions (read before copying code)

- **AGPL-3.0 — SearXNG, Jan:** copying their code into your app forces your app to be AGPL too. SearXNG is safe as a *separate bundled service* you call over HTTP; Jan is *ideas only*.
- **Trafilatura:** Apache-2.0 only from **v1.8.0+** — pin a recent version (older versions are GPLv3).
- **Open WebUI:** recent versions added a **branding clause** — verify the current LICENSE before reusing code.
- **Claude Agent SDK:** MIT code, but **Anthropic branding restrictions** apply (don't present your product as Claude Code).
- **MIT/Apache repos (LangGraph, LiteLLM, AnythingLLM, LibreChat, LanceDB, MCP SDKs, Local-File-Organizer):** copy freely **with attribution** per each LICENSE/NOTICE.

*Licenses change — re-check each repo's LICENSE file at clone time. Items marked "verify" I didn't confirm to a primary source.*
