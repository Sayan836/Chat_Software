# Chat Software App

Local-first chat and file-agent implementation.

## What is implemented

- Tauri 2 shell scaffold with a Vite frontend.
- Python sidecar service with HTTP endpoints:
  - `GET /health`
  - `GET /providers`
  - `POST /chat`
  - `POST /chat/stream` (NDJSON runtime events plus the final response)
  - `GET /sessions/{session_id}`
  - `POST /roots/approve`
  - `POST /files/list`
  - `POST /files/stat`
  - `POST /files/read`
  - `POST /files/glob`
  - `POST /files/grep`
  - `POST /commands/git/status`
  - `POST /commands/git/diff`
- FastAPI-served browser console at `/` for manual API testing.
- Real chat calls use the configured provider by default. Set
  `CHAT_SOFTWARE_OFFLINE_ECHO=1` only for no-model smoke tests.
- Browser controls for selecting Ollama, OpenAI, or Claude models when the
  relevant API key/service is available.
- Automatic Ollama discovery selects the configured model when it is installed,
  otherwise the most recently modified installed model.
- Ollama model management controls:
  - refresh installed local models
  - install/pull a model by name
  - delete an installed model
- Scoped file management controls:
  - approve one or more root directories from the web page
  - list directories, stat paths, read files, glob files, and search text
  - create folders/files, copy, move/rename, and edit files
  - run read-only git status and git diff commands inside approved roots
- Live command events appear in the UI command history and as structured
  `[runtime]` entries in the backend terminal.
- Provider package seeded from `/Users/sayansen/Downloads/edur/utils/llm_provider.py`.
- Local-first provider defaults:
  - default provider: `ollama`
  - optional providers: `openai`, `anthropic`
  - OpenAI SDK call path for Ollama/OpenAI runtime model calls
  - Anthropic SDK call path for Claude runtime model calls
  - model capability metadata for routing and safety decisions
- File-Ops tools:
  - `list_dir`
  - `stat_path`
  - `read_file`
  - `glob_files`
  - `grep_files`
  - `make_dir`
  - `write_file`
  - `copy_path`
  - `move_path`
  - `edit_file`
  - `git_status`
  - `git_diff`
- MCP server exposing the read-only tools when the `mcp` dependency is installed.
- Session checkpointing to SQLite for Phase 1 conversations.
- A deterministic read-only agent fallback, so tests pass without network, Ollama, or cloud keys.

## Local development

From this directory:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[test]"
npm install
npm run dev:backend
```

In another terminal:

```bash
npm install
npm run dev
```

Or serve the plain HTML/CSS/JavaScript console and the API from one FastAPI
process:

```bash
npm run serve
```

Then open `http://127.0.0.1:8765/`.
If port `8765` is already in use, `main.py` automatically starts on the next
available port and prints the URL.

Chat calls use Ollama's OpenAI-compatible endpoint at
`http://localhost:11434/v1` by default. Ollama model management uses the native
API at `http://localhost:11434`. The curated install list is a starter list; you
can type any valid Ollama model tag into the install field.

For Tauri:

```bash
npm run tauri dev
```

## Test without installing app dependencies

The unit tests only use the Python standard library:

```bash
npm run test:backend
```

## Native Tauri prerequisite

`npm run tauri dev` requires Rust/Cargo on `PATH`. If it fails with
`failed to run cargo metadata`, install Rust first:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Then open a new terminal, return to this directory, and run:

```bash
source "$HOME/.cargo/env"
npm run tauri dev
```

## File-write safety

All file operations require a server-approved root. Moves refuse overwrites;
same-filesystem moves use rename, and cross-device file moves quarantine the
source inside the approved root after a verified copy. Organize operations keep
their existing preview/apply/undo workflow.
