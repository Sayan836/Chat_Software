const apiBaseInput = document.querySelector("#api-base");
const healthState = document.querySelector("#health-state");
const providerState = document.querySelector("#provider-state");
const modelState = document.querySelector("#model-state");
const modeState = document.querySelector("#mode-state");
const requestState = document.querySelector("#request-state");
const approveRootPath = document.querySelector("#approve-root-path");
const approvedRootsSelect = document.querySelector("#approved-roots");
const providerSelect = document.querySelector("#provider-select");
const modelInput = document.querySelector("#model-input");
const providerNote = document.querySelector("#provider-note");
const ollamaInstalled = document.querySelector("#ollama-installed");
const ollamaInstall = document.querySelector("#ollama-install");
const ollamaCatalog = document.querySelector("#ollama-catalog");
const sessionInput = document.querySelector("#session-id");
const rootInput = document.querySelector("#root-path");
const messageInput = document.querySelector("#message");
const messages = document.querySelector("#messages");
const chatForm = document.querySelector("#chat-form");
const sendButton = document.querySelector("#send");
const toolCalls = document.querySelector("#tool-calls");
const rawOutput = document.querySelector("#raw-output");
const commandConsole = document.querySelector("#command-console");
const commandSelect = document.querySelector("#command-select");
const commandPath = document.querySelector("#command-path");
const commandTarget = document.querySelector("#command-target");
const commandPattern = document.querySelector("#command-pattern");
const commandQuery = document.querySelector("#command-query");
const commandInclude = document.querySelector("#command-include");
const commandContent = document.querySelector("#command-content");
const commandOverwrite = document.querySelector("#command-overwrite");
const organizeStrategy = document.querySelector("#organize-strategy");
const organizeFilter = document.querySelector("#organize-filter");
const organizePath = document.querySelector("#organize-path");
const organizeMaxFiles = document.querySelector("#organize-max-files");
const organizeState = document.querySelector("#organize-state");
const organizePreview = document.querySelector("#organize-preview");
const organizePlanButton = document.querySelector("#organize-plan");
const organizeApplyButton = document.querySelector("#organize-apply");
const organizeUndoButton = document.querySelector("#organize-undo");
const knowledgeState = document.querySelector("#knowledge-state");
const knowledgeWorkspace = document.querySelector("#knowledge-workspace");
const knowledgeBinding = document.querySelector("#knowledge-binding");
const documentsMode = document.querySelector("#documents-mode");
const sourceViewer = document.querySelector("#source-viewer");
const sourceState = document.querySelector("#source-state");
const ingestPaths = document.querySelector("#ingest-paths");
const ingestPlan = document.querySelector("#ingest-plan");
const ingestState = document.querySelector("#ingest-state");
const ingestApproveButton = document.querySelector("#ingest-approve");
const ingestDiscardButton = document.querySelector("#ingest-discard");

const storedApiBase = localStorage.getItem("chatSoftwareApiBase");
const defaultApiBase = window.location.protocol === "file:" ? "http://127.0.0.1:8765" : window.location.origin;
let currentSessionId = localStorage.getItem("chatSoftwareSessionId") || "";
let providerOptions = [];
let approvedRoots = JSON.parse(localStorage.getItem("chatSoftwareApprovedRoots") || "[]");
let currentOrganizePlan = null;
let commandSequence = 0;
let knowledgeBindings = [];
let currentIngestSubmission = null;
let ingestPollTimer = null;

apiBaseInput.value = storedApiBase || defaultApiBase;
sessionInput.value = currentSessionId;
providerSelect.value = localStorage.getItem("chatSoftwareProvider") || "ollama";
modelInput.value = localStorage.getItem("chatSoftwareModel") || "";

function setRequestState(text, mode = "") {
  requestState.textContent = text;
  requestState.className = mode ? `pill ${mode}` : "pill";
}

function setBusy(isBusy) {
  sendButton.disabled = isBusy;
  document.querySelector("#check-health").disabled = isBusy;
  document.querySelector("#load-providers").disabled = isBusy;
  document.querySelector("#load-session").disabled = isBusy;
  document.querySelector("#approve-root").disabled = isBusy;
  document.querySelector("#remove-root").disabled = isBusy;
  document.querySelector("#run-command").disabled = isBusy;
  document.querySelector("#refresh-ollama").disabled = isBusy;
  document.querySelector("#pull-ollama").disabled = isBusy;
  document.querySelector("#delete-ollama").disabled = isBusy;
  organizePlanButton.disabled = isBusy;
  updateOrganizeActionButtons(isBusy);
}

function apiBase() {
  const value = apiBaseInput.value.trim() || defaultApiBase;
  localStorage.setItem("chatSoftwareApiBase", value);
  return value.replace(/\/+$/, "");
}

function apiUrl(path) {
  return `${apiBase()}${path}`;
}

async function requestJson(path, options = {}) {
  const response = await fetch(apiUrl(path), options);
  const text = await response.text();
  let body = null;

  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      body = text;
    }
  }

  if (!response.ok) {
    const detail = typeof body === "object" && body ? JSON.stringify(body, null, 2) : text;
    throw new Error(`${response.status} ${response.statusText}${detail ? `\n${detail}` : ""}`);
  }

  return body;
}

async function requestNdjson(path, options, onEvent) {
  const response = await fetch(apiUrl(path), options);
  if (!response.ok) {
    const text = await response.text();
    let detail = text;
    try {
      const body = text ? JSON.parse(text) : null;
      detail = body?.detail || text;
    } catch {
      // Keep the plain response text.
    }
    throw new Error(`${response.status} ${response.statusText}${detail ? `\n${detail}` : ""}`);
  }
  if (!response.body) {
    throw new Error("The browser did not expose the chat response stream.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() || "";
    lines.filter(Boolean).forEach((line) => onEvent(JSON.parse(line)));
    if (done) break;
  }
  if (buffer.trim()) onEvent(JSON.parse(buffer));
}

function showEmptyState() {
  messages.replaceChildren();
  const empty = document.createElement("div");
  empty.className = "empty-state";
  empty.textContent = "No messages in this session.";
  messages.append(empty);
}

function appendMessage(role, content) {
  const existingEmpty = messages.querySelector(".empty-state");
  if (existingEmpty) existingEmpty.remove();

  const row = document.createElement("article");
  row.className = `message ${role}`;

  const label = document.createElement("span");
  label.className = "message-role";
  label.textContent = role;

  const text = document.createElement("div");
  text.textContent = content;

  row.append(label, text);
  messages.append(row);
  messages.scrollTop = messages.scrollHeight;
  return row;
}

function documentsFromToolCalls(calls) {
  const call = (calls || []).find((item) => item.name === "knowledge_search");
  return call ? call.result || null : null;
}

function renderCitations(row, documents) {
  if (!row || !documents) return;
  const footer = document.createElement("div");
  footer.className = "citations";
  const citations = documents.citations || [];
  if (citations.length === 0) {
    const note = document.createElement("span");
    note.className = "citation-note";
    note.textContent = documents.refused
      ? `No answer: ${String(documents.reason || "refused").replaceAll("_", " ")}`
      : "No citations";
    footer.append(note);
  }
  citations.forEach((citation) => {
    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = "citation-chip secondary";
    const where = citation.filename || citation.document_id;
    const rows = citation.row_range
      ? citation.row_range[0] === citation.row_range[1]
        ? ` row ${citation.row_range[0]}`
        : ` rows ${citation.row_range[0]}\u2013${citation.row_range[1]}`
      : "";
    chip.textContent = `[${citation.index}] ${where}${citation.page ? ` p.${citation.page}` : ""}${rows}`;
    chip.title = citation.excerpt || "";
    chip.addEventListener("click", () =>
      openSource(documents.root_id, citation.document_id, citation.page || 1, citation.excerpt)
    );
    footer.append(chip);
  });
  if (documents.retrieval_status && documents.retrieval_status.degraded) {
    const warn = document.createElement("span");
    warn.className = "citation-note";
    warn.textContent = "Retrieval degraded";
    footer.append(warn);
  }
  row.append(footer);
}

async function openSource(rootId, documentId, page = 1, excerpt = "") {
  sourceState.textContent = "Opening";
  sourceState.className = "pill busy";
  try {
    const data = await requestJson("/knowledge/source", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ root_id: rootId, document_id: documentId, page })
    });
    const header = `${documentId} \u2022 page ${data.page}${data.truncated ? " (truncated)" : ""}`;
    const body = data.needs_ocr ? "[This page has no extractable text; OCR is needed.]" : data.text;
    sourceViewer.textContent = `${header}\n\n${body}`;
    if (excerpt) sourceViewer.title = `Cited excerpt: ${excerpt}`;
    sourceState.textContent = "Untrusted";
    sourceState.className = "pill ok";
  } catch (error) {
    sourceViewer.textContent = error.message;
    sourceState.textContent = "Denied";
    sourceState.className = "pill error";
  }
}

function selectedKnowledgeRootId() {
  return knowledgeBinding.value || "";
}

function renderKnowledgeBindings() {
  const previous = knowledgeBinding.value;
  knowledgeBinding.replaceChildren();
  if (knowledgeBindings.length === 0) {
    knowledgeBinding.append(option("No bound roots", ""));
    documentsMode.checked = false;
    documentsMode.disabled = true;
    return;
  }
  documentsMode.disabled = false;
  knowledgeBindings.forEach((binding) => {
    knowledgeBinding.append(
      option(
        `${binding.root_name || binding.root_path} \u2192 ${binding.workspace_id}${binding.ingest_enabled ? " (submit on)" : ""}`,
        binding.root_id
      )
    );
  });
  if (knowledgeBindings.some((binding) => binding.root_id === previous)) {
    knowledgeBinding.value = previous;
  }
}

async function loadKnowledge() {
  try {
    const status = await requestJson("/knowledge/status");
    knowledgeState.textContent = status.configured
      ? `Retrieval backend: ${status.base_url}`
      : status.error || "Retrieval backend not configured (CHAT_SOFTWARE_KNOWLEDGE_URL).";
    const data = await requestJson("/knowledge/bindings");
    knowledgeBindings = data.bindings || [];
  } catch (error) {
    knowledgeState.textContent = error.message;
    knowledgeBindings = [];
  }
  renderKnowledgeBindings();
}

async function serverRootIdForPath(path) {
  const data = await requestJson("/roots");
  const match = (data.roots || []).find((root) => root.path === path);
  if (!match) throw new Error(`Root is not approved on the server: ${path}`);
  return match.root_id;
}

async function bindKnowledgeRoot() {
  const roots = activeRoots();
  const workspaceId = knowledgeWorkspace.value.trim();
  if (roots.length !== 1 || !workspaceId) {
    appendMessage("system", "Select exactly one approved root and enter a retrieval workspace ID.");
    return;
  }
  setBusy(true);
  try {
    const rootId = await serverRootIdForPath(roots[0]);
    const data = await requestJson("/knowledge/bindings", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ root_id: rootId, workspace_id: workspaceId })
    });
    renderRaw(data);
    await loadKnowledge();
    knowledgeBinding.value = rootId;
  } catch (error) {
    appendMessage("system", error.message);
  } finally {
    setBusy(false);
  }
}

async function unbindKnowledgeRoot() {
  const rootId = selectedKnowledgeRootId();
  if (!rootId) return;
  try {
    renderRaw(await requestJson(`/knowledge/bindings/${encodeURIComponent(rootId)}`, { method: "DELETE" }));
  } catch (error) {
    appendMessage("system", error.message);
  }
  await loadKnowledge();
}

function setIngestState(text, mode = "") {
  ingestState.textContent = text;
  ingestState.className = mode ? `pill ${mode}` : "pill";
}

function formatBytes(size) {
  if (size >= 1024 * 1024) return `${(size / (1024 * 1024)).toFixed(1)} MB`;
  if (size >= 1024) return `${Math.round(size / 1024)} KB`;
  return `${size} B`;
}

function renderIngestPlan(plan) {
  currentIngestSubmission = plan && plan.submission_id ? plan : null;
  ingestPlan.replaceChildren();
  const pending = Boolean(plan && plan.status === "awaiting_approval");
  ingestApproveButton.disabled = !pending;
  ingestDiscardButton.disabled = !pending;
  if (!plan) return;
  if (plan.destination) {
    const destination = document.createElement("p");
    destination.className = "note";
    destination.textContent =
      `Destination: ${plan.destination.workspace_name || plan.destination.workspace_id} on ` +
      `${plan.destination.location}. ${plan.destination.note}`;
    ingestPlan.append(destination);
  }
  const list = document.createElement("ul");
  list.className = "ingest-files";
  (plan.files || []).forEach((file) => {
    const item = document.createElement("li");
    const detail = file.chunks_indexed ? `, ${file.chunks_indexed} chunks` : "";
    item.textContent = `${file.rel_path} (${formatBytes(file.size_bytes)}) \u2014 ${file.status}${detail}`;
    if (file.error) item.title = file.error;
    item.className = `ingest-file ${file.status}`;
    list.append(item);
  });
  (plan.rejected || []).forEach((file) => {
    const item = document.createElement("li");
    item.textContent = `${file.path} \u2014 not sent: ${file.reason}`;
    item.className = "ingest-file rejected";
    list.append(item);
  });
  ingestPlan.append(list);
  setIngestState(plan.status ? plan.status.replaceAll("_", " ") : "Idle", plan.status === "finished" ? "ok" : "");
}

async function enableIngest() {
  const rootId = selectedKnowledgeRootId();
  if (!rootId) {
    appendMessage("system", "Bind a root to a retrieval workspace first.");
    return;
  }
  try {
    const data = await requestJson("/knowledge/ingest/enable", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ root_id: rootId })
    });
    renderRaw(data);
    await loadKnowledge();
    knowledgeBinding.value = rootId;
  } catch (error) {
    appendMessage("system", error.message);
  }
}

async function previewIngest() {
  const rootId = selectedKnowledgeRootId();
  const paths = ingestPaths.value.split("\n").map((line) => line.trim()).filter(Boolean);
  if (!rootId || paths.length === 0) {
    appendMessage("system", "Select a bound root and list the files to submit.");
    return;
  }
  setIngestState("Checking", "busy");
  try {
    const plan = await requestJson("/knowledge/ingest/preview", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ root_id: rootId, paths })
    });
    renderIngestPlan(plan);
    renderRaw(plan);
  } catch (error) {
    setIngestState("Error", "error");
    appendMessage("system", error.message);
  }
}

function scheduleIngestPoll(submissionId, attempt = 0) {
  window.clearTimeout(ingestPollTimer);
  if (attempt > 150) return;
  ingestPollTimer = window.setTimeout(async () => {
    try {
      const status = await requestJson(`/knowledge/ingest/${encodeURIComponent(submissionId)}`);
      renderIngestPlan(status);
      if (status.status !== "finished") scheduleIngestPoll(submissionId, attempt + 1);
    } catch (error) {
      setIngestState("Error", "error");
      appendMessage("system", error.message);
    }
  }, 2000);
}

async function approveIngest() {
  const plan = currentIngestSubmission;
  if (!plan) return;
  const names = (plan.files || []).map((file) => file.rel_path).join(", ");
  const where = plan.destination ? plan.destination.location : "the retrieval backend";
  if (!window.confirm(`Send ${plan.files.length} file(s) to ${where}? ${names}`)) return;
  ingestApproveButton.disabled = true;
  ingestDiscardButton.disabled = true;
  setIngestState("Sending", "busy");
  try {
    const status = await requestJson(`/knowledge/ingest/${encodeURIComponent(plan.submission_id)}/approve`, {
      method: "POST"
    });
    renderIngestPlan({ ...status, destination: plan.destination });
    renderRaw(status);
    if (status.status !== "finished") scheduleIngestPoll(plan.submission_id);
  } catch (error) {
    setIngestState("Error", "error");
    appendMessage("system", error.message);
  }
}

async function discardIngest() {
  const plan = currentIngestSubmission;
  if (!plan) return;
  try {
    await requestJson(`/knowledge/ingest/${encodeURIComponent(plan.submission_id)}`, { method: "DELETE" });
    renderIngestPlan(null);
    setIngestState("Discarded");
  } catch (error) {
    appendMessage("system", error.message);
  }
}

async function revokeKnowledgeRoot() {
  const rootId = selectedKnowledgeRootId();
  const binding = knowledgeBindings.find((item) => item.root_id === rootId);
  if (!binding) return;
  if (!window.confirm(`Revoke ${binding.root_path}? File tools and document search for it stop immediately.`)) {
    return;
  }
  try {
    renderRaw(await requestJson(`/roots/${encodeURIComponent(rootId)}/revoke`, { method: "POST" }));
    approvedRoots = approvedRoots.filter((root) => root !== binding.root_path);
    saveApprovedRoots();
    renderApprovedRoots();
  } catch (error) {
    appendMessage("system", error.message);
  }
  await loadKnowledge();
}

function renderToolCalls(calls) {
  toolCalls.replaceChildren();

  if (!calls || calls.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-state";
    empty.textContent = "No tool calls for the latest response.";
    toolCalls.append(empty);
    return;
  }

  calls.forEach((call) => {
    const item = document.createElement("article");
    item.className = "tool-call";

    const title = document.createElement("h3");
    title.textContent = call.name || "tool";

    const command = document.createElement("code");
    command.className = "tool-command";
    command.textContent = toolCommand(call);

    const summary = document.createElement("p");
    summary.className = "tool-summary";
    summary.textContent = toolSummary(call);

    const output = document.createElement("pre");
    output.textContent = JSON.stringify(
      {
        command: toolCommand(call),
        arguments: call.arguments || {},
        result: call.result || {}
      },
      null,
      2
    );

    item.append(title, command, summary, output);
    toolCalls.append(item);
  });
}

function appendCommandConsole(command, { status = "completed", source = "agent", detail = "" } = {}) {
  const existingEmpty = commandConsole.querySelector(".empty-state");
  if (existingEmpty) existingEmpty.remove();

  const row = document.createElement("article");
  row.className = `command-line ${status}`;

  const meta = document.createElement("div");
  meta.className = "command-meta";

  const index = document.createElement("span");
  index.className = "command-index";
  index.textContent = `#${++commandSequence}`;

  const statusBadge = document.createElement("span");
  statusBadge.className = "command-status";
  statusBadge.textContent = status;

  const sourceLabel = document.createElement("span");
  sourceLabel.className = "command-source";
  sourceLabel.textContent = source;

  meta.append(index, statusBadge, sourceLabel);

  const code = document.createElement("code");
  code.textContent = command;

  row.append(meta, code);
  if (detail) {
    const description = document.createElement("p");
    description.textContent = detail;
    row.append(description);
  }
  commandConsole.append(row);
  commandConsole.scrollTop = commandConsole.scrollHeight;
  return row;
}

function updateCommandConsole(row, status, detail = "") {
  if (!row) return;
  row.className = `command-line ${status}`;
  const badge = row.querySelector(".command-status");
  if (badge) badge.textContent = status;
  let description = row.querySelector("p");
  if (detail) {
    if (!description) {
      description = document.createElement("p");
      row.append(description);
    }
    description.textContent = detail;
  }
}

function renderCommandConsoleEmpty() {
  commandSequence = 0;
  commandConsole.replaceChildren();
  const empty = document.createElement("div");
  empty.className = "empty-state compact";
  empty.textContent = "No commands have run yet.";
  commandConsole.append(empty);
}

function appendToolCallsToCommandConsole(calls) {
  (calls || []).forEach((call) => {
    appendCommandConsole(toolCommand(call), {
      status: call.status || "completed",
      source: call.name || "tool",
      detail: toolSummary(call)
    });
  });
}

function renderCommandHistoryFromMessages(sessionMessages) {
  renderCommandConsoleEmpty();
  (sessionMessages || []).forEach((message) => {
    const calls = message?.metadata?.tool_calls;
    if (!Array.isArray(calls)) return;
    calls.forEach((call) => {
      if (typeof call === "string") {
        appendCommandConsole(call, { status: "completed", source: "history" });
        return;
      }
      appendCommandConsole(toolCommand(call), {
        status: call.status || "completed",
        source: call.name || "tool",
        detail: toolSummary(call)
      });
    });
  });
}

function shortJson(value) {
  return JSON.stringify(value, (_key, current) => {
    if (typeof current === "string" && current.length > 500) {
      return `${current.slice(0, 500)}... [${current.length - 500} more chars]`;
    }
    if (Array.isArray(current) && current.length > 4) {
      return [...current.slice(0, 4), `... ${current.length - 4} more`];
    }
    return current;
  });
}

function toolCommand(call) {
  if (call?.command) return call.command;
  const name = call?.name || "tool";
  const args = call?.arguments || {};
  if (call?.result?.command && Array.isArray(call.result.command)) {
    const cwd = call.result.cwd || ".";
    return `(cd ${cwd} && ${call.result.command.join(" ")})`;
  }
  const endpoint = toolEndpoint(name);
  if (endpoint) return `POST ${endpoint} ${shortJson(args)}`;
  return `${name}(${Object.entries(args).map(([key, value]) => `${key}=${shortJson(value)}`).join(", ")})`;
}

function toolEndpoint(name) {
  return {
    list: "/files/list",
    list_dir: "/files/list",
    stat: "/files/stat",
    stat_path: "/files/stat",
    read: "/files/read",
    read_file: "/files/read",
    glob: "/files/glob",
    glob_files: "/files/glob",
    grep: "/files/grep",
    grep_files: "/files/grep",
    mkdir: "/files/mkdir",
    make_dir: "/files/mkdir",
    write: "/files/write",
    write_file: "/files/write",
    copy: "/files/copy",
    copy_path: "/files/copy",
    move: "/files/move",
    move_path: "/files/move",
    edit: "/files/edit",
    edit_file: "/files/edit",
    git_status: "/commands/git/status",
    git_diff: "/commands/git/diff",
    organize_plan: "/organize/plan",
    organize_apply: "/organize/apply",
    organize_undo: "/organize/undo"
  }[name];
}

function toolSummary(call) {
  const result = call?.result || {};
  if (result.op === "mkdir") return result.created ? `Created ${result.path}.` : `${result.path} already existed.`;
  if (result.op === "write") return `${result.overwritten ? "Updated" : "Created"} ${result.path}.`;
  if (result.op === "copy") return `Copied ${result.from} to ${result.to}.`;
  if (result.op === "move") return `Moved ${result.from} to ${result.to}.`;
  if (result.op === "edit") return `${result.replacements || 0} replacement(s) in ${result.path}.`;
  if (Array.isArray(result.entries)) return `${result.entries.length} item(s) returned.`;
  if (Array.isArray(result.matches)) return `${result.matches.length} match(es) returned.`;
  if (Array.isArray(result.actions)) return `${result.actions.length} planned action(s).`;
  if (Array.isArray(result.applied)) return `${result.applied.length} action(s) applied.`;
  if (Array.isArray(result.undone)) return `${result.undone.length} action(s) undone.`;
  if (typeof result.returncode !== "undefined") return `Exited with code ${result.returncode}.`;
  if (result.path) return `Read ${result.path}.`;
  return "Tool completed.";
}

function setOrganizeState(text, mode = "") {
  organizeState.textContent = text;
  organizeState.className = mode ? `pill ${mode}` : "pill";
}

function statusMode(status) {
  if (status === "applied" || status === "undone") return "ok";
  if (status === "applying") return "busy";
  if (status === "failed" || status === "partial") return "error";
  return "";
}

function updateOrganizeActionButtons(isBusy = false) {
  const status = currentOrganizePlan?.status || "";
  const actions = Array.isArray(currentOrganizePlan?.actions) ? currentOrganizePlan.actions : [];
  const hasMoves = actions.some((action) => action?.op === "move");
  organizeApplyButton.disabled = isBusy || status !== "planned" || !hasMoves;
  organizeUndoButton.disabled = isBusy || !["applied", "partial"].includes(status);
}

function renderOrganizeEmpty(text = "No organize preview yet.") {
  currentOrganizePlan = null;
  organizePreview.replaceChildren();
  const empty = document.createElement("div");
  empty.className = "empty-state compact";
  empty.textContent = text;
  organizePreview.append(empty);
  setOrganizeState("No plan");
  updateOrganizeActionButtons(false);
}

function createStat(label, value) {
  const item = document.createElement("div");
  item.className = "stat-chip";

  const number = document.createElement("strong");
  number.textContent = String(value ?? 0);

  const caption = document.createElement("span");
  caption.textContent = label;

  item.append(number, caption);
  return item;
}

function createOrganizeRow(action) {
  const row = document.createElement("div");
  row.className = "organize-row";

  const op = document.createElement("span");
  op.className = "op-badge";
  op.textContent = action.op || "op";

  const source = document.createElement("span");
  source.className = "path-cell";
  source.textContent = action.from || action.path || "";

  const target = document.createElement("span");
  target.className = "path-cell muted";
  target.textContent = action.to || "";

  const status = document.createElement("span");
  status.className = "status-badge";
  status.textContent = action.status || currentOrganizePlan?.status || "planned";

  row.append(op, source, target, status);
  return row;
}

function renderOrganizePlan(plan) {
  if (!plan) {
    renderOrganizeEmpty();
    return;
  }

  currentOrganizePlan = plan;
  organizePreview.replaceChildren();
  setOrganizeState(plan.status || "planned", statusMode(plan.status));

  const summary = plan.summary || {};
  const stats = document.createElement("div");
  stats.className = "organize-stats";
  stats.append(
    createStat("moves", summary.moves),
    createStat("folders", summary.new_dirs),
    createStat("skipped", summary.skipped),
    createStat("seen", summary.files_seen)
  );
  organizePreview.append(stats);

  const meta = document.createElement("div");
  meta.className = "organize-meta";
  const fileFilter = plan.file_filter && plan.file_filter !== "all" ? `, ${plan.file_filter} files` : "";
  meta.textContent = `${plan.strategy || "strategy"}${fileFilter} in ${plan.path || "."}`;
  organizePreview.append(meta);

  if (summary.truncated) {
    const truncated = document.createElement("p");
    truncated.className = "note warning-note";
    truncated.textContent = "Preview reached the max file limit.";
    organizePreview.append(truncated);
  }

  const warnings = Array.isArray(plan.warnings) ? plan.warnings : [];
  if (warnings.length > 0) {
    const warningList = document.createElement("ul");
    warningList.className = "warning-list";
    warnings.slice(0, 5).forEach((warning) => {
      const item = document.createElement("li");
      item.textContent = warning;
      warningList.append(item);
    });
    organizePreview.append(warningList);
  }

  const actions = Array.isArray(plan.actions) ? plan.actions : [];
  if (actions.length === 0) {
    const empty = document.createElement("div");
    empty.className = "empty-state compact";
    empty.textContent = "No file moves needed for this preview.";
    organizePreview.append(empty);
  } else {
    const table = document.createElement("div");
    table.className = "organize-table";

    const header = document.createElement("div");
    header.className = "organize-row header";
    ["Op", "Source", "Target", "Status"].forEach((label) => {
      const item = document.createElement("span");
      item.textContent = label;
      header.append(item);
    });
    table.append(header);

    actions.slice(0, 80).forEach((action) => table.append(createOrganizeRow(action)));
    organizePreview.append(table);

    if (actions.length > 80) {
      const note = document.createElement("p");
      note.className = "note";
      note.textContent = `${actions.length - 80} more action(s) hidden in the preview.`;
      organizePreview.append(note);
    }
  }

  const skipped = Array.isArray(plan.skipped) ? plan.skipped : [];
  if (skipped.length > 0) {
    const skippedList = document.createElement("div");
    skippedList.className = "skipped-list";
    const title = document.createElement("strong");
    title.textContent = "Skipped";
    skippedList.append(title);
    skipped.slice(0, 12).forEach((item) => {
      const row = document.createElement("span");
      row.textContent = `${item.path}: ${item.reason}`;
      skippedList.append(row);
    });
    organizePreview.append(skippedList);
  }

  updateOrganizeActionButtons(false);
}

function organizePlanFromToolCalls(calls) {
  const plans = (calls || [])
    .filter((call) => call?.name === "organize_plan" && call?.result?.plan_id)
    .map((call) => call.result);
  return plans.length > 0 ? plans[plans.length - 1] : null;
}

function renderOrganizePreviewFromToolCalls(calls) {
  const plan = organizePlanFromToolCalls(calls);
  if (plan) renderOrganizePlan(plan);
}

function renderRaw(data) {
  rawOutput.textContent = data ? JSON.stringify(data, null, 2) : "";
}

function saveApprovedRoots() {
  localStorage.setItem("chatSoftwareApprovedRoots", JSON.stringify(approvedRoots));
}

function renderApprovedRoots() {
  approvedRootsSelect.replaceChildren();
  approvedRoots.forEach((root) => approvedRootsSelect.append(option(root, root)));
  if (approvedRoots.length > 0 && !rootInput.value.trim()) {
    rootInput.value = approvedRoots[0];
  }
}

function activeRoots() {
  const selected = Array.from(approvedRootsSelect.selectedOptions).map((item) => item.value);
  if (selected.length > 0) return selected;
  if (approvedRoots.length > 0) return approvedRoots;
  const fallback = rootInput.value.trim();
  return fallback ? [fallback] : [];
}

function activeOrganizeRoot() {
  const roots = activeRoots();
  if (roots.length === 0) {
    throw new Error("Approve at least one file root before previewing an organize plan.");
  }
  return roots[0];
}

async function approveRoot() {
  const path = approveRootPath.value.trim() || rootInput.value.trim();
  if (!path) {
    appendMessage("system", "Enter a root directory to approve.");
    return;
  }

  setBusy(true);
  setRequestState("Approving", "busy");
  try {
    const data = await requestJson("/roots/approve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path })
    });
    if (!approvedRoots.includes(data.path)) {
      approvedRoots.push(data.path);
      saveApprovedRoots();
    }
    rootInput.value = data.path;
    approveRootPath.value = data.path;
    renderApprovedRoots();
    renderRaw(data);
    setRequestState("Ready", "ok");
  } catch (error) {
    appendMessage("system", error.message);
    setRequestState("Error", "error");
  } finally {
    setBusy(false);
  }
}

function removeSelectedRoot() {
  const selected = new Set(Array.from(approvedRootsSelect.selectedOptions).map((item) => item.value));
  if (selected.size === 0) return;
  approvedRoots = approvedRoots.filter((root) => !selected.has(root));
  saveApprovedRoots();
  renderApprovedRoots();
}

function option(text, value = text) {
  const item = document.createElement("option");
  item.value = value;
  item.textContent = text;
  return item;
}

function setDatalist(element, values) {
  element.replaceChildren();
  values.forEach((value) => element.append(option(value)));
}

function setModelOptions(values, preferredModel = "") {
  const uniqueModels = [...new Set(values.filter(Boolean))];
  const selectedModel = uniqueModels.includes(preferredModel)
    ? preferredModel
    : uniqueModels[0] || "";
  modelInput.replaceChildren();
  if (uniqueModels.length === 0) {
    modelInput.append(option("No models available", ""));
  } else {
    uniqueModels.forEach((model) => modelInput.append(option(model, model)));
  }
  modelInput.value = selectedModel;
  return selectedModel;
}

function selectedProviderInfo() {
  return providerOptions.find((item) => item.provider === providerSelect.value);
}

function updateProviderNote() {
  const info = selectedProviderInfo();
  const selected = providerSelect.value;
  localStorage.setItem("chatSoftwareProvider", selected);
  localStorage.setItem("chatSoftwareModel", modelInput.value.trim());

  if (!info) {
    providerNote.textContent = "Provider options not loaded yet.";
    return;
  }

  if (selected === "ollama") {
    const installedCount = info.installed_count || 0;
    const endpoint = info.native_base_url || info.base_url || "default endpoint";
    if (info.model_error) {
      providerNote.textContent = `Could not reach local Ollama at ${endpoint}.`;
      return;
    }
    providerNote.textContent = installedCount > 0
      ? `Local Ollama at ${endpoint} with ${installedCount} installed model(s).`
      : `Local Ollama at ${endpoint} has no installed models.`;
    return;
  }

  if (selected === "airllm") {
    const count = (info.models || []).length;
    providerNote.textContent = count
      ? `AirLLM: ${count} local checkpoint(s). The worker starts on first use; large models are slow.`
      : "AirLLM has no models yet. Add one in Settings > AirLLM.";
    return;
  }

  if (selected === "compatible") {
    providerNote.textContent = info.api_available
      ? `OpenAI-compatible endpoint: ${info.base_url}.`
      : "Set the OpenAI-compatible base URL in Settings first.";
    return;
  }

  providerNote.textContent = info.api_available
    ? `${info.label} API key configured.`
    : `${info.label} API key not set; add it in Settings before chatting.`;
}

function suggestedModelsFor(provider, registeredModels, ollamaModels, catalog, providers = []) {
  if (provider === "ollama") {
    const installed = ollamaModels.map(ollamaModelName).filter(Boolean);
    return [...new Set(installed)];
  }
  const info = providers.find((item) => item.provider === provider) || {};
  return [...new Set([
    ...(info.models || []),
    ...registeredModels.filter((item) => item.provider === provider).map((item) => item.model),
    info.default_model
  ].filter(Boolean))];
}

function ollamaModelName(model) {
  return model?.name || model?.model || "";
}

function applyModelSuggestions(data, ollamaModels = []) {
  const provider = providerSelect.value;
  const suggestions = suggestedModelsFor(
    provider,
    data.registered_models || [],
    ollamaModels,
    data.ollama_catalog || [],
    data.providers || []
  );
  const storedModel = localStorage.getItem("chatSoftwareModel") || "";
  const activeModel = data.active?.provider === provider ? data.active?.model || "" : "";
  const currentModel = modelInput.value.trim();
  const selectedModel = setModelOptions(
    suggestions,
    [currentModel, storedModel, activeModel].find((model) => suggestions.includes(model)) || ""
  );
  if (provider === "ollama") {
    ollamaInstalled.value = selectedModel;
  }
  updateProviderNote();
}

function renderOllamaModels(models) {
  const selectedModel = modelInput.value.trim();
  ollamaInstalled.replaceChildren();

  if (!models || models.length === 0) {
    ollamaInstalled.append(option("No local models found", ""));
    return;
  }

  models.forEach((model) => {
    const modelName = ollamaModelName(model);
    const label = model.size
      ? `${modelName} (${Math.round(model.size / 1024 / 1024 / 1024 * 10) / 10} GB)`
      : modelName;
    ollamaInstalled.append(option(label, modelName));
  });
  if (Array.from(ollamaInstalled.options).some((item) => item.value === selectedModel)) {
    ollamaInstalled.value = selectedModel;
  }
}

async function checkHealth() {
  setBusy(true);
  setRequestState("Checking", "busy");
  try {
    const data = await requestJson("/health");
    healthState.textContent = data.ok ? "OK" : "Unavailable";
    providerState.textContent = data.provider || "Unknown";
    modelState.textContent = data.model || "Unknown";
    modeState.textContent = data.chat_mode || "Unknown";
    renderRaw(data);
    setRequestState("Ready", "ok");
  } catch (error) {
    healthState.textContent = "Error";
    appendMessage("system", error.message);
    setRequestState("Error", "error");
  } finally {
    setBusy(false);
  }
}

async function loadProviders() {
  setBusy(true);
  setRequestState("Loading", "busy");
  try {
    const data = await requestJson("/model-options");
    providerOptions = data.providers || [];
    providerState.textContent = data.active?.provider || "Unknown";
    modelState.textContent = data.active?.model || "Unknown";
    if (!modelInput.value.trim()) {
      providerSelect.value = data.active?.provider || providerSelect.value;
      if (providerSelect.value !== "ollama") {
        modelInput.value = data.active?.model || "";
      }
    }
    renderOllamaModels(data.ollama_installed?.models || []);
    setDatalist(ollamaCatalog, data.ollama_catalog || []);
    applyModelSuggestions(data, data.ollama_installed?.models || []);
    renderRaw(data);
    setRequestState("Ready", "ok");
  } catch (error) {
    appendMessage("system", error.message);
    setRequestState("Error", "error");
  } finally {
    setBusy(false);
  }
}

async function refreshOllamaModels() {
  setBusy(true);
  setRequestState("Ollama", "busy");
  try {
    const [optionsData, ollamaData] = await Promise.all([
      requestJson("/model-options"),
      requestJson("/ollama/models")
    ]);
    providerOptions = optionsData.providers || [];
    const models = ollamaData.models || [];
    renderOllamaModels(models);
    setDatalist(ollamaCatalog, optionsData.ollama_catalog || []);
    applyModelSuggestions(optionsData, models);
    renderRaw(ollamaData);
    setRequestState("Ready", "ok");
  } catch (error) {
    renderOllamaModels([]);
    appendMessage("system", error.message);
    setRequestState("Error", "error");
  } finally {
    setBusy(false);
  }
}

async function pullOllamaModel() {
  const model = ollamaInstall.value.trim() || modelInput.value.trim();
  if (!model) return;

  setBusy(true);
  setRequestState("Installing", "busy");
  try {
    const data = await requestJson("/ollama/models/pull", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model })
    });
    modelInput.value = model;
    providerSelect.value = "ollama";
    renderRaw(data);
    await refreshOllamaModels();
  } catch (error) {
    appendMessage("system", error.message);
    setRequestState("Error", "error");
    setBusy(false);
  }
}

async function deleteOllamaModel() {
  const model = ollamaInstalled.value;
  if (!model) return;
  if (!window.confirm(`Delete Ollama model "${model}"?`)) return;

  setBusy(true);
  setRequestState("Deleting", "busy");
  try {
    const data = await requestJson("/ollama/models", {
      method: "DELETE",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model })
    });
    renderRaw(data);
    await refreshOllamaModels();
  } catch (error) {
    appendMessage("system", error.message);
    setRequestState("Error", "error");
    setBusy(false);
  }
}

async function loadSession() {
  const sessionId = sessionInput.value.trim();
  if (!sessionId) {
    showEmptyState();
    return;
  }

  setBusy(true);
  setRequestState("Loading", "busy");
  try {
    const data = await requestJson(`/sessions/${encodeURIComponent(sessionId)}`);
    messages.replaceChildren();
    if (!data.messages || data.messages.length === 0) {
      showEmptyState();
    } else {
      data.messages.forEach((message) => {
        const row = appendMessage(message.role, message.content);
        if (message.metadata && message.metadata.documents) {
          renderCitations(row, message.metadata.documents);
        }
      });
    }
    renderCommandHistoryFromMessages(data.messages || []);
    currentSessionId = data.session_id;
    localStorage.setItem("chatSoftwareSessionId", currentSessionId);
    renderRaw(data);
    setRequestState("Ready", "ok");
  } catch (error) {
    appendMessage("system", error.message);
    setRequestState("Error", "error");
  } finally {
    setBusy(false);
  }
}

async function sendMessage(event) {
  event.preventDefault();
  const text = messageInput.value.trim();
  if (!text) return;

  const roots = activeRoots();
  const provider = providerSelect.value;
  const model = modelInput.value.trim();
  const payload = {
    message: text,
    session_id: currentSessionId || undefined,
    roots: roots.length > 0 ? roots : undefined,
    working_path: commandPath.value.trim() || ".",
    provider,
    model: model || undefined
  };
  if (typeof composerExtras === "function") Object.assign(payload, composerExtras());
  if (documentsMode.checked) {
    const knowledgeRootId = selectedKnowledgeRootId();
    if (!knowledgeRootId) {
      appendMessage("system", "Bind a root to a retrieval workspace before using Documents mode.");
      return;
    }
    payload.knowledge_root_id = knowledgeRootId;
  }

  appendMessage("user", text);
  messageInput.value = "";
  setBusy(true);
  setRequestState("Sending", "busy");
  const requestRow = appendCommandConsole(`POST /chat/stream ${shortJson(payload)}`, {
    status: "running",
    source: "user request"
  });
  const runtimeRows = new Map();
  let data = null;

  try {
    await requestNdjson("/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }, (runtimeEvent) => {
      if (runtimeEvent.type === "chat_started") {
        providerState.textContent = runtimeEvent.provider || provider;
        modelState.textContent = runtimeEvent.model || model || "Unknown";
        updateCommandConsole(
          requestRow,
          "running",
          `Using ${runtimeEvent.provider}/${runtimeEvent.model}`
        );
        return;
      }
      if (runtimeEvent.type === "tool_started") {
        const row = appendCommandConsole(toolCommand(runtimeEvent), {
          status: "running",
          source: runtimeEvent.name || "tool",
          detail: "Tool is running..."
        });
        runtimeRows.set(runtimeEvent.tool_id, row);
        return;
      }
      if (runtimeEvent.type === "tool_completed") {
        const row = runtimeRows.get(runtimeEvent.tool_id);
        if (row && runtimeEvent.command) {
          const code = row.querySelector("code");
          if (code) code.textContent = runtimeEvent.command;
        }
        updateCommandConsole(row, runtimeEvent.status || "completed", toolSummary(runtimeEvent));
        return;
      }
      if (runtimeEvent.type === "tool_failed") {
        updateCommandConsole(
          runtimeRows.get(runtimeEvent.tool_id),
          "failed",
          runtimeEvent.error || "Tool failed."
        );
        return;
      }
      if (runtimeEvent.type === "chat_completed") {
        data = runtimeEvent.data;
        return;
      }
      if (runtimeEvent.type === "chat_failed") {
        throw new Error(runtimeEvent.error || "Chat request failed.");
      }
    });

    if (!data) throw new Error("The chat stream ended without a final response.");
    currentSessionId = data.session_id;
    sessionInput.value = currentSessionId;
    localStorage.setItem("chatSoftwareSessionId", currentSessionId);
    const assistantRow = appendMessage("assistant", data.response || "");
    renderCitations(assistantRow, documentsFromToolCalls(data.tool_calls));
    providerState.textContent = data.provider || provider;
    modelState.textContent = data.model || model || modelState.textContent;
    if (data.model) {
      modelInput.value = data.model;
      localStorage.setItem("chatSoftwareModel", data.model);
    }
    renderToolCalls(data.tool_calls || []);
    updateCommandConsole(requestRow, "completed", `${(data.tool_calls || []).length} tool command(s) returned.`);
    renderOrganizePreviewFromToolCalls(data.tool_calls || []);
    renderRaw(data);
    setRequestState("Ready", "ok");
  } catch (error) {
    updateCommandConsole(requestRow, "failed", error.message);
    appendMessage("system", error.message);
    setRequestState("Error", "error");
  } finally {
    setBusy(false);
    messageInput.focus();
  }
}

function commandEndpoint(command) {
  return {
    list: "/files/list",
    stat: "/files/stat",
    read: "/files/read",
    glob: "/files/glob",
    grep: "/files/grep",
    mkdir: "/files/mkdir",
    write: "/files/write",
    copy: "/files/copy",
    move: "/files/move",
    edit: "/files/edit",
    git_status: "/commands/git/status",
    git_diff: "/commands/git/diff"
  }[command];
}

function commandPayload(command) {
  const roots = activeRoots();
  if (roots.length === 0) {
    throw new Error("Approve at least one file root before running file commands.");
  }

  const path = commandPath.value.trim() || ".";
  if (command === "list") return { path, roots };
  if (command === "stat") return { path, roots };
  if (command === "read") return { path, roots };
  if (command === "mkdir") return { path, roots };
  if (command === "write") return {
    path,
    roots,
    content: commandContent.value,
    overwrite: commandOverwrite.checked
  };
  if (command === "copy") {
    const destination = commandTarget.value.trim();
    if (!destination) throw new Error("Enter a destination path.");
    return { source: path, destination, roots };
  }
  if (command === "move") {
    const destination = commandTarget.value.trim();
    if (!destination) throw new Error("Enter a destination path.");
    return { source: path, destination, roots };
  }
  if (command === "edit") {
    const oldText = commandQuery.value;
    if (!oldText) throw new Error("Enter text to replace.");
    return {
      path,
      roots,
      old_text: oldText,
      new_text: commandContent.value,
      count: 1
    };
  }
  if (command === "glob") {
    return { root: path, pattern: commandPattern.value.trim() || "*", roots };
  }
  if (command === "grep") {
    const query = commandQuery.value.trim();
    if (!query) throw new Error("Enter a search query.");
    return {
      root: path,
      query,
      include: commandInclude.value.trim() || "*",
      roots
    };
  }
  if (command === "git_status") return { repo_path: path, roots };
  if (command === "git_diff") return { repo_path: path, roots, stat_only: true };
  throw new Error(`Unsupported command: ${command}`);
}

async function runCommand() {
  const command = commandSelect.value;
  setBusy(true);
  setRequestState("Running", "busy");
  let commandRow = null;
  try {
    const payload = commandPayload(command);
    commandRow = appendCommandConsole(`POST ${commandEndpoint(command)} ${shortJson(payload)}`, {
      status: "running",
      source: command
    });
    const data = await requestJson(commandEndpoint(command), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    renderRaw(data);
    renderToolCalls([
      {
        name: command,
        command: `POST ${commandEndpoint(command)} ${shortJson(payload)}`,
        arguments: payload,
        result: data
      }
    ]);
    updateCommandConsole(commandRow, "completed", toolSummary({ name: command, result: data }));
    setRequestState("Ready", "ok");
  } catch (error) {
    updateCommandConsole(commandRow, "failed", error.message);
    appendMessage("system", error.message);
    setRequestState("Error", "error");
  } finally {
    setBusy(false);
  }
}

function organizePayload() {
  const maxFiles = Number.parseInt(organizeMaxFiles.value, 10);
  return {
    root: activeOrganizeRoot(),
    path: organizePath.value.trim() || ".",
    strategy: organizeStrategy.value,
    file_filter: organizeFilter.value,
    max_files: Number.isFinite(maxFiles) ? Math.min(Math.max(maxFiles, 1), 5000) : 1000
  };
}

async function previewOrganizePlan() {
  let payload;
  try {
    payload = organizePayload();
  } catch (error) {
    appendMessage("system", error.message);
    setRequestState("Error", "error");
    return;
  }

  setBusy(true);
  setRequestState("Planning", "busy");
  const commandRow = appendCommandConsole(`POST /organize/plan ${shortJson(payload)}`, {
    status: "running",
    source: "organize_plan"
  });
  try {
    const data = await requestJson("/organize/plan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    renderOrganizePlan(data);
    renderToolCalls([{ name: "organize_plan", command: `POST /organize/plan ${shortJson(payload)}`, arguments: payload, result: data }]);
    updateCommandConsole(commandRow, "completed", toolSummary({ name: "organize_plan", result: data }));
    renderRaw(data);
    setRequestState("Ready", "ok");
  } catch (error) {
    updateCommandConsole(commandRow, "failed", error.message);
    appendMessage("system", error.message);
    setRequestState("Error", "error");
  } finally {
    setBusy(false);
  }
}

async function reloadOrganizePlan(planId) {
  return requestJson(`/organize/plans/${encodeURIComponent(planId)}`);
}

async function applyOrganizePlan() {
  const planId = currentOrganizePlan?.plan_id;
  if (!planId) return;
  if (!window.confirm(`Apply organize plan ${planId}? This will move files inside the approved root.`)) return;

  setBusy(true);
  setRequestState("Applying", "busy");
  const commandRow = appendCommandConsole(`POST /organize/apply ${shortJson({ plan_id: planId })}`, {
    status: "running",
    source: "organize_apply"
  });
  try {
    const result = await requestJson("/organize/apply", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ plan_id: planId })
    });
    const plan = await reloadOrganizePlan(planId);
    renderOrganizePlan(plan);
    renderToolCalls([{ name: "organize_apply", command: `POST /organize/apply ${shortJson({ plan_id: planId })}`, arguments: { plan_id: planId }, result }]);
    updateCommandConsole(commandRow, "completed", toolSummary({ name: "organize_apply", result }));
    renderRaw({ result, plan });
    setRequestState("Ready", "ok");
  } catch (error) {
    updateCommandConsole(commandRow, "failed", error.message);
    appendMessage("system", error.message);
    setRequestState("Error", "error");
  } finally {
    setBusy(false);
  }
}

async function undoOrganizePlan() {
  const planId = currentOrganizePlan?.plan_id;
  if (!planId) return;
  if (!window.confirm(`Undo organize plan ${planId}? This will move files back where possible.`)) return;

  setBusy(true);
  setRequestState("Undoing", "busy");
  const commandRow = appendCommandConsole(`POST /organize/undo ${shortJson({ plan_id: planId })}`, {
    status: "running",
    source: "organize_undo"
  });
  try {
    const result = await requestJson("/organize/undo", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ plan_id: planId })
    });
    const plan = await reloadOrganizePlan(planId);
    renderOrganizePlan(plan);
    renderToolCalls([{ name: "organize_undo", command: `POST /organize/undo ${shortJson({ plan_id: planId })}`, arguments: { plan_id: planId }, result }]);
    updateCommandConsole(commandRow, "completed", toolSummary({ name: "organize_undo", result }));
    renderRaw({ result, plan });
    setRequestState("Ready", "ok");
  } catch (error) {
    updateCommandConsole(commandRow, "failed", error.message);
    appendMessage("system", error.message);
    setRequestState("Error", "error");
  } finally {
    setBusy(false);
  }
}

document.querySelector("#check-health").addEventListener("click", checkHealth);
document.querySelector("#load-providers").addEventListener("click", loadProviders);
document.querySelector("#approve-root").addEventListener("click", approveRoot);
document.querySelector("#remove-root").addEventListener("click", removeSelectedRoot);
document.querySelector("#run-command").addEventListener("click", runCommand);
organizePlanButton.addEventListener("click", previewOrganizePlan);
organizeApplyButton.addEventListener("click", applyOrganizePlan);
organizeUndoButton.addEventListener("click", undoOrganizePlan);
document.querySelector("#refresh-ollama").addEventListener("click", refreshOllamaModels);
document.querySelector("#pull-ollama").addEventListener("click", pullOllamaModel);
document.querySelector("#delete-ollama").addEventListener("click", deleteOllamaModel);
document.querySelector("#load-session").addEventListener("click", loadSession);
document.querySelector("#bind-knowledge").addEventListener("click", bindKnowledgeRoot);
document.querySelector("#unbind-knowledge").addEventListener("click", unbindKnowledgeRoot);
document.querySelector("#revoke-root").addEventListener("click", revokeKnowledgeRoot);
document.querySelector("#enable-ingest").addEventListener("click", enableIngest);
document.querySelector("#ingest-preview").addEventListener("click", previewIngest);
ingestApproveButton.addEventListener("click", approveIngest);
ingestDiscardButton.addEventListener("click", discardIngest);
providerSelect.addEventListener("change", () => {
  localStorage.removeItem("chatSoftwareModel");
  updateProviderNote();
  loadProviders();
});
modelInput.addEventListener("change", () => {
  const model = modelInput.value;
  localStorage.setItem("chatSoftwareModel", model);
  if (providerSelect.value === "ollama") {
    ollamaInstalled.value = model;
  }
  modelState.textContent = model || "Unknown";
  updateProviderNote();
});
ollamaInstalled.addEventListener("change", () => {
  const model = ollamaInstalled.value;
  if (!model) return;
  providerSelect.value = "ollama";
  modelInput.value = model;
  localStorage.setItem("chatSoftwareProvider", "ollama");
  localStorage.setItem("chatSoftwareModel", model);
  providerState.textContent = "ollama";
  modelState.textContent = model;
  updateProviderNote();
});
document.querySelector("#new-session").addEventListener("click", () => {
  currentSessionId = "";
  sessionInput.value = "";
  localStorage.removeItem("chatSoftwareSessionId");
  showEmptyState();
  renderToolCalls([]);
  renderOrganizeEmpty();
  renderCommandConsoleEmpty();
  renderRaw(null);
  setRequestState("Idle");
});
document.querySelector("#clear-log").addEventListener("click", () => {
  renderToolCalls([]);
  renderRaw(null);
});
document.querySelector("#clear-command-console").addEventListener("click", renderCommandConsoleEmpty);
chatForm.addEventListener("submit", sendMessage);

showEmptyState();
renderToolCalls([]);
renderCommandConsoleEmpty();
renderOrganizeEmpty();
renderApprovedRoots();
checkHealth();
loadProviders();
loadKnowledge();
