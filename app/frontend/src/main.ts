import "./styles.css";

type ChatResponse = {
  session_id: string;
  provider?: string;
  model?: string;
  response: string;
  tool_calls: Array<{
    name: string;
    command?: string;
    status?: string;
    arguments: Record<string, unknown>;
    result?: OrganizePlan & Record<string, unknown>;
  }>;
};

type ModelOptions = {
  active?: { provider?: string; model?: string };
  providers?: Array<{ provider: string; installed_count?: number; model_error?: string | null }>;
  ollama_installed?: {
    models?: Array<{ name?: string; model?: string }>;
    recommended_model?: string | null;
  };
  registered_models?: Array<{ provider?: string; model?: string }>;
};

type OrganizeAction = {
  op?: string;
  from?: string;
  to?: string;
  path?: string;
  status?: string;
};

type OrganizePlan = {
  plan_id?: string;
  status?: string;
  strategy?: string;
  path?: string;
  summary?: Record<string, number | boolean>;
  actions?: OrganizeAction[];
  warnings?: string[];
  skipped?: Array<{ path?: string; reason?: string }>;
};

const backendUrl = localStorage.getItem("chatSoftwareBackendUrl") ?? "http://127.0.0.1:8765";
let sessionId: string | null = localStorage.getItem("chatSoftwareSessionId");
let currentPlan: OrganizePlan | null = null;

const app = document.querySelector<HTMLDivElement>("#app");

if (!app) {
  throw new Error("Missing #app mount point");
}

app.innerHTML = `
  <section class="workspace">
    <aside class="sidebar">
      <div>
        <p class="eyebrow">Local agent</p>
        <h1>Chat Software</h1>
      </div>
      <label>
        Backend
        <input id="backend-url" value="${escapeHtml(backendUrl)}" />
      </label>
      <label>
        Approved root
        <input id="root-path" placeholder="/path/to/folder" />
      </label>
      <button id="approve-root" type="button">Approve Root</button>
      <label>
        Provider
        <select id="provider-select">
          <option value="ollama">Ollama</option>
          <option value="openai">OpenAI</option>
          <option value="anthropic">Claude</option>
        </select>
      </label>
      <label>
        Model
        <input id="model-input" list="model-suggestions" placeholder="Loading installed models..." />
        <datalist id="model-suggestions"></datalist>
      </label>
      <button id="refresh-models" type="button">Refresh Models</button>
      <p id="model-note" class="note">Models have not been checked.</p>
      <button id="health" type="button">Check</button>
      <pre id="status" class="status">Backend not checked.</pre>
    </aside>
    <section class="chat">
      <div id="messages" class="messages"></div>
      <form id="chat-form" class="composer">
        <textarea id="message" rows="3" placeholder="Find files about Acme and summarize"></textarea>
        <button type="submit">Send</button>
      </form>
    </section>
    <aside class="preview-panel">
      <header class="preview-header">
        <div>
          <p class="eyebrow">Safe file management</p>
          <h2>Organize Preview</h2>
        </div>
        <span id="plan-state" class="pill">No plan</span>
      </header>
      <section class="command-console-panel">
        <div class="preview-header">
          <div>
            <p class="eyebrow">Agent runtime</p>
            <h2>Command Console</h2>
          </div>
          <button id="clear-command-console" type="button" class="secondary compact-button">Clear</button>
        </div>
        <div id="command-console" class="command-console"></div>
      </section>
      <label>
        Strategy
        <select id="organize-strategy">
          <option value="by_type">By type</option>
          <option value="by_date">By date</option>
        </select>
      </label>
      <label>
        Path
        <input id="organize-path" value="." />
      </label>
      <label>
        Max files
        <input id="organize-max-files" type="number" min="1" max="5000" value="1000" />
      </label>
      <div class="preview-actions">
        <button id="organize-plan" type="button">Preview</button>
        <button id="organize-apply" type="button" class="secondary" disabled>Apply</button>
        <button id="organize-undo" type="button" class="secondary" disabled>Undo</button>
      </div>
      <div id="organize-preview" class="organize-preview"></div>
    </aside>
  </section>
`;

const backendInput = document.querySelector<HTMLInputElement>("#backend-url")!;
const rootInput = document.querySelector<HTMLInputElement>("#root-path")!;
const approveRootButton = document.querySelector<HTMLButtonElement>("#approve-root")!;
const providerSelect = document.querySelector<HTMLSelectElement>("#provider-select")!;
const modelInput = document.querySelector<HTMLInputElement>("#model-input")!;
const modelSuggestions = document.querySelector<HTMLDataListElement>("#model-suggestions")!;
const refreshModelsButton = document.querySelector<HTMLButtonElement>("#refresh-models")!;
const modelNote = document.querySelector<HTMLParagraphElement>("#model-note")!;
const statusBox = document.querySelector<HTMLPreElement>("#status")!;
const messages = document.querySelector<HTMLDivElement>("#messages")!;
const form = document.querySelector<HTMLFormElement>("#chat-form")!;
const messageInput = document.querySelector<HTMLTextAreaElement>("#message")!;
const strategyInput = document.querySelector<HTMLSelectElement>("#organize-strategy")!;
const organizePathInput = document.querySelector<HTMLInputElement>("#organize-path")!;
const maxFilesInput = document.querySelector<HTMLInputElement>("#organize-max-files")!;
const planState = document.querySelector<HTMLSpanElement>("#plan-state")!;
const organizePreview = document.querySelector<HTMLDivElement>("#organize-preview")!;
const previewButton = document.querySelector<HTMLButtonElement>("#organize-plan")!;
const applyButton = document.querySelector<HTMLButtonElement>("#organize-apply")!;
const undoButton = document.querySelector<HTMLButtonElement>("#organize-undo")!;
const commandConsole = document.querySelector<HTMLDivElement>("#command-console")!;
const clearCommandConsoleButton = document.querySelector<HTMLButtonElement>("#clear-command-console")!;

function activeBackend(): string {
  const value = backendInput.value.trim() || "http://127.0.0.1:8765";
  localStorage.setItem("chatSoftwareBackendUrl", value);
  return value;
}

function appendMessage(role: "user" | "assistant", content: string): void {
  const row = document.createElement("article");
  row.className = `message ${role}`;
  row.textContent = content;
  messages.append(row);
  messages.scrollTop = messages.scrollHeight;
}

async function requestJson<T>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`${activeBackend()}${path}`, options);
  const text = await response.text();
  const data = text ? JSON.parse(text) : null;
  if (!response.ok) {
    throw new Error(typeof data === "object" ? JSON.stringify(data, null, 2) : response.statusText);
  }
  return data as T;
}

async function requestNdjson(
  path: string,
  options: RequestInit,
  onEvent: (event: Record<string, any>) => void
): Promise<void> {
  const response = await fetch(`${activeBackend()}${path}`, options);
  if (!response.ok) {
    const text = await response.text();
    let detail = text;
    try {
      const parsed = text ? JSON.parse(text) : null;
      detail = parsed?.detail ?? text;
    } catch {
      // Keep the plain response text.
    }
    throw new Error(`${response.status} ${response.statusText}${detail ? `: ${detail}` : ""}`);
  }
  if (!response.body) throw new Error("The chat response stream is unavailable.");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value ?? new Uint8Array(), { stream: !done });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) {
      if (line.trim()) onEvent(JSON.parse(line));
    }
    if (done) break;
  }
  if (buffer.trim()) onEvent(JSON.parse(buffer));
}

function modelName(model: { name?: string; model?: string }): string {
  return model.name ?? model.model ?? "";
}

async function loadModels(): Promise<void> {
  modelNote.textContent = "Loading installed models...";
  const data = await requestJson<ModelOptions>("/model-options");
  const selectedProvider = providerSelect.value || data.active?.provider || "ollama";
  providerSelect.value = selectedProvider;
  let suggestions: string[] = [];
  if (selectedProvider === "ollama") {
    suggestions = (data.ollama_installed?.models ?? []).map(modelName).filter(Boolean);
  } else {
    suggestions = (data.registered_models ?? [])
      .filter((item) => item.provider === selectedProvider)
      .map((item) => item.model ?? "")
      .filter(Boolean);
  }
  modelSuggestions.replaceChildren(
    ...suggestions.map((name) => {
      const option = document.createElement("option");
      option.value = name;
      return option;
    })
  );
  if (!modelInput.value.trim() || !suggestions.includes(modelInput.value.trim())) {
    modelInput.value = suggestions[0] ?? data.active?.model ?? "";
  }
  localStorage.setItem("chatSoftwareProvider", providerSelect.value);
  localStorage.setItem("chatSoftwareModel", modelInput.value.trim());
  const providerInfo = (data.providers ?? []).find((item) => item.provider === selectedProvider);
  modelNote.textContent = providerInfo?.model_error
    ? `Model service error: ${providerInfo.model_error}`
    : selectedProvider === "ollama"
      ? `${providerInfo?.installed_count ?? suggestions.length} installed Ollama model(s).`
      : `${suggestions.length} configured ${selectedProvider} model(s).`;
}

async function checkHealth(): Promise<void> {
  statusBox.textContent = "Checking...";
  const data = await requestJson<unknown>("/health");
  statusBox.textContent = JSON.stringify(data, null, 2);
}

async function approveRoot(): Promise<void> {
  const path = rootInput.value.trim();
  if (!path) throw new Error("Enter a root directory first.");
  const data = await requestJson<{ path: string }>("/roots/approve", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path })
  });
  rootInput.value = data.path;
  statusBox.textContent = JSON.stringify(data, null, 2);
}

function updatePlanButtons(): void {
  const status = currentPlan?.status ?? "";
  const hasMoves = (currentPlan?.actions ?? []).some((action) => action.op === "move");
  applyButton.disabled = status !== "planned" || !hasMoves;
  undoButton.disabled = !["applied", "partial"].includes(status);
}

function escapeHtml(value: unknown): string {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "\"": "&quot;",
    "'": "&#39;"
  })[char] ?? char);
}

function shortJson(value: unknown): string {
  return JSON.stringify(value, (_key, current) => {
    if (Array.isArray(current) && current.length > 4) {
      return [...current.slice(0, 4), `... ${current.length - 4} more`];
    }
    return current;
  });
}

function renderEmptyCommandConsole(): void {
  commandConsole.innerHTML = `<div class="empty">No agent commands have run yet.</div>`;
}

function appendCommandConsole(
  command: string,
  options: { status?: string; source?: string; detail?: string } = {}
): HTMLElement {
  const existingEmpty = commandConsole.querySelector(".empty");
  if (existingEmpty) existingEmpty.remove();
  const status = options.status ?? "completed";
  const row = document.createElement("article");
  row.className = `command-line ${status}`;
  row.innerHTML = `
    <div class="command-meta">
      <span class="command-status">${escapeHtml(status)}</span>
      <span class="command-source">${escapeHtml(options.source ?? "agent")}</span>
    </div>
    <code>${escapeHtml(command)}</code>
    ${options.detail ? `<p>${escapeHtml(options.detail)}</p>` : ""}
  `;
  commandConsole.append(row);
  commandConsole.scrollTop = commandConsole.scrollHeight;
  return row;
}

function updateCommandConsole(row: HTMLElement | null, status: string, detail = ""): void {
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

function toolCommand(call: ChatResponse["tool_calls"][number]): string {
  if (call.command) return call.command;
  return `${call.name}(${Object.entries(call.arguments ?? {}).map(([key, value]) => `${key}=${shortJson(value)}`).join(", ")})`;
}

function toolSummary(call: ChatResponse["tool_calls"][number]): string {
  const result = call.result ?? {};
  if (result.op === "copy") return `Copied ${result.from} to ${result.to}.`;
  if (result.op === "move") return `Moved ${result.from} to ${result.to}.`;
  if (result.op === "edit") return `${result.replacements ?? 0} replacement(s) completed.`;
  if (result.op === "mkdir") return result.created ? `Created ${result.path}.` : `${result.path} already exists.`;
  if (result.op === "write") return `${result.overwritten ? "Updated" : "Created"} ${result.path}.`;
  return "Tool completed.";
}

function appendToolCallsToConsole(calls: ChatResponse["tool_calls"]): void {
  calls.forEach((call) => {
    appendCommandConsole(toolCommand(call), {
      status: "completed",
      source: call.name,
      detail: "Tool completed."
    });
  });
}

function renderEmptyPlan(text = "No organize preview yet."): void {
  currentPlan = null;
  planState.textContent = "No plan";
  planState.className = "pill";
  organizePreview.innerHTML = `<div class="empty">${text}</div>`;
  updatePlanButtons();
}

function renderPlan(plan: OrganizePlan | null): void {
  if (!plan) {
    renderEmptyPlan();
    return;
  }

  currentPlan = plan;
  const summary = plan.summary ?? {};
  planState.textContent = plan.status ?? "planned";
  planState.className = `pill ${["applied", "undone"].includes(plan.status ?? "") ? "ok" : ""}`;
  const actions = plan.actions ?? [];
  const warnings = plan.warnings ?? [];
  const skipped = plan.skipped ?? [];

  const rows = actions.slice(0, 80).map((action) => `
    <div class="plan-row">
      <span class="op">${escapeHtml(action.op)}</span>
      <span>${escapeHtml(action.from ?? action.path)}</span>
      <span>${escapeHtml(action.to)}</span>
      <span>${escapeHtml(action.status ?? plan.status ?? "planned")}</span>
    </div>
  `).join("");

  organizePreview.innerHTML = `
    <div class="plan-stats">
      <div><strong>${summary.moves ?? 0}</strong><span>moves</span></div>
      <div><strong>${summary.new_dirs ?? 0}</strong><span>folders</span></div>
      <div><strong>${summary.skipped ?? 0}</strong><span>skipped</span></div>
      <div><strong>${summary.files_seen ?? 0}</strong><span>seen</span></div>
    </div>
    <p class="plan-meta">${escapeHtml(plan.strategy ?? "strategy")} in ${escapeHtml(plan.path ?? ".")}</p>
    ${warnings.length ? `<ul class="warnings">${warnings.slice(0, 5).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>` : ""}
    <div class="plan-table">
      <div class="plan-row header"><span>Op</span><span>Source</span><span>Target</span><span>Status</span></div>
      ${rows || `<div class="empty">No file moves needed.</div>`}
    </div>
    ${skipped.length ? `<div class="skipped">${skipped.slice(0, 8).map((item) => `<span>${escapeHtml(item.path)}: ${escapeHtml(item.reason)}</span>`).join("")}</div>` : ""}
  `;
  updatePlanButtons();
}

function activeRoot(): string {
  const root = rootInput.value.trim();
  if (!root) throw new Error("Enter and approve a root directory before previewing.");
  return root;
}

function organizePayload(): Record<string, unknown> {
  const maxFiles = Number.parseInt(maxFilesInput.value, 10);
  return {
    root: activeRoot(),
    strategy: strategyInput.value,
    path: organizePathInput.value.trim() || ".",
    max_files: Number.isFinite(maxFiles) ? Math.min(Math.max(maxFiles, 1), 5000) : 1000
  };
}

async function previewOrganizePlan(): Promise<void> {
  const payload = organizePayload();
  const row = appendCommandConsole(`POST /organize/plan ${shortJson(payload)}`, {
    status: "running",
    source: "organize_plan"
  });
  const plan = await requestJson<OrganizePlan>("/organize/plan", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  renderPlan(plan);
  updateCommandConsole(row, "completed", `${plan.actions?.length ?? 0} planned action(s).`);
  statusBox.textContent = JSON.stringify(plan, null, 2);
}

async function reloadPlan(planId: string): Promise<OrganizePlan> {
  return requestJson<OrganizePlan>(`/organize/plans/${encodeURIComponent(planId)}`);
}

async function applyPlan(): Promise<void> {
  const planId = currentPlan?.plan_id;
  if (!planId) return;
  if (!window.confirm(`Apply organize plan ${planId}?`)) return;
  const row = appendCommandConsole(`POST /organize/apply ${shortJson({ plan_id: planId })}`, {
    status: "running",
    source: "organize_apply"
  });
  const result = await requestJson<unknown>("/organize/apply", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ plan_id: planId })
  });
  const plan = await reloadPlan(planId);
  renderPlan(plan);
  updateCommandConsole(row, "completed", "Apply completed.");
  statusBox.textContent = JSON.stringify({ result, plan }, null, 2);
}

async function undoPlan(): Promise<void> {
  const planId = currentPlan?.plan_id;
  if (!planId) return;
  if (!window.confirm(`Undo organize plan ${planId}?`)) return;
  const row = appendCommandConsole(`POST /organize/undo ${shortJson({ plan_id: planId })}`, {
    status: "running",
    source: "organize_undo"
  });
  const result = await requestJson<unknown>("/organize/undo", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ plan_id: planId })
  });
  const plan = await reloadPlan(planId);
  renderPlan(plan);
  updateCommandConsole(row, "completed", "Undo completed.");
  statusBox.textContent = JSON.stringify({ result, plan }, null, 2);
}

function renderPlanFromToolCalls(calls: ChatResponse["tool_calls"]): void {
  const plan = calls.find((call) => call.name === "organize_plan" && call.result?.plan_id)?.result;
  if (plan) renderPlan(plan);
}

document.querySelector<HTMLButtonElement>("#health")!.addEventListener("click", () => {
  checkHealth().catch((error: unknown) => {
    statusBox.textContent = error instanceof Error ? error.message : String(error);
  });
});

approveRootButton.addEventListener("click", () => {
  approveRoot().catch((error: unknown) => {
    statusBox.textContent = error instanceof Error ? error.message : String(error);
  });
});

refreshModelsButton.addEventListener("click", () => {
  loadModels().catch((error: unknown) => {
    modelNote.textContent = error instanceof Error ? error.message : String(error);
  });
});

providerSelect.addEventListener("change", () => {
  modelInput.value = "";
  loadModels().catch((error: unknown) => {
    modelNote.textContent = error instanceof Error ? error.message : String(error);
  });
});

modelInput.addEventListener("change", () => {
  localStorage.setItem("chatSoftwareModel", modelInput.value.trim());
});

previewButton.addEventListener("click", () => {
  previewOrganizePlan().catch((error: unknown) => {
    statusBox.textContent = error instanceof Error ? error.message : String(error);
  });
});

applyButton.addEventListener("click", () => {
  applyPlan().catch((error: unknown) => {
    statusBox.textContent = error instanceof Error ? error.message : String(error);
  });
});

undoButton.addEventListener("click", () => {
  undoPlan().catch((error: unknown) => {
    statusBox.textContent = error instanceof Error ? error.message : String(error);
  });
});

clearCommandConsoleButton.addEventListener("click", renderEmptyCommandConsole);

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = messageInput.value.trim();
  if (!text) return;

  messageInput.value = "";
  appendMessage("user", text);

  const roots = rootInput.value.trim() ? [rootInput.value.trim()] : undefined;
  const payload = {
    message: text,
    session_id: sessionId,
    roots,
    working_path: organizePathInput.value.trim() || ".",
    provider: providerSelect.value,
    model: modelInput.value.trim() || undefined
  };
  const row = appendCommandConsole(`POST /chat/stream ${shortJson(payload)}`, {
    status: "running",
    source: "user request"
  });
  const runtimeRows = new Map<string, HTMLElement>();
  let data: ChatResponse | null = null;
  try {
    await requestNdjson("/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }, (runtimeEvent) => {
      if (runtimeEvent.type === "chat_started") {
        updateCommandConsole(row, "running", `Using ${runtimeEvent.provider}/${runtimeEvent.model}`);
      } else if (runtimeEvent.type === "tool_started") {
        const toolEvent = runtimeEvent as ChatResponse["tool_calls"][number] & { tool_id: string };
        runtimeRows.set(runtimeEvent.tool_id, appendCommandConsole(toolCommand(toolEvent), {
          status: "running",
          source: runtimeEvent.name,
          detail: "Tool is running..."
        }));
      } else if (runtimeEvent.type === "tool_completed") {
        const toolEvent = runtimeEvent as ChatResponse["tool_calls"][number] & { tool_id: string };
        const runtimeRow = runtimeRows.get(runtimeEvent.tool_id) ?? null;
        const code = runtimeRow?.querySelector("code");
        if (code && runtimeEvent.command) code.textContent = runtimeEvent.command;
        updateCommandConsole(runtimeRow, runtimeEvent.status ?? "completed", toolSummary(toolEvent));
      } else if (runtimeEvent.type === "tool_failed") {
        updateCommandConsole(
          runtimeRows.get(runtimeEvent.tool_id) ?? null,
          "failed",
          runtimeEvent.error ?? "Tool failed."
        );
      } else if (runtimeEvent.type === "chat_completed") {
        data = runtimeEvent.data as ChatResponse;
      } else if (runtimeEvent.type === "chat_failed") {
        throw new Error(runtimeEvent.error ?? "Chat request failed.");
      }
    });

    if (!data) throw new Error("The chat stream ended without a final response.");
    const completed = data as ChatResponse;
    sessionId = completed.session_id;
    localStorage.setItem("chatSoftwareSessionId", sessionId);
    appendMessage("assistant", completed.response);
    if (completed.model) {
      modelInput.value = completed.model;
      localStorage.setItem("chatSoftwareModel", completed.model);
    }
    updateCommandConsole(row, "completed", `${completed.tool_calls?.length ?? 0} tool command(s) returned.`);
    renderPlanFromToolCalls(completed.tool_calls ?? []);
  } catch (error: unknown) {
    const detail = error instanceof Error ? error.message : String(error);
    updateCommandConsole(row, "failed", detail);
    appendMessage("assistant", `Backend request failed: ${detail}`);
  }
});

renderEmptyCommandConsole();
renderEmptyPlan();
providerSelect.value = localStorage.getItem("chatSoftwareProvider") ?? "ollama";
modelInput.value = localStorage.getItem("chatSoftwareModel") ?? "";
loadModels().catch((error: unknown) => {
  modelNote.textContent = error instanceof Error ? error.message : String(error);
});
