// Platform views: conversations, chat skills/tools, research, teams, skills, sandbox and settings.
// Loaded after app.js and reuses its helpers (requestJson, requestNdjson, appendMessage, option, ...).
// Model and web output is always inserted as text, never as HTML.

const viewTitles = {
  chat: "Conversation",
  research: "Deep research",
  teams: "Agent teams",
  skills: "Skills",
  sandbox: "Sandbox",
  settings: "Settings"
};
const KNOWN_SKILL_TOOLS = ["web_search", "fetch_url", "search_documents", "run_code"];
let skillsCache = [];
let teamsCache = [];
let currentTeamSession = null;
let pullPollTimer = null;

function el(tag, text, className) {
  const node = document.createElement(tag);
  if (text !== undefined && text !== null) node.textContent = text;
  if (className) node.className = className;
  return node;
}

function safeLink(url, text) {
  const link = el("a", text || url);
  if (/^https?:\/\//i.test(url)) {
    link.href = url;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
  }
  return link;
}

function showView(name) {
  document.querySelectorAll(".tab").forEach((tab) => {
    const active = tab.dataset.view === name;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", active ? "true" : "false");
  });
  document.querySelectorAll(".view").forEach((view) => {
    view.hidden = view.id !== `view-${name}`;
  });
  document.querySelector("#view-title").textContent = viewTitles[name] || "";
  try {
    localStorage.setItem("chatSoftwareView", name);
  } catch (error) {
    /* storage unavailable */
  }
  if (name === "research") refreshResearchStatus();
  if (name === "teams") loadTeams();
  if (name === "skills") loadSkills();
  if (name === "sandbox") refreshSandbox();
  if (name === "settings") loadSettings();
}

// ---- Conversations -------------------------------------------------------------------------
async function loadConversations() {
  const list = document.querySelector("#session-list");
  const q = document.querySelector("#session-search").value.trim();
  const filter = document.querySelector("#session-filter").value;
  try {
    const data = await requestJson(`/sessions?archived=${filter}${q ? `&q=${encodeURIComponent(q)}` : ""}`);
    list.replaceChildren();
    if (!data.sessions.length) list.append(el("li", "No conversations", "note"));
    data.sessions.forEach((session) => list.append(conversationRow(session)));
  } catch (error) {
    list.replaceChildren(el("li", error.message, "note"));
  }
}

function conversationRow(session) {
  const row = el("li", null, `session-row${session.session_id === currentSessionId ? " current" : ""}`);
  const open = el("button", null, "session-open");
  open.type = "button";
  open.append(el("span", `${session.pinned ? "★ " : ""}${session.title}`, "session-title"),
              el("span", session.kind, `kind-badge ${session.kind}`));
  open.title = `${session.message_count} messages`;
  open.addEventListener("click", () => openConversation(session));
  const actions = el("div", null, "session-actions");
  const button = (label, title, handler, cls = "secondary") => {
    const b = el("button", label, `${cls} mini`);
    b.type = "button";
    b.title = title;
    b.addEventListener("click", handler);
    actions.append(b);
  };
  button(session.pinned ? "Unpin" : "Pin", "Pin to top", () => patchSession(session.session_id, { pinned: !session.pinned }));
  button("Rename", "Rename", () => {
    const title = window.prompt("Conversation title", session.title);
    if (title !== null && title.trim()) patchSession(session.session_id, { title: title.trim() });
  });
  button(session.archived ? "Restore" : "Archive", "Archive keeps it but hides it from the list",
         () => patchSession(session.session_id, { archived: !session.archived }));
  button("Delete", "Delete permanently", async () => {
    if (!window.confirm(`Permanently delete "${session.title}" and all its messages? This cannot be undone.`)) return;
    try {
      await requestJson(`/sessions/${encodeURIComponent(session.session_id)}`, { method: "DELETE" });
      if (session.session_id === currentSessionId) {
        currentSessionId = "";
        sessionInput.value = "";
        showEmptyState();
      }
    } catch (error) {
      appendMessage("system", error.message);
    }
    loadConversations();
  }, "danger");
  row.append(open, actions);
  return row;
}

async function patchSession(sessionId, body) {
  try {
    await requestJson(`/sessions/${encodeURIComponent(sessionId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
  } catch (error) {
    appendMessage("system", error.message);
  }
  loadConversations();
}

async function openConversation(session) {
  if (session.kind === "team" && session.meta && session.meta.team_id) {
    showView("teams");
    await loadTeams(session.meta.team_id);
    await openTeamSession(session.session_id, session.title);
    return;
  }
  showView("chat");
  sessionInput.value = session.session_id;
  await loadSession();
  loadConversations();
}

// ---- Chat composer: skills and tools -------------------------------------------------------
function composerExtras() {
  const extras = {};
  const skills = Array.from(document.querySelector("#chat-skills").selectedOptions).map((o) => o.value);
  if (skills.length) extras.skill_ids = skills;
  const tools = [];
  if (document.querySelector("#chat-tool-web").checked) tools.push("web_search", "fetch_url");
  if (document.querySelector("#chat-tool-code").checked) tools.push("run_code");
  if (tools.length) extras.tools = tools;
  return extras;
}

async function loadSkillOptions() {
  try {
    const data = await requestJson("/skills");
    skillsCache = data.skills;
    const select = document.querySelector("#chat-skills");
    const selected = new Set(Array.from(select.selectedOptions).map((o) => o.value));
    select.replaceChildren(...data.skills.map((skill) => {
      const opt = option(skill.name, skill.id);
      opt.title = skill.description;
      opt.selected = selected.has(skill.id);
      return opt;
    }));
  } catch (error) {
    /* skills are optional for chat */
  }
}

// ---- Research ------------------------------------------------------------------------------
async function refreshResearchStatus() {
  const note = document.querySelector("#research-status");
  try {
    const data = await requestJson("/research/status");
    note.textContent = data.web_search_provider
      ? `Web search provider: ${data.web_search_provider}.`
      : "No web search provider configured yet. Add a SearXNG URL or a Brave/Tavily key in Settings, or research bound documents only.";
  } catch (error) {
    note.textContent = error.message;
  }
}

function researchStep(text, cls = "") {
  const item = el("li", text, cls);
  document.querySelector("#research-steps").append(item);
  return item;
}

async function startResearch() {
  const question = document.querySelector("#research-question").value.trim();
  if (question.length < 3) return;
  const useDocs = document.querySelector("#research-docs").checked;
  const payload = {
    question,
    depth: document.querySelector("#research-depth").value,
    use_web: document.querySelector("#research-web").checked,
    documents_root_id: useDocs ? selectedKnowledgeRootId() || undefined : undefined,
    provider: providerSelect.value,
    model: modelInput.value || undefined
  };
  const steps = document.querySelector("#research-steps");
  const report = document.querySelector("#research-report");
  steps.replaceChildren();
  report.hidden = true;
  const button = document.querySelector("#research-start");
  button.disabled = true;
  try {
    await requestNdjson("/research/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }, (event) => {
      if (event.type === "session") researchStep(`Using ${event.provider}/${event.model}`, "muted");
      if (event.type === "plan") {
        researchStep(`Plan: ${event.approach}`, "step-plan");
        event.sub_questions.forEach((q, i) => researchStep(`${i + 1}. ${q.question}`, "step-sub"));
      }
      if (event.type === "step_started") researchStep(`Researching: ${event.question}`, "step-head");
      if (event.type === "search") {
        const label = event.documents ? "Searched documents" : `Searched "${event.query}"`;
        researchStep(event.error ? `${label} — failed: ${event.error}` : `${label} — ${event.hits.length} results`,
                     event.error ? "step-error" : "");
      }
      if (event.type === "read") {
        const item = researchStep(event.ok ? `Read [${event.source}] ` : "Could not read ", event.ok ? "" : "step-error");
        item.append(safeLink(event.url, event.title || event.url));
      }
      if (event.type === "notes") researchStep(`Notes ready (${event.sources.length} sources)`, "muted");
      if (event.type === "synthesis_started") researchStep(`Writing report from ${event.sources} sources…`, "step-head");
      if (event.type === "report") renderReport(event);
      if (event.type === "research_failed") researchStep(event.error, "step-error");
    });
  } catch (error) {
    researchStep(error.message, "step-error");
  } finally {
    button.disabled = false;
    loadConversations();
  }
}

function renderReport(event) {
  const report = document.querySelector("#research-report");
  report.replaceChildren();
  const body = event.report.split("\n## Sources\n")[0];
  report.append(el("div", body, "report-body"));
  report.append(el("h3", "Sources"));
  const list = el("ol", null, "sources");
  event.sources.forEach((source) => {
    const item = el("li");
    item.append(source.kind === "web" ? safeLink(source.url, source.title) : el("span", `${source.title} (document)`));
    list.append(item);
  });
  report.append(list);
  if (event.invalid_citations_removed && event.invalid_citations_removed.length) {
    report.append(el("p", `Removed citations to non-existent sources: ${event.invalid_citations_removed.join(", ")}`, "note"));
  }
  report.hidden = false;
}

// ---- Teams ---------------------------------------------------------------------------------
async function loadTeams(selectId) {
  try {
    const [teams, templates] = await Promise.all([requestJson("/teams"), requestJson("/teams/templates")]);
    teamsCache = teams.teams;
    const select = document.querySelector("#team-select");
    const keep = selectId || select.value;
    select.replaceChildren(option("New team…", ""), ...teamsCache.map((t) => option(t.name, t.id)));
    if (teamsCache.some((t) => t.id === keep)) select.value = keep;
    document.querySelector("#team-template").replaceChildren(
      ...templates.templates.map((t) => option(`${t.name} (${t.members.length})`, t.id)));
    showTeamDefinition();
  } catch (error) {
    appendMessage("system", error.message);
  }
}

function selectedTeam() {
  return teamsCache.find((t) => t.id === document.querySelector("#team-select").value) || null;
}

function showTeamDefinition(definition) {
  const team = definition || selectedTeam();
  const editable = team
    ? { name: team.name, description: team.description, lead: team.lead, max_turns: team.max_turns,
        max_tool_steps: team.max_tool_steps, documents_root_id: team.documents_root_id, members: team.members }
    : { name: "My team", description: "", lead: "lead", max_turns: 6, max_tool_steps: 4,
        members: [{ id: "lead", name: "Lead", role: "Coordinates the team", instructions: "", tools: [], skills: [] }] };
  document.querySelector("#team-editor").value = JSON.stringify(editable, null, 2);
}

async function saveTeam() {
  let body;
  try {
    body = JSON.parse(document.querySelector("#team-editor").value);
  } catch (error) {
    appendMessage("system", `Team definition is not valid JSON: ${error.message}`);
    return null;
  }
  const team = selectedTeam();
  try {
    const saved = await requestJson(team ? `/teams/${encodeURIComponent(team.id)}` : "/teams", {
      method: team ? "PUT" : "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    await loadTeams(saved.id);
    return saved;
  } catch (error) {
    appendMessage("system", error.message);
    return null;
  }
}

async function createTeamFromTemplate() {
  try {
    const saved = await requestJson("/teams/from-template", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ template_id: document.querySelector("#team-template").value })
    });
    await loadTeams(saved.id);
  } catch (error) {
    appendMessage("system", error.message);
  }
}

async function draftTeam() {
  const goal = document.querySelector("#team-goal").value.trim();
  if (goal.length < 5) return;
  const button = document.querySelector("#team-draft");
  button.disabled = true;
  try {
    const data = await requestJson("/teams/draft", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ goal, provider: providerSelect.value, model: modelInput.value || undefined })
    });
    document.querySelector("#team-select").value = "";
    showTeamDefinition(data.draft);
    appendMessage("system", "Team draft ready in the Teams tab. Review it, then Save team.");
  } catch (error) {
    appendMessage("system", error.message);
  } finally {
    button.disabled = false;
  }
}

async function deleteTeam() {
  const team = selectedTeam();
  if (!team || !window.confirm(`Delete team "${team.name}"? Its past conversations are kept.`)) return;
  try {
    await requestJson(`/teams/${encodeURIComponent(team.id)}`, { method: "DELETE" });
  } catch (error) {
    appendMessage("system", error.message);
  }
  loadTeams();
}

function teamFeedEntry(speaker, content, cls = "assistant") {
  const feed = document.querySelector("#team-feed");
  const row = el("article", null, `message ${cls}`);
  row.append(el("span", speaker, "message-role"), el("div", content));
  feed.append(row);
  feed.scrollTop = feed.scrollHeight;
  return row;
}

function handleTeamEvent(event) {
  if (event.type === "plan") {
    teamFeedEntry(`${event.agent} · plan`, `${event.plan}\n` +
      event.assignments.map((a) => `→ ${a.agent}: ${a.task}`).join("\n"), "system");
  } else if (event.type === "moderator") {
    teamFeedEntry(`${event.agent} · moderator`, event.decision === "done"
      ? `Discussion complete. ${event.reason || ""}` : `Next: ${event.decision}. ${event.reason || ""}`, "system");
  } else if (event.type === "agent_started") {
    teamFeedEntry(event.agent, `${event.role} · ${event.provider}/${event.model}` +
      (event.tools.length ? ` · tools: ${event.tools.join(", ")}` : "") + " …", "system");
  } else if (event.type === "tool_started") {
    teamFeedEntry(`${event.actor} · tool`, `${event.name} ${JSON.stringify(event.arguments).slice(0, 300)}`, "system");
  } else if (event.type === "tool_failed") {
    teamFeedEntry(`${event.actor} · tool failed`, event.error || "failed", "system");
  } else if (event.type === "agent_message") {
    teamFeedEntry(event.final ? `${event.agent} · final answer` : event.agent, event.content);
  } else if (event.type === "team_failed") {
    teamFeedEntry("Error", event.error, "system");
  }
}

async function runTeamTask() {
  let team = selectedTeam();
  const task = document.querySelector("#team-task").value.trim();
  if (!task) return;
  if (!team) team = await saveTeam();
  if (!team) return;
  const panel = document.querySelector("#team-chat");
  panel.hidden = false;
  currentTeamSession = null;
  document.querySelector("#team-chat-title").textContent = `— ${team.name} (task run)`;
  document.querySelector("#team-feed").replaceChildren();
  teamFeedEntry("You", task, "user");
  try {
    await requestNdjson(`/teams/${encodeURIComponent(team.id)}/run/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ task, provider: providerSelect.value, model: modelInput.value || undefined })
    }, handleTeamEvent);
  } catch (error) {
    teamFeedEntry("Error", error.message, "system");
  }
  loadConversations();
}

async function openTeamChat() {
  let team = selectedTeam();
  if (!team) team = await saveTeam();
  if (!team) return;
  try {
    const session = await requestJson(`/teams/${encodeURIComponent(team.id)}/sessions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ provider: providerSelect.value, model: modelInput.value || undefined })
    });
    await openTeamSession(session.session_id, session.title);
    loadConversations();
  } catch (error) {
    appendMessage("system", error.message);
  }
}

async function openTeamSession(sessionId, title) {
  currentTeamSession = sessionId;
  document.querySelector("#team-chat").hidden = false;
  document.querySelector("#team-chat-title").textContent = `— ${title || ""}`;
  const feed = document.querySelector("#team-feed");
  feed.replaceChildren();
  try {
    const data = await requestJson(`/sessions/${encodeURIComponent(sessionId)}`);
    data.messages.forEach((m) => {
      if (m.role === "user") teamFeedEntry("You", m.content, "user");
      else if (m.role === "assistant") teamFeedEntry((m.metadata && m.metadata.agent) || "Assistant", m.content);
    });
  } catch (error) {
    teamFeedEntry("Error", error.message, "system");
  }
}

async function sendTeamMessage() {
  const input = document.querySelector("#team-message");
  const message = input.value.trim();
  if (!message) return;
  if (!currentTeamSession) {
    appendMessage("system", "Open a team chat first.");
    return;
  }
  input.value = "";
  teamFeedEntry("You", message, "user");
  const button = document.querySelector("#team-send");
  button.disabled = true;
  try {
    await requestNdjson(`/teams/sessions/${encodeURIComponent(currentTeamSession)}/messages/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message })
    }, handleTeamEvent);
  } catch (error) {
    teamFeedEntry("Error", error.message, "system");
  } finally {
    button.disabled = false;
  }
}

// ---- Skills --------------------------------------------------------------------------------
async function loadSkills(selectId) {
  await loadSkillOptions();
  const select = document.querySelector("#skill-select");
  const keep = selectId || select.value;
  select.replaceChildren(...skillsCache.map((s) => option(`${s.name}${s.builtin ? " (built-in)" : ""}`, s.id)));
  if (skillsCache.some((s) => s.id === keep)) select.value = keep;
  const tools = document.querySelector("#skill-tools");
  if (!tools.children.length) {
    KNOWN_SKILL_TOOLS.forEach((name) => {
      const label = el("label", null, "checkbox-row");
      const box = el("input");
      box.type = "checkbox";
      box.value = name;
      label.append(box, el("span", name));
      tools.append(label);
    });
  }
  showSkill();
}

async function showSkill() {
  const id = document.querySelector("#skill-select").value;
  if (!id) return;
  try {
    const skill = await requestJson(`/skills/${encodeURIComponent(id)}`);
    fillSkillForm(skill);
    document.querySelector("#skill-note").textContent = skill.builtin
      ? "Built-in skills are read-only. Duplicate to customise."
      : "Your skill. It is saved as a Markdown file in the app's state folder.";
    document.querySelector("#skill-save").disabled = skill.builtin;
    document.querySelector("#skill-delete").disabled = skill.builtin;
  } catch (error) {
    appendMessage("system", error.message);
  }
}

function fillSkillForm(skill) {
  document.querySelector("#skill-name").value = skill.name || "";
  document.querySelector("#skill-description").value = skill.description || "";
  document.querySelector("#skill-instructions").value = skill.instructions || "";
  document.querySelectorAll("#skill-tools input").forEach((box) => {
    box.checked = (skill.tools || []).includes(box.value);
  });
}

function newSkill(draft) {
  document.querySelector("#skill-select").value = "";
  fillSkillForm(draft || { name: "", description: "", instructions: "", tools: [] });
  document.querySelector("#skill-note").textContent = draft ? "AI draft — review before saving." : "New skill.";
  document.querySelector("#skill-save").disabled = false;
  document.querySelector("#skill-delete").disabled = true;
}

async function saveSkill() {
  const id = document.querySelector("#skill-select").value;
  const current = skillsCache.find((s) => s.id === id);
  const body = {
    name: document.querySelector("#skill-name").value.trim(),
    description: document.querySelector("#skill-description").value.trim(),
    instructions: document.querySelector("#skill-instructions").value,
    tools: Array.from(document.querySelectorAll("#skill-tools input:checked")).map((b) => b.value)
  };
  const editing = current && !current.builtin;
  try {
    const saved = await requestJson(editing ? `/skills/${encodeURIComponent(id)}` : "/skills", {
      method: editing ? "PUT" : "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    });
    await loadSkills(saved.id);
  } catch (error) {
    appendMessage("system", error.message);
  }
}

async function duplicateSkill() {
  const id = document.querySelector("#skill-select").value;
  if (!id) return;
  try {
    const copy = await requestJson(`/skills/${encodeURIComponent(id)}/duplicate`, { method: "POST" });
    await loadSkills(copy.id);
  } catch (error) {
    appendMessage("system", error.message);
  }
}

async function deleteSkill() {
  const id = document.querySelector("#skill-select").value;
  if (!id || !window.confirm("Delete this skill?")) return;
  try {
    await requestJson(`/skills/${encodeURIComponent(id)}`, { method: "DELETE" });
  } catch (error) {
    appendMessage("system", error.message);
  }
  loadSkills();
}

async function draftSkill() {
  const goal = document.querySelector("#skill-goal").value.trim();
  if (goal.length < 5) return;
  const button = document.querySelector("#skill-draft");
  button.disabled = true;
  try {
    const data = await requestJson("/skills/draft", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ goal, provider: providerSelect.value, model: modelInput.value || undefined })
    });
    newSkill(data.draft);
  } catch (error) {
    appendMessage("system", error.message);
  } finally {
    button.disabled = false;
  }
}

// ---- Sandbox -------------------------------------------------------------------------------
async function refreshSandbox() {
  const note = document.querySelector("#sandbox-status");
  const pull = document.querySelector("#sandbox-pull");
  try {
    const status = await requestJson("/sandbox/status");
    const docker = status.docker || {};
    note.textContent = status.active
      ? `Active backend: ${status.active} (${status.isolated ? "isolated, no network" : "NOT isolated"}).` +
        (docker.available && !docker.image_present ? ` Docker is available; download ${docker.image} to use containers.` : "")
      : status.reason || "No sandbox backend available.";
    pull.hidden = !(docker.available && !docker.image_present);
  } catch (error) {
    note.textContent = error.message;
  }
}

async function runSandbox() {
  const output = document.querySelector("#sandbox-output");
  output.textContent = "Running…";
  try {
    const result = await requestJson("/sandbox/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        code: document.querySelector("#sandbox-code").value,
        language: document.querySelector("#sandbox-language").value,
        timeout: Number(document.querySelector("#sandbox-timeout").value) || 30
      })
    });
    output.textContent =
      `[${result.backend}${result.isolated ? ", isolated" : ", NOT isolated"}] exit ${result.exit_code}` +
      `${result.timed_out ? " (timed out)" : ""} in ${result.duration_s}s\n\n` +
      (result.stdout || "") + (result.stderr ? `\n--- stderr ---\n${result.stderr}` : "");
  } catch (error) {
    output.textContent = error.message;
  }
}

async function pullSandboxImage() {
  if (!window.confirm("Download the sandbox container image now? This may take a few minutes.")) return;
  const note = document.querySelector("#sandbox-status");
  note.textContent = "Downloading image…";
  try {
    const result = await requestJson("/sandbox/image/pull", { method: "POST" });
    note.textContent = result.ok ? "Image downloaded." : `Download failed: ${result.output}`;
  } catch (error) {
    note.textContent = error.message;
  }
  refreshSandbox();
}

// ---- Settings: keys, verification, Hugging Face installs ------------------------------------
async function loadSettings() {
  const grid = document.querySelector("#settings-values");
  try {
    const data = await requestJson("/settings/providers");
    grid.replaceChildren();
    Object.entries(data.values).forEach(([name, info]) => {
      const row = el("div", null, "settings-row");
      const label = el("label", info.label);
      const input = el("input");
      input.type = info.secret ? "password" : "text";
      input.autocomplete = "off";
      input.placeholder = info.secret ? (info.set ? `set (${info.source})` : "not set") : (info.value || "");
      if (!info.secret && info.value) input.value = info.value;
      label.append(input);
      const save = el("button", "Save", "secondary mini");
      save.type = "button";
      save.addEventListener("click", () => saveSetting(name, input.value));
      const clear = el("button", "Clear", "secondary mini");
      clear.type = "button";
      clear.disabled = info.source !== "app";
      clear.addEventListener("click", () => saveSetting(name, null));
      row.append(label, save, clear);
      grid.append(row);
    });
  } catch (error) {
    grid.replaceChildren(el("p", error.message, "note"));
  }
  refreshPulls();
  loadAirllm();
}

async function saveSetting(name, value) {
  try {
    await requestJson("/settings/values", {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, value: value && value.trim() ? value.trim() : null })
    });
  } catch (error) {
    appendMessage("system", error.message);
  }
  loadSettings();
  if (typeof loadProviders === "function") loadProviders();
}

async function verifyProvider() {
  const provider = document.querySelector("#verify-provider").value;
  const output = document.querySelector("#verify-output");
  output.textContent = "Contacting provider…";
  try {
    const data = await requestJson(`/providers/${provider}/models`);
    output.textContent = `Key works. ${data.models.length} models available:\n` + data.models.map((m) => m.id).join("\n");
  } catch (error) {
    output.textContent = error.message;
  }
}

async function searchHuggingFace() {
  const q = document.querySelector("#hf-query").value.trim();
  const list = document.querySelector("#hf-results");
  if (!q) return;
  list.replaceChildren(el("li", "Searching…", "note"));
  try {
    const format = document.querySelector("#hf-format").value;
    const data = await requestJson(`/models/hf/search?q=${encodeURIComponent(q)}&format=${format}`);
    list.replaceChildren();
    data.results.forEach((repo) => {
      const item = el("li");
      const head = el("button", `${repo.repo} · ${repo.downloads || 0} downloads${repo.gated ? " · gated" : ""}`, "linklike");
      head.type = "button";
      head.addEventListener("click", () => {
        if (format === "transformers") {
          document.querySelector("#airllm-repo").value = repo.repo;
          checkAirllmRepo();
        } else {
          showQuantizations(repo.repo, item);
        }
      });
      item.append(head);
      list.append(item);
    });
    if (!data.results.length) list.append(el("li", "No matching models found.", "note"));
  } catch (error) {
    list.replaceChildren(el("li", error.message, "note"));
  }
}

async function showQuantizations(repo, item) {
  const existing = item.querySelector("ul");
  if (existing) {
    existing.remove();
    return;
  }
  const sub = el("ul", null, "quant-list");
  sub.append(el("li", "Loading files…", "note"));
  item.append(sub);
  try {
    const data = await requestJson(`/models/hf/quantizations?repo=${encodeURIComponent(repo)}`);
    sub.replaceChildren();
    data.quantizations.forEach((q) => {
      const row = el("li");
      const gb = (q.size_bytes / 1e9).toFixed(2);
      const install = el("button", `Install ${q.quant} (${gb} GB)`, "secondary mini");
      install.type = "button";
      install.addEventListener("click", () => startPull(q.ollama_model, gb));
      row.append(install, el("code", ` ${q.ollama_model}`));
      sub.append(row);
    });
    if (!data.quantizations.length) sub.append(el("li", "No GGUF files found.", "note"));
  } catch (error) {
    sub.replaceChildren(el("li", error.message, "note"));
  }
}

async function startPull(model, gb) {
  if (!window.confirm(`Download ${model}${gb ? ` (~${gb} GB)` : ""} into Ollama?`)) return;
  try {
    await requestJson("/ollama/pulls", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model })
    });
  } catch (error) {
    appendMessage("system", error.message);
  }
  refreshPulls();
}

async function refreshPulls() {
  const list = document.querySelector("#pull-jobs");
  window.clearTimeout(pullPollTimer);
  try {
    const data = await requestJson("/ollama/pulls");
    list.replaceChildren();
    if (!data.pulls.length) list.append(el("li", "No downloads yet.", "note"));
    let running = false;
    data.pulls.forEach((job) => {
      running = running || job.status === "running" || job.status === "queued";
      const item = el("li");
      item.append(el("span", `${job.model} — ${job.status}${job.percent !== null ? ` ${job.percent}%` : ""}` +
        `${job.error ? `: ${job.error}` : ""}`));
      if (job.percent !== null) {
        const bar = el("progress");
        bar.max = 100;
        bar.value = job.percent;
        item.append(bar);
      }
      list.append(item);
    });
    if (running) pullPollTimer = window.setTimeout(refreshPulls, 1500);
    else if (typeof refreshOllamaModels === "function" && data.pulls.some((j) => j.status === "completed")) {
      refreshOllamaModels();
    }
  } catch (error) {
    list.replaceChildren(el("li", error.message, "note"));
  }
}

// ---- AirLLM --------------------------------------------------------------------------------
let airllmPollTimer = null;

function gb(bytes) {
  return `${(bytes / 1e9).toFixed(2)} GB`;
}

async function loadAirllm() {
  const note = document.querySelector("#airllm-status");
  window.clearTimeout(airllmPollTimer);
  try {
    const data = await requestJson("/airllm/status");
    const runtime = data.runtime || {};
    const worker = data.worker || {};
    const health = worker.health || {};
    const progress = health.progress || {};
    const busy = progress.phase && progress.phase !== "idle";
    note.textContent =
      `${runtime.ready ? `Runtime ready (${runtime.python}).` : `Runtime not ready: ${runtime.reason}`} ` +
      `Worker: ${worker.running
        ? `running, backend ${worker.backend}` +
          (health.loaded_model ? `, loaded ${health.loaded_model}` : "") +
          (busy ? `, ${progress.phase}${progress.generated_tokens ? ` (${progress.generated_tokens} tokens)` : ""}` : "")
        : "stopped"}. ${data.note}`;
    document.querySelector("#airllm-setup").hidden = Boolean(runtime.ready);
    const list = document.querySelector("#airllm-models");
    list.replaceChildren();
    if (!data.models.length) list.append(el("li", "No AirLLM models yet.", "note"));
    data.models.forEach((model) => {
      const item = el("li");
      item.append(el("span", `${model.id} · ${model.architecture || "?"} · ${model.context_tokens} ctx` +
        `${model.present ? "" : " · files missing"}${model.repository ? ` · ${model.repository}` : ""} `));
      const remove = el("button", "Remove", "secondary mini");
      remove.type = "button";
      remove.addEventListener("click", async () => {
        if (!window.confirm(`Remove ${model.id} from AirLLM?`)) return;
        const deleteFiles = window.confirm("Also delete its files? (Only files inside the AirLLM folder are ever deleted.)");
        try {
          await requestJson(`/airllm/models/${encodeURIComponent(model.id)}?delete_files=${deleteFiles}`, { method: "DELETE" });
        } catch (error) {
          appendMessage("system", error.message);
        }
        loadAirllm();
      });
      item.append(remove);
      list.append(item);
    });
    const downloads = document.querySelector("#airllm-downloads");
    downloads.replaceChildren();
    let active = false;
    data.downloads.forEach((job) => {
      active = active || job.status === "running" || job.status === "queued";
      const item = el("li", `${job.repo} → ${job.model_id}: ${job.status}` +
        `${job.percent !== null ? ` ${job.percent}%` : ""}${job.error ? ` — ${job.error}` : ""}`);
      if (job.percent !== null) {
        const bar = el("progress");
        bar.max = 100;
        bar.value = job.percent;
        item.append(bar);
      }
      downloads.append(item);
    });
    if (active || busy) airllmPollTimer = window.setTimeout(loadAirllm, 2000);
  } catch (error) {
    note.textContent = error.message;
  }
}

async function airllmAction(path, label, confirmText) {
  if (confirmText && !window.confirm(confirmText)) return;
  const note = document.querySelector("#airllm-status");
  note.textContent = `${label}…`;
  try {
    const result = await requestJson(path, { method: "POST" });
    if (result.output) appendMessage("system", result.output.slice(-1500));
  } catch (error) {
    appendMessage("system", error.message);
  }
  await loadAirllm();
  if (typeof loadProviders === "function") loadProviders();
}

async function checkAirllmRepo() {
  const repo = document.querySelector("#airllm-repo").value.trim();
  const box = document.querySelector("#airllm-check-result");
  if (!repo) return;
  box.replaceChildren(el("span", "Checking…"));
  try {
    const info = await requestJson(`/airllm/models/check?repo=${encodeURIComponent(repo)}`);
    box.replaceChildren(el("span", `${info.support}. Weights ${gb(info.weights_bytes)}; needs about ` +
      `${gb(info.disk_needed_bytes)} of disk (weights + AirLLM layer split); ${gb(info.free_bytes)} free.` +
      `${info.gated ? " Gated: needs an HF token and an accepted licence." : ""} `));
    if (info.supported) {
      const button = el("button", `Download as ${info.suggested_id}`, "secondary mini");
      button.type = "button";
      button.addEventListener("click", async () => {
        if (!window.confirm(`Download ${repo} (${gb(info.weights_bytes)}) for AirLLM?`)) return;
        try {
          await requestJson("/airllm/models/download", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ repo, model_id: info.suggested_id, context_tokens: Math.min(info.context_tokens, 8192) })
          });
        } catch (error) {
          appendMessage("system", error.message);
        }
        loadAirllm();
      });
      box.append(button);
    }
  } catch (error) {
    box.replaceChildren(el("span", error.message));
  }
}

async function registerAirllmModel() {
  const path = document.querySelector("#airllm-path").value.trim();
  const id = document.querySelector("#airllm-id").value.trim();
  if (!path || !id) {
    appendMessage("system", "Enter the checkpoint folder and a model id.");
    return;
  }
  try {
    await requestJson("/airllm/models/register", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path, model_id: id })
    });
  } catch (error) {
    appendMessage("system", error.message);
  }
  await loadAirllm();
  if (typeof loadProviders === "function") loadProviders();
}

// ---- Wiring--------------------------------------------------------------------------------
document.querySelectorAll(".tab").forEach((tab) => tab.addEventListener("click", () => showView(tab.dataset.view)));
document.querySelector("#session-search").addEventListener("input", () => {
  window.clearTimeout(loadConversations.timer);
  loadConversations.timer = window.setTimeout(loadConversations, 250);
});
document.querySelector("#session-filter").addEventListener("change", loadConversations);
document.querySelector("#research-start").addEventListener("click", startResearch);
document.querySelector("#team-select").addEventListener("change", () => showTeamDefinition());
document.querySelector("#team-from-template").addEventListener("click", createTeamFromTemplate);
document.querySelector("#team-draft").addEventListener("click", draftTeam);
document.querySelector("#team-save").addEventListener("click", saveTeam);
document.querySelector("#team-delete").addEventListener("click", deleteTeam);
document.querySelector("#team-open-chat").addEventListener("click", openTeamChat);
document.querySelector("#team-run").addEventListener("click", runTeamTask);
document.querySelector("#team-send").addEventListener("click", sendTeamMessage);
document.querySelector("#skill-select").addEventListener("change", showSkill);
document.querySelector("#skill-new").addEventListener("click", () => newSkill());
document.querySelector("#skill-duplicate").addEventListener("click", duplicateSkill);
document.querySelector("#skill-delete").addEventListener("click", deleteSkill);
document.querySelector("#skill-draft").addEventListener("click", draftSkill);
document.querySelector("#skill-save").addEventListener("click", saveSkill);
document.querySelector("#sandbox-run").addEventListener("click", runSandbox);
document.querySelector("#sandbox-pull").addEventListener("click", pullSandboxImage);
document.querySelector("#verify-run").addEventListener("click", verifyProvider);
document.querySelector("#hf-search").addEventListener("click", searchHuggingFace);
document.querySelector("#airllm-setup").addEventListener("click", () => airllmAction(
  "/airllm/runtime/setup", "Setting up the AirLLM runtime (this can take 10+ minutes)",
  "Set up the AirLLM runtime now? It clones AirLLM and installs PyTorch and Transformers (about 1–2 GB)."));
document.querySelector("#airllm-start").addEventListener("click", () => airllmAction("/airllm/worker/start", "Starting worker"));
document.querySelector("#airllm-stop").addEventListener("click", () => airllmAction("/airllm/worker/stop", "Stopping worker"));
document.querySelector("#airllm-cancel").addEventListener("click", () => airllmAction("/airllm/worker/cancel", "Cancelling"));
document.querySelector("#airllm-check").addEventListener("click", checkAirllmRepo);
document.querySelector("#airllm-register").addEventListener("click", registerAirllmModel);
chatForm.addEventListener("submit", () => window.setTimeout(loadConversations, 1500));

loadSkillOptions();
loadConversations();
let initialView = "chat";
try {
  initialView = localStorage.getItem("chatSoftwareView") || "chat";
} catch (error) {
  /* storage unavailable */
}
showView(viewTitles[initialView] ? initialView : "chat");
