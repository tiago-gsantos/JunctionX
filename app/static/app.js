// Stock Sense: one-screen staff app. All data comes from the server (app/server.py); this file only
// renders it and sends staff actions back.

const $ = (sel) => document.querySelector(sel);
const STEPS = ["Finding the shelf areas", "Looking at the products", "Comparing with the usual shelf",
               "Checking stock and products"];
const MODES = ["priority", "location", "product", "type"];
const ICON = {
  check: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 8.5l3.2 3L13 4.5"/></svg>',
  undo: '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M4 6h6a3 3 0 0 1 0 6H7"/><path d="M6.5 3.5 4 6l2.5 2.5"/></svg>',
  clear: '<svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M5 12.5l4.5 4.5L19 7.5"/></svg>',
};
const INSTRUCTION = {
  empty: (p) => `The area is empty. Restock ${p}.`,
  low: (p) => `The area is half full. Top up ${p}.`,
  wrong: (p) => `Another product is in this area. Put ${p} back in its place.`,
};

const state = {
  data: null,
  selected: null,
  mode: MODES.includes(load("mode")) ? load("mode") : "priority",
  showDone: false,
  uploadInfo: null,
};

function load(key) { try { return localStorage.getItem(`stocksense.${key}`); } catch { return null; } }
function save(key, value) { try { localStorage.setItem(`stocksense.${key}`, value); } catch { /* private mode */ } }
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);

async function api(path, options = {}) {
  try {
    const res = await fetch(path, options);
    let body = null;
    try { body = await res.json(); } catch { /* not JSON */ }
    return { ok: res.ok, status: res.status, body };
  } catch {
    return { ok: false, status: 0, body: { error: "Cannot reach the Stock Sense server." } };
  }
}

// ---------------------------------------------------------------- loading and processing
async function refreshStatus() {
  const { ok, body: s } = await api("/api/status");
  if (!ok) return showToast(s.error, { error: true });
  if (s.state === "processing") {
    showProcessing(s);
    return setTimeout(refreshStatus, 600);
  }
  $("#processing").hidden = true;
  if (s.state === "error") showToast(`Could not process the photos: ${s.message}`, { error: true, sticky: true });
  if (s.has_run) await loadDashboard();
  else showWelcome();
}

async function loadDashboard() {
  const { ok, body } = await api("/api/dashboard");
  if (!ok) return showWelcome();
  const fresh = !state.data || state.data.photos_until !== body.photos_until || state.data.date !== body.date;
  state.data = body;
  if (fresh) { state.selected = null; state.showDone = false; }
  render();
}

function showWelcome() {
  state.data = null;
  $("#app").hidden = true;
  $("#welcome").hidden = false;
  $("#stats").innerHTML = "";
  $("#context").textContent = "";
  $("#download").hidden = true;
  $("#uploadBtn").hidden = true;
}

function showProcessing(s) {
  const n = state.uploadInfo?.photos;
  $("#procTitle").textContent = n ? `Analysing ${n} photos` : "Analysing photos";
  $("#procBar").style.width = `${Math.round((s.progress || 0) * 100)}%`;
  const current = Math.max(0, STEPS.indexOf(s.step));
  $("#procSteps").innerHTML = STEPS.map((step, i) =>
    `<li class="${i < current ? "complete" : i === current ? "active" : ""}">${step}</li>`).join("");
  $("#processing").hidden = false;
}

// ---------------------------------------------------------------- view model
function taskById(id) { return state.data.tasks.find((t) => t.task_id === id); }

function view() {
  const d = state.data;
  const byId = new Map(d.tasks.map((t) => [t.task_id, t]));
  const open = d.tasks.filter((t) => !t.done);
  let sections;
  if (state.mode === "priority") {
    sections = [{ tasks: open }];  // the server's order; unsure tasks already sit last within their level
  } else {
    sections = d.groups[state.mode]
      .map((g) => ({ title: g.title, sub: g.subtitle, tasks: g.task_ids.map((id) => byId.get(id)).filter((t) => !t.done) }))
      .filter((s) => s.tasks.length);
  }
  return { sections, done: d.tasks.filter((t) => t.done), order: sections.flatMap((s) => s.tasks) };
}

// ---------------------------------------------------------------- rendering
// keepScroll: leave the list where it is (after an action). reveal: scroll the selected card into view.
function render({ keepScroll = false, reveal = true } = {}) {
  const d = state.data;
  $("#welcome").hidden = true;
  $("#app").hidden = false;
  $("#download").hidden = false;
  $("#uploadBtn").hidden = false;
  $("#context").textContent = `${d.store} · Photos up to ${d.photos_until}, ${d.date}`;
  $("#stats").innerHTML = [["Availability", d.summary.availability_pct], ["Compliance", d.summary.compliance_pct]]
    .map(([label, value]) => `<div class="stat"><span class="stat-label">${label}</span><span class="stat-value">${value}%</span></div>`).join("");
  $("#todoCount").textContent = d.summary.todo;
  $("#todoDone").textContent = d.summary.done ? `${d.summary.done} done` : "";
  for (const b of document.querySelectorAll("#groupBy button")) b.setAttribute("aria-selected", b.dataset.mode === state.mode);

  const v = view();
  const ids = new Set(d.tasks.map((t) => t.task_id));
  if (!state.selected || !ids.has(state.selected)) state.selected = v.order[0]?.task_id ?? null;
  const scrollTop = $("#list").scrollTop;
  renderList(v);
  if (keepScroll) $("#list").scrollTop = scrollTop;
  if (reveal) revealSelected();
  renderDetail();
}

function revealSelected() { $("#list .task.selected")?.scrollIntoView({ block: "nearest" }); }


function card(t) {
  const where = `Camera ${t.camera_id} · Area ${t.area_number}`;
  const title = state.mode === "product" ? where : t.product;
  const meta = [state.mode === "location" ? `Area ${t.area_number}` : state.mode === "product" ? null : where,
                `since ${t.since}`];
  const action = t.done
    ? `<button class="task-action" type="button" data-action="undo" aria-label="Undo: ${esc(t.product)}">${ICON.undo} Undo</button>`
    : `<button class="task-action" type="button" data-action="done" aria-label="Mark done: ${esc(t.product)}">${ICON.check} Mark done</button>`;
  return `<div class="task${t.task_id === state.selected ? " selected" : ""}${t.done ? " done" : ""}" data-id="${t.task_id}">
    <div class="task-top"><span class="status pill status-${t.status}">${esc(t.status_label)}</span></div>
    <span class="conf conf-${t.confidence}" title="How sure the AI is that this needs doing">${t.confidence_pct === null
      ? "Corrected by staff" : `<i class="bars" aria-hidden="true"><b></b><b></b><b></b></i><strong>${t.confidence_pct}%</strong> confidence`}</span>
    <div class="task-title">${esc(title)}</div>
    <div class="task-meta">${esc(meta.filter(Boolean).join(" · "))}</div>
    ${action}
  </div>`;
}

function renderList(v) {
  let html = v.sections.map((s) => `${s.title ? `<div class="group-head"><span class="group-title">${esc(s.title)}</span><span class="group-sub">${esc(s.sub)}</span></div>` : ""}
    ${s.tasks.map(card).join("")}`).join("");
  if (!v.order.length) {
    html = `<div class="all-clear"><div class="mark">${ICON.clear}</div><h3>All shelves look good</h3>
      <p>Nothing to restock or fix.</p></div>`;
  }
  if (v.done.length) {
    html += `<button class="done-toggle" type="button" aria-expanded="${state.showDone}"><span class="chev">›</span> Done · ${v.done.length}</button>`;
    if (state.showDone) html += v.done.map(card).join("");
  }
  $("#list").innerHTML = html;
}

function quadBox(quad, w, h, pad) {
  const xs = quad.map((p) => p[0]), ys = quad.map((p) => p[1]);
  let x0 = Math.max(0, Math.min(...xs)), x1 = Math.min(w, Math.max(...xs));
  let y0 = Math.max(0, Math.min(...ys)), y1 = Math.min(h, Math.max(...ys));
  const bw = x1 - x0, bh = y1 - y0;
  // Pad, then widen the shorter side to a 4:3 box so the close-up isn't distorted.
  let cw = bw * (1 + pad), ch = bh * (1 + pad);
  if (cw / ch < 4 / 3) cw = ch * 4 / 3; else ch = cw * 3 / 4;
  const cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
  return [cx - cw / 2, cy - ch / 2, cw, ch].map((n) => Math.round(n)).join(" ");
}

function polygons(cam, selectedArea) {
  const tasks = new Map(state.data.tasks.filter((t) => t.camera_id === cam.camera_id).map((t) => [t.area_id, t]));
  const areas = [...cam.areas].sort((a, b) => (a.area_id === selectedArea) - (b.area_id === selectedArea));
  return areas.map((a) => {
    const t = tasks.get(a.area_id);
    const cls = ["area", a.status, t ? "task-area" : "", t?.done ? "done-area" : "", a.area_id === selectedArea ? "selected" : ""].join(" ");
    return `<polygon class="${cls}" ${t ? `data-id="${t.task_id}"` : ""} points="${a.quad.map((p) => p.join(",")).join(" ")}"><title>Area ${a.number} · ${esc(a.product)}</title></polygon>`;
  }).join("");
}

function renderDetail() {
  const t = state.selected && taskById(state.selected);
  if (!t) {
    $("#detail").innerHTML = `<div class="detail-empty"><p>Select a task to see where it is.</p></div>`;
    return;
  }
  const cam = state.data.cameras.find((c) => c.camera_id === t.camera_id);
  const area = cam.areas.find((a) => a.area_id === t.area_id);
  const image = `<image href="${cam.image_url}" width="${cam.width}" height="${cam.height}"/>`;
  const pack = t.pack && t.pack !== "loose" ? ` <span class="detail-pack">${esc(t.pack)}</span>` : "";

  let actions;
  if (t.done) {
    actions = `<div class="actions"><button class="btn btn-ghost" type="button" data-action="undo">Undo done</button></div>`;
  } else {
    actions = `<div class="actions">
      <button class="btn btn-primary" type="button" data-action="done">Mark done <span class="btn-check" aria-hidden="true"></span></button>
      <button class="btn btn-ghost" type="button" data-action="feedback-open">AI was wrong</button></div>`;
  }

  $("#detail").innerHTML = `
    <div class="detail-top">
      <div>
        <span class="status status-${t.status}">${esc(t.status_label)}</span>
        <h2 class="detail-title">${esc(t.product)}${pack}</h2>
        <p class="detail-where">Camera ${t.camera_id} · Area ${t.area_number} · ${esc(t.status_label.toLowerCase())} since ${t.since}</p>
        <p class="detail-catalogue" title="Catalogue name · product id">${esc(t.product_raw)} · ${esc(t.product_id)}</p>
        <p class="detail-instruction">${esc(INSTRUCTION[t.status](t.product))}</p>
        ${t.type === "check" ? `<p class="detail-unsure">The AI isn't sure about this one. Take a look before acting.</p>` : ""}
        ${actions}
      </div>
      <div>
        <button class="closeup" type="button" data-action="zoom" aria-label="Enlarge the close-up of area ${t.area_number}">${closeupSvg(cam, area)}</button>
        <div class="caption">Close-up of area ${t.area_number} · click to enlarge</div>
      </div>
    </div>
    <figure class="camera">
      <div class="camera-frame"><svg viewBox="0 0 ${cam.width} ${cam.height}" preserveAspectRatio="xMidYMid slice"
        data-w="${cam.width}" data-h="${cam.height}" data-cy="${areaCentreY(area.quad, cam.height)}">${image}${polygons(cam, t.area_id)}</svg></div>
      <figcaption><span>Camera ${cam.camera_id} · photo taken at ${cam.photo_time}</span>
        <span class="legend"><span class="status l-empty">Empty</span><span class="status l-low">Half</span>
        <span class="status l-wrong">Wrong product</span><span class="status l-ok">OK</span></span></figcaption>
    </figure>`;
  fitCamera();
}

function closeupSvg(cam, area) {
  return `<svg viewBox="${quadBox(area.quad, cam.width, cam.height, 0.35)}" preserveAspectRatio="xMidYMid slice">
    <image href="${cam.image_url}" width="${cam.width}" height="${cam.height}"/>
    <polygon class="area ${area.status} selected" points="${area.quad.map((p) => p.join(",")).join(" ")}"/></svg>`;
}

function areaCentreY(quad, height) {
  const ys = quad.map((p) => Math.min(Math.max(p[1], 0), height));
  return Math.round((Math.min(...ys) + Math.max(...ys)) / 2);
}

// The camera photo spans the panel's full width. If the window is too short for the whole 16:9 photo
// plus caption, it is cropped top and bottom, centred on the selected area so that stays visible.
function fitCamera() {
  const panel = $("#detail"), frame = panel.querySelector(".camera-frame");
  if (!frame) return;
  const svg = frame.querySelector("svg"), caption = frame.nextElementSibling;
  const w = Number(svg.dataset.w), h = Number(svg.dataset.h);
  const top = frame.getBoundingClientRect().top - panel.getBoundingClientRect().top + panel.scrollTop;
  const below = caption.offsetHeight + parseFloat(getComputedStyle(caption).marginTop) + parseFloat(getComputedStyle(panel).paddingBottom);
  const full = frame.clientWidth * h / w;
  const height = Math.min(full, Math.max(panel.clientHeight - top - below, 240));
  frame.style.height = `${height}px`;
  const visible = h * height / full;
  const y0 = Math.min(Math.max(Number(svg.dataset.cy) - visible / 2, 0), h - visible);
  svg.setAttribute("viewBox", `0 ${y0.toFixed(1)} ${w} ${visible.toFixed(1)}`);
}

// ---------------------------------------------------------------- actions
async function act(id, action, body) {
  const t = taskById(id);
  const before = view().order.findIndex((x) => x.task_id === id);
  const selectedBefore = state.selected;
  const r = await api(`/api/tasks/${encodeURIComponent(id)}/${action}`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
  if (!r.ok) return showToast(r.body?.error || "Something went wrong.", { error: true });
  state.data = r.body;
  if (action === "undo") {
    state.selected = id;
  } else {
    // Keep working down the list: select whatever now sits where the finished task was.
    const order = view().order;
    if (!order.some((x) => x.task_id === state.selected)) {
      state.selected = order[Math.min(Math.max(before, 0), order.length - 1)]?.task_id ?? null;
    }
  }
  // Stay where the user is scrolled; only move the list when the selection moved (finished the selected
  // card, or undo brings a card back).
  render({ keepScroll: true, reveal: state.selected !== selectedBefore || action === "undo" });
  if (action === "done") showToast(`${t.product} · area ${t.area_number} marked done`, { undo: () => act(id, "undo") });
  if (action === "feedback") showToast("Thanks. Correction saved.", { undo: () => act(id, "undo") });
}

function select(id) {
  if (state.selected === id) return;
  state.selected = id;
  for (const el of document.querySelectorAll("#list .task")) el.classList.toggle("selected", el.dataset.id === id);
  revealSelected();
  renderDetail();
}

// "AI was wrong": a small dialog where staff pick what is really there; one click saves it.
function openFix(t) {
  const where = `${esc(t.product)} · Camera ${t.camera_id} · Area ${t.area_number}`;
  const options = t.status === "wrong"
    ? `<div class="fix-options single"><button class="fix-option" type="button" data-fb='{"planogram_compliant":true}'>
         ${esc(t.product)} is in its place<small>The product is correct</small></button></div>`
    : `<div class="fix-options">${[["empty", "Empty"], ["half", "Half"], ["full", "Full"]].map(([level, label]) =>
         `<button class="fix-option" type="button" data-fb='{"stock_level":"${level}"}' ${t.stock_level === level ? "disabled" : ""}>
            ${label}<small>${t.stock_level === level ? "AI said this" : "&nbsp;"}</small></button>`).join("")}</div>`;
  const dialog = $("#fixDialog");
  dialog.innerHTML = `<div class="dialog fix-dialog" role="dialog" aria-modal="true" aria-labelledby="fixTitle">
      <div class="fix-head"><h2 id="fixTitle">${t.status === "wrong" ? "Correct the product" : "Correct the stock"}</h2>
        <button class="fix-close" type="button" data-close aria-label="Close">✕</button></div>
      <p class="fix-sub">${where}</p>
      ${options}
      <div class="fix-foot"><button class="btn btn-ghost" type="button" data-close>Cancel</button></div>
    </div>`;
  dialog.hidden = false;
  dialog.onclick = (e) => {
    const fb = e.target.closest("[data-fb]");
    if (fb) { closeFix(); return act(t.task_id, "feedback", JSON.parse(fb.dataset.fb)); }
    if (e.target === dialog || e.target.closest("[data-close]")) closeFix();
  };
  dialog.querySelector(".fix-option:not(:disabled)")?.focus();
}

function closeFix() { $("#fixDialog").hidden = true; }

// Clicking the close-up opens it large, so staff can see what is on the shelf.
function openZoom(t) {
  const cam = state.data.cameras.find((c) => c.camera_id === t.camera_id);
  const area = cam.areas.find((a) => a.area_id === t.area_id);
  const dialog = $("#zoomDialog");
  dialog.innerHTML = `<div class="dialog zoom-dialog" role="dialog" aria-modal="true" aria-labelledby="zoomTitle">
      <div class="fix-head"><h2 id="zoomTitle">${esc(t.product)} <span class="zoom-where">Camera ${t.camera_id} · Area ${t.area_number} · photo at ${cam.photo_time}</span></h2>
        <button class="fix-close" type="button" data-close aria-label="Close">✕</button></div>
      <div class="zoom-frame">${closeupSvg(cam, area)}</div>
    </div>`;
  dialog.hidden = false;
  dialog.onclick = (e) => { if (e.target === dialog || e.target.closest("[data-close]")) dialog.hidden = true; };
  dialog.querySelector(".fix-close").focus();
}

let toastTimer = null;
function showToast(message, { undo = null, error = false, sticky = false } = {}) {
  const el = $("#toast");
  el.className = `toast${error ? " error" : ""}`;
  el.innerHTML = `<span>${esc(message)}</span>${undo ? '<button type="button" data-toast="undo">Undo</button>' : ""}<button type="button" data-toast="close" aria-label="Dismiss">✕</button>`;
  el.hidden = false;
  el.onclick = (e) => {
    const which = e.target.closest("[data-toast]")?.dataset.toast;
    if (!which) return;
    el.hidden = true;
    if (which === "undo") undo();
  };
  clearTimeout(toastTimer);
  if (!sticky) toastTimer = setTimeout(() => { el.hidden = true; }, 6000);
}

// ---------------------------------------------------------------- upload
async function upload(items) {
  const files = items.filter(({ file }) => /\.(jpe?g|zip)$/i.test(file.name));
  if (!files.length) return showToast("No photos found. Choose the folder that holds one sub-folder per camera.", { error: true });
  const form = new FormData();
  for (const { file, path } of files) form.append("files", file, path);
  $("#procTitle").textContent = "Uploading photos";
  $("#procBar").style.width = "0%";
  $("#procSteps").innerHTML = "";
  $("#processing").hidden = false;
  const r = await api("/api/upload", { method: "POST", body: form });
  if (!r.ok) {
    $("#processing").hidden = true;
    return showToast(r.body?.error || "Upload failed.", { error: true });
  }
  state.uploadInfo = r.body;
  refreshStatus();
}

async function filesFromDrop(dataTransfer) {
  const entries = [...dataTransfer.items].map((i) => i.webkitGetAsEntry?.()).filter(Boolean);
  if (!entries.length) return [...dataTransfer.files].map((file) => ({ file, path: file.name }));
  const out = [];
  async function walk(entry) {
    if (entry.isFile) {
      const file = await new Promise((resolve, reject) => entry.file(resolve, reject));
      out.push({ file, path: entry.fullPath.replace(/^\//, "") });
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      for (let batch; (batch = await new Promise((resolve, reject) => reader.readEntries(resolve, reject))).length;) {
        for (const e of batch) await walk(e);
      }
    }
  }
  for (const e of entries) await walk(e);
  return out;
}

// ---------------------------------------------------------------- events
$("#list").addEventListener("click", (e) => {
  if (e.target.closest(".done-toggle")) {
    state.showDone = !state.showDone;
    const scrollTop = $("#list").scrollTop;
    renderList(view());
    $("#list").scrollTop = scrollTop;
    return;
  }
  const taskEl = e.target.closest(".task");
  if (!taskEl) return;
  const button = e.target.closest("[data-action]");
  if (button) return act(taskEl.dataset.id, button.dataset.action);
  select(taskEl.dataset.id);
});

$("#detail").addEventListener("click", (e) => {
  const id = state.selected;
  const action = e.target.closest("[data-action]")?.dataset.action;
  if (action === "feedback-open") return openFix(taskById(id));
  if (action === "zoom") return openZoom(taskById(id));
  if (action) return act(id, action);
  const area = e.target.closest("polygon[data-id]");
  if (area) select(area.dataset.id);
});

$("#groupBy").addEventListener("click", (e) => {
  const mode = e.target.closest("button")?.dataset.mode;
  if (!mode || mode === state.mode) return;
  state.mode = mode;
  save("mode", mode);
  render();
});

document.addEventListener("keydown", (e) => {
  const typing = e.target instanceof Element && e.target.closest("input, textarea, select");
  const modal = ["#fixDialog", "#zoomDialog"].map((sel) => $(sel)).find((m) => !m.hidden);
  if (modal) { if (e.key === "Escape") modal.hidden = true; return; }
  if (!state.data || !$("#processing").hidden || typing || e.metaKey || e.ctrlKey || e.altKey) return;
  const order = view().order;
  const i = order.findIndex((t) => t.task_id === state.selected);
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    const next = order[e.key === "ArrowDown" ? Math.min(i + 1, order.length - 1) : Math.max(i - 1, 0)];
    if (next) select(next.task_id);
  } else if (e.key.toLowerCase() === "d" && i >= 0) {
    act(state.selected, "done");
  }
});

for (const id of ["#uploadBtn", "#chooseBtn"]) $(id).addEventListener("click", () => $("#folderInput").click());
$("#folderInput").addEventListener("change", (e) => {
  const items = [...e.target.files].map((file) => ({ file, path: file.webkitRelativePath || file.name }));
  e.target.value = "";
  upload(items);
});

let dragDepth = 0;
const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes("Files");
window.addEventListener("dragenter", (e) => {
  if (!hasFiles(e)) return;
  dragDepth += 1;
  if ($("#welcome").hidden) $("#dropOverlay").hidden = false; else $("#dropzone").classList.add("over");
});
window.addEventListener("dragleave", () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (!dragDepth) { $("#dropOverlay").hidden = true; $("#dropzone").classList.remove("over"); }
});
window.addEventListener("dragover", (e) => { if (hasFiles(e)) e.preventDefault(); });
window.addEventListener("drop", async (e) => {
  if (!hasFiles(e)) return;
  e.preventDefault();
  dragDepth = 0;
  $("#dropOverlay").hidden = true;
  $("#dropzone").classList.remove("over");
  upload(await filesFromDrop(e.dataTransfer));
});

window.addEventListener("resize", fitCamera);
document.fonts?.ready.then(fitCamera);
refreshStatus();
