/* BiomedCAT console: job queue dashboard (vanilla JS, polls the local API every 3 s). */
(function () {
  "use strict";
  const $ = (sel) => document.querySelector(sel);
  const state = { files: [], profiles: [], jobs: [], selected: null, status: null, pagesCache: new Map(),
                  branches: null, customs: [], builder: null };

  // ---------------------------------------------------------------------------------------
  // helpers
  // ---------------------------------------------------------------------------------------
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const fmtDur = (s) => {
    if (s == null || isNaN(s)) return "–";
    s = Math.round(s);
    if (s < 60) return `${s} s`;
    const m = Math.floor(s / 60), h = Math.floor(m / 60);
    if (h) return `${h} h ${String(m % 60).padStart(2, "0")} min`;
    return `${m} min ${String(s % 60).padStart(2, "0")} s`;
  };
  const fmtTime = (iso) => (iso ? new Date(iso).toLocaleString() : "–");
  const fmtBytes = (b) => (b > 1e6 ? (b / 1e6).toFixed(1) + " MB" : Math.round(b / 1e3) + " kB");
  async function api(path, opts) {
    const r = await fetch(path, opts);
    if (!r.ok) {
      let msg = r.statusText;
      try { msg = (await r.json()).detail || msg; } catch (_) { /* not JSON */ }
      throw new Error(msg);
    }
    return r.json();
  }

  // ---------------------------------------------------------------------------------------
  // status and profiles
  // ---------------------------------------------------------------------------------------
  async function loadStatus() {
    try {
      const s = await api("/api/status");
      state.status = s;
      const missing = Object.entries(s.models_installed).filter(([, ok]) => !ok).map(([k]) => s.models_required[k]);
      const parts = [];
      parts.push(`<span class="dot ${s.ollama_reachable ? "" : "bad"}"></span>Ollama ${s.ollama_reachable ? "reachable" : "not reachable"}`);
      if (s.ollama_reachable && missing.length) parts.push(`<br>missing models: <code>${esc(missing.join(", "))}</code> (ollama pull)`);
      parts.push(`<br><span class="dot ${s.kg2c ? "" : "bad"}"></span>RTX-KG2c ${s.kg2c ? "ready" : "missing (context graphs disabled)"}`);
      if (s.kg2c && !(s.extras.node_details && s.extras.edge_publications))
        parts.push(`<br><span class="muted">KG2c descriptions and publications: not built (scripts/build_kg2c_extras.py)</span>`);
      parts.push(`<br>${s.worker ? "worker running" : "worker off"}${s.running ? ", job in progress" : ""} · ~${fmtDur(s.sec_per_page)} per page`);
      $("#status").innerHTML = parts.join("");
    } catch (e) {
      $("#status").innerHTML = `<span class="dot bad"></span>console API unreachable`;
    }
  }

  async function loadProfiles() {
    state.profiles = await api("/api/profiles");
    const box = $("#profiles");
    box.innerHTML = "<legend>Profiles (one context graph per selected profile)</legend>";
    for (const p of state.profiles) {
      const div = document.createElement("div");
      div.className = "profile-option";
      const checked = p.id !== "uniform" ? "checked" : "";
      div.innerHTML = `<label><input type="checkbox" name="profile" value="${esc(p.id)}" ${checked}> <strong>${esc(p.name)}</strong>
        <span class="muted small">(${p.n_predicates} predicates${p.stratum ? ", " + esc(p.stratum) + " stratum" : ""})</span>
        <span class="desc">${esc(p.description)}</span></label>`;
      box.appendChild(div);
    }
    box.addEventListener("change", updateEstimate);
    const base = $("#custom-base");
    for (const p of state.profiles) {
      if (!p.branch_weights || !Object.keys(p.branch_weights).length) continue;
      const opt = document.createElement("option");
      opt.value = p.id; opt.textContent = p.name;
      base.appendChild(opt);
    }
  }
  const selectedProfiles = () => [...document.querySelectorAll('input[name="profile"]:checked')].map((i) => i.value);
  const nSelectedProfiles = () => selectedProfiles().length + state.customs.length;

  // ---------------------------------------------------------------------------------------
  // custom profile builder: one priority (0 / 0.5 / 1) per entity branch of the Biolink model
  // ---------------------------------------------------------------------------------------
  const LEVELS = [0, 0.5, 1];
  async function loadBranches() {
    if (state.branches) return state.branches;
    state.branches = await api("/api/branches");
    return state.branches;
  }

  function renderBranches() {
    const box = $("#custom-branches");
    box.innerHTML = "";
    const groups = [["Biology (biological entity)", (b) => b.group === "biological entity"], ["Other entities", (b) => b.group !== "biological entity"]];
    for (const [title, keep] of groups) {
      const h = document.createElement("h4"); h.textContent = title; box.appendChild(h);
      const items = state.branches.branches.filter(keep).sort((a, b) => b.n_nodes_kg2c - a.n_nodes_kg2c || a.name.localeCompare(b.name));
      for (const b of items) {
        const w = state.builder.weights[b.name] || 0;
        const div = document.createElement("div");
        div.className = "branch" + (b.n_nodes_kg2c ? "" : " empty");
        div.title = (b.description || "") + (b.examples && b.examples.length ? "\nIncludes: " + b.examples.join(", ") : "");
        div.innerHTML = `<span class="bname">${esc(b.name)}<span class="muted">${b.n_classes} class(es), ${b.n_nodes_kg2c.toLocaleString()} KG2c node(s)</span></span>
          <span class="seg" data-branch="${esc(b.name)}">${LEVELS.map((l) => `<button type="button" data-level="${l}" class="${l === w ? "on" + (l === 0.5 ? " half" : "") : ""}">${l}</button>`).join("")}</span>`;
        box.appendChild(div);
      }
    }
    box.querySelectorAll(".seg button").forEach((btn) => btn.addEventListener("click", () => {
      const branch = btn.parentElement.dataset.branch, level = parseFloat(btn.dataset.level);
      if (level > 0) state.builder.weights[branch] = level; else delete state.builder.weights[branch];
      btn.parentElement.querySelectorAll("button").forEach((b) => { b.className = parseFloat(b.dataset.level) === level ? "on" + (level === 0.5 ? " half" : "") : ""; });
    }));
  }

  function applyBase() {
    const id = $("#custom-base").value;
    const p = state.profiles.find((x) => x.id === id);
    state.builder.weights = p ? Object.assign({}, p.branch_weights) : {};
    if (p && p.directionality) {
      $("#custom-dir").value = p.directionality.mode || "on";
      $("#custom-inv").value = p.directionality.inverse_factor != null ? p.directionality.inverse_factor : 0.5;
    }
    if (p && !$("#custom-name").value) $("#custom-name").value = p.name + " (custom)";
    renderBranches();
  }

  async function openBuilder() {
    const err = $("#custom-error"); err.hidden = true;
    try { await loadBranches(); } catch (e) { err.hidden = false; err.textContent = "Cannot load the Biolink branches: " + e.message; return; }
    state.builder = { weights: {} };
    $("#custom-name").value = "";
    $("#custom-base").value = "";
    $("#custom-dir").value = "on"; $("#custom-inv").value = "0.5";
    $("#custom-builder").hidden = false;
    renderBranches();
  }

  function addCustom() {
    const err = $("#custom-error"); err.hidden = true;
    const weights = state.builder.weights;
    if (!Object.keys(weights).length) { err.hidden = false; err.textContent = "Give at least one branch a priority of 0.5 or 1."; return; }
    const name = $("#custom-name").value.trim() || `custom profile ${state.customs.length + 1}`;
    const inv = Math.min(1, Math.max(0, parseFloat($("#custom-inv").value) || 0));
    state.customs.push({ name, branch_weights: weights, directionality: { mode: $("#custom-dir").value, inverse_factor: inv } });
    $("#custom-builder").hidden = true;
    renderCustoms();
    updateEstimate();
  }

  function renderCustoms() {
    const box = $("#custom-list");
    if (!state.customs.length) {
      box.innerHTML = `<p class="muted small">None yet. A custom profile sets, for each family of Biolink entities, whether it is of primary interest (1), secondary interest (0.5) or out of scope (0); the class scope and the predicate weights are derived automatically and shipped with the results.</p>`;
      return;
    }
    box.innerHTML = "";
    state.customs.forEach((c, i) => {
      const primary = Object.entries(c.branch_weights).filter(([, w]) => w >= 1).map(([b]) => b);
      const secondary = Object.entries(c.branch_weights).filter(([, w]) => w < 1).map(([b]) => b);
      const div = document.createElement("div");
      div.className = "custom-item";
      div.innerHTML = `<span><strong>${esc(c.name)}</strong> <span class="muted small">directionality ${esc(c.directionality.mode)}${c.directionality.mode === "on" ? " (reverse ×" + c.directionality.inverse_factor + ")" : ""}</span><br>
        <span class="small">1: ${esc(primary.join(", ") || "–")}${secondary.length ? " · 0.5: " + esc(secondary.join(", ")) : ""}</span></span>
        <button class="link" data-i="${i}">remove</button>`;
      box.appendChild(div);
    });
    box.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => { state.customs.splice(+b.dataset.i, 1); renderCustoms(); updateEstimate(); }));
  }

  // ---------------------------------------------------------------------------------------
  // file intake: drop zone, folder traversal, page counting for the estimate
  // ---------------------------------------------------------------------------------------
  const SUPPORTED = [".pdf", ".png", ".jpg", ".jpeg"];
  const supported = (name) => SUPPORTED.some((ext) => name.toLowerCase().endsWith(ext));

  function addFile(file, path) {
    if (!supported(file.name)) return;
    const key = path || file.name;
    if (state.files.some((f) => f.path === key)) return;
    state.files.push({ file, path: key });
  }

  function walkEntry(entry, prefix) {
    return new Promise((resolve) => {
      if (entry.isFile) {
        entry.file((file) => { addFile(file, prefix + file.name); resolve(); }, () => resolve());
      } else if (entry.isDirectory) {
        const reader = entry.createReader();
        const all = [];
        const readBatch = () => reader.readEntries(async (entries) => {
          if (!entries.length) {
            for (const e of all) await walkEntry(e, prefix + entry.name + "/");
            resolve();
          } else { all.push(...entries); readBatch(); }
        }, () => resolve());
        readBatch();
      } else resolve();
    });
  }

  async function handleDrop(ev) {
    ev.preventDefault();
    $("#drop").classList.remove("over");
    const items = ev.dataTransfer.items;
    if (items && items.length && items[0].webkitGetAsEntry) {
      const entries = [...items].map((it) => it.webkitGetAsEntry()).filter(Boolean);
      for (const entry of entries) await walkEntry(entry, "");
    } else {
      for (const f of ev.dataTransfer.files) addFile(f, f.name);
    }
    renderFiles();
  }

  function renderFiles() {
    const ul = $("#file-list");
    ul.innerHTML = "";
    state.files.forEach((f, i) => {
      const li = document.createElement("li");
      li.innerHTML = `<span class="path">${esc(f.path)}</span><span class="muted">${fmtBytes(f.file.size)} <button class="link" data-i="${i}">remove</button></span>`;
      ul.appendChild(li);
    });
    ul.querySelectorAll("button").forEach((b) => b.addEventListener("click", () => { state.files.splice(+b.dataset.i, 1); renderFiles(); }));
    if (state.files.length && !$("#job-name").value) {
      const first = state.files[0].path;
      $("#job-name").placeholder = first.includes("/") ? first.split("/")[0] : first.replace(/\.[^.]+$/, "");
    }
    $("#submit").disabled = !state.files.length;
    updateEstimate();
  }

  /* Rough client-side page count of a PDF (server recounts with poppler): occurrences of "/Type /Page". */
  async function countPages(f) {
    if (state.pagesCache.has(f.path)) return state.pagesCache.get(f.path);
    let pages = 1;
    if (f.file.name.toLowerCase().endsWith(".pdf") && f.file.size < 60e6) {
      try {
        const text = await f.file.slice(0, f.file.size).text();
        const n = (text.match(/\/Type\s*\/Page(?![s\w])/g) || []).length;
        if (n > 0) pages = n;
      } catch (_) { /* binary decoding issues: keep 1 */ }
    }
    state.pagesCache.set(f.path, pages);
    return pages;
  }

  let estimateTimer = null;
  function updateEstimate() {
    clearTimeout(estimateTimer);
    estimateTimer = setTimeout(async () => {
      const box = $("#estimate");
      if (!state.files.length) { box.hidden = true; return; }
      let pages = 0;
      for (const f of state.files) pages += await countPages(f);
      const nProfiles = nSelectedProfiles();
      try {
        const { seconds } = await api("/api/estimate", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ pages, n_profiles: nProfiles }) });
        box.hidden = false;
        box.innerHTML = `<strong>Estimated duration: about ${fmtDur(seconds)}</strong> for ${pages} page(s) and ${nProfiles} profile(s).
          Everything runs on this computer: do not shut it down, put it to sleep or close the console while the job runs.
          Jobs are processed one at a time; the estimate improves as jobs complete.`;
      } catch (_) { box.hidden = true; }
    }, 250);
  }

  async function submit() {
    const btn = $("#submit"), err = $("#submit-error");
    err.hidden = true;
    const profiles = selectedProfiles();
    if (!profiles.length && !state.customs.length) { err.hidden = false; err.textContent = "Select at least one profile or build a custom one."; return; }
    btn.disabled = true; btn.textContent = "Uploading…";
    try {
      const fd = new FormData();
      for (const f of state.files) fd.append("files", f.file, f.file.name);
      fd.append("paths", JSON.stringify(state.files.map((f) => f.path)));
      fd.append("name", $("#job-name").value.trim());
      fd.append("profiles", JSON.stringify(profiles));
      fd.append("custom_profiles", JSON.stringify(state.customs));
      fd.append("options", JSON.stringify({ review: $("#opt-review").checked, offline: $("#opt-offline").checked, gate: $("#opt-gate").value }));
      const job = await api("/api/jobs", { method: "POST", body: fd });
      state.files = []; state.customs = []; renderCustoms(); $("#job-name").value = ""; renderFiles();
      state.selected = job.id;
      await refresh();
    } catch (e) {
      err.hidden = false; err.textContent = "Could not start the job: " + e.message;
    } finally {
      btn.textContent = "Start analysis"; btn.disabled = !state.files.length;
    }
  }

  // ---------------------------------------------------------------------------------------
  // job list
  // ---------------------------------------------------------------------------------------
  function jobTime(j) {
    if (j.status === "running") return `running ${fmtDur(j.running_s)} / ~${fmtDur(j.estimate_s)}`;
    if (j.status === "queued") return `queued · ~${fmtDur(j.estimate_s)} once started`;
    if (j.status === "review") return `waiting for your review (${fmtDur(j.elapsed_s || j.running_s)})`;
    return `${j.status} · ${fmtDur(j.elapsed_s)}${j.status === "done" ? " of ~" + fmtDur(j.estimate_s) + " estimated" : ""}`;
  }

  function renderJobs() {
    const box = $("#jobs");
    $("#jobs-count").textContent = state.jobs.length ? `(${state.jobs.length})` : "";
    if (!state.jobs.length) { box.innerHTML = '<p class="muted small">No job yet. Drop a slide deck on the right to start.</p>'; return; }
    box.innerHTML = "";
    for (const j of state.jobs) {
      const div = document.createElement("div");
      div.className = "job" + (j.id === state.selected ? " selected" : "");
      const pct = j.progress && j.progress.total_units ? Math.min(100, Math.round(100 * (j.progress.done_units || 0) / j.progress.total_units)) : 0;
      const docs = j.documents.length, pages = j.documents.reduce((a, d) => a + (d.pages || 1), 0);
      const actions = [];
      if (j.status === "done") actions.push(`<button data-act="results">Results</button>`);
      if (j.status === "review") actions.push(`<button data-act="results" class="primary">Review entities</button>`);
      if (j.status === "queued" || j.status === "running") actions.push(`<button data-act="cancel">Cancel</button>`);
      if (j.status === "failed" || j.status === "cancelled") actions.push(`<button data-act="requeue">Requeue</button>`);
      if (j.status === "done" || j.status === "failed") actions.push(`<a href="/api/jobs/${esc(j.id)}/download"><button>Download</button></a>`);
      if (j.status !== "running") actions.push(`<button data-act="delete" class="danger">Delete</button>`);
      div.innerHTML = `<div class="row"><span class="name">${esc(j.name)}</span><span class="badge ${esc(j.status)}">${esc(j.status)}</span></div>
        <div class="meta">${docs} document(s), ${pages} page(s) · ${j.profiles.map((p) => `<span class="chip">${esc(p)}</span>`).join("")}</div>
        <div class="meta">${esc(jobTime(j))}${j.status === "running" ? " · " + esc(j.progress.stage || "") + (j.live_detail ? ": " + esc(j.live_detail) : "") : ""}</div>
        <div class="progress ${esc(j.status)}"><div style="width:${j.status === "done" ? 100 : pct}%"></div></div>
        <div class="actions">${actions.join("")}</div>`;
      div.addEventListener("click", (ev) => { if (!ev.target.closest("button, a")) { state.selected = j.id; refresh(); } });
      div.querySelectorAll("button[data-act]").forEach((b) => b.addEventListener("click", (ev) => { ev.stopPropagation(); jobAction(j, b.dataset.act); }));
      box.appendChild(div);
    }
  }

  async function jobAction(j, act) {
    try {
      if (act === "results") { state.selected = j.id; await refresh(); $("#detail").scrollIntoView({ behavior: "smooth" }); }
      if (act === "cancel") await api(`/api/jobs/${j.id}/cancel`, { method: "POST" });
      if (act === "requeue") await api(`/api/jobs/${j.id}/requeue`, { method: "POST" });
      if (act === "delete") {
        if (!confirm(`Delete job "${j.name}" and its results?`)) return;
        await api(`/api/jobs/${j.id}`, { method: "DELETE" });
        if (state.selected === j.id) state.selected = null;
      }
      await refresh();
    } catch (e) { alert(e.message); }
  }

  // ---------------------------------------------------------------------------------------
  // job detail
  // ---------------------------------------------------------------------------------------
  function renderDetail(j) {
    const box = $("#detail");
    if (!j) { box.hidden = true; return; }
    box.hidden = false;
    const history = (j.progress && j.progress.history) || [];
    const opts = j.options || {};
    let html = `<div class="row" style="display:flex;justify-content:space-between;align-items:baseline;gap:10px;flex-wrap:wrap">
      <h2 style="margin:0">${esc(j.name)} <span class="badge ${esc(j.status)}">${esc(j.status)}</span></h2>
      <span class="muted small">${esc(j.id)}</span></div>
      <dl class="kv">
        <dt>Created</dt><dd>${fmtTime(j.created_at)}</dd>
        <dt>Started / finished</dt><dd>${fmtTime(j.started_at)} → ${fmtTime(j.finished_at)} (${esc(jobTime(j))})</dd>
        <dt>Documents</dt><dd>${j.documents.map((d) => `${esc(d.relative || d.stem)} (${d.pages} p.)`).join("<br>")}</dd>
        <dt>Profiles</dt><dd>${j.profiles.map((p) => `<span class="chip">${esc(p)}</span>`).join("")}</dd>
        <dt>Options</dt><dd>${opts.review ? "human review · " : ""}${opts.offline ? "offline linking · " : "resolvers online · "}existence gate: ${esc(opts.gate || "default")}</dd>
      </dl>`;
    if (j.status === "running") html += `<div class="notice">Running: <strong>${esc(j.progress.stage)}</strong> on ${esc(j.progress.document)}${j.live_detail ? " — " + esc(j.live_detail) : ""}. Keep the computer awake.</div>`;
    if (j.error) html += `<div class="error">${esc(j.error)}</div>`;
    if (history.length) {
      html += `<h3>Stage timings</h3><table><tr><th>Document</th><th>Stage</th><th>Seconds</th><th>Detail</th></tr>` +
        history.map((h) => `<tr><td>${esc(h.document)}</td><td>${esc(h.stage)}</td><td>${fmtDur(h.seconds)}</td><td class="muted">${esc(h.detail || "")}</td></tr>`).join("") + `</table>`;
    }
    if (j.status === "review") html += `<div id="review-box"><p class="muted small">loading the entity sheet…</p></div>`;
    const outputs = j.outputs || {};
    const stems = Object.keys(outputs);
    if (stems.length) {
      html += `<h3>Results</h3>`;
      for (const stem of stems) {
        const o = outputs[stem];
        html += `<h4 style="margin:10px 0 4px">${esc(stem)}</h4><div class="grid">`;
        html += `<div class="stat"><div class="big">${o.n_slides ?? "–"}</div><div class="label">slides read</div></div>
                 <div class="stat"><div class="big">${o.n_entities ?? "–"}</div><div class="label">entities typed</div></div>
                 <div class="stat"><div class="big">${o.n_linked ?? "–"}</div><div class="label">linked to an identifier</div></div>
                 <div class="stat"><div class="big">${o.n_in_kg2c ?? "–"}</div><div class="label">present in RTX-KG2c (seeds)</div></div>`;
        html += `</div>`;
        if (o.result_json) html += `<p class="small"><a href="/api/jobs/${esc(j.id)}/file?path=${encodeURIComponent("documents/" + stem + "/" + stem + "_BiomedCAT.json")}">entities JSON</a>
          · <a href="/api/jobs/${esc(j.id)}/file?path=${encodeURIComponent("documents/" + stem + "/" + stem + "_stage1.json")}">slide texts and typed entities</a></p>`;
        const graphs = o.graphs || {};
        const gkeys = Object.keys(graphs);
        if (gkeys.length) {
          html += `<div class="grid">`;
          for (const p of gkeys) {
            const g = graphs[p];
            if (g.error) { html += `<div class="stat"><div class="label"><span class="chip">${esc(p)}</span></div><div class="error">${esc(g.error)}</div></div>`; continue; }
            html += `<div class="stat"><div class="label"><span class="chip">${esc(p)}</span></div>
              <div><b>${g.n_nodes}</b> nodes · <b>${g.n_edges}</b> edges · ${g.n_seeds_in_graph}/${g.n_seeds} seeds</div>
              <div class="muted small">diameter ${g.diameter ?? "–"} · ${g.n_components} component(s) · seed pairs connected ${esc(g.seed_pairs_connected || "–")} · ×${g.vocabulary_expansion} vocabulary</div>
              <p><button class="primary" data-viewer="/viewer?job=${encodeURIComponent(j.id)}&doc=${encodeURIComponent(stem)}&profile=${encodeURIComponent(p)}">Visualize my results</button>
              <a href="/api/jobs/${esc(j.id)}/summary?doc=${encodeURIComponent(stem)}&profile=${encodeURIComponent(p)}" target="_blank" rel="noopener"><button>Graph summary</button></a></p></div>`;
          }
          html += `</div>`;
        }
      }
      if (j.status === "done" || j.status === "failed") html += `<p><a href="/api/jobs/${esc(j.id)}/download"><button>Download everything (zip with trace)</button></a>
        <span class="muted small">results, graphs (CSV, GraphML, Cytoscape JSON), KG2c details and publications when available, GO enrichment, the full log.</span></p>`;
    }
    html += `<details><summary>Trace (last lines of job.log)</summary><div class="log">${esc(j.log_tail || "(empty)")}</div></details>`;
    box.innerHTML = html;
    // The viewer opens in its own window (full screen for the graph); the console stays behind.
    box.querySelectorAll("button[data-viewer]").forEach((b) => b.addEventListener("click", () =>
      window.open(b.dataset.viewer, "biomedcat-viewer-" + Date.now(), "popup=yes,width=" + Math.round(screen.availWidth * 0.95) + ",height=" + Math.round(screen.availHeight * 0.9))));
    if (j.status === "review") loadReview(j);
  }

  async function loadReview(j) {
    const box = $("#review-box");
    if (!box) return;
    try {
      const data = await api(`/api/jobs/${j.id}/review`);
      const stems = Object.keys(data.sheets);
      if (!stems.length) { box.innerHTML = '<p class="muted">No review sheet found.</p>'; return; }
      let html = `<div class="notice">The reading stage is done and nothing has left this computer yet. For each entity choose
        <b>send</b> (may be sent to the name resolvers), <b>local</b> (linked offline by exact name only) or <b>drop</b>.
        Personal data and identifier-like strings are pre-set to local.</div>`;
      for (const stem of stems) {
        html += `<h4>${esc(stem)}</h4><table class="review" data-doc="${esc(stem)}"><tr><th>Page</th><th>Entity</th><th>Type</th><th>Decision</th><th>Context</th></tr>`;
        data.sheets[stem].forEach((row, i) => {
          const key = esc(row.text + "||" + row.type);
          const radio = (v) => `<label style="display:inline;margin-right:6px"><input type="radio" name="r${stem}-${i}" value="${v}" data-key="${key}" ${row.send === v ? "checked" : ""}> ${v === "yes" ? "send" : v}</label>`;
          html += `<tr><td>${esc(row.page)}</td><td><b>${esc(row.text)}</b></td><td>${esc(row.type)}</td><td>${radio("yes")}${radio("local")}${radio("drop")}</td><td class="muted small">${esc(row.segment)}</td></tr>`;
        });
        html += `</table>`;
      }
      html += `<p><button class="primary" id="review-continue">Continue with these decisions</button></p>`;
      box.innerHTML = html;
      $("#review-continue").addEventListener("click", async () => {
        const body = stems.map((stem) => {
          const decisions = {};
          box.querySelectorAll(`table[data-doc="${CSS.escape(stem)}"] input[type=radio]:checked`).forEach((r) => { decisions[r.dataset.key] = r.value; });
          return { document: stem, decisions };
        });
        try { await api(`/api/jobs/${j.id}/review`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) }); await refresh(); }
        catch (e) { alert(e.message); }
      });
    } catch (e) { box.innerHTML = `<div class="error">${esc(e.message)}</div>`; }
  }

  // ---------------------------------------------------------------------------------------
  // polling
  // ---------------------------------------------------------------------------------------
  /* Re-render only when the data changed: a DOM rebuilt every 3 s would steal clicks and focus. */
  let lastJobs = "", lastDetail = "";
  async function refresh() {
    try {
      const jobs = await api("/api/jobs");
      const key = JSON.stringify(jobs) + "|" + state.selected;
      if (key !== lastJobs) { state.jobs = jobs; renderJobs(); lastJobs = key; }
      if (state.selected) {
        try {
          const detail = await api(`/api/jobs/${state.selected}`);
          const dkey = JSON.stringify(detail);
          if (dkey !== lastDetail) { renderDetail(detail); lastDetail = dkey; }
        } catch (_) { state.selected = null; lastDetail = ""; renderDetail(null); }
      } else if (lastDetail !== "") { lastDetail = ""; renderDetail(null); }
    } catch (e) { /* the console may be restarting; the next poll retries */ }
  }

  // ---------------------------------------------------------------------------------------
  // wiring
  // ---------------------------------------------------------------------------------------
  const drop = $("#drop");
  ["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", handleDrop);
  $("#pick-files").addEventListener("click", () => $("#file-input").click());
  $("#pick-folder").addEventListener("click", () => $("#folder-input").click());
  $("#file-input").addEventListener("change", (e) => { for (const f of e.target.files) addFile(f, f.name); renderFiles(); e.target.value = ""; });
  $("#folder-input").addEventListener("change", (e) => { for (const f of e.target.files) addFile(f, f.webkitRelativePath || f.name); renderFiles(); e.target.value = ""; });
  $("#submit").addEventListener("click", submit);
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  $("#custom-open").addEventListener("click", openBuilder);
  $("#custom-cancel").addEventListener("click", () => { $("#custom-builder").hidden = true; });
  $("#custom-add").addEventListener("click", addCustom);
  $("#custom-base").addEventListener("change", applyBase);
  $("#custom-dir").addEventListener("change", () => { $("#custom-inv-label").hidden = $("#custom-dir").value !== "on"; });
  loadStatus(); loadProfiles(); refresh();
  setInterval(refresh, 3000);
  setInterval(loadStatus, 20000);
})();
