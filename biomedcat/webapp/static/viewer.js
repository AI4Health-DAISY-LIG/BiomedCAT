/* BiomedCAT viewer: Cytoscape.js rendering of a context graph, grouped by Biolink group or profile stratum. */
(function () {
  "use strict";
  const $ = (sel) => document.querySelector(sel);
  const params = new URLSearchParams(location.search);
  const state = { job: params.get("job"), doc: params.get("doc"), profile: params.get("profile"), graph: null, cy: null,
                  hidden: { categories: new Set(), clusters: new Set() }, budget: 400, groupMode: "cluster" };
  const groupKey = (n) => (state.groupMode === "scope" ? (n.data("scopeGroup") || "other") : (n.data("cluster") || "other"));
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  async function api(path, opts) {
    const r = await fetch(path, opts);
    if (!r.ok) { let m = r.statusText; try { m = (await r.json()).detail || m; } catch (_) { } throw new Error(m); }
    return r.json();
  }
  /* External links open in their own window (a popup), never on top of the graph. */
  const ext = (url, text) => `<a href="${esc(url)}" data-ext="1">${esc(text)}</a>`;
  const pubLink = (p) => {
    const m = /^PMID:(\d+)$/i.exec(p);
    if (m) return `<span class="pub">${ext(`https://pubmed.ncbi.nlm.nih.gov/${m[1]}/`, p)}</span>`;
    const d = /^(doi|DOI):(.+)$/.exec(p);
    if (d) return `<span class="pub">${ext(`https://doi.org/${encodeURIComponent(d[2])}`, p)}</span>`;
    return `<span class="pub">${esc(p)}</span>`;
  };
  const curieLink = (id) => ext(`https://bioregistry.io/${encodeURIComponent(id)}`, id);
  function openWindow(url) {
    const w = window.open(url, "biomedcat-ext-" + Date.now(), "popup=yes,width=1000,height=760,noopener");
    if (!w) location.href = url;  // popup blocked: fall back to the same tab
  }
  document.addEventListener("click", (ev) => {
    const a = ev.target.closest("a[data-ext]");
    if (a) { ev.preventDefault(); openWindow(a.getAttribute("href")); }
  });

  // ---------------------------------------------------------------------------------------
  // cluster layout: groups on a ring, nodes inside each group on a sunflower spiral
  // (seeds and best scores at the centre). Deterministic, no plugin needed.
  // ---------------------------------------------------------------------------------------
  function clusterLayout(cy) {
    const groups = new Map();
    cy.nodes().filter((n) => !n.data("isCluster")).forEach((n) => {
      const key = groupKey(n);
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key).push(n);
    });
    const spacing = 40;
    const clusters = [...groups.entries()].map(([id, nodes]) => {
      nodes.sort((a, b) => (b.data("isSeed") - a.data("isSeed")) || (b.data("score") - a.data("score")));
      return { id, nodes, r: spacing * Math.sqrt(nodes.length) * 0.62 + 60 };
    }).sort((a, b) => b.nodes.length - a.nodes.length);
    const perimeter = clusters.reduce((s, c) => s + 2 * c.r + 80, 0);
    const R = clusters.length > 1 ? Math.max(perimeter / (2 * Math.PI), 200) : 0;
    const positions = {};
    clusters.forEach((c, k) => {
      const angle = (2 * Math.PI * k) / clusters.length - Math.PI / 2;
      const cx = R * Math.cos(angle), cy0 = R * Math.sin(angle);
      c.nodes.forEach((n, i) => {
        const rr = spacing * 0.62 * Math.sqrt(i), th = i * 2.399963;  // golden angle
        positions[n.id()] = { x: cx + rr * Math.cos(th), y: cy0 + rr * Math.sin(th) };
      });
    });
    cy.layout({ name: "preset", positions: (n) => positions[n.id()] || { x: 0, y: 0 }, fit: true, padding: 30, animate: false }).run();
  }

  // ---------------------------------------------------------------------------------------
  // rendering
  // ---------------------------------------------------------------------------------------
  function style() {
    return [
      { selector: "node", style: { "background-color": "data(color)", "label": "data(label)", "font-size": 9, "color": "#1f2933",
        "text-valign": "bottom", "text-halign": "center", "text-margin-y": 3, "text-wrap": "ellipsis", "text-max-width": 95,
        "text-outline-color": "#fff", "text-outline-width": 2, "width": "mapData(size, 0, 1, 14, 34)", "height": "mapData(size, 0, 1, 14, 34)",
        "border-width": 0.5, "border-color": "#ffffff", "min-zoomed-font-size": 6 } },
      // Thin dark outline for the entities in the profile's reading scope.
      { selector: "node[?inScope]", style: { "border-width": 0.7, "border-color": "#4b5563" } },
      { selector: "node[?isSeed]", style: { "border-width": 2, "border-color": "#b91c1c", "font-weight": "bold", "font-size": 11, "z-index": 10 } },
      { selector: "node[?isCluster]", style: { "background-color": "data(color)", "background-opacity": 0.55, "border-width": 1, "border-style": "dashed",
        "border-color": "#9aa5b1", "label": "data(label)", "font-size": 14, "font-weight": "bold", "color": "#52606d", "text-valign": "top",
        "text-halign": "center", "text-margin-y": -6, "padding": 28, "shape": "round-rectangle", "text-outline-width": 0, "events": "no" } },
      { selector: "edge", style: { "width": "mapData(weight, 0, 1, 0.5, 3)", "line-color": "#aab4bf", "curve-style": "straight", "opacity": 0.75,
        "target-arrow-shape": "triangle", "target-arrow-color": "#aab4bf", "arrow-scale": 0.6, "label": "", "font-size": 7, "color": "#4b5563",
        "text-rotation": "autorotate", "text-background-color": "#ffffff", "text-background-opacity": 0.85, "text-background-padding": 1, "min-zoomed-font-size": 5 } },
      { selector: "edge.labeled", style: { "label": "data(predicate)" } },
      { selector: ".faded", style: { "opacity": 0.12, "text-opacity": 0.1 } },
      { selector: ".highlight", style: { "border-width": 3, "border-color": "#2563eb", "z-index": 20 } },
      { selector: "edge.highlight", style: { "line-color": "#2563eb", "target-arrow-color": "#2563eb", "width": 3, "opacity": 1, "label": "data(predicate)" } },
      { selector: ".hidden", style: { "display": "none" } },
    ];
  }

  function nodeSize(d) {
    if (d.isSeed) return 1;
    return Math.min(1, 0.25 + 0.15 * Math.log2(1 + (d.coverage || 0)) + 0.12 * Math.log10(1 + (d.degree || 0)));
  }

  function render(graph) {
    state.graph = graph;
    const elements = graph.elements.map((e) => ({ data: { ...e.data, size: e.data.isCluster || "source" in e.data ? undefined : nodeSize(e.data) } }));
    if (state.cy) state.cy.destroy();
    const cy = cytoscape({ container: $("#cy"), elements, style: style(), wheelSensitivity: 0.25, minZoom: 0.05, maxZoom: 6,
                           textureOnViewport: elements.length > 3000, hideEdgesOnViewport: elements.length > 3000 });
    state.cy = cy;
    cy.nodes("[?isCluster]").forEach((p) => p.toggleClass("hidden", p.children().length === 0));
    clusterLayout(cy);
    applyEdgeLabels();
    cy.on("tap", "node", (ev) => { if (!ev.target.data("isCluster")) showNode(ev.target); });
    cy.on("tap", "edge", (ev) => showEdge(ev.target));
    cy.on("tap", (ev) => { if (ev.target === cy) clearHighlight(); });
    cy.on("zoom", applyEdgeLabels);
    renderLegend(graph);
    renderClusters(graph);
    renderStatus(graph);
    renderMeta(graph);
    renderFigureLegend(graph);
    applyFilters();
  }

  /* Move every node under the compound parent of the chosen grouping (cluster:* or scope:*). */
  function applyGrouping() {
    const cy = state.cy;
    if (!cy) return;
    cy.batch(() => {
      cy.nodes().filter((n) => !n.data("isCluster")).forEach((n) => {
        const target = state.groupMode === "scope" ? `scope:${n.data("scopeGroup") || "other"}` : `cluster:${n.data("cluster") || "other"}`;
        if (n.data("parent") !== target && cy.getElementById(target).length) n.move({ parent: target });
      });
      cy.nodes("[?isCluster]").forEach((p) => p.toggleClass("hidden", p.children().length === 0));
    });
    state.hidden.clusters.clear();
    renderClusters(state.graph);
    clusterLayout(cy);
    applyFilters();
  }

  function applyEdgeLabels() {
    const cy = state.cy;
    if (!cy) return;
    const on = $("#edge-labels").checked && (cy.zoom() > 0.9 || cy.edges().length < 400);
    cy.batch(() => cy.edges().toggleClass("labeled", on));
  }

  function renderLegend(graph) {
    const box = $("#legend");
    box.innerHTML = graph.legend.map((l) => `<label class="legend-item"><input type="checkbox" data-cat="${esc(l.category)}" ${state.hidden.categories.has(l.category) ? "" : "checked"}>
      <span class="sw" style="background:${l.color}"></span>${esc(l.label)}<span class="n">${l.n}</span></label>`).join("");
    box.querySelectorAll("input").forEach((i) => i.addEventListener("change", () => { i.checked ? state.hidden.categories.delete(i.dataset.cat) : state.hidden.categories.add(i.dataset.cat); applyFilters(); }));
  }

  function renderClusters(graph) {
    const box = $("#clusters");
    const groups = state.groupMode === "scope" ? (graph.scope_groups || []) : graph.clusters;
    $("#clusters-title").textContent = state.groupMode === "scope" ? "Groups (profile stratum)" : "Clusters (Biolink groups)";
    box.innerHTML = groups.map((c) => `<label class="legend-item"><input type="checkbox" data-cl="${esc(c.id)}" ${state.hidden.clusters.has(c.id) ? "" : "checked"}>${esc(c.label)}<span class="n">${c.n}</span></label>`).join("");
    box.querySelectorAll("input").forEach((i) => i.addEventListener("change", () => { i.checked ? state.hidden.clusters.delete(i.dataset.cl) : state.hidden.clusters.add(i.dataset.cl); applyFilters(); }));
  }

  /* Figure legend drawn over the graph (and composed into the PNG export). */
  function legendEntries(graph) {
    const shown = new Set(state.cy ? state.cy.nodes().not(".hidden").map((n) => n.data("categoryFull")) : []);
    const cats = graph.legend.filter((l) => shown.size === 0 || shown.has(l.category)).slice(0, 14);
    return { cats, rest: Math.max(0, graph.legend.length - cats.length) };
  }

  function renderFigureLegend(graph) {
    const box = $("#figure-legend");
    box.hidden = !$("#show-legend").checked;
    const { cats, rest } = legendEntries(graph);
    box.innerHTML = `<h4>${esc(state.doc)} · ${esc(graph.meta.profile_name || graph.meta.profile)}</h4>
      <div class="cols">${cats.map((l) => `<div class="row"><span class="sw" style="background:${l.color}"></span>${esc(l.label)}</div>`).join("")}</div>
      ${rest ? `<div class="caption">+ ${rest} other categories (see the panel)</div>` : ""}
      <div class="row"><span class="sw seed"></span>seed: entity read on the slides</div>
      <div class="row"><span class="sw scope"></span>thin outline: in the profile's reading scope</div>
      <div class="row"><span class="edge"></span>edge width: profile weight of the relation</div>
      <div class="caption">Node size grows with the number of seeds reaching it. Boxes: ${state.groupMode === "scope" ? "profile stratum groups" : "Biolink groups"}.</div>`;
  }

  function applyFilters() {
    const cy = state.cy;
    if (!cy) return;
    const onlyScope = $("#only-scope").checked, onlySeedNb = $("#only-seed-nb").checked;
    cy.batch(() => {
      cy.nodes().forEach((n) => {
        if (n.data("isCluster")) return;
        const d = n.data();
        let hide = state.hidden.categories.has(d.categoryFull) || state.hidden.clusters.has(groupKey(n));
        if (onlyScope && !d.inScope && !d.isSeed) hide = true;
        if (onlySeedNb && !d.isSeed && d.hop > 1) hide = true;
        n.toggleClass("hidden", hide);
      });
      cy.nodes("[?isCluster]").forEach((p) => p.toggleClass("hidden", p.children().not(".hidden").length === 0));
    });
    const shown = cy.nodes().not(".hidden").not("[?isCluster]").length;
    const shownEdges = cy.edges().filter((e) => !e.source().hasClass("hidden") && !e.target().hasClass("hidden")).length;
    const s = $("#statusbar").querySelector("#shown");
    if (s) s.innerHTML = `<b>${shown}</b> nodes, <b>${shownEdges}</b> edges displayed`;
    if (state.graph) renderFigureLegend(state.graph);
  }

  function renderStatus(graph) {
    const m = graph.meta;
    $("#statusbar").innerHTML = `<span id="shown"></span><span>graph: <b>${m.n_nodes}</b> nodes, <b>${m.n_edges}</b> edges, <b>${m.n_seeds_in_graph ?? m.n_seeds}</b>/${m.n_seeds} seeds</span>
      <span>diameter <b>${m.diameter ?? "–"}</b>, <b>${m.n_components ?? "–"}</b> component(s), seed pairs connected <b>${esc(m.seed_pairs_connected ?? "–")}</b></span>
      <span>${m.hops ?? 2} hops · profile <b>${esc(m.profile_name || m.profile)}</b></span>
      <span class="muted">${m.extras && m.extras.node_details ? "KG2c descriptions and publications: on" : "KG2c extras not built (no descriptions/publications)"}</span>`;
  }

  function renderMeta(graph) {
    const m = graph.meta;
    $("#title").textContent = `${state.doc} · ${m.profile_name || m.profile}`;
    $("#meta").innerHTML = `Seeds are outlined in red; entities in the profile's reading scope have a thin dark outline. Node size grows with the number of seeds reaching it. ` +
      (m.entity_scope && m.entity_scope.length ? `Profile scope: ${esc(m.entity_scope.slice(0, 12).join(", "))}${m.entity_scope.length > 12 ? "…" : ""}.` : "");
  }

  // ---------------------------------------------------------------------------------------
  // search (top right of the graph)
  // ---------------------------------------------------------------------------------------
  function runSearch() {
    const q = $("#search").value.trim().toLowerCase(), cy = state.cy, box = $("#search-results");
    if (!cy || q.length < 2) { box.innerHTML = ""; if (q.length === 0) clearHighlight(); return; }
    const hits = cy.nodes().filter((n) => !n.data("isCluster") && !n.hasClass("hidden") && String(n.data("label")).toLowerCase().includes(q));
    const sorted = hits.sort((a, b) => (b.data("isSeed") - a.data("isSeed")) || a.data("label").localeCompare(b.data("label")));
    box.innerHTML = sorted.slice(0, 12).map((n) => `<div data-id="${esc(n.id())}"><span>${esc(n.data("label"))}${n.data("isSeed") ? " ●" : ""}</span><span class="cat">${esc(n.data("category"))}</span></div>`).join("")
      + (hits.length > 12 ? `<div class="cat">${hits.length} matches, showing 12</div>` : hits.length === 0 ? `<div class="cat">no node name contains "${esc(q)}"</div>` : "");
    box.querySelectorAll("div[data-id]").forEach((el) => el.addEventListener("click", () => { const t = cy.getElementById(el.dataset.id); cy.animate({ center: { eles: t }, zoom: Math.max(cy.zoom(), 1.4) }, { duration: 300 }); showNode(t); }));
    cy.elements().addClass("faded").removeClass("highlight");
    hits.removeClass("faded").addClass("highlight");
    if (hits.length) cy.animate({ fit: { eles: hits, padding: 80 } }, { duration: 300 });
  }

  // ---------------------------------------------------------------------------------------
  // selection details
  // ---------------------------------------------------------------------------------------
  function clearHighlight() {
    const cy = state.cy;
    if (cy) cy.elements().removeClass("faded highlight");
    $("#info").innerHTML = '<p class="muted">Click a node or an edge for its details.</p>';
  }

  function showNode(n) {
    const cy = state.cy, d = n.data();
    const nb = n.closedNeighborhood();
    cy.elements().addClass("faded");
    nb.removeClass("faded");
    cy.elements().removeClass("highlight");
    n.addClass("highlight");
    const neighbours = n.neighborhood("node").not(".hidden").map((x) => ({ id: x.id(), label: x.data("label"), cat: x.data("category"),
      pred: n.edgesWith(x).map((e) => e.data("predicate")).join(", ") })).sort((a, b) => a.label.localeCompare(b.label));
    let html = `<h2>${esc(d.label)}</h2><p>${curieLink(d.id)} <span class="chip">${esc(d.category)}</span>${d.isSeed ? ' <span class="badge running">seed</span>' : ""}${d.inScope ? ' <span class="badge done">in profile scope</span>' : ""}</p>`;
    if (d.description) html += `<p>${esc(d.description)}</p>`;
    html += `<dl class="kv"><dt>Cluster</dt><dd>${esc(d.cluster)}</dd><dt>KG degree</dt><dd>${d.degree}</dd>`;
    if (d.isSeed) html += `<dt>Role</dt><dd>seed: mentioned on the slides and linked to RTX-KG2c</dd>`;
    else html += `<dt>Hop</dt><dd>${d.hop}</dd><dt>Score</dt><dd>${d.score}</dd><dt>Reached from</dt><dd>${d.coverage} seed(s)${d.shared ? " (" + esc(d.shared) + ")" : ""}</dd>`;
    html += `<dt>Seeds</dt><dd>${(d.seedNames || []).map(esc).join(", ") || "–"}</dd><dt>Slides</dt><dd>${(d.slides || []).join(", ") || "–"}</dd>`;
    if (d.synonyms && d.synonyms.length) html += `<dt>Synonyms</dt><dd>${d.synonyms.map(esc).join("; ")}</dd>`;
    if (d.iri) html += `<dt>IRI</dt><dd>${ext(d.iri, d.iri)}</dd>`;
    html += `</dl>`;
    if (d.publications && d.publications.length) html += `<h3>Publications (KG2c)</h3><p>${d.publications.map(pubLink).join(" ")}</p>`;
    html += `<h3>Neighbours (${neighbours.length})</h3><table>` + neighbours.slice(0, 80).map((x) => `<tr class="clickable" data-id="${esc(x.id)}"><td>${esc(x.label)}</td><td class="muted">${esc(x.cat)}</td><td class="muted">${esc(x.pred)}</td></tr>`).join("") + `</table>`;
    $("#info").innerHTML = html;
    $("#info").querySelectorAll("tr[data-id]").forEach((tr) => tr.addEventListener("click", () => { const t = cy.getElementById(tr.dataset.id); cy.animate({ center: { eles: t } }, { duration: 300 }); showNode(t); }));
  }

  function showEdge(e) {
    const cy = state.cy, d = e.data();
    cy.elements().addClass("faded").removeClass("highlight");
    e.removeClass("faded").addClass("highlight");
    e.connectedNodes().removeClass("faded");
    let html = `<h2>${esc(e.source().data("label"))} → ${esc(e.target().data("label"))}</h2>
      <p><b>${esc(d.predicate)}</b> <span class="muted mono">${esc(d.predicateFull)}</span></p>
      <dl class="kv"><dt>Weight</dt><dd>${d.weight} (profile)</dd><dt>Source</dt><dd>${esc(d.sourceKb)}</dd><dt>Knowledge level</dt><dd>${esc(d.level)}</dd>
      <dt>Subject</dt><dd>${curieLink(d.source)}</dd><dt>Object</dt><dd>${curieLink(d.target)}</dd></dl>`;
    if (d.publications && d.publications.length) html += `<h3>Publications (KG2c)</h3><p>${d.publications.map(pubLink).join(" ")}</p>`;
    else html += `<p class="muted small">No publication attached in RTX-KG2c for this edge${state.graph.meta.extras && state.graph.meta.extras.edge_publications ? "" : " (extras tables not built)"}.</p>`;
    $("#info").innerHTML = html;
  }

  // ---------------------------------------------------------------------------------------
  // PNG export with the legend composed into the image
  // ---------------------------------------------------------------------------------------
  function exportPng() {
    const cy = state.cy, graph = state.graph;
    const img = new Image();
    img.onload = () => {
      const pad = 24, lineH = 22, colW = 260;
      const { cats, rest } = legendEntries(graph);
      const extra = ["seed: entity read on the slides", "thin outline: in the profile's reading scope", "edge width: profile weight", "node size: number of seeds reaching it"];
      const rows = Math.ceil(cats.length / 2) + extra.length + (rest ? 1 : 0) + 1;
      const legendH = rows * lineH + pad;
      const canvas = document.createElement("canvas");
      canvas.width = Math.max(img.width, 2 * colW + pad * 2);
      canvas.height = img.height + legendH + pad;
      const ctx = canvas.getContext("2d");
      ctx.fillStyle = "#ffffff"; ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);
      let y = img.height + pad;
      ctx.fillStyle = "#1f2933"; ctx.font = "bold 16px system-ui, sans-serif";
      ctx.fillText(`${state.doc} · ${graph.meta.profile_name || graph.meta.profile} (BiomedCAT context graph, RTX-KG2c)`, pad, y);
      y += lineH;
      ctx.font = "14px system-ui, sans-serif";
      cats.forEach((l, i) => {
        const x = pad + (i % 2) * colW, yy = y + Math.floor(i / 2) * lineH;
        ctx.fillStyle = l.color; ctx.beginPath(); ctx.arc(x + 7, yy - 5, 7, 0, 2 * Math.PI); ctx.fill();
        ctx.fillStyle = "#1f2933"; ctx.fillText(l.label, x + 22, yy);
      });
      y += Math.ceil(cats.length / 2) * lineH;
      if (rest) { ctx.fillStyle = "#5f6b7a"; ctx.fillText(`+ ${rest} other categories`, pad, y); y += lineH; }
      ctx.strokeStyle = "#b91c1c"; ctx.lineWidth = 2; ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(pad + 7, y - 5, 7, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      ctx.fillStyle = "#1f2933"; ctx.fillText(extra[0], pad + 22, y); y += lineH;
      ctx.strokeStyle = "#4b5563"; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(pad + 7, y - 5, 7, 0, 2 * Math.PI); ctx.stroke();
      ctx.fillText(extra[1], pad + 22, y); y += lineH;
      ctx.strokeStyle = "#aab4bf"; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(pad, y - 5); ctx.lineTo(pad + 16, y - 5); ctx.stroke();
      ctx.fillText(extra[2], pad + 22, y); y += lineH;
      ctx.fillText(extra[3], pad + 22, y);
      const a = document.createElement("a");
      a.href = canvas.toDataURL("image/png");
      a.download = `biomedcat_${state.doc}_${state.profile}.png`;
      a.click();
    };
    img.src = cy.png({ full: true, scale: 2, bg: "#ffffff" });
  }

  // ---------------------------------------------------------------------------------------
  // enrichment
  // ---------------------------------------------------------------------------------------
  async function runEnrichment() {
    const backend = $("#enrich-backend").value, scope = $("#enrich-scope").value, box = $("#enrich-result");
    if (backend === "gprofiler" && !confirm("The NCBI gene identifiers of this graph will be sent to g:Profiler (biit.cs.ut.ee). Continue?")) return;
    box.innerHTML = '<p class="muted">computing…</p>';
    try {
      const r = await api(`/api/jobs/${encodeURIComponent(state.job)}/enrich`, { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doc: state.doc, profile: state.profile, backend, scope }) });
      if (r.error) { box.innerHTML = `<div class="error">${esc(r.error)}</div>`; return; }
      let html = `<p class="muted">${r.n_query} gene(s) queried${r.n_query_in_universe != null ? ", " + r.n_query_in_universe + " annotated" : ""}${r.universe ? " · universe " + r.universe : ""} · ${r.results.length} term(s) shown. ${esc(r.note || "")}</p>`;
      if (!r.results.length) html += `<p class="muted">No enriched term at FDR ≤ 0.25 (small gene sets rarely reach significance).</p>`;
      else html += `<table class="enrich-table"><tr><th>Term</th><th>Aspect</th><th>FDR</th><th>k/K</th></tr>` +
        r.results.map((t) => `<tr class="clickable" data-genes="${esc((t.genes || []).join(","))}"><td>${esc(t.name)}<br><span class="muted mono">${esc(t.term)}</span></td><td>${esc(t.aspect)}</td><td>${t.fdr}</td><td>${t.overlap}/${t.term_size}</td></tr>`).join("") + `</table><p class="hint">Click a term to highlight its genes. The graph summary includes these results.</p>`;
      box.innerHTML = html;
      box.querySelectorAll("tr[data-genes]").forEach((tr) => tr.addEventListener("click", () => {
        const ids = tr.dataset.genes.split(",").filter(Boolean);
        const cy = state.cy;
        cy.elements().addClass("faded").removeClass("highlight");
        const sel = cy.nodes().filter((n) => ids.includes(n.id()));
        sel.removeClass("faded").addClass("highlight");
        if (sel.length) cy.animate({ fit: { eles: sel, padding: 60 } }, { duration: 400 });
      }));
    } catch (e) { box.innerHTML = `<div class="error">${esc(e.message)}</div>`; }
  }

  // ---------------------------------------------------------------------------------------
  // loading
  // ---------------------------------------------------------------------------------------
  async function loadGraph() {
    $("#statusbar").textContent = "loading the graph…";
    try {
      const g = await api(`/api/jobs/${encodeURIComponent(state.job)}/graph?doc=${encodeURIComponent(state.doc)}&profile=${encodeURIComponent(state.profile)}&max_nodes=${state.budget}`);
      render(g);
      history.replaceState(null, "", `/viewer?job=${encodeURIComponent(state.job)}&doc=${encodeURIComponent(state.doc)}&profile=${encodeURIComponent(state.profile)}`);
    } catch (e) { $("#statusbar").textContent = "could not load the graph: " + e.message; }
  }

  async function loadJob() {
    const job = await api(`/api/jobs/${encodeURIComponent(state.job)}`);
    const out = job.outputs || {};
    const sel = $("#profile-select");
    sel.innerHTML = "";
    for (const doc of Object.keys(out)) {
      for (const p of Object.keys(out[doc].graphs || {})) {
        if (out[doc].graphs[p].error) continue;
        const o = document.createElement("option");
        o.value = `${doc}||${p}`; o.textContent = `${doc} · ${p}`;
        if (doc === state.doc && p === state.profile) o.selected = true;
        sel.appendChild(o);
      }
    }
    if (!state.doc || !state.profile) { const [d, p] = (sel.value || "||").split("||"); state.doc = d; state.profile = p; }
    sel.addEventListener("change", () => { const [d, p] = sel.value.split("||"); state.doc = d; state.profile = p; $("#enrich-result").innerHTML = ""; loadGraph(); });
  }

  window.biomedcatViewer = state;  // debugging handle (the Cytoscape instance is state.cy)

  window.initViewer = async function () {
    if (!state.job) { $("#statusbar").textContent = "no job given: open a graph from the console"; return; }
    let t = null;
    $("#budget").addEventListener("input", () => { $("#budget-val").textContent = $("#budget").value; clearTimeout(t); t = setTimeout(() => { state.budget = +$("#budget").value; loadGraph(); }, 500); });
    $("#fit").addEventListener("click", () => state.cy && state.cy.fit(undefined, 30));
    $("#relayout").addEventListener("click", () => state.cy && clusterLayout(state.cy));
    $("#cose").addEventListener("click", () => state.cy && state.cy.layout({ name: "cose", animate: false, nodeRepulsion: 9000, idealEdgeLength: 60, numIter: 400, fit: true, padding: 30 }).run());
    $("#png").addEventListener("click", exportPng);
    $("#edge-labels").addEventListener("change", applyEdgeLabels);
    $("#show-legend").addEventListener("change", () => state.graph && renderFigureLegend(state.graph));
    $("#group-mode").addEventListener("change", () => { state.groupMode = $("#group-mode").value; applyGrouping(); });
    $("#only-scope").addEventListener("change", applyFilters);
    $("#only-seed-nb").addEventListener("change", applyFilters);
    $("#search").addEventListener("input", runSearch);
    $("#search").addEventListener("keydown", (ev) => { if (ev.key === "Enter") { const first = $("#search-results div[data-id]"); if (first) first.click(); } });
    $("#summary-html").addEventListener("click", () => openWindow(`/api/jobs/${encodeURIComponent(state.job)}/summary?doc=${encodeURIComponent(state.doc)}&profile=${encodeURIComponent(state.profile)}`));
    $("#summary-md").addEventListener("click", () => { location.href = `/api/jobs/${encodeURIComponent(state.job)}/summary?doc=${encodeURIComponent(state.doc)}&profile=${encodeURIComponent(state.profile)}&format=md`; });
    $("#enrich-run").addEventListener("click", runEnrichment);
    try { await loadJob(); await loadGraph(); } catch (e) { $("#statusbar").textContent = e.message; }
  };
})();
