"""Narrative summary of a context graph, in Markdown and HTML, with every node name in bold.

The summary is regenerated on request by the console (so a GO enrichment run from the viewer is
included) and written next to graph.json by the runner (graph_summary.md, graph_summary.html),
hence shipped in the download package.
"""
from __future__ import annotations

import html as _html
import json
import re
from datetime import datetime
from pathlib import Path


def _b(name: str) -> str:
    """Bold a node name in Markdown, escaping the characters that would break the emphasis."""
    return "**" + str(name).replace("*", "\\*").replace("\n", " ").strip() + "**"


def graph_summary_markdown(graph: dict, job: dict | None = None, enrichment: dict | None = None, top_per_group: int = 15) -> str:
    """Seeds, size, composition, neighbourhood by group (all node names in bold), seed-to-seed
    connections, GO enrichment when available, provenance."""
    meta = graph.get("meta", {})
    els = graph.get("elements", [])
    nodes = [e["data"] for e in els if "source" not in e["data"] and not e["data"].get("isCluster")]
    edges = [e["data"] for e in els if "source" in e["data"]]
    by_id = {n["id"]: n for n in nodes}
    seeds = [n for n in nodes if n.get("isSeed")]
    clusters = graph.get("clusters", [])
    doc = (job or {}).get("_doc", "") or "document"
    profile = meta.get("profile_name") or meta.get("profile", "")
    when = datetime.now().strftime("%Y-%m-%d %H:%M")
    slides = sorted({s for n in seeds for s in n.get("slides", [])}, key=lambda s: (len(s), s))
    n_pub_edges = sum(1 for e in edges if e.get("publications"))
    scope = meta.get("entity_scope") or []

    out: list[str] = [f"# Context graph summary: {doc} · {profile}", ""]
    out.append(f"Generated on {when} by the BiomedCAT console" + (f" (job {job.get('id')})" if job and job.get("id") else "") + ".")
    out += ["", "## Reading", ""]
    out.append(
        f"This graph is the neighbourhood of RTX-KG2c around {_b(str(len(seeds)))} seed entities read on the slides"
        + (f" (slides {', '.join(slides)})" if slides else "") + f", extracted with the profile {_b(profile)}"
        + (f" over {meta.get('hops')} hops" if meta.get("hops") else "") + ". "
        f"It holds {_b(str(meta.get('n_nodes', len(nodes))))} nodes and {_b(str(meta.get('n_edges', len(edges))))} edges"
        + (f", diameter {meta.get('diameter')}" if meta.get("diameter") is not None else "")
        + (f", {meta.get('n_components')} connected component(s)" if meta.get("n_components") is not None else "")
        + (f", {meta.get('seed_pairs_connected')} seed pairs connected inside the graph" if meta.get("seed_pairs_connected") else "") + ". "
        "Edge weights come from the profile (predicate, source and knowledge-level weights); node scores sum the best "
        "profile-weighted paths from the seeds, penalised by the degree of the intermediate nodes, so generic hubs "
        "contribute little. This is a presence map of the knowledge around the slides, not a ranking of hypotheses.")
    if scope:
        out += ["", "Reading scope of the profile (entity kinds the slide reader was asked to name): " + ", ".join(scope) + "."]
    out += ["", "## Seeds (entities read on the slides and linked to RTX-KG2c)", ""]
    for n in sorted(seeds, key=lambda n: str(n["label"]).lower()):
        desc = (n.get("description") or "").split(". ")[0].strip()
        out.append(f"- {_b(n['label'])} ({n.get('category')}, {n['id']}"
                   + (f", slides {', '.join(n.get('slides', []))}" if n.get("slides") else "") + ")" + (f": {desc[:220]}" if desc else ""))
    out += ["", "## Composition", "", "| Group | Nodes | Main Biolink categories |", "|---|---|---|"]
    for c in clusters:
        members = [n for n in nodes if n.get("cluster") == c["id"]]
        cats: dict[str, int] = {}
        for n in members:
            cats[n.get("category", "")] = cats.get(n.get("category", ""), 0) + 1
        top = ", ".join(f"{k} ({v})" for k, v in sorted(cats.items(), key=lambda kv: -kv[1])[:4])
        out.append(f"| {c['label']} | {len(members)} | {top} |")
    sg = graph.get("scope_groups") or []
    if sg:
        out += ["", "Position relative to the profile stratum: " + ", ".join(f"{g['n']} node(s) {str(g['label']).lower()}" for g in sg) + "."]
    out += ["", "## Neighbourhood by group", ""]
    for c in clusters:
        members = sorted((n for n in nodes if n.get("cluster") == c["id"]),
                         key=lambda n: (-int(bool(n.get("isSeed"))), -float(n.get("score") or 0)))
        if not members:
            continue
        out += [f"### {c['label']} ({len(members)} nodes)", ""]
        best = [n for n in members if not n.get("isSeed")][:top_per_group]
        if best:
            parts = []
            for n in best:
                via = ", ".join(n.get("seedNames", [])[:3])
                parts.append(f"{_b(n['label'])} ({n.get('category')}" + (f"; reached from {via}" if via else "") + ")")
            out += ["Best-scored nodes: " + "; ".join(parts) + ".", ""]
        out += ["All nodes: " + ", ".join(_b(n["label"]) for n in sorted(members, key=lambda n: str(n["label"]).lower())) + ".", ""]
    seed_ids = {n["id"] for n in seeds}
    seed_edges = [e for e in edges if e["source"] in seed_ids and e["target"] in seed_ids]
    if seed_edges:
        out += ["## Direct connections between seeds", ""]
        for e in sorted(seed_edges, key=lambda e: -float(e.get("weight") or 0))[:60]:
            s, t = by_id.get(e["source"], {}), by_id.get(e["target"], {})
            pubs = e.get("publications") or []
            out.append(f"- {_b(s.get('label', e['source']))} —{e.get('predicate')}→ {_b(t.get('label', e['target']))} "
                       f"({e.get('sourceKb') or 'unknown source'}, {e.get('level') or ''}, weight {e.get('weight')}"
                       + (f", {len(pubs)} publication(s): {', '.join(pubs[:5])}" + ("…" if len(pubs) > 5 else "") if pubs else "") + ")")
        out.append("")
    if enrichment and enrichment.get("results"):
        out += [f"## GO enrichment ({enrichment.get('backend')})", ""]
        out.append(f"{enrichment.get('n_query')} gene(s) of the graph queried"
                   + (f", {enrichment.get('n_query_in_universe')} annotated" if enrichment.get("n_query_in_universe") else "")
                   + ". " + str(enrichment.get("note", "")))
        out += ["", "| Term | Aspect | FDR | Overlap | Genes |", "|---|---|---|---|---|"]
        for r in enrichment["results"][:25]:
            genes = ", ".join(_b(by_id.get(g, {}).get("label", g)) for g in r.get("genes", [])[:8]) + ("…" if len(r.get("genes", [])) > 8 else "")
            out.append(f"| {r.get('name')} ({r.get('term')}) | {r.get('aspect')} | {r.get('fdr')} | {r.get('overlap')}/{r.get('term_size')} | {genes} |")
        out.append("")
    out += ["## Provenance", ""]
    out.append(f"Knowledge graph: RTX-KG2c 2.10.1 (filtered build, see data/kg2c/stats.json). {n_pub_edges} of the {len(edges)} edges carry "
               "publications in RTX-KG2c"
               + ("; descriptions, synonyms and publications of nodes come from the same release." if (meta.get("extras") or {}).get("node_details") else ".")
               + " Node identifiers resolve at https://bioregistry.io/<identifier>. Files: nodes.csv, edges.csv, context_graph.graphml, "
                 "node_details.csv, edge_details.csv, graph.json.")
    return "\n".join(out) + "\n"


def markdown_to_html(md: str, title: str = "Graph summary") -> str:
    """Minimal converter for the summary's own Markdown subset: headings, paragraphs, bullet lists,
    tables, **bold**, [text](url) and bare links. No external dependency."""

    def inline(text: str) -> str:
        text = _html.escape(text, quote=False)
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"\[([^\]]+)\]\((https?://[^\s)]+)\)", r'<a href="\2" target="_blank" rel="noopener">\1</a>', text)
        text = re.sub(r"(?<![\"'>])(https?://[^\s<)]+)", r'<a href="\1" target="_blank" rel="noopener">\1</a>', text)
        return text

    body: list[str] = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1
        elif line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            body.append(f"<h{level}>{inline(line[level:].strip())}</h{level}>")
            i += 1
        elif line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            rows = [r for r in rows if not all(re.fullmatch(r"-+", c or "") for c in r)]
            if rows:
                head, *rest = rows
                body.append("<table><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr>"
                            + "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in rest) + "</table>")
        elif line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{inline(lines[i][2:])}</li>")
                i += 1
            body.append("<ul>" + "".join(items) + "</ul>")
        else:
            para = []
            while i < len(lines) and lines[i].strip() and not lines[i].startswith(("#", "|", "- ")):
                para.append(lines[i])
                i += 1
            body.append(f"<p>{inline(' '.join(para))}</p>")
    css = ("body{font:14px/1.5 system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;color:#1f2933;max-width:900px;margin:32px auto;padding:0 20px}"
           "h1{font-size:22px}h2{font-size:17px;margin-top:28px}h3{font-size:14px;margin-top:18px}table{border-collapse:collapse;font-size:13px;margin:8px 0}"
           "th,td{border-bottom:1px solid #e1e6eb;padding:4px 8px;text-align:left;vertical-align:top}th{color:#5f6b7a}b{color:#0f3d3a}a{color:#2563eb}")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>{_html.escape(title)}</title><style>{css}</style></head>"
            f"<body>{''.join(body)}</body></html>")


def write_graph_summary(graph_dir: str | Path, job: dict | None = None, doc: str = "") -> tuple[Path, Path]:
    """Write graph_summary.md and graph_summary.html next to graph.json (uses the cached local enrichment if any)."""
    graph_dir = Path(graph_dir)
    graph = json.loads((graph_dir / "graph.json").read_text(encoding="utf-8"))
    enrichment = None
    for cand in sorted(graph_dir.glob("enrichment_local_*.json")):
        try:
            enrichment = json.loads(cand.read_text(encoding="utf-8"))
        except ValueError:
            continue
    doc = doc or graph_dir.parent.name
    md = graph_summary_markdown(graph, dict(job or {}, _doc=doc), enrichment)
    md_path, html_path = graph_dir / "graph_summary.md", graph_dir / "graph_summary.html"
    md_path.write_text(md, encoding="utf-8")
    html_path.write_text(markdown_to_html(md, f"Graph summary: {doc} · {graph.get('meta', {}).get('profile_name', '')}"), encoding="utf-8")
    return md_path, html_path
