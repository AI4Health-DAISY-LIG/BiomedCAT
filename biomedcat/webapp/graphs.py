"""Cytoscape export, KG2c enrichment tables and zip packaging of a context graph.

graph.json (built once per graph by the runner, filtered on the fly by the server):

    {
      "meta":     {profile, n_nodes, n_edges, n_seeds, ...},
      "clusters": [{id, label, n, color}],                # Biolink-derived groups (compound nodes)
      "legend":   [{category, label, color, n}],          # one colour per Biolink category
      "elements": [{"data": {...}}, ...]                   # cluster parents, nodes, edges
    }

Clusters come from the Biolink class hierarchy (data/biolink_classes_flat.json): a node's
category is walked up its ancestors until it meets one of the anchor classes of a cluster
(genes and gene products, chemicals and drugs, diseases and phenotypes, anatomy and cells,
processes and functions, organisms, clinical and exposures, other). The profile's reading
scope marks the categories the user asked for (`inScope`).

When scripts/build_kg2c_extras.py has been run, node descriptions, IRIs, synonyms and edge
publications from RTX-KG2c are joined into the details tables and the viewer.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import re
import zipfile
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from biomedcat.config import settings, ROOT_PATH

logger = logging.getLogger(__name__)

# Ordered: the first cluster whose anchors meet the class or one of its ancestors wins, so
# anatomy is tested before organisms (anatomical entity is_a organismal entity).
CLUSTERS: list[tuple[str, str, set[str]]] = [
    ("genes", "Genes & gene products", {
        "gene", "protein", "protein isoform", "polypeptide", "transcript", "RNA product", "RNA product isoform",
        "noncoding RNA product", "microRNA", "siRNA", "gene family", "protein family", "protein domain",
        "macromolecular complex", "genomic entity", "nucleic acid entity", "sequence variant", "snv", "genotype",
        "haplotype", "genome", "gene product mixin", "gene or gene product", "regulatory region", "coding sequence",
        "accessible dna region", "reagent targeted gene", "nucleosome modification", "posttranslational modification",
        "epigenomic entity", "nucleic acid sequence motif", "transcription factor binding site"}),
    ("chemicals", "Chemicals & drugs", {
        "chemical entity", "drug", "small molecule", "molecular entity", "chemical mixture", "molecular mixture",
        "complex molecular mixture", "food", "food additive", "environmental food contaminant", "processed material",
        "chemical role", "chemical or drug or treatment", "chemical entity or gene or gene product"}),
    ("diseases", "Diseases & phenotypes", {
        "disease", "phenotypic feature", "disease or phenotypic feature", "clinical finding", "behavioral feature",
        "clinical attribute", "clinical measurement", "clinical modifier", "clinical course", "onset", "severity value",
        "phenotypic quality", "biological sex", "genotypic sex", "phenotypic sex", "organism attribute"}),
    ("anatomy", "Anatomy & cells", {
        "anatomical entity", "gross anatomical structure", "cell", "cellular component", "cell line",
        "pathological anatomical structure", "pathological anatomical outcome", "cellular organism"}),
    ("processes", "Processes & functions", {
        "biological process or activity", "biological process", "molecular activity", "pathway", "physiological process",
        "behavior", "pathological process", "physical essence", "physical essence or occurrent", "occurrent",
        "molecular activity", "planetary entity"}),
    ("organisms", "Organisms & populations", {
        "organism taxon", "organismal entity", "individual organism", "population of individual organisms", "virus",
        "bacterium", "fungus", "plant", "mammal", "human", "invertebrate", "vertebrate", "life stage", "cohort",
        "study population", "subject of investigation"}),
    ("clinical", "Clinical, procedures & exposures", {
        "procedure", "treatment", "clinical intervention", "clinical entity", "clinical trial", "study", "device",
        "diagnostic aid", "exposure event", "drug exposure", "chemical exposure", "environmental exposure",
        "behavioral exposure", "genomic background exposure", "socioeconomic exposure", "geographic exposure",
        "pathological anatomical exposure", "complex chemical exposure", "drug to gene interaction exposure",
        "biotic exposure", "environmental process", "environmental feature", "geographic location", "event",
        "activity", "information content entity", "publication", "dataset", "material sample", "agent",
        "administrative entity", "socioeconomic attribute", "attribute", "evidence type", "confidence level"}),
]
OTHER = ("other", "Other")

# Colour per Biolink category: a fixed palette in a stable order, then generated hues.
PALETTE = ["#4c72b0", "#dd8452", "#55a868", "#c44e52", "#8172b3", "#937860", "#da8bc3", "#8c8c8c", "#ccb974",
           "#64b5cd", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
           "#bcbd22", "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2",
           "#dbdb8d", "#9edae5", "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173"]
CLUSTER_COLORS = {"genes": "#e8f0fb", "chemicals": "#fff1e6", "diseases": "#fde8e8", "anatomy": "#e9f7ec",
                  "processes": "#f1ecfa", "organisms": "#f3efe9", "clinical": "#fbeaf5", "other": "#f0f0f0"}


@lru_cache(maxsize=1)
def flat_classes() -> dict:
    path = Path(settings.internal_data_path) / "biolink_classes_flat.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


@lru_cache(maxsize=1)
def _class_by_norm() -> dict[str, str]:
    return {re.sub(r"[\s_\-]", "", name).lower(): name for name in flat_classes()}


def class_name(category: str) -> str:
    """'biolink:SmallMolecule' -> 'small molecule' (the Biolink class name used by the profiles)."""
    key = re.sub(r"[\s_\-]", "", (category or "").replace("biolink:", "")).lower()
    name = _class_by_norm().get(key)
    if name:
        return name
    words = re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", " ", (category or "").replace("biolink:", ""))
    return words.lower().strip() or "named thing"


def category_label(category: str) -> str:
    return (category or "").replace("biolink:", "")


def cluster_for(category: str) -> tuple[str, str]:
    """(cluster id, cluster label) of a Biolink category, from its ancestors in the class tree."""
    name = class_name(category)
    meta = (flat_classes().get(name) or {}).get("metadata", {})
    chain = [name] + list(meta.get("ancestors") or []) + list(meta.get("mixins") or [])
    chain_set = set(chain)
    for cid, label, anchors in CLUSTERS:
        if chain_set & anchors:
            return cid, label
    return OTHER


# ------------------------------------------------------------------------------------------
# KG2c lookups (DuckDB over the filtered Parquet files; optional extras tables)
# ------------------------------------------------------------------------------------------

def _kg_dir() -> Path:
    return Path(settings.kg2c_dir)


def _sql_path(path: Path) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def kg_node_info(ids: Iterable[str]) -> dict[str, dict]:
    """id -> {name, category, all_categories} from nodes.parquet (empty when the KG is absent)."""
    ids = [i for i in dict.fromkeys(ids) if i]
    path = _kg_dir() / "nodes.parquet"
    if not ids or not path.is_file():
        return {}
    import duckdb

    con = duckdb.connect()
    con.execute("CREATE TABLE want (id VARCHAR)")
    con.executemany("INSERT INTO want VALUES (?)", [(i,) for i in ids])
    degrees = _kg_dir() / "degrees.parquet"
    degree_join = f"LEFT JOIN read_parquet('{_sql_path(degrees)}') d ON d.node = n.id" if degrees.is_file() else ""
    degree_col = "d.degree" if degrees.is_file() else "NULL"
    rows = con.execute(f"SELECT n.id, n.name, n.category, n.all_categories, {degree_col} FROM read_parquet('{_sql_path(path)}') n "
                       f"JOIN want USING (id) {degree_join}").fetchall()
    con.close()
    return {r[0]: {"name": r[1], "category": r[2], "all_categories": list(r[3] or []), "degree": int(r[4] or 0)} for r in rows}


def kg_node_extras(ids: Iterable[str]) -> dict[str, dict]:
    """id -> {description, iri, synonyms, publications} from node_details.parquet, if built."""
    ids = [i for i in dict.fromkeys(ids) if i]
    path = _kg_dir() / "node_details.parquet"
    if not ids or not path.is_file():
        return {}
    import duckdb

    con = duckdb.connect()
    con.execute("CREATE TABLE want (id VARCHAR)")
    con.executemany("INSERT INTO want VALUES (?)", [(i,) for i in ids])
    rows = con.execute(f"SELECT d.id, d.description, d.iri, d.synonyms, d.publications FROM read_parquet('{_sql_path(path)}') d JOIN want USING (id)").fetchall()
    con.close()
    return {r[0]: {"description": r[1] or "", "iri": r[2] or "", "synonyms": list(r[3] or []), "publications": list(r[4] or [])} for r in rows}


def kg_edge_publications(triples: Iterable[tuple[str, str, str]]) -> dict[tuple[str, str, str], list[str]]:
    """(subject, predicate, object) -> publications from edge_publications.parquet, if built."""
    triples = list(dict.fromkeys(triples))
    path = _kg_dir() / "edge_publications.parquet"
    if not triples or not path.is_file():
        return {}
    import duckdb

    con = duckdb.connect()
    con.execute("CREATE TABLE want (subject VARCHAR, predicate VARCHAR, object VARCHAR)")
    con.executemany("INSERT INTO want VALUES (?, ?, ?)", triples)
    rows = con.execute(f"""
        SELECT p.subject, p.predicate, p.object, p.publications
        FROM read_parquet('{_sql_path(path)}') p JOIN want USING (subject, predicate, object)""").fetchall()
    con.close()
    return {(r[0], r[1], r[2]): list(r[3] or []) for r in rows}


def extras_available() -> dict[str, bool]:
    d = _kg_dir()
    return {"node_details": (d / "node_details.parquet").is_file(), "edge_publications": (d / "edge_publications.parquet").is_file(),
            "gene_go": (d / "gene_go.parquet").is_file(), "kg2c": (d / "nodes.parquet").is_file()}


# ------------------------------------------------------------------------------------------
# graph.json
# ------------------------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _job_dir_of(graph_dir: Path | None) -> Path | None:
    """<job>/graphs/<stem>/<profile> -> <job>, so custom profiles of the job can be found."""
    if graph_dir is None:
        return None
    graph_dir = Path(graph_dir)
    return graph_dir.parents[2] if len(graph_dir.parents) >= 3 and graph_dir.parents[1].name == "graphs" else None


def _load_job_profile(profile_id: str, graph_dir: Path | None = None):
    from biomedcat.profiles import load_profile
    from biomedcat.webapp.jobs import profile_path

    return load_profile(profile_path(profile_id, _job_dir_of(graph_dir)))


def _profile_scope(profile_id: str, graph_dir: Path | None = None) -> set[str]:
    try:
        return set(_load_job_profile(profile_id, graph_dir).entity_scope)
    except (OSError, ValueError):
        return set()


def _profile_strata(profile_id: str, graph_dir: Path | None = None) -> tuple[set[str], set[str]]:
    """(core classes, shared classes) of the profile: its stratum from data/biolink_strata.json, or,
    for a custom profile without stratum, the classes of its primary (1) and secondary (0.5) branches."""
    try:
        profile = _load_job_profile(profile_id, graph_dir)
        if not profile.stratum and profile.branch_weights:
            from biomedcat.weights import BiolinkModel

            model = BiolinkModel.load(Path(settings.internal_data_path) / "biolink_strata.json")
            core, shared = set(), set()
            for b, w in profile.branch_weights.items():
                (core if w >= 1.0 else shared).update([b] + sorted(model.descendants.get(b, ())))
            return core, shared
        strata = json.loads((Path(settings.internal_data_path) / "biolink_strata.json").read_text(encoding="utf-8")).get("strata", {})
        d = strata.get(profile.stratum, {})
        return set(d.get("core_classes", [])), set(d.get("shared_classes", []))
    except (OSError, ValueError, KeyError):
        return set(), set()


SCOPE_GROUPS = {"core": "Profile core classes", "shared": "Profile shared classes", "other": "Outside the profile stratum"}


def _f(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def build_graph_json(graph_dir: str | Path, profile_id: str, use_kg: bool = True) -> dict:
    """Cytoscape elements for one context-graph export directory (nodes.csv + edges.csv)."""
    graph_dir = Path(graph_dir)
    nodes = _read_csv(graph_dir / "nodes.csv")
    edges = _read_csv(graph_dir / "edges.csv")
    summary = {}
    if (graph_dir / "summary.json").is_file():
        try:
            summary = json.loads((graph_dir / "summary.json").read_text(encoding="utf-8"))
        except ValueError:
            summary = {}
    scope = _profile_scope(profile_id, graph_dir)
    scope_norm = {re.sub(r"[\s_\-]", "", s).lower() for s in scope}
    core, shared = _profile_strata(profile_id, graph_dir)
    scope_groups: dict[str, int] = {}

    # Seeds are exported with category "seed": their Biolink category comes from the KG table.
    seed_ids = [n["id"] for n in nodes if n.get("is_seed") in ("1", "True", "true")]
    info = kg_node_info(seed_ids) if use_kg else {}
    extras = kg_node_extras([n["id"] for n in nodes]) if use_kg else {}
    seed_names = {n["id"]: (n.get("name") or info.get(n["id"], {}).get("name") or n["id"]) for n in nodes if n["id"] in seed_ids}

    categories: dict[str, int] = {}
    clusters: dict[str, dict] = {}
    elements: list[dict] = []
    present: set[str] = set()
    for n in nodes:
        nid = n["id"]
        if nid in present:
            continue
        present.add(nid)
        is_seed = nid in seed_ids
        category = n.get("category") or ""
        if category == "seed" or not category:
            category = info.get(nid, {}).get("category") or "biolink:NamedThing"
        cid, clabel = cluster_for(category)
        categories[category] = categories.get(category, 0) + 1
        clusters.setdefault(cid, {"id": cid, "label": clabel, "n": 0})["n"] += 1
        seeds = [s for s in (n.get("seeds") or "").split(";") if s]
        ex = extras.get(nid, {})
        cname = class_name(category)
        scope_group = "core" if cname in core else ("shared" if cname in shared else "other")
        scope_groups[scope_group] = scope_groups.get(scope_group, 0) + 1
        elements.append({"data": {
            "id": nid, "label": n.get("name") or info.get(nid, {}).get("name") or nid, "parent": f"cluster:{cid}",
            "category": category_label(category), "categoryFull": category, "cluster": cid, "scopeGroup": scope_group,
            "hop": int(_f(n.get("hop"), 0)), "score": round(_f(n.get("score")), 5),
            "degree": int(_f(n.get("degree"), 0)) or int(info.get(nid, {}).get("degree") or 0),
            "coverage": int(_f(n.get("coverage"), 0)), "shared": n.get("shared") or "",
            "seeds": seeds, "seedNames": [seed_names.get(s, s) for s in seeds],
            "slides": [s for s in (n.get("slides") or "").split(";") if s],
            "isSeed": is_seed, "inScope": re.sub(r"[\s_\-]", "", class_name(category)).lower() in scope_norm,
            # GO comments ("// COMMENTS: ...") are shipped as descriptions by KG2c: not a description.
            "description": "" if ex.get("description", "").lstrip().startswith("//") else ex.get("description", ""),
            "iri": ex.get("iri", ""), "synonyms": ex.get("synonyms", [])[:12],
            "publications": ex.get("publications", [])[:50],
        }})

    parents = [{"data": {"id": f"cluster:{c['id']}", "label": c["label"], "isCluster": True, "n": c["n"],
                         "color": CLUSTER_COLORS.get(c["id"], "#f0f0f0")}} for c in clusters.values()]
    # Alternative grouping by the profile stratum (core / shared / outside); the viewer moves nodes
    # between the two parent sets, so both are shipped.
    parents += [{"data": {"id": f"scope:{g}", "label": SCOPE_GROUPS[g], "isCluster": True, "n": n, "mode": "scope",
                          "color": {"core": "#e6f4f2", "shared": "#f1ecfa", "other": "#f0f0f0"}[g]}} for g, n in scope_groups.items()]

    triples = [(e["subject"], e["predicate"], e["object"]) for e in edges]
    pubs = kg_edge_publications(triples) if use_kg else {}
    edge_elements = []
    for i, e in enumerate(edges):
        if e["subject"] not in present or e["object"] not in present:
            continue
        predicate = e.get("predicate") or ""
        edge_elements.append({"data": {
            "id": f"e{i}", "source": e["subject"], "target": e["object"],
            "predicate": predicate.replace("biolink:", "").replace("biolink_", "").replace("_", " "), "predicateFull": predicate,
            "weight": round(_f(e.get("weight")), 4), "sourceKb": e.get("primary_knowledge_source") or "",
            "level": e.get("knowledge_level") or "", "publications": pubs.get((e["subject"], predicate, e["object"]), [])[:50],
        }})

    ordered = sorted(categories.items(), key=lambda kv: -kv[1])
    legend = [{"category": cat, "label": category_label(cat), "color": PALETTE[i % len(PALETTE)], "n": n} for i, (cat, n) in enumerate(ordered)]
    color_of = {l["category"]: l["color"] for l in legend}
    for el in elements:
        el["data"]["color"] = color_of.get(el["data"]["categoryFull"], "#999999")

    return {
        "meta": {"profile": profile_id, "profile_name": summary.get("profile", profile_id), "n_nodes": len(elements),
                 "n_edges": len(edge_elements), "n_seeds": len(seed_ids), "n_seeds_in_graph": summary.get("n_seeds_in_graph"),
                 "diameter": summary.get("diameter"), "n_components": summary.get("n_components"),
                 "seed_pairs_connected": summary.get("seed_pairs_connected"), "vocabulary_expansion": summary.get("vocabulary_expansion"),
                 "entity_scope": sorted(scope), "extras": extras_available(), "hops": (summary.get("params") or {}).get("hops")},
        "clusters": sorted(clusters.values(), key=lambda c: -c["n"]),
        "scope_groups": [{"id": g, "label": SCOPE_GROUPS[g], "n": n} for g, n in sorted(scope_groups.items(), key=lambda kv: -kv[1])],
        "legend": legend,
        "elements": parents + elements + edge_elements,
    }


def filter_graph(graph: dict, max_nodes: int | None = None, min_score: float = 0.0, hops: int | None = None) -> dict:
    """Keep the seeds and the best-scored nodes up to `max_nodes`; drop dangling edges and empty clusters."""
    els = graph.get("elements", [])
    parents = [e for e in els if e["data"].get("isCluster")]
    nodes = [e for e in els if "source" not in e["data"] and not e["data"].get("isCluster")]
    edges = [e for e in els if "source" in e["data"]]
    kept = [n for n in nodes if n["data"]["isSeed"] or (n["data"]["score"] >= min_score and (hops is None or n["data"]["hop"] <= hops))]
    if max_nodes is not None and len(kept) > max_nodes:
        # Stratified by cluster: each Biolink group keeps a share of the budget proportional to
        # its size (at least a few nodes), best scores first, so one group with many high scores
        # cannot crowd out the others; seeds are always kept.
        seeds = [n for n in kept if n["data"]["isSeed"]]
        rest = [n for n in kept if not n["data"]["isSeed"]]
        budget = max(0, max_nodes - len(seeds))
        by_cluster: dict[str, list] = {}
        for n in rest:
            by_cluster.setdefault(n["data"]["cluster"], []).append(n)
        for members in by_cluster.values():
            members.sort(key=lambda n: -n["data"]["score"])
        total = len(rest)
        if budget < 3 * len(by_cluster):
            # Too small a budget to stratify: plain best scores.
            kept = seeds + sorted(rest, key=lambda n: -n["data"]["score"])[:budget]
        else:
            quota = {c: min(len(m), max(3, round(budget * len(m) / total))) for c, m in by_cluster.items()}
            # Fit the quotas to the budget: drop the weakest marginal node first, then hand back
            # leftovers to the groups that still have candidates.
            while sum(quota.values()) > budget:
                c = min((c for c in quota if quota[c] > 0), key=lambda c: by_cluster[c][quota[c] - 1]["data"]["score"])
                quota[c] -= 1
            leftover = budget - sum(quota.values())
            for c in sorted(by_cluster, key=lambda c: -len(by_cluster[c])):
                extra = min(leftover, len(by_cluster[c]) - quota[c])
                quota[c] += extra
                leftover -= extra
            kept = seeds + [n for c, m in by_cluster.items() for n in m[: quota[c]]]
    ids = {n["data"]["id"] for n in kept}
    kept_edges = [e for e in edges if e["data"]["source"] in ids and e["data"]["target"] in ids]
    used_parents = {n["data"]["parent"] for n in kept} | {f"scope:{n['data'].get('scopeGroup', 'other')}" for n in kept}
    kept_parents = [p for p in parents if p["data"]["id"] in used_parents]
    counts: dict[str, int] = {}
    for n in kept:
        counts[n["data"]["categoryFull"]] = counts.get(n["data"]["categoryFull"], 0) + 1
    legend = [dict(l, n=counts.get(l["category"], 0)) for l in graph.get("legend", []) if counts.get(l["category"])]
    clusters = [dict(c, n=sum(1 for n in kept if n["data"]["cluster"] == c["id"])) for c in graph.get("clusters", []) if f"cluster:{c['id']}" in used_parents]
    meta = dict(graph.get("meta", {}), n_nodes_shown=len(kept), n_edges_shown=len(kept_edges))
    scope_counts: dict[str, int] = {}
    for n in kept:
        g = n["data"].get("scopeGroup", "other")
        scope_counts[g] = scope_counts.get(g, 0) + 1
    scope_groups = [{"id": g, "label": SCOPE_GROUPS.get(g, g), "n": c} for g, c in sorted(scope_counts.items(), key=lambda kv: -kv[1])]
    return {"meta": meta, "clusters": clusters, "scope_groups": scope_groups, "legend": legend, "elements": kept_parents + kept + kept_edges}


# ------------------------------------------------------------------------------------------
# Details tables and packaging
# ------------------------------------------------------------------------------------------

def write_details_tables(graph_dir: str | Path) -> None:
    """node_details.csv / edge_details.csv: the exports joined with the KG2c information available."""
    graph_dir = Path(graph_dir)
    graph_path = graph_dir / "graph.json"
    if not graph_path.is_file():
        return
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    nodes = [e["data"] for e in graph["elements"] if "source" not in e["data"] and not e["data"].get("isCluster")]
    edges = [e["data"] for e in graph["elements"] if "source" in e["data"]]
    info = kg_node_info([n["id"] for n in nodes])
    with open(graph_dir / "node_details.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "name", "category", "all_categories", "cluster", "in_profile_scope", "is_seed", "hop", "score", "degree",
                    "coverage", "seeds", "seed_names", "slides", "description", "iri", "synonyms", "publications"])
        for n in nodes:
            w.writerow([n["id"], n["label"], n["categoryFull"], ";".join(info.get(n["id"], {}).get("all_categories", [])), n["cluster"],
                        int(n["inScope"]), int(n["isSeed"]), n["hop"], n["score"], n["degree"], n["coverage"], ";".join(n["seeds"]),
                        ";".join(n["seedNames"]), ";".join(n["slides"]), n.get("description", ""), n.get("iri", ""),
                        ";".join(n.get("synonyms", [])), ";".join(n.get("publications", []))])
    with open(graph_dir / "edge_details.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject", "predicate", "object", "weight", "primary_knowledge_source", "knowledge_level", "publications"])
        for e in edges:
            w.writerow([e["source"], e["predicateFull"], e["target"], e["weight"], e["sourceKb"], e["level"], ";".join(e.get("publications", []))])


def trace_readme(job: dict) -> str:
    """Human-readable description of the package contents and of the run (the trace)."""
    lines = [f"# BiomedCAT job {job.get('id')}", "", f"Name: {job.get('name')}", f"Created: {job.get('created_at')}",
             f"Status: {job.get('status')}  Started: {job.get('started_at')}  Finished: {job.get('finished_at')}  Elapsed: {job.get('elapsed_s')} s",
             f"Profiles: {', '.join(job.get('profiles', []))}", f"Options: {json.dumps(job.get('options', {}))}", "",
             "## Documents", ""]
    for d in job.get("documents", []):
        lines.append(f"- {d.get('file')} ({d.get('pages')} page(s))")
    lines += ["", "## Stage timings", ""]
    for h in (job.get("progress") or {}).get("history", []):
        lines.append(f"- {h.get('document')}: {h.get('stage')} {h.get('seconds')} s ({h.get('detail')})")
    lines += ["", "## Files", "",
              "- job.json: job state, parameters, timings.",
              "- job.log: full trace (OCR, every agent step, linking judgements, context-graph statistics).",
              "- profile_merged.json: reading profile (union of the selected profiles) used for OCR and NER.",
              "- profiles/<profile>.json: the profiles of the job (entity-branch priorities, directionality); profiles/derived/<profile>.derived.json and .review.md: the class scope and predicate weights derived from them, with the per-predicate review table.",
              "- documents/<stem>/<stem>_stage1.json: slide texts and typed entities before any linking.",
              "- documents/<stem>/<stem>_review.csv: the entity review sheet (review option only).",
              "- documents/<stem>/<stem>_BiomedCAT.json: linked entities with CURIE and RTX-KG2c id, run metadata.",
              "- graphs/<stem>/<profile>/nodes.csv, edges.csv: the context graph with provenance (seeds, slides, hop, score).",
              "- graphs/<stem>/<profile>/context_graph.graphml: the same graph for Cytoscape desktop or Gephi.",
              "- graphs/<stem>/<profile>/summary.json: presence metrics (size, categories, diameter, components).",
              "- graphs/<stem>/<profile>/graph.json: the Cytoscape.js elements used by the console viewer.",
              "- graphs/<stem>/<profile>/node_details.csv, edge_details.csv: exports joined with RTX-KG2c descriptions, synonyms, IRIs and publications when available.",
              "- graphs/<stem>/<profile>/enrichment_*.json: GO enrichment results, when run from the viewer.",
              "", f"Generated by the BiomedCAT console on {datetime.now().isoformat(timespec='seconds')}."]
    return "\n".join(lines) + "\n"


def build_zip(job_dir: str | Path, job: dict) -> Path:
    """Zip the job directory (minus previous zips) with the trace README; returns the zip path."""
    job_dir = Path(job_dir)
    zip_path = job_dir / f"{job_dir.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr(f"{job_dir.name}/README_trace.md", trace_readme(job))
        for path in sorted(job_dir.rglob("*")):
            if path.is_file() and path.suffix != ".zip" and not path.name.endswith(".tmp"):
                z.write(path, f"{job_dir.name}/{path.relative_to(job_dir).as_posix()}")
    return zip_path
