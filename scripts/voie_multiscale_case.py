#!/usr/bin/env python
"""Multi-scale case for the context-graph paper: three mechanistic hypotheses drawn on three slides
by the FSHD researchers, and their union, read by the same pipeline with the same profile.

The FSHD1 inflammation deck (v13) explains the muscle vulnerability of FSHD1 by three pathways that
share one root, DUX4, and diverge in scale:

  VoieA     molecular    DUX4 -> retroelements, dsRNA -> innate sensors (RIG-I, MDA5, TLR3) -> IFN,
                         ISGs, STAT1 -> MHC-I presentation (viral mimicry)
  VoieB     organelle    DUX4 -> mitochondrial dysfunction -> ROS, lipid oxidation, mtDNA damage ->
                         DAMPs (HMGB1), inflammasome, IL-6 -> bioenergetic deficit, fatigue
  VoieC     tissue       damaged fibre -> macrophages, FAPs -> fibro-adipogenic remodelling,
                         fibrosis -> force transmission, MRI/STIR
  VoiesA-C  integrated   the three pathways on one slide

Every slide was run through the published pipeline and its linked entities seeded a context graph
of RTX-KG2c under the biochemical_actions profile (2 hops, per-category cap 300). This script reads
those graphs and reports, for each deck:

  structure    size, diameter, components, connected seed pairs, seed-to-seed distances, share of
               saturated categories; with the complexity of the slide (mentions, linked entities,
               distinct Biolink types) read from the pipeline JSON;
  signatures   presence and prominence of pathway signature nodes (Reactome / GO / gene nodes that
               are seeds of no deck but that a reader of the pathway expects):
               a 3 x N matrix (signature A/B/C x deck), which says whether the enrichment follows the
               biology of the slide or falls back on a generic "muscle disease" neighbourhood;
  distances    shortest paths inside each kept graph between the three signatures, and from DUX4 to
               each signature, with the bridge nodes on those paths (the DUX4 use case);
  overlap      Jaccard overlap of the kept node sets between decks, and how much of the union slide's
               graph is already in the single-pathway graphs;
  mechanism    rank of the six Reactome nodes of DUX4 target activation (fshd_dux4_mechanism_case.py)
               in every deck.

None of this ranks the hypotheses: distances and ranks are presence metrics of the kept graph.

Usage:
  uv run python scripts/voie_multiscale_case.py [--graphs-dir output] [--out data/eval/voie_multiscale]
                                                [--decks VoieA VoieB VoieC VoiesA-C FSHD1multiscales]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from itertools import combinations
from pathlib import Path

B = Path(__file__).resolve().parents[1]

DEFAULT_DECKS = ["VoieA", "VoieB", "VoieC", "VoiesA-C", "FSHD1multiscales"]

# How the researchers meant each slide (used as labels only).
SCALE = {
    "VoieA": "molecular (innate antiviral sensing)",
    "VoieB": "organelle (mitochondria, redox, bioenergetics)",
    "VoieC": "tissue (immune infiltrate, fibro-adipogenic remodelling)",
    "VoiesA-C": "integrated (A + B + C on one slide)",
    "FSHD1multiscales": "summary (nucleus to clinic, inflammation)",
    "FSHD": "reference deck (two slides)",
}

DUX4_ID = "NCBIGene:100288687"

# Pathway signatures: RTX-KG2c node names a reader of the pathway expects to find around the slide
# entities. None of them is a seed of any deck (they were either not written on the slide, or written
# but not linked, e.g. HMGB1 and the inflammasome on slide B), so every hit is produced by the
# expansion. A pattern is a case-insensitive substring of the node name; a leading "=" asks for an
# exact (case-insensitive) name match.
SIGNATURES = {
    "A": [
        "DDX58/IFIH1-mediated induction of interferon",
        "RIG-I signaling pathway",
        "Toll Like Receptor 3 (TLR3) Cascade",
        "ISG15 antiviral mechanism",
        "Activation of IRF3, IRF7 mediated by TBK1",
        "TRAF6 mediated IRF7 activation",
        "=antiviral innate immune response",
        "cGAS/STING signaling pathway",
        "antigen processing and presentation of peptide antigen via MHC class I",
        "=cellular response to dsRNA",
    ],
    "B": [
        "Mitochondrial complex I (NADH dehydrogenase)",
        "=reactive oxygen species metabolic process",
        "Aerobic respiration and respiratory electron transport",
        "The NLRP3 inflammasome",
        "=NLRP3",
        "=HMGB1",
        "=4-hydroxynonenal",
        "cell death in response to oxidative stress",
        "Interleukin-6 family signaling",
        "Detoxification of Reactive Oxygen Species",
    ],
    "C": [
        "Extracellular matrix organization",
        "Degradation of the extracellular matrix",
        "=collagen-containing extracellular matrix",
        "Signaling by TGF-beta Receptor Complex",
        "=TGFB1",
        "=Chemokine CCL2",
        "=TNF-alpha",
        "=Tumor Necrosis Factors",
        "=execution phase of apoptosis",
        "=Collagen",
    ],
}

# The Reactome nodes of the DUX4 target-activation mechanism (same list as fshd_dux4_mechanism_case.py).
MECHANISM_NODE_NAMES = (
    "Expression of DUX4 in the zygote",
    "DUX4 binds the KDM4E gene",
    "DUX4 binds the LEUTX gene",
    "DUX4 binds HERVL LTR",
    "DUX4 binds MaLR",
    "DUX4 binds HSATII pericentric repeat",
)


# ----------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------
def graph_dir(root: Path, deck: str) -> Path:
    """`output/<deck>_context_graph` (pipeline layout) or `<root>/<deck>` (data/context_graph layout)."""
    for d in (root / f"{deck}_context_graph", root / deck):
        if (d / "nodes.csv").is_file():
            return d
    raise FileNotFoundError(f"no context graph for {deck} under {root}")


def result_json(root: Path, deck: str) -> dict | None:
    for p in (root / f"_final_{deck}" / f"{deck}_BiomedCAT.json", root / f"{deck}_BiomedCAT.json"):
        if p.is_file():
            return json.loads(p.read_text(encoding="utf-8"))
    return None


def load_graph(d: Path):
    """nodes (list of dicts, unique ids), edges, undirected simple igraph graph, id -> vertex index.

    Parallel edges between two nodes are collapsed to the heaviest one; the graph keeps that weight
    (`w`), the edge's predicate, source and knowledge level (`label`), so that a path can be read as
    a chain of assertions, and a `cost` reproducing the stage's path_weight: the cheapest path is the
    one with the largest product of edge weights divided by log2(2 + degree) of every intermediate
    node (degree in RTX-KG2c, column `degree` of nodes.csv). Without the degree term, weight-1 edges
    (subclass_of, manifestation_of) are free and the best path wanders through ontology hierarchies.
    """
    import math

    import igraph as ig

    with open(d / "nodes.csv", encoding="utf-8", newline="") as f:
        nodes = list(csv.DictReader(f))
    with open(d / "edges.csv", encoding="utf-8", newline="") as f:
        edges = list(csv.DictReader(f))
    index = {n["id"]: i for i, n in enumerate(nodes)}
    best: dict[tuple[int, int], dict] = {}
    for e in edges:
        if e["subject"] not in index or e["object"] not in index:
            continue
        a, b = index[e["subject"]], index[e["object"]]
        if a == b:
            continue
        key = (min(a, b), max(a, b))
        w = float(e["weight"] or 0)
        if key not in best or w > best[key]["w"]:
            best[key] = {"w": w, "subject": e["subject"], "object": e["object"], "predicate": e["predicate"].replace("biolink:", ""),
                         "source": e.get("primary_knowledge_source", ""), "level": e.get("knowledge_level", "")}
    g = ig.Graph(n=len(nodes), directed=False)
    g.vs["id"] = [n["id"] for n in nodes]
    g.vs["name"] = [n["name"] for n in nodes]
    g.add_edges(list(best))
    g.es["w"] = [v["w"] for v in best.values()]
    g.es["label"] = [v for v in best.values()]
    # Degree penalty of the stage, log2(2 + degree), split in two halves on the edges touching the
    # node: the endpoints of a path then carry a constant, and the sum over a path orders paths
    # between two given nodes exactly as path_weight does. Seeds have no KG2c degree in nodes.csv
    # (they are not scored); their kept-graph degree stands in.
    kept_deg = g.degree()
    pen = []
    for i, n in enumerate(nodes):
        deg = int(float(n["degree"])) if n.get("degree") not in (None, "") else kept_deg[i]
        pen.append(math.log(math.log2(2 + max(deg, 0))))
    g.es["cost"] = [-math.log(max(v["w"], 1e-6)) + 0.5 * (pen[a] + pen[b]) for (a, b), v in best.items()]
    summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    return nodes, edges, g, index, summary


# ----------------------------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------------------------
def complexity(result: dict | None) -> dict:
    """Size of the hypothesis as the pipeline read it: mentions, linked mentions, seeds, Biolink types."""
    if not result:
        return {}
    ents = result.get("entities") or []
    linked = [e for e in ents if e.get("kg2c_id")]
    return {
        "n_mentions": len(ents),
        "n_linked_mentions": len(linked),
        "n_distinct_seeds": len({e["kg2c_id"] for e in linked}),
        "n_linked_biolink_types": len({e.get("type") for e in linked}),
        "n_segments": len({(e.get("page"), e.get("segment")) for e in ents}),
        "n_slides": len(result.get("slides") or []),
    }


def structure(nodes, g, index, summary) -> dict:
    seeds = [n["id"] for n in nodes if n["is_seed"] == "1"]
    seed_idx = [index[s] for s in seeds]
    dist = g.distances(source=seed_idx, target=seed_idx)
    finite = [dist[i][j] for i in range(len(seed_idx)) for j in range(i + 1, len(seed_idx))
              if dist[i][j] != float("inf")]
    n_pairs = len(seed_idx) * (len(seed_idx) - 1) // 2
    cap = (summary.get("params") or {}).get("per_category_cap", 300)
    cats = summary.get("categories") or {}
    hops = [int(n["hop"]) for n in nodes if n["is_seed"] != "1"]
    return {
        "n_seeds": len(seeds),
        "n_nodes": summary["n_nodes"],
        "n_edges": summary["n_edges"],
        "n_reached": summary.get("n_reached"),
        "vocabulary_expansion": summary.get("vocabulary_expansion"),
        "diameter": summary.get("diameter"),
        "n_components": summary.get("n_components"),
        "seed_pairs_connected": summary.get("seed_pairs_connected"),
        "seed_pairs_connected_frac": round(len(finite) / n_pairs, 3) if n_pairs else None,
        "seed_distance_mean": round(statistics.mean(finite), 2) if finite else None,
        "seed_distance_max": max(finite) if finite else None,
        "share_hop1": round(sum(1 for h in hops if h == 1) / len(hops), 3) if hops else None,
        "n_categories": len(cats),
        "n_saturated_categories": sum(1 for v in cats.values() if v >= cap),
        "categories": cats,
        "profile": summary.get("profile"),
    }


def category_ranks(nodes) -> dict[str, tuple[int, int]]:
    """node id -> (rank within its Biolink category by score, category size); seeds are not ranked."""
    by_cat: dict[str, list[dict]] = {}
    for n in nodes:
        if n["is_seed"] != "1":
            by_cat.setdefault(n["category"], []).append(n)
    out = {}
    for cat, rows in by_cat.items():
        rows.sort(key=lambda r: float(r["score"] or 0), reverse=True)
        for i, r in enumerate(rows):
            out[r["id"]] = (i + 1, len(rows))
    return out


def match(pattern: str, name: str) -> bool:
    if pattern.startswith("="):
        return name.lower() == pattern[1:].lower()
    return pattern.lower() in name.lower()


def signature_hits(nodes, patterns, ranks) -> list[dict]:
    """One row per pattern: whether it is present, as a seed or not, and the best-ranked hit."""
    rows = []
    for p in patterns:
        hits = [n for n in nodes if match(p, n["name"])]
        non_seed = [n for n in hits if n["is_seed"] != "1"]
        best = min(non_seed, key=lambda n: ranks[n["id"]][0] / ranks[n["id"]][1], default=None)
        rows.append({
            "pattern": p,
            "present": bool(hits),
            "as_seed": bool(hits) and not non_seed,
            "n_hits": len(hits),
            "best_id": best["id"] if best else None,
            "best_name": best["name"] if best else None,
            "best_category": best["category"].replace("biolink:", "") if best else None,
            "best_hop": int(best["hop"]) if best else None,
            "best_coverage": int(best["coverage"]) if best else None,
            "best_rank": ranks[best["id"]][0] if best else None,
            "best_category_size": ranks[best["id"]][1] if best else None,
            "best_rank_pct": round(ranks[best["id"]][0] / ranks[best["id"]][1], 3) if best else None,
        })
    return rows


def signature_summary(rows: list[dict]) -> dict:
    pcts = [r["best_rank_pct"] for r in rows if r["best_rank_pct"] is not None]
    return {
        "n_patterns": len(rows),
        "n_present": sum(1 for r in rows if r["present"]),
        "n_present_as_seed": sum(1 for r in rows if r["as_seed"]),
        "recovered_frac": round(sum(1 for r in rows if r["present"]) / len(rows), 3),
        "median_rank_pct": round(statistics.median(pcts), 3) if pcts else None,
        "n_hop1": sum(1 for r in rows if r["best_hop"] == 1),
    }


def _annotated(g, vpath: list[int]) -> list[dict]:
    """A vertex path as a list of nodes, each step carrying the edge that leads to it."""
    out = []
    for k, v in enumerate(vpath):
        node = {"id": g.vs[v]["id"], "name": g.vs[v]["name"]}
        if k:
            e = g.es[g.get_eid(vpath[k - 1], v)]
            lab = e["label"]
            # Read the step in the direction of the path; mark it with "<-" when the stored edge points back.
            forward = lab["subject"] == g.vs[vpath[k - 1]]["id"]
            node["edge"] = {"predicate": lab["predicate"], "source": lab["source"], "level": lab["level"],
                            "weight": round(lab["w"], 4), "forward": forward}
        out.append(node)
    return out


def shortest_between(g, index, src_ids: list[str], dst_ids: list[str]) -> dict:
    """Shortest connection between two node sets inside the kept graph.

    `distance` is the minimum number of hops; `path` is the path with the largest product of edge
    weights (sum of -log w), the measure the stage itself uses to score a node, so that a chain of
    co-localisation edges (DUX4 -located_in-> cytoplasm <-located_in- X) does not win over a direct,
    specific assertion only because it is as short.
    """
    src = [index[i] for i in src_ids if i in index]
    dst = [index[i] for i in dst_ids if i in index]
    if not src or not dst:
        return {"distance": None, "path": [], "hop_path": []}
    dist = g.distances(source=src, target=dst)
    best = min(((dist[i][j], i, j) for i in range(len(src)) for j in range(len(dst))), key=lambda t: t[0])
    if best[0] == float("inf"):
        return {"distance": None, "path": [], "hop_path": []}
    hop_path = g.get_shortest_paths(src[best[1]], to=dst[best[2]], output="vpath")[0]
    wdist = g.distances(source=src, target=dst, weights="cost")
    wbest = min(((wdist[i][j], i, j) for i in range(len(src)) for j in range(len(dst))), key=lambda t: t[0])
    wpath = g.get_shortest_paths(src[wbest[1]], to=dst[wbest[2]], weights="cost", output="vpath")[0]
    return {"distance": int(best[0]), "weighted_hops": len(wpath) - 1, "weighted_cost": round(wbest[0], 3),
            "path": _annotated(g, wpath), "hop_path": _annotated(g, hop_path)}


def mechanism_ranks(nodes, ranks) -> list[dict]:
    out = []
    for n in nodes:
        if any(m in n["name"] for m in MECHANISM_NODE_NAMES) and n["id"] in ranks:
            rank, size = ranks[n["id"]]
            out.append({"name": n["name"], "rank": rank, "category_size": size, "hop": int(n["hop"]),
                        "coverage": int(n["coverage"]), "score": float(n["score"] or 0)})
    return sorted(out, key=lambda r: r["rank"])


# ----------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------
def path_text(p: list[dict]) -> str:
    parts = [p[0]["name"]]
    for n in p[1:]:
        e = n["edge"]
        arrow = "-->" if e["forward"] else "<--"
        parts.append(f" --{e['predicate']} [{e['source'].replace('infores:', '')}, {e['level']}, {e['weight']}]{arrow} {n['name']}")
    return "".join(parts)


def write_report(out: Path, R: dict) -> None:
    decks = R["decks"]
    L = ["# Multi-scale case: Voie A, B, C and their union", "",
         f"Graphs: {R['graphs_dir']}; profile: {R['profile']}", "",
         "## Structure", "",
         "| deck | scale | mentions | linked | seeds | types | nodes | edges | diameter | components | seed pairs connected | mean seed distance | saturated categories |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for d in decks:
        s, c = R["structure"][d], R["complexity"][d]
        L.append(f"| {d} | {SCALE.get(d, '')} | {c.get('n_mentions', '')} | {c.get('n_linked_mentions', '')} | {s['n_seeds']} | "
                 f"{c.get('n_linked_biolink_types', '')} | {s['n_nodes']} | {s['n_edges']} | {s['diameter']} | {s['n_components']} | "
                 f"{s['seed_pairs_connected']} | {s['seed_distance_mean']} | {s['n_saturated_categories']}/{s['n_categories']} |")
    L += ["", "## Signature recovery (patterns present / patterns; median rank percentile of the best hit inside its category)", "",
          "| signature | " + " | ".join(decks) + " |", "|---|" + "---|" * len(decks)]
    for sig in SIGNATURES:
        cells = []
        for d in decks:
            s = R["signatures"][d][sig]["summary"]
            cells.append(f"{s['n_present']}/{s['n_patterns']} (p{s['median_rank_pct']})" if s["median_rank_pct"] is not None else f"{s['n_present']}/{s['n_patterns']}")
        L.append(f"| {sig} | " + " | ".join(cells) + " |")
    L += ["", "## Distances inside the kept graph (minimum hops, undirected; in brackets the hops of the best-weighted path)", "",
          "| deck | A-B | A-C | B-C | DUX4-A | DUX4-B | DUX4-C |", "|---|---|---|---|---|---|---|"]
    for d in decks:
        D = R["distances"][d]
        L.append(f"| {d} | " + " | ".join(f"{D[k]['distance']} [{D[k].get('weighted_hops')}]" if D[k]["distance"] is not None else "-"
                                          for k in ["A-B", "A-C", "B-C", "DUX4-A", "DUX4-B", "DUX4-C"]) + " |")
    L += ["", "### DUX4 paths (best-weighted path; each step reads predicate [source, knowledge level, weight])", ""]
    for d in decks:
        for k in ["DUX4-A", "DUX4-B", "DUX4-C"]:
            p = R["distances"][d][k]["path"]
            if p:
                L.append(f"- {d} {k}: " + path_text(p))

    L += ["", "## Node-set overlap (Jaccard)", "", "| | " + " | ".join(decks) + " |", "|---|" + "---|" * len(decks)]
    for a in decks:
        L.append(f"| {a} | " + " | ".join(f"{R['overlap']['jaccard'][a][b]:.2f}" for b in decks) + " |")
    if "union_coverage" in R["overlap"]:
        u = R["overlap"]["union_coverage"]
        L += ["", f"Nodes of the VoiesA-C graph already present in VoieA or VoieB or VoieC: {u['in_any_single']}/{u['n_union_nodes']} "
                  f"({u['frac']:.2f}); in none of them: {u['new']}."]
    L += ["", "## DUX4 target-activation mechanism (rank inside the kept MolecularActivity category)", "",
          "| deck | found / 6 | best rank | worst rank | category size |", "|---|---|---|---|---|"]
    for d in decks:
        m = R["mechanism"][d]
        if m:
            L.append(f"| {d} | {len(m)} | {m[0]['rank']} | {m[-1]['rank']} | {m[0]['category_size']} |")
        else:
            L.append(f"| {d} | 0 | | | |")
    (out / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--graphs-dir", default=str(B / "output"), help="Directory holding <deck>_context_graph/ (or <deck>/).")
    ap.add_argument("--results-dir", default=None, help="Directory holding _final_<deck>/<deck>_BiomedCAT.json (default: --graphs-dir).")
    ap.add_argument("--decks", nargs="*", default=DEFAULT_DECKS)
    ap.add_argument("--out", default=str(B / "data/eval/voie_multiscale"))
    args = ap.parse_args()

    root = Path(args.graphs_dir)
    rroot = Path(args.results_dir) if args.results_dir else root
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    R = {"graphs_dir": str(root), "decks": [], "profile": None, "scale": SCALE, "signatures_def": SIGNATURES,
         "complexity": {}, "structure": {}, "signatures": {}, "distances": {}, "overlap": {}, "mechanism": {}}
    node_sets: dict[str, set[str]] = {}
    for deck in args.decks:
        try:
            d = graph_dir(root, deck)
        except FileNotFoundError as e:
            print(f"{deck}: {e}")
            continue
        nodes, edges, g, index, summary = load_graph(d)
        R["decks"].append(deck)
        R["profile"] = R["profile"] or summary.get("profile")
        if summary.get("profile") != R["profile"]:
            print(f"warning: {deck} was built with profile {summary.get('profile')!r}, others with {R['profile']!r}")
        R["complexity"][deck] = complexity(result_json(rroot, deck))
        R["structure"][deck] = structure(nodes, g, index, summary)
        ranks = category_ranks(nodes)
        sig_rows = {s: signature_hits(nodes, pats, ranks) for s, pats in SIGNATURES.items()}
        R["signatures"][deck] = {s: {"rows": rows, "summary": signature_summary(rows)} for s, rows in sig_rows.items()}
        hit_ids = {s: [n["id"] for n in nodes if any(match(p, n["name"]) for p in pats)] for s, pats in SIGNATURES.items()}
        D = {}
        for a, b in combinations(SIGNATURES, 2):
            D[f"{a}-{b}"] = shortest_between(g, index, hit_ids[a], hit_ids[b])
        for s in SIGNATURES:
            D[f"DUX4-{s}"] = shortest_between(g, index, [DUX4_ID], hit_ids[s])
        R["distances"][deck] = D
        R["mechanism"][deck] = mechanism_ranks(nodes, ranks)
        node_sets[deck] = {n["id"] for n in nodes}
        s = R["structure"][deck]
        print(f"{deck:18s} seeds {s['n_seeds']:3d}  nodes {s['n_nodes']:5d}  diameter {s['diameter']}  components {s['n_components']}  "
              f"pairs {s['seed_pairs_connected']:>8s}  mean seed dist {s['seed_distance_mean']}  "
              + "  ".join(f"sig{k} {v['summary']['n_present']}/{v['summary']['n_patterns']}" for k, v in R['signatures'][deck].items()))

    decks = R["decks"]
    R["overlap"]["jaccard"] = {a: {b: (len(node_sets[a] & node_sets[b]) / len(node_sets[a] | node_sets[b]) if node_sets[a] | node_sets[b] else 0.0)
                                   for b in decks} for a in decks}
    if "VoiesA-C" in node_sets and all(k in node_sets for k in ("VoieA", "VoieB", "VoieC")):
        single = node_sets["VoieA"] | node_sets["VoieB"] | node_sets["VoieC"]
        u = node_sets["VoiesA-C"]
        R["overlap"]["union_coverage"] = {"n_union_nodes": len(u), "in_any_single": len(u & single),
                                          "frac": round(len(u & single) / len(u), 3), "new": len(u - single)}

    (out / "summary.json").write_text(json.dumps(R, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "signatures.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["deck", "signature", "pattern", "present", "as_seed", "n_hits", "best_name", "best_category", "best_hop",
                    "best_coverage", "best_rank", "best_category_size", "best_rank_pct"])
        for deck in decks:
            for sig, block in R["signatures"][deck].items():
                for r in block["rows"]:
                    w.writerow([deck, sig, r["pattern"], int(r["present"]), int(r["as_seed"]), r["n_hits"], r["best_name"],
                                r["best_category"], r["best_hop"], r["best_coverage"], r["best_rank"], r["best_category_size"], r["best_rank_pct"]])
    write_report(out, R)
    print(f"\n-> {out / 'summary.json'}, signatures.csv, report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
