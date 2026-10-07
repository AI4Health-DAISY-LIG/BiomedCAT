#!/usr/bin/env python
"""Figure 4 of the manuscript: from DUX4 to each pathway (readable redesign, 7 Oct 2026).

Panels a-c: a window on each single-pathway context graph (slides A, B, C of the FSHD1 inflammation
deck) drawn as two rings. Inner ring: the seeds of the slide (white, black outline), ordered so that
seeds sharing neighbours sit next to each other. Outer ring: the best-scored kept neighbours of each
seed (`per_seed`), the best-scored nodes of the whole graph (`n_top`) and the nodes of the
best-weighted path from DUX4 to the slide's own pathway signature, each placed at the angle of the
seed(s) it is attached to and coloured by Biolink branch. Labels are radial, outside the outer ring,
so they never overlap. Edges of the kept graph between shown nodes are thin grey lines; the DUX4 path
is drawn in the signature colour.
Panel d: the best-weighted path from DUX4 to each signature, in the slide of that pathway and in
the slide that integrates the three, as chains of RTX-KG2c assertions (predicate, primary source,
profile weight), reusing `chain` from make_voie_panel.py.

Inputs: data/eval/voie_multiscale/summary.json (scripts/voie_multiscale_case.py) and the context
graphs it was computed from (Output/<deck>_context_graph, runs of 4 Oct 2026, the same as Figure 3).
Outputs: fig4_dux4_use_case.{pdf,png} (180 mm wide composite), fig4a-c windows, fig4d paths and the
legend as stand-alone files, plus fig4_dux4_use_case_data.csv (the nodes shown in each window).

Usage: uv run python scripts/make_fig4_dux4.py [--out ../manuscripts/biomedcat_manuscript/figures]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

B = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(B / "scripts"))
from make_voie_panel import BRANCH, DECK_SHORT, SIG_COLOR, branch_color, chain  # noqa: E402
from voie_multiscale_case import graph_dir  # noqa: E402

logger = logging.getLogger("make_fig4_dux4")
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
                     "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9, "figure.dpi": 100})
SIG_OF_DECK = {"VoieA": "A", "VoieB": "B", "VoieC": "C"}
SCALE_SHORT = {"VoieA": "molecular scale", "VoieB": "organelle scale", "VoieC": "tissue scale"}


def load_graph(d: Path):
    with open(d / "nodes.csv", encoding="utf-8", newline="") as f:
        nodes = list(csv.DictReader(f))
    with open(d / "edges.csv", encoding="utf-8", newline="") as f:
        edges = list(csv.DictReader(f))
    info = {n["id"]: n for n in nodes}
    adj: dict[str, set[str]] = {}
    for e in edges:
        a, b = e["subject"], e["object"]
        if a != b and a in info and b in info:
            adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)
    return nodes, info, adj


def short(name: str, n: int = 26) -> str:
    name = name.replace(" (Homo sapiens)", "")
    return name if len(name) <= n else name[: n - 1].rstrip() + "…"


def order_seeds(seeds: list[str], adj: dict, outer: set[str]) -> list[str]:
    """Greedy ordering of the seeds on the ring: each next seed is the one sharing most shown
    neighbours with the previous one, so that shared neighbours sit between their seeds."""
    rem = set(seeds); order = [max(seeds, key=lambda s: len(adj.get(s, set()) & outer))]; rem.discard(order[0])
    while rem:
        last = adj.get(order[-1], set()) & outer
        nxt = max(rem, key=lambda s: (len((adj.get(s, set()) & outer) & last), len(adj.get(s, set()) & outer)))
        order.append(nxt); rem.discard(nxt)
    return order


def window(ax, d: Path, path_nodes: list[str], sig_color: str, n_top: int = 10, per_seed: int = 2):
    """One ring: seeds (white, black outline, bold label) interleaved with the nodes attached to them,
    every node at the same angular spacing, labels radial outside the ring. Edges are chords."""
    nodes, info, adj = load_graph(d)
    seeds = [n["id"] for n in nodes if n["is_seed"] == "1"]
    score = {n["id"]: float(n["score"] or 0) for n in nodes}
    sset = set(seeds)
    outer: set[str] = set()
    for sd in seeds:
        outer.update(sorted((v for v in adj.get(sd, ()) if v not in sset), key=lambda v: -score[v])[:per_seed])
    outer.update(n["id"] for n in sorted((n for n in nodes if n["is_seed"] != "1"), key=lambda n: -score[n["id"]])[:n_top])
    outer.update(x for x in path_nodes if x in info and x not in sset)
    order_s = order_seeds(seeds, adj, outer)
    # ring order: each seed followed by its not-yet-placed attached nodes (best score first)
    ring: list[str] = []
    placed: set[str] = set()
    for sd in order_s:
        ring.append(sd); placed.add(sd)
        for v in sorted((v for v in adj.get(sd, ()) if v in outer and v not in placed), key=lambda v: -score[v]):
            ring.append(v); placed.add(v)
    for v in sorted((v for v in outer if v not in placed), key=lambda v: -score[v]):  # attached to no seed: next to a shown neighbour
        nb = [u for u in adj.get(v, set()) if u in placed]
        if nb:
            i = ring.index(max(nb, key=lambda u: score.get(u, 0))); ring.insert(i + 1, v)
        else:
            ring.append(v)
        placed.add(v)
    n = len(ring)
    ang = {v: 2 * math.pi * (k / n) + math.pi / 2 for k, v in enumerate(ring)}  # start at the top, clockwise order kept simple
    Rr = 1.0
    pos = {v: (Rr * math.cos(a), Rr * math.sin(a)) for v, a in ang.items()}
    path_edges = {frozenset((path_nodes[i], path_nodes[i + 1])) for i in range(len(path_nodes) - 1)}
    shown = set(ring)
    for a in shown:
        for b in adj.get(a, ()):
            if b in shown and a < b:
                (x1, y1), (x2, y2) = pos[a], pos[b]
                if frozenset((a, b)) in path_edges:
                    ax.plot([x1, x2], [y1, y2], color=sig_color, lw=2.0, alpha=0.95, zorder=2)
                else:
                    ax.plot([x1, x2], [y1, y2], color="#c4c4c4", lw=0.4, alpha=0.55, zorder=1)
    for v in ring:
        x, y = pos[v]; a = ang[v]; deg = math.degrees(a) % 360
        is_seed = v in sset; on_path = v in path_nodes
        if is_seed:
            ax.scatter([x], [y], s=70, color="white", edgecolors=sig_color if on_path else "black", linewidths=1.5 if on_path else 1.0, zorder=5)
        else:
            ax.scatter([x], [y], s=34, color=branch_color(info[v]["category"]), edgecolors=sig_color if on_path else "white",
                       linewidths=1.3 if on_path else 0.4, zorder=4)
        flip = 90 < deg < 270
        ax.text((Rr + 0.06) * math.cos(a), (Rr + 0.06) * math.sin(a), short(info[v]["name"], 24), rotation=deg + 180 if flip else deg,
                rotation_mode="anchor", ha="right" if flip else "left", va="center", fontsize=5.5,
                color="#111" if (is_seed or on_path) else "#444", weight="bold" if (is_seed or on_path) else "normal", zorder=6)
    ax.set_xlim(-2.05, 2.05); ax.set_ylim(-2.05, 2.2); ax.set_aspect("equal"); ax.axis("off")
    return [{"id": v, "name": info[v]["name"], "category": info[v]["category"], "seed": v in sset, "on_path": v in path_nodes,
             "score": score.get(v, 0.0)} for v in ring]


def legend_handles():
    h = [Patch(color=col, label=lab) for lab, _, col in BRANCH]
    h.append(Line2D([], [], marker="o", ls="", mfc="white", mec="black", label="seed (slide entity)", ms=8))
    h.append(Line2D([], [], color="#333", lw=2.2, label="best-weighted path from DUX4"))
    h.append(Line2D([], [], color="#b0b0b0", lw=0.6, label="other edge of the kept graph"))
    return h


def branch_name(category: str) -> str:
    c = category.replace("biolink:", "")
    for lab, cats, _ in BRANCH:
        if c in cats:
            return lab
    return "other"


def export_cytoscape(cy: Path, deck: str, d: Path, shown: list[dict], path: list[dict]) -> None:
    """Node table, edge table and GraphML of one window, ready for Cytoscape (File > Import > Network from File
    for the edge table or the GraphML; File > Import > Table from File for the node table, key column `id`)."""
    import networkx as nx
    ids = {r["id"] for r in shown}
    with open(d / "edges.csv", encoding="utf-8", newline="") as f:
        edges = [e for e in csv.DictReader(f) if e["subject"] in ids and e["object"] in ids and e["subject"] != e["object"]]
    path_ids = [n["id"] for n in path]
    path_pairs = {frozenset((path_ids[i], path_ids[i + 1])) for i in range(len(path_ids) - 1)}
    tag = DECK_SHORT[deck]
    nrows = [{"id": r["id"], "name": r["name"].replace(" (Homo sapiens)", ""), "category": r["category"].replace("biolink:", ""),
              "branch": "seed" if r["seed"] else branch_name(r["category"]), "color": "#ffffff" if r["seed"] else branch_color(r["category"]),
              "is_seed": int(r["seed"]), "on_dux4_path": int(r["on_path"]), "score": r["score"]} for r in shown]
    erows = [{"source": e["subject"], "target": e["object"], "interaction": e["predicate"].replace("biolink:", ""), "weight": e["weight"],
              "primary_knowledge_source": e["primary_knowledge_source"].replace("infores:", ""), "knowledge_level": e["knowledge_level"],
              "on_dux4_path": int(frozenset((e["subject"], e["object"])) in path_pairs)} for e in edges]
    with open(cy / f"fig4_{tag}_window_nodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(nrows[0])); w.writeheader(); w.writerows(nrows)
    with open(cy / f"fig4_{tag}_window_edges.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(erows[0])); w.writeheader(); w.writerows(erows)
    G = nx.MultiDiGraph()
    for r in nrows:
        G.add_node(r["id"], **{k: v for k, v in r.items() if k != "id"})
    for r in erows:
        G.add_edge(r["source"], r["target"], **{k: v for k, v in r.items() if k not in ("source", "target")})
    nx.write_graphml(G, cy / f"fig4_{tag}_window.graphml")


def export_paths(cy: Path, R: dict) -> None:
    rows = []
    for deck in ("VoieA", "VoieB", "VoieC", "VoiesA-C", "FSHD1multiscales"):
        for sg in "ABC":
            pth = R["distances"][deck][f"DUX4-{sg}"]["path"]
            for i in range(1, len(pth)):
                e = pth[i]["edge"]
                a, b = (pth[i - 1], pth[i]) if e["forward"] else (pth[i], pth[i - 1])
                rows.append({"slide": DECK_SHORT[deck], "signature": sg, "step": i, "source": a["id"], "source_name": a["name"],
                             "target": b["id"], "target_name": b["name"], "interaction": e["predicate"], "weight": e["weight"],
                             "primary_knowledge_source": e["source"].replace("infores:", ""), "knowledge_level": e["level"]})
    with open(cy / "fig4d_dux4_paths_edges.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", default=str(B / "data/eval/voie_multiscale/summary.json"))
    ap.add_argument("--graphs-dir", default=str(B / "Output"))
    ap.add_argument("--out", default=str(B.parent / "manuscripts/biomedcat_manuscript/figures"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    R = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    root, out = Path(args.graphs_dir), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    cy = out / "fig4_cytoscape"; cy.mkdir(exist_ok=True)
    export_paths(cy, R)

    # composite, 180 mm wide: windows a, b (row 1), c and the legend (row 2), the paths (row 3)
    fig = plt.figure(figsize=(7.09, 9.3))
    gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.9], hspace=0.0, wspace=0.0)
    cells = [(0, 0), (0, 1), (1, 0)]
    for k, deck in enumerate(("VoieA", "VoieB", "VoieC")):
        sig = SIG_OF_DECK[deck]
        path = [nd["id"] for nd in R["distances"][deck][f"DUX4-{sig}"]["path"]]
        ax = fig.add_subplot(gs[cells[k]])
        shown = window(ax, graph_dir(root, deck), path, SIG_COLOR[sig])
        rows += [{"panel": "abc"[k], "deck": deck, **r} for r in shown]
        export_cytoscape(cy, deck, graph_dir(root, deck), shown, R["distances"][deck][f"DUX4-{sig}"]["path"])
        st = R["structure"][deck]
        ax.text(-2.0, 2.18, f"({'abc'[k]}) slide {DECK_SHORT[deck]}, {SCALE_SHORT[deck]}: {st['n_seeds']} seeds; window of {len(shown)} of the {st['n_nodes']:,} kept nodes",
                fontsize=6.8, va="top", ha="left")
        f2, a2 = plt.subplots(figsize=(6, 6)); window(a2, graph_dir(root, deck), path, SIG_COLOR[sig])
        a2.set_title(f"slide {DECK_SHORT[deck]}, {SCALE_SHORT[deck]}", fontsize=9, loc="left"); f2.tight_layout()
        f2.savefig(out / f"fig4{'abc'[k]}_window_{DECK_SHORT[deck]}.pdf"); f2.savefig(out / f"fig4{'abc'[k]}_window_{DECK_SHORT[deck]}.png", dpi=300); plt.close(f2)
    axl = fig.add_subplot(gs[1, 1]); axl.axis("off")
    axl.legend(handles=legend_handles(), loc="center", ncol=1, fontsize=7.5, frameon=False, handlelength=1.8, labelspacing=1.1,
               title="Biolink branch of the kept nodes", title_fontsize=8)

    axd = fig.add_subplot(gs[2, :]); axd.axis("off")
    D = R["distances"]
    prows = [("slide A  \u2192  signature A", D["VoieA"]["DUX4-A"]["path"], SIG_COLOR["A"]),
             ("slide B  \u2192  signature B   (the same edge is the path in the slide-C graph)", D["VoieB"]["DUX4-B"]["path"], SIG_COLOR["B"]),
             ("slide C  \u2192  signature C   (the same path in the slide A+B+C graph)", D["VoieC"]["DUX4-C"]["path"], SIG_COLOR["C"]),
             ("slide A+B+C  \u2192  signature B", D["VoiesA-C"]["DUX4-B"]["path"], SIG_COLOR["B"])]
    axd.text(0.0, 1.0, "(d) best-weighted path from DUX4 to each pathway signature (predicate above each edge; primary source and profile weight below)",
             fontsize=6.8, va="top", transform=axd.transAxes)
    sub = axd.inset_axes([0.0, 0.0, 1.0, 0.93]); sub.axis("off")
    band = 1.0 / len(prows)
    for r, (label, path, color) in enumerate(prows):
        top = 1.0 - r * band
        sub.text(0.0, top, label, ha="left", va="top", fontsize=6.5, weight="bold", color=color, transform=sub.transAxes)
        chain(sub, top - 0.6 * band, path, color)
    sub.text(1.0, 0.0, "slide A+B+C \u2192 signature A: as slide A, with RELA in place of DHX15", ha="right", va="bottom", fontsize=6, color="#555", transform=sub.transAxes)
    fig.subplots_adjust(left=0.0, right=1.0, top=0.995, bottom=0.005)
    fig.savefig(out / "fig4_dux4_use_case.pdf"); fig.savefig(out / "fig4_dux4_use_case.png", dpi=300); plt.close(fig)

    # panel d alone (same drawing as in the composite)
    fdd = plt.figure(figsize=(7.09, 3.4)); ad = fdd.add_axes([0.01, 0.01, 0.98, 0.98]); ad.axis("off")
    D = R["distances"]
    prows_d = [("slide A  \u2192  signature A", D["VoieA"]["DUX4-A"]["path"], SIG_COLOR["A"]),
               ("slide B  \u2192  signature B   (the same edge is the path in the slide-C graph)", D["VoieB"]["DUX4-B"]["path"], SIG_COLOR["B"]),
               ("slide C  \u2192  signature C   (the same path in the slide A+B+C graph)", D["VoieC"]["DUX4-C"]["path"], SIG_COLOR["C"]),
               ("slide A+B+C  \u2192  signature B", D["VoiesA-C"]["DUX4-B"]["path"], SIG_COLOR["B"])]
    bd = 1.0 / len(prows_d)
    for r, (label, pth, color) in enumerate(prows_d):
        top = 1.0 - r * bd
        ad.text(0.0, top, label, ha="left", va="top", fontsize=6.5, weight="bold", color=color, transform=ad.transAxes)
        chain(ad, top - 0.6 * bd, pth, color)
    fdd.savefig(out / "fig4d_dux4_paths.pdf"); fdd.savefig(out / "fig4d_dux4_paths.png", dpi=300); plt.close(fdd)

    # legend alone and the paths alone
    fl, al = plt.subplots(figsize=(6, 2)); al.axis("off"); al.legend(handles=legend_handles(), loc="center", ncol=2, fontsize=9, frameon=False)
    fl.savefig(out / "fig4_legend.pdf"); fl.savefig(out / "fig4_legend.png", dpi=300); plt.close(fl)
    with open(out / "fig4_dux4_use_case_data.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["panel", "deck", "id", "name", "category", "seed", "on_path", "score"]); w.writeheader(); w.writerows(rows)
    logger.info("-> %s", out / "fig4_dux4_use_case.pdf")
    return 0


if __name__ == "__main__":
    sys.exit(main())
