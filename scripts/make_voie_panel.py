#!/usr/bin/env python
"""Figures of the multi-scale case (Voie A, B, C, their union) and of the DUX4 use case.

Reads data/eval/voie_multiscale/summary.json written by scripts/voie_multiscale_case.py and the
context graphs it was computed from, and writes every panel as a stand-alone square figure (PDF
+ PNG, 300 dpi, Arial, text kept as text in the PDF) to be assembled by hand, plus a captions file
next to them. Numbers follow the manuscript (Figures 3 and 4).

  Figure 3, multi-scale case
  fig3_multiscale        the five panels assembled, 180 mm wide at 300 dpi, no explanatory text
                         (it belongs to the caption); text overlaps are checked before saving
  fig3a_structure        kept graph structure per deck: diameter, components, connected seed pairs,
                         mean seed-to-seed distance
  fig3b_categories       kept nodes per Biolink category where the graphs differ (the categories at
                         the per-category cap in every graph are left out)
  fig3c_signatures       pathway-signature recovery, signature x deck (the specificity matrix)
  fig3d_overlap          Jaccard overlap of the kept node sets
  fig3e_dux4_ranks       rank of the six Reactome nodes of DUX4 target activation inside the
                         MolecularActivity category

  Figure 4, DUX4 use case
  fig4a_window_A         a window on the slide-A graph: seeds, the 3 best-scored neighbours of each
  fig4b_window_B         seed and the 15 best-scored nodes overall, coloured by Biolink branch
  fig4c_window_C
  fig4_legend            the Biolink-branch legend shared by panels a-c
  fig4d_dux4_paths       best-weighted path from DUX4 to each pathway signature, in the slide of that
                         pathway and in the slide that integrates the three, read as chains of
                         RTX-KG2c assertions

Usage:
  uv run python scripts/make_voie_panel.py [--summary data/eval/voie_multiscale/summary.json]
                                           [--graphs-dir output] [--out ../manuscripts/biomedcat_manuscript/figures]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.text
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

B = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(B / "scripts"))
from voie_multiscale_case import SCALE, graph_dir  # noqa: E402

logger = logging.getLogger("make_voie_panel")
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "pdf.fonttype": 42, "ps.fonttype": 42,          # TrueType in the PDF: text stays editable when assembling
    "font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100,
})

SQ = 6.0   # side of every panel, in inches
FS = {"tick": 10, "value": 10, "label": 10, "note": 9, "node": 6.5, "seed": 7.5, "path": 7}

DECK_LABEL = {"VoieA": "A\nmolecular", "VoieB": "B\norganelle", "VoieC": "C\ntissue", "VoiesA-C": "A+B+C\nintegrated",
              "FSHD1multiscales": "summary\nslide", "FSHD": "FSHD\ndeck"}
DECK_SHORT = {"VoieA": "A", "VoieB": "B", "VoieC": "C", "VoiesA-C": "A+B+C", "FSHD1multiscales": "summary", "FSHD": "FSHD"}
SIG_LABEL = {"A": "A  innate antiviral", "B": "B  mitochondria / redox", "C": "C  remodelling / fibrosis"}
SIG_COLOR = {"A": "#4c72b0", "B": "#dd8452", "C": "#55a868"}

# Biolink branches used to colour the graph windows.
BRANCH = [
    ("gene & product", {"Gene", "Protein", "Transcript", "GenomicEntity", "GeneFamily", "Polypeptide", "NucleicAcidEntity", "NoncodingRNAProduct"}, "#4c72b0"),
    ("chemical / drug", {"SmallMolecule", "ChemicalEntity", "Drug", "MolecularMixture", "ChemicalMixture", "MolecularEntity"}, "#dd8452"),
    ("process / pathway", {"Pathway", "BiologicalProcess", "MolecularActivity", "PhysiologicalProcess", "Activity", "Phenomenon", "Behavior"}, "#55a868"),
    ("disease / phenotype", {"Disease", "PhenotypicFeature", "ClinicalAttribute", "BehavioralFeature"}, "#c44e52"),
    ("cell / anatomy", {"Cell", "CellularComponent", "AnatomicalEntity", "GrossAnatomicalStructure"}, "#8172b3"),
]


def branch_color(category: str) -> str:
    c = category.replace("biolink:", "")
    for _, cats, col in BRANCH:
        if c in cats:
            return col
    return "#9e9e9e"


def square(nrows: int = 1, ncols: int = 1):
    return plt.subplots(nrows, ncols, figsize=(SQ, SQ))


def save(fig, out: Path, name: str) -> None:
    """Save at the exact square size (no bbox trimming, which would break the aspect)."""
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)
    logger.info("-> %s", out / name)


def _deck_axis(ax, decks, labels=None, rotation: float = 0) -> None:
    ax.set_xticks(range(len(decks)))
    ax.set_xticklabels([(labels or DECK_LABEL)[d] for d in decks], fontsize=FS["tick"], rotation=rotation,
                       ha="right" if rotation else "center", rotation_mode="anchor" if rotation else "default")


def _heat_frame(ax) -> None:
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)


# ----------------------------------------------------------------------------------------------
# Figure 3: multi-scale case
# ----------------------------------------------------------------------------------------------
def fig3a(R: dict, out: Path) -> None:
    """Structure of the kept graph: four bar charts sharing the deck axis."""
    decks, S = R["decks"], R["structure"]
    metrics = [("diameter", "Diameter (hops)", "{:g}"), ("n_components", "Connected components", "{:g}"),
               ("seed_pairs_connected_frac", "Seed pairs connected", "{:.2f}"), ("seed_distance_mean", "Mean seed–seed\ndistance (hops)", "{:.2f}")]
    fig, axes = square(2, 2)
    x = np.arange(len(decks))
    for ax, (key, title, fmt) in zip(axes.flat, metrics):
        vals = [S[d].get(key) or 0 for d in decks]
        ax.bar(x, vals, color="#4c72b0", width=0.62)
        top = max(vals) if vals else 1
        for i, v in enumerate(vals):
            ax.text(i, v + 0.02 * top, fmt.format(v), ha="center", va="bottom", fontsize=FS["note"])
        _deck_axis(ax, decks, DECK_SHORT, rotation=40)
        ax.set_title(title, fontsize=FS["label"])
        ax.tick_params(axis="y", labelsize=FS["tick"])
        ax.set_ylim(0, 1.12 if key == "seed_pairs_connected_frac" else top * 1.18)
    fig.tight_layout()
    save(fig, out, "fig3a_structure")


def fig3b(R: dict, out: Path) -> None:
    """Kept nodes per Biolink category, for the categories in which the graphs differ.

    The per-category cap puts the most populated categories at exactly the cap in every graph,
    so a composition chart shows identical shares; the information is in the other categories."""
    decks, S = R["decks"], R["structure"]
    cap = max((S[d].get("categories") or {}).get("biolink:Gene", 300) for d in decks)
    cats_all = sorted({c for d in decks for c in S[d]["categories"]}, key=lambda c: -sum(S[d]["categories"].get(c, 0) for d in decks))
    always_full = [c for c in cats_all if all(S[d]["categories"].get(c, 0) >= cap for d in decks)]
    varying = [c for c in cats_all if c not in always_full and sum(S[d]["categories"].get(c, 0) for d in decks) >= 25]
    M = np.array([[S[d]["categories"].get(c, 0) for d in decks] for c in varying], float)
    fig, ax = square()
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=cap, aspect="auto")
    for i in range(len(varying)):
        for j in range(len(decks)):
            v = int(M[i, j])
            ax.text(j, i, f"{v}" + ("*" if v >= cap else ""), ha="center", va="center", fontsize=FS["note"],
                    color="white" if v > 0.6 * cap else "black")
    _deck_axis(ax, decks, DECK_SHORT)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(varying))); ax.set_yticklabels([c.replace("biolink:", "") for c in varying], fontsize=FS["note"])
    _heat_frame(ax)
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label("kept nodes", fontsize=FS["label"]); cb.ax.tick_params(labelsize=FS["note"])
    ax.set_xlabel(f"{len(always_full)} categories at the cap of {cap} in all\ngraphs are not shown; * = at the cap",
                  fontsize=FS["note"], color="#616161")
    fig.tight_layout()
    save(fig, out, "fig3b_categories")


def fig3c(R: dict, out: Path) -> None:
    """Signature recovery matrix: signature x deck."""
    decks = R["decks"]
    sigs = list(R["signatures_def"])
    M = np.array([[R["signatures"][d][s]["summary"]["recovered_frac"] for d in decks] for s in sigs])
    fig, ax = square()
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i, s in enumerate(sigs):
        for j, d in enumerate(decks):
            sm = R["signatures"][d][s]["summary"]
            txt = f"{sm['n_present']}/{sm['n_patterns']}"
            if sm["median_rank_pct"] is not None:
                txt += f"\np = {sm['median_rank_pct']:.2f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=FS["value"], color="white" if M[i, j] > 0.55 else "black")
    _deck_axis(ax, decks)
    ax.set_yticks(range(len(sigs)))
    ax.set_yticklabels([SIG_LABEL[s].replace("  ", "\n", 1) for s in sigs], fontsize=FS["tick"])
    for t, s in zip(ax.get_yticklabels(), sigs):
        t.set_color(SIG_COLOR[s])
    _heat_frame(ax)
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.16)
    cb.set_label("share of the 10 signature nodes present", fontsize=FS["label"]); cb.ax.tick_params(labelsize=FS["note"])
    ax.set_title("p = median rank percentile of the hits\ninside their Biolink category (0 = first)", fontsize=FS["note"], color="#616161")
    fig.tight_layout()
    save(fig, out, "fig3c_signatures")


def fig3d(R: dict, out: Path) -> None:
    """Jaccard overlap of the kept node sets."""
    decks = R["decks"]
    J = np.array([[R["overlap"]["jaccard"][a][b] for b in decks] for a in decks])
    fig, ax = square()
    masked = np.ma.masked_where(np.eye(len(decks), dtype=bool), J)
    im = ax.imshow(masked, cmap="Greens", vmin=0, vmax=0.5, aspect="equal")
    for i in range(len(decks)):
        for j in range(len(decks)):
            if i != j:
                ax.text(j, i, f"{J[i, j]:.2f}", ha="center", va="center", fontsize=FS["value"], color="white" if J[i, j] > 0.33 else "black")
    _deck_axis(ax, decks, DECK_SHORT)
    ax.set_yticks(range(len(decks))); ax.set_yticklabels([DECK_SHORT[d] for d in decks], fontsize=FS["tick"])
    _heat_frame(ax)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("Jaccard index of kept node sets", fontsize=FS["label"]); cb.ax.tick_params(labelsize=FS["note"])
    if "union_coverage" in R["overlap"]:
        u = R["overlap"]["union_coverage"]
        ax.set_xlabel(f"A+B+C graph: {u['frac']:.0%} of its nodes are in A, B or C", fontsize=FS["note"], color="#616161")
    fig.tight_layout()
    save(fig, out, "fig3d_overlap")


def fig3e(R: dict, out: Path) -> None:
    """Rank of the six Reactome reactions of DUX4 target activation in each graph."""
    decks = R["decks"]
    fig, ax = square()
    size = 300
    for j, d in enumerate(decks):
        ranks = [m["rank"] for m in R["mechanism"][d]]
        if R["mechanism"][d]:
            size = R["mechanism"][d][0]["category_size"]
        ax.scatter([j] * len(ranks), ranks, s=40, color="#c44e52", zorder=3)
        if ranks:
            ax.plot([j, j], [min(ranks), max(ranks)], color="#c44e52", lw=1.4, zorder=2)
            ax.text(j + 0.13, max(ranks), f"{len(ranks)}/6", ha="left", va="center", fontsize=FS["value"])
    worst = max((m["rank"] for d in decks for m in R["mechanism"][d]), default=30)
    _deck_axis(ax, decks, DECK_SHORT)
    ax.set_xlim(-0.5, len(decks) - 0.3)
    ax.set_ylim(worst * 1.25, 0)
    ax.set_ylabel(f"rank among the {size} kept\nMolecularActivity nodes (1 = best)", fontsize=FS["label"])
    ax.tick_params(axis="y", labelsize=FS["tick"])
    fig.tight_layout()
    save(fig, out, "fig3e_dux4_ranks")


MM = 1 / 25.4
COMPOSITE_WIDTH_MM = 180      # journal maximum width
CFS = {"tick": 7.5, "value": 7, "label": 8, "panel": 10}   # composite font sizes, in pt at print size


def text_overlaps(fig) -> list[str]:
    """Pairs of visible, non-empty text artists whose rendered boxes intersect, and texts that
    leave the canvas. Used to check the composite before saving."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    W, H = fig.canvas.get_width_height()
    boxes = []
    for t in fig.findobj(matplotlib.text.Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        bb = t.get_window_extent(renderer)
        if bb.width == 0 or bb.height == 0:
            continue
        boxes.append((t.get_text().replace("\n", " ") + f" @({bb.x0:.0f},{bb.y0:.0f})", bb))
    problems = []
    for name, bb in boxes:
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > W + 1 or bb.y1 > H + 1:
            problems.append(f"outside the canvas: {name!r}")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i][1], boxes[j][1]
            # shrink by 0.5 px so that boxes that merely touch are not reported
            if a.x0 + 0.5 < b.x1 and b.x0 + 0.5 < a.x1 and a.y0 + 0.5 < b.y1 and b.y0 + 0.5 < a.y1:
                problems.append(f"overlap: {boxes[i][0]!r} / {boxes[j][0]!r}")
    return problems


def fig3_composite(R: dict, out: Path) -> None:
    """Figure 3 as one figure, 180 mm wide at 300 dpi; every explanation is left to the caption."""
    decks, S = R["decks"], R["structure"]
    # Short deck labels, defined in the caption: ABC = integrated slide, S = summary slide.
    short = [{"VoiesA-C": "ABC", "FSHD1multiscales": "S"}.get(d, DECK_SHORT[d]) for d in decks]
    fig = plt.figure(figsize=(COMPOSITE_WIDTH_MM * MM, 165 * MM))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.15, 1], hspace=0.32, left=0.07, right=0.985, top=0.95, bottom=0.05)
    top = outer[0].subgridspec(1, 2, width_ratios=[1, 0.82], wspace=0.78)
    bottom = outer[1].subgridspec(1, 3, width_ratios=[1.15, 1, 0.9], wspace=0.42)

    def letter(ax, s, dx=-0.02):
        # Panel letter at the top-left corner of the panel, above its title or top tick labels.
        fig.canvas.draw()
        bb = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
        fig.text(ax.get_position().x0 + dx, bb.y1 + 0.004, s, fontsize=CFS["panel"], weight="bold", ha="right", va="bottom")

    def deck_ticks(ax, rotation=0):
        ax.set_xticks(range(len(decks)))
        ax.set_xticklabels(short, fontsize=CFS["tick"], rotation=rotation,
                           ha="right" if rotation else "center", rotation_mode="anchor" if rotation else "default")

    # (a) structure, 2 x 2 bars
    ga = top[0].subgridspec(2, 2, hspace=0.75, wspace=0.42)
    metrics = [("diameter", "Diameter", "{:g}"), ("n_components", "Components", "{:g}"),
               ("seed_pairs_connected_frac", "Seed pairs connected", "{:.2f}"), ("seed_distance_mean", "Mean seed distance", "{:.1f}")]
    a_axes = []
    for k, (key, title, fmt) in enumerate(metrics):
        ax = fig.add_subplot(ga[k // 2, k % 2]); a_axes.append(ax)
        vals = [S[d].get(key) or 0 for d in decks]
        ax.bar(range(len(decks)), vals, color="#4c72b0", width=0.65)
        hi = max(vals) if vals else 1
        for i, v in enumerate(vals):
            ax.text(i, v + 0.03 * hi, fmt.format(v), ha="center", va="bottom", fontsize=CFS["value"] - 1)
        deck_ticks(ax)
        ax.set_title(title, fontsize=CFS["label"], pad=3)
        ax.tick_params(axis="y", labelsize=CFS["tick"], pad=1); ax.tick_params(axis="x", pad=1)
        ax.set_ylim(0, 1.2 if key == "seed_pairs_connected_frac" else hi * 1.25)
        if key == "seed_pairs_connected_frac":
            ax.set_yticks([0, 0.5, 1])

    # (b) kept nodes per category where the graphs differ (cap marked *)
    axb = fig.add_subplot(top[1])
    cap = max((S[d].get("categories") or {}).get("biolink:Gene", 300) for d in decks)
    cats_all = sorted({c for d in decks for c in S[d]["categories"]}, key=lambda c: -sum(S[d]["categories"].get(c, 0) for d in decks))
    varying = [c for c in cats_all if not all(S[d]["categories"].get(c, 0) >= cap for d in decks)
               and sum(S[d]["categories"].get(c, 0) for d in decks) >= 25]
    M = np.array([[S[d]["categories"].get(c, 0) for d in decks] for c in varying], float)
    axb.imshow(M, cmap="Blues", vmin=0, vmax=cap, aspect="auto")
    for i in range(len(varying)):
        for j in range(len(decks)):
            v = int(M[i, j])
            axb.text(j, i, f"{v}" + ("*" if v >= cap else ""), ha="center", va="center", fontsize=CFS["value"],
                     color="white" if v > 0.6 * cap else "black")
    axb.set_xticks(range(len(decks))); axb.set_xticklabels(short, fontsize=CFS["tick"]); axb.xaxis.tick_top()
    axb.set_yticks(range(len(varying))); axb.set_yticklabels([c.replace("biolink:", "") for c in varying], fontsize=CFS["tick"])
    _heat_frame(axb)

    # (c) signature recovery: n/10 and median rank percentile
    axc = fig.add_subplot(bottom[0])
    sigs = list(R["signatures_def"])
    C = np.array([[R["signatures"][d][s]["summary"]["recovered_frac"] for d in decks] for s in sigs])
    axc.imshow(C, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    for i, s in enumerate(sigs):
        for j, d in enumerate(decks):
            sm = R["signatures"][d][s]["summary"]
            txt = f"{sm['n_present']}/{sm['n_patterns']}" + (f"\n{sm['median_rank_pct']:.2f}" if sm["median_rank_pct"] is not None else "")
            axc.text(j, i, txt, ha="center", va="center", fontsize=CFS["value"], color="white" if C[i, j] > 0.55 else "black", linespacing=1.1)
    deck_ticks(axc)
    axc.set_yticks(range(len(sigs))); axc.set_yticklabels(sigs, fontsize=CFS["tick"])
    axc.set_ylabel("Signature", fontsize=CFS["label"])
    _heat_frame(axc)

    # (d) Jaccard overlap
    axd = fig.add_subplot(bottom[1])
    J = np.array([[R["overlap"]["jaccard"][a][b] for b in decks] for a in decks])
    axd.imshow(np.ma.masked_where(np.eye(len(decks), dtype=bool), J), cmap="Greens", vmin=0, vmax=0.5, aspect="auto")
    for i in range(len(decks)):
        for j in range(len(decks)):
            if i != j:
                axd.text(j, i, f"{J[i, j]:.2f}", ha="center", va="center", fontsize=CFS["value"], color="white" if J[i, j] > 0.33 else "black")
    deck_ticks(axd)
    axd.set_yticks(range(len(decks))); axd.set_yticklabels(short, fontsize=CFS["tick"])
    _heat_frame(axd)

    # (e) ranks of the six DUX4 target-activation reactions
    axe = fig.add_subplot(bottom[2])
    for j, d in enumerate(decks):
        ranks = [m["rank"] for m in R["mechanism"][d]]
        axe.scatter([j] * len(ranks), ranks, s=12, color="#c44e52", zorder=3)
        if ranks:
            axe.plot([j, j], [min(ranks), max(ranks)], color="#c44e52", lw=1.0, zorder=2)
    worst = max((m["rank"] for d in decks for m in R["mechanism"][d]), default=30)
    deck_ticks(axe)
    axe.set_xlim(-0.6, len(decks) - 0.4); axe.set_ylim(worst * 1.15, 0); axe.set_yticks(range(0, int(worst * 1.15) + 1, 10))
    axe.set_ylabel("Rank", fontsize=CFS["label"], labelpad=2)
    axe.tick_params(axis="y", labelsize=CFS["tick"])

    for ax, s in [(a_axes[0], "a"), (axb, "b"), (axc, "c"), (axd, "d"), (axe, "e")]:
        letter(ax, s, dx=-0.045 if ax in (axb, axc) else -0.035)

    problems = text_overlaps(fig)
    for p in problems:
        logger.warning("fig3 composite: %s", p)
    save(fig, out, "fig3_multiscale")
    if not problems:
        logger.info("fig3 composite: no overlapping or clipped text")


def fig3(R: dict, out: Path) -> None:
    for panel in (fig3a, fig3b, fig3c, fig3d, fig3e, fig3_composite):
        panel(R, out)


# ----------------------------------------------------------------------------------------------
# Figure 4: graph windows and DUX4 paths
# ----------------------------------------------------------------------------------------------
def window(ax, d: Path, n_top: int = 15, per_seed: int = 3) -> int:
    """Seeds, the `per_seed` best-scored kept neighbours of each seed, and the `n_top` best-scored
    nodes overall: a window on the kept graph (a few dozen of its ~5,500 nodes), chosen so that
    every seed is shown with the context the expansion gave it. Returns the number of nodes shown."""
    import networkx as nx

    with open(d / "nodes.csv", encoding="utf-8", newline="") as f:
        nodes = list(csv.DictReader(f))
    with open(d / "edges.csv", encoding="utf-8", newline="") as f:
        edges = list(csv.DictReader(f))
    info = {n["id"]: n for n in nodes}
    seeds = {n["id"] for n in nodes if n["is_seed"] == "1"}
    score = {n["id"]: float(n["score"] or 0) for n in nodes}
    adj: dict[str, set[str]] = {}
    for e in edges:
        a, b = e["subject"], e["object"]
        if a != b and a in info and b in info:
            adj.setdefault(a, set()).add(b); adj.setdefault(b, set()).add(a)
    keep = set(seeds)
    for s in seeds:
        keep.update(sorted((v for v in adj.get(s, ()) if v not in seeds), key=lambda v: -score[v])[:per_seed])
    keep.update(n["id"] for n in sorted((n for n in nodes if n["is_seed"] != "1"), key=lambda n: -score[n["id"]])[:n_top])
    G = nx.Graph()
    G.add_nodes_from(keep)
    G.add_edges_from((a, b) for a in keep for b in adj.get(a, ()) if b in keep and a < b)
    pos = nx.spring_layout(G, seed=7, k=0.5)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25, width=0.6)
    others = [n for n in G if n not in seeds]
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=others, node_size=28, node_color=[branch_color(info[n]["category"]) for n in others], alpha=0.9, linewidths=0)
    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=[n for n in G if n in seeds], node_size=80, node_color="white", edgecolors="black", linewidths=1.1)
    nx.draw_networkx_labels(G, pos, ax=ax, labels={n: info[n]["name"][:22] for n in G if n in seeds}, font_size=FS["seed"], font_family="Arial")
    nx.draw_networkx_labels(G, pos, ax=ax, labels={n: info[n]["name"][:20] for n in others}, font_size=FS["node"] - 1.5,
                            font_color="#424242", font_family="Arial")
    ax.axis("off"); ax.margins(0.06)
    return G.number_of_nodes()


def fig4_windows(R: dict, root: Path, out: Path) -> None:
    for k, d in enumerate(d for d in ("VoieA", "VoieB", "VoieC") if d in R["decks"]):
        fig, ax = square()
        n_shown = window(ax, graph_dir(root, d))
        s = R["structure"][d]
        ax.text(0.0, 1.0, f"slide {DECK_SHORT[d]}, {SCALE[d].split(' (')[0]} scale: {s['n_seeds']} seeds, {s['n_nodes']} kept "
                          f"nodes, diameter {s['diameter']}, {s['n_components']} component(s)",
                transform=ax.transAxes, fontsize=FS["note"], color="#616161", va="bottom")
        ax.text(1.0, 0.0, f"window: {n_shown} of {s['n_nodes']} kept nodes", transform=ax.transAxes,
                fontsize=FS["note"], color="#616161", ha="right", va="top")
        fig.tight_layout()
        save(fig, out, f"fig4{'abc'[k]}_window_{DECK_SHORT[d]}")


def fig4_legend(out: Path) -> None:
    fig, ax = square()
    ax.axis("off")
    handles = [plt.Line2D([], [], marker="o", ls="", color=col, label=lab, ms=9) for lab, _, col in BRANCH]
    handles.append(plt.Line2D([], [], marker="o", ls="", mfc="white", mec="black", label="seed (slide entity)", ms=11))
    ax.legend(handles=handles, loc="center", fontsize=FS["label"] + 2, frameon=False, title="Biolink branch",
              title_fontsize=FS["label"] + 2, labelspacing=1.0)
    save(fig, out, "fig4_legend")


def chain(ax, y: float, path: list[dict], color: str) -> None:
    """One path as boxes joined by arrows, with the predicate above and the source below each arrow."""
    n = len(path)
    if n == 0:
        return
    margin = 0.01
    bw = min(0.2, (1 - 2 * margin) / n * 0.62)
    xs = np.linspace(margin + bw / 2, 1 - margin - bw / 2, n) if n > 1 else [0.5]
    for k, (node, x) in enumerate(zip(path, xs)):
        name = node["name"].replace(" (Homo sapiens)", "")
        name = "\n".join(textwrap.wrap(name, 15, break_long_words=False)[:3])
        box = FancyBboxPatch((x - bw / 2, y - 0.04), bw, 0.08, boxstyle="round,pad=0.004,rounding_size=0.012",
                             fc="white" if k else "#f3d1d1", ec=color if k else "#c44e52", lw=1.1, transform=ax.transAxes, zorder=3)
        ax.add_patch(box)
        ax.text(x, y, name, ha="center", va="center", fontsize=FS["path"], transform=ax.transAxes, zorder=4, linespacing=1.0)
        if k:
            e = node["edge"]
            xa, xb = xs[k - 1] + bw / 2, x - bw / 2
            if not e["forward"]:
                xa, xb = xb, xa
            ax.annotate("", xy=(xb, y), xytext=(xa, y), xycoords="axes fraction", textcoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0, shrinkA=0, shrinkB=0), zorder=2)
            mid = (xs[k - 1] + x) / 2
            ax.text(mid, y + 0.012, "\n".join(textwrap.wrap(e["predicate"].replace("_", " "), 9, break_long_words=False)), ha="center", va="bottom",
                    fontsize=FS["path"] - 1, color=color, transform=ax.transAxes, linespacing=1.0)
            ax.text(mid, y - 0.012, f"{e['source'].replace('infores:', '')}\nw={e['weight']:.2f}", ha="center", va="top",
                    fontsize=FS["path"] - 1.5, color="#616161", transform=ax.transAxes, linespacing=1.0)


def fig4d(R: dict, out: Path) -> None:
    decks = R["decks"]
    singles = [d for d in ("VoieA", "VoieB", "VoieC") if d in decks]
    rows = [(f"slide {DECK_SHORT[d]} → signature {s}", R["distances"][d][f"DUX4-{s}"]["path"], SIG_COLOR[s]) for s, d in zip("ABC", singles)]
    if "VoiesA-C" in decks:
        rows += [(f"slide A+B+C → signature {s}", R["distances"]["VoiesA-C"][f"DUX4-{s}"]["path"], SIG_COLOR[s]) for s in "ABC"]
    fig, ax = square()
    ax.axis("off")
    band = 1.0 / len(rows)
    for r, (label, path, color) in enumerate(rows):
        top = 1.0 - r * band
        ax.text(0.0, top - 0.005, label, ha="left", va="top", fontsize=FS["label"], weight="bold", color=color, transform=ax.transAxes)
        y = top - 0.58 * band
        if path:
            chain(ax, y, path, color)
        else:
            ax.text(0.5, y, "no path inside the kept graph", ha="center", va="center", fontsize=FS["note"], color="#9e9e9e", transform=ax.transAxes)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
    save(fig, out, "fig4d_dux4_paths")


def fig4(R: dict, root: Path, out: Path) -> None:
    fig4_windows(R, root, out)
    fig4_legend(out)
    fig4d(R, out)


def captions(R: dict, out: Path) -> None:
    decks = R["decks"]
    S = R["structure"]
    u = R["overlap"].get("union_coverage", {})
    txt = f"""# Captions, multi-scale case

## Figure 3 (fig3a-e)
Context graphs of the three pathway slides of the FSHD1 inflammation deck (A, innate antiviral sensing; B, mitochondria and redox; C, immune infiltrate and fibro-adipogenic remodelling), of the slide that draws the three pathways together (A+B+C) and of the deck's multi-scale summary slide, all extracted with the biochemical profile (2 hops, per-category cap 300). (a) Structure of the kept graph: diameter {', '.join(str(S[d]['diameter']) for d in decks)}; connected components {', '.join(str(S[d]['n_components']) for d in decks)}; share of seed pairs connected {', '.join(f"{S[d]['seed_pairs_connected_frac']:.2f}" for d in decks)}. (b) Kept nodes per Biolink category where the graphs differ (13 categories are at the cap of 300 in every graph). (c) Recovery of pathway signatures: for each pathway, ten RTX-KG2c nodes (Reactome reactions and pathways, GO processes, genes) that a reader expects and that are seeds of no deck; cells give the number present in the kept graph and the median rank percentile of the best hit inside its Biolink category. (d) Jaccard overlap of the kept node sets; {u.get('frac', 0):.0%} of the A+B+C graph is already in A, B or C. (e) Rank of the six Reactome reactions of DUX4 target activation inside the capped MolecularActivity category of each graph.

## Figure 4 (fig4a-d, fig4_legend)
(a-c) A window on each single-pathway graph: seeds (white, labelled), the three best-scored neighbours of each seed and the fifteen best-scored nodes overall, coloured by Biolink branch. (d) Best-weighted path from DUX4 to the signature of each pathway, in the slide of that pathway and in the slide that integrates the three; each step is an RTX-KG2c edge with its predicate, primary source and profile weight.
"""
    (out / "voie_panel_captions.md").write_text(txt, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--summary", default=str(B / "data/eval/voie_multiscale/summary.json"))
    ap.add_argument("--graphs-dir", default=str(B / "output"))
    ap.add_argument("--out", default=str(B.parent / "manuscripts/biomedcat_manuscript/figures"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    R = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    out = Path(args.out)
    fig3(R, out)
    fig4(R, Path(args.graphs_dir), out)
    captions(R, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
