#!/usr/bin/env python
"""Venn diagrams: FSHD drugs (trials and repurposed marketed drugs) found in the context graphs.

Input: data/eval/fshd_drugs_presence.json (one row per drug: kg2c_id, presence per profile graph and
per 1-hop baseline). Output: figures/figS7_fshd_drugs_venn.{pdf,png} in the manuscript directory
and a Markdown table next to the JSON.

Two panels: (a) three-set Venn of the drugs present in the uniform, biochemical and clinical
profile graphs (universe = drugs that exist in the filtered RTX-KG2c); (b) presence per drug as a
matrix, including the plain 1-hop expansions, so the reader sees which drug each graph reaches.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

B = Path(__file__).resolve().parents[1]


def venn3(ax, sets: dict[str, set], labels: dict[str, str], colors: dict[str, str]) -> None:
    keys = list(sets)
    a, b, c = (sets[k] for k in keys)
    centers = {keys[0]: (-0.55, 0.35), keys[1]: (0.55, 0.35), keys[2]: (0.0, -0.6)}
    # Set labels sit outside the circles so they never overlap the member lists.
    label_pos = {keys[0]: (-1.55, 1.55), keys[1]: (1.55, 1.55), keys[2]: (0.0, -2.05)}
    for k, (x, y) in centers.items():
        ax.add_patch(Circle((x, y), 1.0, alpha=0.35, color=colors[k], lw=1.5, ec=colors[k]))
        ax.text(*label_pos[k], f"{labels[k]} ({len(sets[k])})", ha="center", va="center", fontsize=9, fontweight="bold", color=colors[k])
    regions = {
        "100": (a - b - c, (-1.05, 0.55)), "010": (b - a - c, (1.05, 0.55)), "001": (c - a - b, (0.0, -1.05)),
        "110": ((a & b) - c, (0.0, 0.75)), "101": ((a & c) - b, (-0.6, -0.25)), "011": ((b & c) - a, (0.6, -0.25)),
        "111": (a & b & c, (0.0, 0.05)),
    }
    for _, (members, (x, y)) in regions.items():
        text = str(len(members)) + ("\n" + "\n".join(sorted(members))[:200] if members and len(members) <= 6 else "")
        ax.text(x, y, text, ha="center", va="center", fontsize=7.5)
    ax.set_xlim(-2.6, 2.6); ax.set_ylim(-2.4, 1.9); ax.set_aspect("equal"); ax.axis("off")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--presence", default=str(B / "data/eval/fshd_drugs_presence.json"))
    parser.add_argument("--out", default=str(B.parent / "manuscripts/biomedcat_manuscript/figures"))
    args = parser.parse_args()
    rows = json.loads(Path(args.presence).read_text(encoding="utf-8"))
    # Accept both the original drug table (key "drug") and graph_presence.py output (key "label", kind "drug").
    rows = [dict(r, drug=r.get("drug") or r.get("label")) for r in rows if r.get("kind", "drug") == "drug"]
    in_kg = [r for r in rows if r.get("kg2c_id")]
    absent = [r["drug"] for r in rows if not r.get("kg2c_id")]
    short = lambda r: r["drug"].split(" (")[0].split(" / ")[0]
    profiles = ["uniform", "biochemical_actions", "clinical_mechanisms"]
    sets = {p: {short(r) for r in in_kg if r.get(p)} for p in profiles}
    labels = {"uniform": "uniform profile", "biochemical_actions": "biochemical profile", "clinical_mechanisms": "clinical profile"}
    colors = {"uniform": "#8c8c8c", "biochemical_actions": "#4c72b0", "clinical_mechanisms": "#c44e52"}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [1.1, 1.3]})
    venn3(ax1, sets, labels, colors)
    none = {short(r) for r in in_kg} - sets["uniform"] - sets["biochemical_actions"] - sets["clinical_mechanisms"]
    ax1.set_title(f"FSHD drugs present in the context graphs\n(universe: {len(in_kg)} drugs present in RTX-KG2c; {len(none)} in no graph)", fontsize=10)
    # (b) presence matrix
    cols = profiles + [c for c in ("FSHD_plover_1hop", "FSHD_gold_kg2c_1hop") if c in in_kg[0]]
    col_labels = [labels.get(c, c).replace("FSHD_", "").replace("_", " ") for c in cols]
    names = [short(r) for r in in_kg]
    ax2.set_xlim(0, len(cols)); ax2.set_ylim(0, len(names))
    for i, r in enumerate(in_kg):
        for j, c in enumerate(cols):
            ax2.add_patch(plt.Rectangle((j, len(names) - 1 - i), 1, 1, color=("#55a868" if r.get(c) else "#f0f0f0"), ec="white"))
            if r.get(c):
                ax2.text(j + 0.5, len(names) - 0.5 - i, "✓", ha="center", va="center", fontsize=9, color="white")
    ax2.set_yticks([len(names) - 0.5 - i for i in range(len(names))]); ax2.set_yticklabels(names, fontsize=8)
    ax2.set_xticks([j + 0.5 for j in range(len(cols))]); ax2.set_xticklabels(col_labels, rotation=25, ha="right", fontsize=8)
    ax2.set_title("Presence per drug (green = node present)", fontsize=10)
    for s in ("top", "right", "left", "bottom"):
        ax2.spines[s].set_visible(False)
    fig.suptitle("Drugs tested in FSHD (ClinicalTrials.gov, systematic review) retrieved implicitly around the slide entities", fontsize=11)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "figS7_fshd_drugs_venn.pdf"); fig.savefig(out / "figS7_fshd_drugs_venn.png", dpi=300)
    md = ["| drug | status | KG2c id | uniform | biochemical | clinical | 1-hop (pipeline seeds) | 1-hop (gold seeds) |", "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['drug']} | {r['status']} | {r.get('kg2c_id') or 'absent from KG2c'} | " + " | ".join("yes" if r.get(c) else "no" for c in profiles + ["FSHD_plover_1hop", "FSHD_gold_kg2c_1hop"]) + " |")
    Path(args.presence).with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("absent from KG2c:", absent)
    print({k: sorted(v) for k, v in sets.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
