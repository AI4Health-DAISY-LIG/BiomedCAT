#!/usr/bin/env python
"""Compelling case for the profiles paper: does a profile surface a known FSHD mechanism that
the uniform (no-profile) baseline buries?

FSHD's textbook molecular mechanism is DUX4 acting as an aberrant transcription factor: normally
expressed only in the early embryo/germline, its epigenetic de-repression in muscle (D4Z4
contraction in FSHD1, SMCHD1 loss-of-function in FSHD2) lets it bind and activate repetitive
elements (MaLR, HSATII, HERVL LTR) and cleavage-stage genes (KDM4E, LEUTX) that are toxic once
switched back on in a differentiated muscle cell. RTX-KG2c carries this exact mechanism as six
Reactome nodes reachable at hop 1 from the DUX4/SMCHD1 seeds of the FSHD slide deck ("Expression
of DUX4 in the zygote", "DUX4 binds <element>"). This script checks, for each profile, whether
those six nodes are not just present in the kept context graph but prominent in it: their rank
inside the (capped) MolecularActivity category, out of ~300 kept nodes.

Requires the context graphs already built under data/context_graph/FSHD_<profile>/ (see the
context-graph CLI, biomedcat.stages.context_graph, run once per profile with
--result-json <a BiomedCAT result JSON with DUX4 and SMCHD1 as linked entities>).

Usage:
  uv run python scripts/fshd_dux4_mechanism_case.py [--profiles-dir data/context_graph] [--doc FSHD]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

B = Path(__file__).resolve().parents[1]

DEFAULT_PROFILES = ["uniform", "biochemical_actions", "clinical_mechanisms", "genetic_determinants",
                    "pharmacological_intervention", "epidemiological_risk"]

# The Reactome nodes of the DUX4 target-activation mechanism, as they read in RTX-KG2c.
MECHANISM_NODE_NAMES = (
    "Expression of DUX4 in the zygote",
    "DUX4 binds the KDM4E gene",
    "DUX4 binds the LEUTX gene",
    "DUX4 binds HERVL LTR",
    "DUX4 binds MaLR",
    "DUX4 binds HSATII pericentric repeat",
)
CATEGORY = "biolink:MolecularActivity"


def load_nodes(d: Path) -> list[dict]:
    with open(d / "nodes.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def mechanism_ranks(nodes: list[dict]) -> tuple[list[tuple[str, int, int, float, int]], int]:
    """[(name, rank, coverage, score, hop), ...] for the mechanism nodes, and the category size."""
    cat = [n for n in nodes if n["category"] == CATEGORY]
    cat.sort(key=lambda n: float(n["score"] or 0), reverse=True)
    hits = []
    for i, n in enumerate(cat):
        if any(m in n["name"] for m in MECHANISM_NODE_NAMES):
            hits.append((n["name"], i + 1, int(n["coverage"]), float(n["score"]), int(n["hop"])))
    return hits, len(cat)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profiles-dir", default=str(B / "data/context_graph"))
    parser.add_argument("--doc", default="FSHD")
    parser.add_argument("--profiles", nargs="*", default=DEFAULT_PROFILES)
    parser.add_argument("--out", default=str(B / "data/eval/fshd_dux4_case.json"),
                        help="JSON record of the ranks (read by scripts/build_master_results.py); '' to skip.")
    args = parser.parse_args()

    base = Path(args.profiles_dir)
    record = {"doc": args.doc, "category": CATEGORY, "mechanism_nodes": list(MECHANISM_NODE_NAMES), "profiles": []}
    print(f"DUX4 target-activation mechanism: rank within the kept {CATEGORY} category\n")
    print(f"{'profile':30s} {'best rank':>10s} {'worst rank':>11s} {'category size':>14s} {'n found / 6':>12s}")
    for prof in args.profiles:
        d = base / f"{args.doc}_{prof}"
        if not (d / "nodes.csv").is_file():
            print(f"{prof:30s}  (no graph at {d})")
            continue
        hits, n_cat = mechanism_ranks(load_nodes(d))
        record["profiles"].append({"profile": prof, "category_size": n_cat, "n_found": len(hits),
                                   "best_rank": min((r for _, r, *_ in hits), default=None),
                                   "worst_rank": max((r for _, r, *_ in hits), default=None),
                                   "hits": [{"name": n, "rank": r, "coverage": c, "score": s, "hop": h} for n, r, c, s, h in sorted(hits, key=lambda h: h[1])]})
        if hits:
            ranks = [r for _, r, *_ in hits]
            print(f"{prof:30s} {min(ranks):10d} {max(ranks):11d} {n_cat:14d} {len(hits):12d}")
        else:
            print(f"{prof:30s}  none of the 6 mechanism nodes reached the kept graph  {n_cat:14d}")

    print("\nDetail:")
    for p in record["profiles"]:
        print(f"\n  {p['profile']} ({p['category_size']} kept {CATEGORY} nodes):")
        for h in p["hits"]:
            print(f"    rank {h['rank']:>3d}/{p['category_size']}  coverage={h['coverage']:>2d}  score={h['score']:.6f}  hop={h['hop']}  {h['name']}")
    if args.out:
        Path(args.out).write_text(json.dumps(record, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
