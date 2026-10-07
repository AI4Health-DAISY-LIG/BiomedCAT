#!/usr/bin/env python
"""Presence of reference entities in every context graph of a document.

Two reference lists are checked against the `nodes.csv` of each `data/context_graph/<doc>_<profile>/`
directory (and the two 1-hop baselines when present):

* the drug list of data/eval/fshd_drugs_presence.json (column kg2c_id; the older per-profile
  columns of that file are ignored and recomputed here), and
* an extra list of KG2c ids given with --extra (genes, targets), so that the same table also
  reports "reached but pruned" through the `ranks` block of each summary.json when the graph
  was built with --find.

Output: <out>.json (one row per entity, one boolean column per graph, plus rank/kept when known)
and <out>.md (table). The Venn figure script reads the JSON with --presence.

Usage:
  uv run python scripts/graph_presence.py --doc FSHD --out data/eval/fshd_presence_v5
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

B = Path(__file__).resolve().parents[1]
PROFILES = ["uniform", "biochemical_actions", "clinical_mechanisms", "genetic_determinants",
            "pharmacological_intervention", "epidemiological_risk"]
BASELINES = ["FSHD_plover_1hop", "FSHD_gold_kg2c_1hop"]


def node_ids(d: Path) -> set[str]:
    p = d / "nodes.csv"
    if not p.is_file():
        return set()
    with open(p, encoding="utf-8", newline="") as f:
        return {row["id"] for row in csv.DictReader(f)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--doc", default="FSHD")
    ap.add_argument("--drugs", default=str(B / "data/eval/fshd_drugs_presence.json"))
    ap.add_argument("--extra", default="", help="Comma-separated 'label=CURIE' pairs (genes, targets).")
    ap.add_argument("--graph-dir", default=str(B / "data/context_graph"))
    ap.add_argument("--out", default=str(B / "data/eval/fshd_presence_v5"))
    args = ap.parse_args()

    root = Path(args.graph_dir)
    graphs = {p: root / f"{args.doc}_{p}" for p in PROFILES}
    for b in BASELINES:
        if (root / b).is_dir():
            graphs[b] = root / b
    ids = {k: node_ids(v) for k, v in graphs.items()}
    ranks = {}
    for k, v in graphs.items():
        s = v / "summary.json"
        if s.is_file():
            ranks[k] = (json.load(open(s, encoding="utf-8")).get("ranks") or {})

    entities = []
    for row in json.load(open(args.drugs, encoding="utf-8")):
        entities.append({"label": row["drug"], "kind": "drug", "status": row.get("status", ""), "kg2c_id": row.get("kg2c_id")})
    for pair in filter(None, args.extra.split(",")):
        label, curie = pair.split("=", 1)
        entities.append({"label": label.strip(), "kind": "reference", "status": "", "kg2c_id": curie.strip()})

    rows = []
    for e in entities:
        r = dict(e)
        r["in_kg2c"] = any(e["kg2c_id"] in s for s in ids.values()) if e["kg2c_id"] else False
        for k in graphs:
            r[k] = bool(e["kg2c_id"]) and e["kg2c_id"] in ids[k]
            info = ranks.get(k, {}).get(e["kg2c_id"] or "", None)
            if isinstance(info, dict):
                r[f"{k}__reached"] = info.get("reached")
                r[f"{k}__rank"] = info.get("rank_in_category")   # "rank/size" inside the capped category
                r[f"{k}__coverage"] = info.get("coverage")
                r[f"{k}__hop"] = info.get("hop")
        rows.append(r)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(rows, open(out.with_suffix(".json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    cols = list(graphs)
    lines = ["| entity | kind | in graph? " + " | ".join(c.replace("_", " ") for c in cols) + " |", "|---|---|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for c in cols:
            if r.get(c):
                cells.append("kept")
            elif r.get(f"{c}__reached"):
                cells.append(f"reached, pruned (rank {r.get(f'{c}__rank')})")
            elif not r["kg2c_id"]:
                cells.append("absent from KG2c")
            else:
                cells.append("-")
        lines.append(f"| {r['label']} | {r['kind']} | " + " | ".join(cells) + " |")
    kept = {c: sum(1 for r in rows if r.get(c)) for c in cols}
    lines.append("")
    lines.append("Kept per graph: " + ", ".join(f"{c} {kept[c]}/{sum(1 for r in rows if r['kg2c_id'])}" for c in cols))
    out.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
