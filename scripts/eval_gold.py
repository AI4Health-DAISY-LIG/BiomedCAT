#!/usr/bin/env python
"""Expert gold standard for the FSHD slides: conversion from the spreadsheet and scoring.

The spreadsheet data/NER_test_Suite.xlsx has one row per term:
  ID | Biolink class | input | compound term | expected, given context
Rows with a CURIE are expert-validated positives extracted from the two FSHD slides. Rows
whose expected value is "NA" are adversarial terms (nonsense words, absurd compounds) that
must receive neither a Biolink type nor a CURIE.

Two commands:

  convert   xlsx -> data/gold/fshd_slides_gold.json
  score     gold JSON + a BiomedCAT result JSON (pipeline output) -> metrics JSON + Markdown

Scoring on the pipeline output (positives):
  - term recall: a gold term is found when an extracted mention equals it (exact) or overlaps
    it by containment (partial), after case and whitespace normalization;
  - typing accuracy on found terms, exact Biolink class, plus the mean hierarchical distance
    between predicted and gold classes (Biolink class tree from data/biolink_classes_nested.json);
  - linking accuracy on found terms: exact CURIE, and canonical KG2c id equality as a second,
    more lenient criterion (different CURIEs for the same concept).
Scoring on an adversarial run (scripts/run_adversarial.py output):
  - false positive rate: adversarial terms that received a type, and that received a CURIE.
Inferred mentions (description-mode reading):
  - the vision model describes figures, so it names entities beyond the expert term list. Every
    extracted mention that matches no gold term is written to creativity_candidates.csv with an
    empty `judgment` column for expert curation (correct = a true inference from the slide or
    its figures, incorrect = a hallucination). Precision is reported on the gold terms only.
  - `creativity` then reads the curated file and reports the true-creativity rate, i.e. the
    precision of the inferred mentions after curation.

Usage (from the BiomedCAT root):
  uv run python scripts/eval_gold.py convert --xlsx data/NER_test_Suite.xlsx
  uv run python scripts/eval_gold.py score --gold data/gold/fshd_slides_gold.json \
      --result output/FSHD_BiomedCAT.json --adversarial output/adversarial_run.json --out data/eval/fshd
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

# Class names in the spreadsheet that differ from the Biolink class names. "Molecular Sequence"
# is not a Biolink class; the D4Z4 array and its repeats are scored as nucleic acid entities.
CLASS_ALIASES = {
    "cellcomponent": "cellularcomponent",
    "rnaproduct": "rnaproduct",
    "molecularsequence": "nucleicacidentity",
}


def norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def norm_class(s: str) -> str:
    key = re.sub(r"[\s_\-]", "", (s or "").strip().lower())
    return CLASS_ALIASES.get(key, key)


def norm_curie(s: str) -> str:
    """Keep the CURIE as written (KG2c prefixes are case-sensitive); comparisons use same_curie()."""
    return (s or "").strip()


def same_curie(a: str, b: str) -> bool:
    return bool(a) and bool(b) and a.strip().lower() == b.strip().lower()


def tokens(s: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", norm_text(s)))


def partial_match(gold_term: str, mention: str) -> bool:
    """Containment either way, or one term's tokens all contained in the other's."""
    g, m = norm_text(gold_term), norm_text(mention)
    if not g or not m:
        return False
    if g in m or m in g:
        return True
    tg, tm = tokens(g), tokens(m)
    return bool(tg) and bool(tm) and (tg <= tm or tm <= tg)


# ----------------------------------------------------------------------------------------
# convert
# ----------------------------------------------------------------------------------------

def convert(xlsx: Path, out: Path) -> None:
    import openpyxl

    wb = openpyxl.load_workbook(xlsx, read_only=True, data_only=True)
    ws = wb.worksheets[0]
    positives, adversarial = [], []
    for i, row in enumerate(ws.iter_rows(values_only=True), start=1):
        if i == 1 or not row or row[0] is None:
            continue
        rid, biolink_class, term, compound, expected = (list(row) + [None] * 5)[:5]
        term = str(term or "").strip()
        expected = str(expected or "").strip()
        if not term:
            continue
        entry = {"id": int(rid), "term": term, "compound": bool(int(compound or 0))}
        if expected and expected.upper() != "NA":
            entry["biolink_class"] = str(biolink_class or "").strip()
            entry["curie"] = norm_curie(expected)
            positives.append(entry)
        else:
            adversarial.append(entry)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"source": str(xlsx), "positives": positives, "adversarial": adversarial}, indent=2), encoding="utf-8")
    print(f"{len(positives)} positives, {len(adversarial)} adversarial -> {out}")


# ----------------------------------------------------------------------------------------
# score
# ----------------------------------------------------------------------------------------

def load_class_tree(nested_path: Path) -> dict[str, list[str]]:
    """Map normalized class name -> path from the root, from the nested Biolink JSON."""
    paths: dict[str, list[str]] = {}

    def walk(name: str, node: dict, ancestors: list[str]) -> None:
        path = ancestors + [norm_class(name)]
        paths[norm_class(name)] = path
        for child, sub in (node.get("children") or {}).items():
            walk(child, sub, path)

    data = json.loads(nested_path.read_text(encoding="utf-8"))
    for root, node in data.items():
        walk(root, node, [])
    return paths


def hier_distance(a: str, b: str, paths: dict[str, list[str]]) -> int | None:
    pa, pb = paths.get(a), paths.get(b)
    if pa is None or pb is None:
        return None
    common = 0
    for x, y in zip(pa, pb):
        if x != y:
            break
        common += 1
    return (len(pa) - common) + (len(pb) - common)


NODENORM_URL = "https://nodenorm.transltr.io/1.5/get_normalized_nodes"


def nodenorm_cliques(curies: list[str]) -> dict[str, str | None]:
    """Map each CURIE to the preferred id of its Translator Node Normalizer clique (None if unknown).

    Two CURIEs name the same concept when their clique ids are equal, whatever the prefixes
    (OMIM:158900 and MONDO:0008030 both normalize to MONDO:0008030). Conflation is OFF: a gene
    and its protein stay distinct cliques, as do drugs and chemicals.
    """
    import requests

    out: dict[str, str | None] = {}
    todo = sorted({c for c in curies if c})
    for i in range(0, len(todo), 50):
        batch = todo[i:i + 50]
        try:
            r = requests.get(NODENORM_URL, params={"curie": batch, "conflate": "false", "drug_chemical_conflate": "false"}, timeout=60)
            r.raise_for_status()
            data = r.json()
        except requests.RequestException as e:
            print(f"[!] NodeNorm lookup failed for {len(batch)} CURIEs: {e}")
            data = {}
        for c in batch:
            entry = data.get(c)
            out[c] = entry["id"]["identifier"] if entry else None
    return out


def canonical_ids(curies: list[str], kg_dir: Path) -> dict[str, str | None]:
    table = kg_dir / "equivalents.parquet"
    if not table.is_file() or not curies:
        return {c: None for c in curies}
    import duckdb

    con = duckdb.connect()
    rows = con.execute(
        f"SELECT curie, canonical_id FROM read_parquet('{str(table).replace(chr(92), '/')}') WHERE curie IN (SELECT UNNEST(?))",
        [curies],
    ).fetchall()
    con.close()
    found = dict(rows)
    return {c: found.get(c) for c in curies}


def creativity_candidates(entities: list[dict], gold_terms: set[str], out_dir: Path) -> dict:
    """Write the mentions that match no gold term to a CSV for expert curation."""
    rows = []
    n_gold = n_out = 0
    seen: set[tuple] = set()
    for e in entities:
        text = norm_text(e.get("text"))
        matches_gold = any(partial_match(t, text) for t in gold_terms)
        if matches_gold:
            n_gold += 1
            continue
        n_out += 1
        key = (text, norm_class(e.get("type")))
        if key in seen:
            continue
        seen.add(key)
        rows.append({"page": e.get("page"), "text": e.get("text"), "type": e.get("type"), "curie": e.get("curie"),
                     "label": e.get("label"), "segment": (e.get("segment") or "")[:200], "judgment": ""})
    out = out_dir / "creativity_candidates.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["page", "text", "type", "curie", "label", "segment", "judgment"])
        w.writeheader()
        w.writerows(rows)
    return {"mentions_matching_gold": n_gold, "mentions_inferred": n_out, "distinct_inferred": len(rows),
            "inferred_share": round(n_out / (n_gold + n_out), 3) if (n_gold + n_out) else None,
            "candidates_file": str(out)}


def creativity_rate(curated_csv: Path) -> dict:
    """True-creativity rate from the curated candidates (judgment = correct | incorrect)."""
    with open(curated_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    judged = [r for r in rows if (r.get("judgment") or "").strip().lower() in ("correct", "incorrect")]
    correct = sum(1 for r in judged if r["judgment"].strip().lower() == "correct")
    return {"n_candidates": len(rows), "n_judged": len(judged), "n_correct": correct,
            "true_creativity_rate": round(correct / len(judged), 3) if judged else None,
            "unjudged": len(rows) - len(judged)}


def score(gold_path: Path, result_path: Path | None, adversarial_path: Path | None, out_dir: Path, kg_dir: Path,
          nested_path: Path) -> None:
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    paths = load_class_tree(nested_path) if nested_path.is_file() else {}
    metrics: dict = {"gold": str(gold_path)}
    lines = ["# FSHD slides: evaluation against the expert gold standard", ""]
    out_dir.mkdir(parents=True, exist_ok=True)

    if result_path:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        entities = result.get("entities", [])
        extracted = [
            {"text": norm_text(e.get("text")), "type": norm_class(e.get("type")), "curie": norm_curie(e.get("curie")), "kg2c_id": e.get("kg2c_id")}
            for e in entities
        ]
        all_curies = [g["curie"] for g in gold["positives"]] + [x["curie"] for x in extracted if x["curie"]]
        canon = canonical_ids(sorted(set(all_curies)), kg_dir)
        clique = nodenorm_cliques(sorted(set(all_curies)))
        link_clique = 0

        rows = []
        exact_found = partial_found = 0
        type_ok = 0
        n_typed_gold = 0
        distances = []
        link_exact = link_canon = 0
        for g in gold["positives"]:
            gt = norm_text(g["term"])
            exact = [x for x in extracted if x["text"] == gt]
            partial = exact or [x for x in extracted if partial_match(gt, x["text"])]
            match = exact[0] if exact else (partial[0] if partial else None)
            status = "exact" if exact else ("partial" if partial else "missed")
            exact_found += bool(exact)
            partial_found += bool(partial)
            row = {"id": g["id"], "term": g["term"], "match": status, "gold_class": g["biolink_class"], "gold_curie": g["curie"]}
            if match:
                gc, pc = norm_class(g["biolink_class"]), match["type"]
                row["pred_class"] = match["type"]
                # Gold rows without a class (HD, TAD, damage) count for recall and linking only.
                row["class_ok"] = (gc == pc) if gc else None
                if gc:
                    n_typed_gold += 1
                    type_ok += row["class_ok"]
                d = hier_distance(gc, pc, paths) if gc else None
                row["hier_distance"] = d
                if d is not None:
                    distances.append(d)
                row["pred_curie"] = match["curie"]
                row["curie_exact"] = same_curie(match["curie"], g["curie"])
                row["curie_canonical"] = bool(canon.get(g["curie"])) and canon.get(g["curie"]) == (match["kg2c_id"] or canon.get(match["curie"]))
                gold_clique, pred_clique = clique.get(g["curie"]), clique.get(match["curie"])
                row["gold_clique"], row["pred_clique"] = gold_clique, pred_clique
                row["curie_same_clique"] = bool(gold_clique) and gold_clique == pred_clique
                link_exact += row["curie_exact"]
                link_canon += row["curie_canonical"] or row["curie_exact"]
                link_clique += row["curie_same_clique"] or row["curie_exact"]
            rows.append(row)

        n_gold = len(gold["positives"])
        n_found = partial_found
        linkable_ids = {g["id"] for g in gold["positives"] if g.get("linkable", True)}
        n_found_linkable = sum(1 for r in rows if r.get("match") != "missed" and r["id"] in linkable_ids)
        link_clique_linkable = sum(1 for r in rows if r["id"] in linkable_ids and (r.get("curie_same_clique") or r.get("curie_exact")))
        gold_terms = {norm_text(g["term"]) for g in gold["positives"]}
        tp_mentions = sum(1 for x in extracted if any(partial_match(t, x["text"]) for t in gold_terms))
        # Description-mode reading repeats a concept across several numbered statements, so the
        # same (text, page) mention can be extracted many times; precision is also reported on
        # distinct (text, page) mentions so that one repeated off-gold phrase cannot swing it.
        distinct = {(norm_text(x["text"]), x.get("page")) for x in extracted}
        tp_distinct = sum(1 for t_, _ in distinct if any(partial_match(t, t_) for t in gold_terms))
        metrics["positives"] = {
            "n_gold": n_gold,
            "n_extracted": len(extracted),
            "recall_exact": round(exact_found / n_gold, 3),
            "recall_partial": round(partial_found / n_gold, 3),
            "precision_mentions": round(tp_mentions / len(extracted), 3) if extracted else None,
            "n_extracted_distinct": len(distinct),
            "precision_mentions_distinct": round(tp_distinct / len(distinct), 3) if distinct else None,
            "typing_accuracy_on_found": round(type_ok / n_typed_gold, 3) if n_typed_gold else None,
            "n_found_with_gold_class": n_typed_gold,
            "mean_hierarchical_distance": round(sum(distances) / len(distances), 2) if distances else None,
            "linking_accuracy_exact_on_found": round(link_exact / n_found, 3) if n_found else None,
            "linking_accuracy_canonical_on_found": round(link_canon / n_found, 3) if n_found else None,
            "linking_accuracy_nodenorm_clique_on_found": round(link_clique / n_found, 3) if n_found else None,
            # Gold v2 marks terms with no identifier in KG2c / NodeNorm as non-linkable: linking is
            # also reported over found AND linkable terms, the only denominator a linker can reach.
            "n_found_linkable": n_found_linkable,
            "linking_accuracy_nodenorm_clique_on_found_linkable": round(link_clique_linkable / n_found_linkable, 3) if n_found_linkable else None,
            "gold_curies_unknown_to_nodenorm": [g["curie"] for g in gold["positives"] if not clique.get(g["curie"])],
            "rows": rows,
        }
        p = metrics["positives"]
        lines += [
            f"Result: `{result_path}`",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Gold terms | {n_gold} |",
            f"| Extracted mentions | {len(extracted)} |",
            f"| Term recall, exact / partial | {p['recall_exact']} / {p['recall_partial']} |",
            f"| Mention precision (gold assumed exhaustive) | {p['precision_mentions']} |",
            f"| Distinct (text, page) mentions | {p['n_extracted_distinct']} |",
            f"| Mention precision on distinct mentions (gold assumed exhaustive) | {p['precision_mentions_distinct']} |",
            f"| Typing accuracy on found terms | {p['typing_accuracy_on_found']} |",
            f"| Mean hierarchical distance (found terms) | {p['mean_hierarchical_distance']} |",
            f"| Linking accuracy, exact CURIE | {p['linking_accuracy_exact_on_found']} |",
            f"| Linking accuracy, canonical KG2c id | {p['linking_accuracy_canonical_on_found']} |",
            f"| Linking accuracy, same Node Normalizer clique | {p['linking_accuracy_nodenorm_clique_on_found']} |",
            f"| Linking accuracy, same clique, linkable terms only | {p['linking_accuracy_nodenorm_clique_on_found_linkable']} ({p['n_found_linkable']} found linkable) |",
            f"| Gold CURIEs unknown to Node Normalizer | {len(p['gold_curies_unknown_to_nodenorm'])} |",
            "",
            "| ID | term | match | gold class | pred class | dist | gold CURIE | pred CURIE | same clique |",
            "|---|---|---|---|---|---:|---|---|---|",
        ]
        for r in rows:
            same = "" if "curie_same_clique" not in r else ("yes" if (r["curie_same_clique"] or r.get("curie_exact")) else "no")
            lines.append(
                f"| {r['id']} | {r['term']} | {r['match']} | {r['gold_class']} | {r.get('pred_class', '')} | "
                f"{r.get('hier_distance', '')} | {r['gold_curie']} | {r.get('pred_curie', '')} | {same} |"
            )
        lines.append("")

        metrics["creativity"] = creativity_candidates(entities, gold_terms, out_dir)
        c = metrics["creativity"]
        lines += [
            "Mentions beyond the expert term list (description-mode reading)",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Mentions matching a gold term | {c['mentions_matching_gold']} |",
            f"| Mentions beyond the gold list | {c['mentions_inferred']} ({c['distinct_inferred']} distinct) |",
            f"| Share beyond the gold list | {c['inferred_share']} |",
            "",
            f"To curate: `{c['candidates_file']}` (fill `judgment` with correct / incorrect, then run `creativity`).",
            "",
        ]

    if adversarial_path:
        adv = json.loads(adversarial_path.read_text(encoding="utf-8"))
        items = adv.get("adversarial", [])
        typed = sum(1 for a in items if a.get("types"))
        linked = sum(1 for a in items if a.get("curies"))
        metrics["adversarial"] = {
            "n": len(items),
            "typed_rate": round(typed / len(items), 3) if items else None,
            "linked_rate": round(linked / len(items), 3) if items else None,
            "rows": items,
        }
        pos_items = adv.get("positives", [])
        if pos_items:
            typed_pos = sum(1 for a in pos_items if a.get("types"))
            metrics["adversarial"]["positives_typed_rate"] = round(typed_pos / len(pos_items), 3)
        a = metrics["adversarial"]
        lines += [
            f"Adversarial run: `{adversarial_path}`",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Adversarial terms | {a['n']} |",
            f"| Received a Biolink type (false positive) | {a['typed_rate']} |",
            f"| Received a CURIE (false positive) | {a['linked_rate']} |",
        ]
        if "positives_typed_rate" in a:
            lines.append(f"| Positive terms typed in isolation | {a['positives_typed_rate']} |")
        lines.append("")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "metrics.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(l for l in lines if l.startswith("|") and "---" not in l))
    print(f"\n-> {out_dir / 'metrics.md'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("convert")
    c.add_argument("--xlsx", default="data/NER_test_Suite.xlsx")
    c.add_argument("--out", default="data/gold/fshd_slides_gold.json")
    s = sub.add_parser("score")
    s.add_argument("--gold", default="data/gold/fshd_slides_gold.json")
    s.add_argument("--result", default=None, help="BiomedCAT pipeline output JSON.")
    s.add_argument("--adversarial", default=None, help="Output of scripts/run_adversarial.py.")
    s.add_argument("--out", default="data/eval/fshd")
    s.add_argument("--kg-dir", default="data/kg2c")
    s.add_argument("--nested", default="data/biolink_classes_nested.json")
    k = sub.add_parser("creativity")
    k.add_argument("--curated", required=True, help="creativity_candidates.csv with the judgment column filled.")
    args = parser.parse_args()
    if args.cmd == "convert":
        convert(Path(args.xlsx), Path(args.out))
    elif args.cmd == "creativity":
        print(json.dumps(creativity_rate(Path(args.curated)), indent=2))
    else:
        if not args.result and not args.adversarial:
            parser.error("--result and/or --adversarial is required")
        score(Path(args.gold), Path(args.result) if args.result else None,
              Path(args.adversarial) if args.adversarial else None, Path(args.out), Path(args.kg_dir), Path(args.nested))
    return 0


if __name__ == "__main__":
    sys.exit(main())
