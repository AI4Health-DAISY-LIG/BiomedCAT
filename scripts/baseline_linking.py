#!/usr/bin/env python
"""Ablation of the entity-linking stage on the expert gold terms.

Every condition receives the same mentions (the gold terms, optionally with a Biolink type) and
must return one CURIE, or nothing. Conditions:

  nameres_top1            RENCI Name Resolver, first candidate, no type given
  nameres_top1_typed      same, with the Biolink type passed as `biolink_type` (gold type, or the
                          type predicted by BiomedCAT's NER when --types-from is given)
  retrieval_top1          BiomedCAT's candidate pool (NameRes + ARAX merged and ranked), first
                          candidate, no LLM judge
  judge                   BiomedCAT's full normalization: candidate pool + local LLM judge
  llm_direct:<model>      a large model asked to write the CURIE directly, no retrieval (Albert
                          API, OpenAI-compatible); measures identifier hallucination

Scoring per condition: exact CURIE, same Node Normalizer clique (conflation OFF), and for the
direct-LLM conditions the share of predicted CURIEs unknown to the Node Normalizer
(hallucinated or non-Translator identifiers). Gold CURIEs that the Node Normalizer does not know
are reported separately (`linkable` subset = gold CURIEs known to NodeNorm).

Usage (from the BiomedCAT root):
  uv run python scripts/baseline_linking.py --out data/eval/linking
  uv run python scripts/baseline_linking.py --out data/eval/linking_predtypes --types-from output/FSHD_BiomedCAT.json
  uv run python scripts/baseline_linking.py --out data/eval/linking --llm openai/gpt-oss-120b,gemma-4-31b-it,mistral-small-3-2-24b-instruct-2506
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_gold import nodenorm_cliques, norm_class, norm_text, partial_match, same_curie  # noqa: E402

BIOLINK_CLASS_BY_NORM: dict[str, str] = {}


def biolink_curie_for_type(type_name: str) -> str | None:
    """'gross anatomical structure' -> 'biolink:GrossAnatomicalStructure'."""
    if not type_name:
        return None
    words = re.sub(r"[_\-]", " ", type_name.replace("biolink:", "")).split()
    return "biolink:" + "".join(w[:1].upper() + w[1:] for w in words)


def types_from_result(path: Path, gold_terms: list[dict]) -> dict[str, str]:
    """Predicted type per gold term, from a BiomedCAT result JSON (first matching mention)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    out = {}
    for g in gold_terms:
        for e in payload.get("entities", []):
            if partial_match(g["term"], e.get("text", "")):
                out[g["term"]] = e.get("type", "")
                break
    return out


def cond_nameres(terms: list[dict], types: dict[str, str] | None) -> dict[str, str | None]:
    from biomedcat.retrieval import renci_lookup

    out = {}
    for g in terms:
        bt = biolink_curie_for_type(types.get(g["term"], "")) if types else None
        hits = renci_lookup(g["term"], biolink_type=bt)
        out[g["term"]] = hits[0].curie if hits else None
    return out


def cond_retrieval(terms: list[dict], types: dict[str, str] | None) -> dict[str, str | None]:
    from biomedcat.retrieval import build_pool

    out = {}
    for g in terms:
        pool = build_pool(g["term"], {types.get(g["term"], "")} if types else set())
        out[g["term"]] = sorted(pool, key=lambda c: c.rank)[0].curie if pool else None
    return out


def cond_judge(terms: list[dict], types: dict[str, str]) -> dict[str, str | None]:
    from biomedcat.stages.norm import run_norm
    from biomedcat.types import Entity

    ents = [Entity(text=g["term"], type=types.get(g["term"], ""), segment=g["term"], page=None) for g in terms]
    return {r.text: r.curie for r in run_norm(ents)}


def cond_llm_direct(terms: list[dict], types: dict[str, str] | None, model: str, env_path: Path) -> dict[str, str | None]:
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(env_path)
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_API_BASE"])
    out: dict[str, str | None] = {}
    batch = 10
    for i in range(0, len(terms), batch):
        chunk = terms[i:i + batch]
        items = [{"mention": g["term"], **({"biolink_type": types.get(g["term"], "")} if types else {})} for g in chunk]
        user = (
            "For each biomedical mention, return the identifier (CURIE) of the concept it refers to, using the "
            "preferred Biolink prefix for its type (NCBIGene for genes, MONDO for diseases, CHEBI for small molecules, "
            "UBERON for anatomy, GO for processes, HP for phenotypes, UniProtKB for proteins, MESH otherwise). "
            "Return null if you are not certain of the exact identifier; do not invent numbers.\n"
            f"Mentions: {json.dumps(items)}\n"
            'Answer with a JSON object: {"curies": {"<mention>": "<PREFIX:id or null>", ...}}'
        )
        try:
            resp = client.chat.completions.create(model=model, temperature=0.0, response_format={"type": "json_object"},
                                                  messages=[{"role": "system", "content": "You are a biomedical curator. JSON only."},
                                                            {"role": "user", "content": user}])
            text = resp.choices[0].message.content or ""
            m = re.search(r"\{.*\}", text, re.DOTALL)
            labels = (json.loads(m.group(0)) if m else {}).get("curies", {})
        except Exception as e:
            print(f"[!] {model}: {str(e)[:120]}")
            labels = {}
        for g in chunk:
            v = labels.get(g["term"])
            out[g["term"]] = str(v).strip() if v and str(v).lower() != "null" else None
        time.sleep(0.5)
    return out


def score(terms: list[dict], preds: dict[str, str | None], cliques: dict[str, str | None]) -> dict:
    n = len(terms)
    linkable = [g for g in terms if cliques.get(g["curie"])]
    exact = sum(1 for g in terms if same_curie(preds.get(g["term"]) or "", g["curie"]))
    same_cl = sum(1 for g in terms if (preds.get(g["term"]) and cliques.get(preds[g["term"]]) and cliques.get(preds[g["term"]]) == cliques.get(g["curie"])) or same_curie(preds.get(g["term"]) or "", g["curie"]))
    same_cl_linkable = sum(1 for g in linkable if (preds.get(g["term"]) and cliques.get(preds[g["term"]]) == cliques.get(g["curie"])) or same_curie(preds.get(g["term"]) or "", g["curie"]))
    answered = [p for p in preds.values() if p]
    unknown = sum(1 for p in answered if not cliques.get(p))
    return {
        "n": n, "answered": len(answered), "exact": exact, "same_clique": same_cl,
        "accuracy_exact": round(exact / n, 3), "accuracy_clique": round(same_cl / n, 3),
        "n_linkable": len(linkable), "accuracy_clique_on_linkable": round(same_cl_linkable / len(linkable), 3) if linkable else None,
        "predicted_curies_unknown_to_nodenorm": unknown,
        "unknown_rate_among_answered": round(unknown / len(answered), 3) if answered else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gold", default="data/gold/fshd_slides_gold.json")
    parser.add_argument("--out", required=True)
    parser.add_argument("--types-from", default=None, help="BiomedCAT result JSON: use its predicted types instead of the gold classes.")
    parser.add_argument("--llm", default="", help="Comma-separated Albert model ids for the direct-CURIE condition.")
    parser.add_argument("--env", default=".env")
    parser.add_argument("--skip-judge", action="store_true", help="Skip the local LLM judge condition (no Ollama).")
    args = parser.parse_args()

    gold = json.loads(Path(args.gold).read_text(encoding="utf-8"))
    terms = [g for g in gold["positives"] if g.get("curie")]
    if args.types_from:
        types = types_from_result(Path(args.types_from), terms)
        type_source = f"predicted ({args.types_from})"
    else:
        types = {g["term"]: g.get("biolink_class", "") for g in terms}
        type_source = "gold"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    conditions: dict[str, dict[str, str | None]] = {}
    t0 = time.perf_counter()
    conditions["nameres_top1"] = cond_nameres(terms, None)
    conditions["nameres_top1_typed"] = cond_nameres(terms, types)
    conditions["retrieval_top1"] = cond_retrieval(terms, None)
    conditions["retrieval_top1_typed"] = cond_retrieval(terms, types)
    if not args.skip_judge:
        conditions["judge"] = cond_judge(terms, types)
    for model in [m.strip() for m in args.llm.split(",") if m.strip()]:
        conditions[f"llm_direct:{model}"] = cond_llm_direct(terms, types, model, Path(args.env))
        conditions[f"llm_direct_untyped:{model}"] = cond_llm_direct(terms, None, model, Path(args.env))

    all_curies = {g["curie"] for g in terms} | {p for c in conditions.values() for p in c.values() if p}
    cliques = nodenorm_cliques(sorted(all_curies))
    report = {"gold": args.gold, "type_source": type_source, "n_terms": len(terms),
              "gold_unknown_to_nodenorm": [g["curie"] for g in terms if not cliques.get(g["curie"])],
              "conditions": {name: score(terms, preds, cliques) for name, preds in conditions.items()},
              "predictions": {name: preds for name, preds in conditions.items()},
              "seconds": round(time.perf_counter() - t0, 1)}
    (out_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [f"# Linking ablation on {len(terms)} gold terms (types: {type_source}; {len(report['gold_unknown_to_nodenorm'])} gold CURIEs outside Node Normalizer)", "",
             "| Condition | answered | exact | same clique | acc exact | acc clique | acc clique, linkable | unknown ids among answered |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, s in report["conditions"].items():
        lines.append(f"| {name} | {s['answered']} | {s['exact']} | {s['same_clique']} | {s['accuracy_exact']} | {s['accuracy_clique']} | {s['accuracy_clique_on_linkable']} | {s['unknown_rate_among_answered']} |")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
