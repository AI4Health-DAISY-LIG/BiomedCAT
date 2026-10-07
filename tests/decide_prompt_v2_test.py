"""Prompt test (20 Sept 2026): the v2 decision prompt of the "search then decide" mode, measured
on the 300 reference terms WITHOUT changing the agent code. The candidate list, the retrieval,
the definition cut (250 characters), the model call options and the verdict parsing are those of
the agent; only the prompt text differs from NERAgentPipeline._decide_prompt:
  - provenance line (retriever matches + parents / children / siblings), menu grouped by parent
    with children indented,
  - none only when the mention denotes no entity of any kind in the list,
  - private reasoning steps: what the mention denotes (abbreviations expanded first), classify
    what it is and not what it does, contrast the two or three best classes against parent and
    children, most specific class the evidence supports.
Reference: results/agent_decide_raw_s300 (current prompt, 150-char menu) = 0.470.
    AGENT_MODE=decide uv run python output/decide_prompt_v2_test.py [--variant all|base]
"""
import argparse, os, sys, time
import pandas as pd
os.environ.setdefault("AGENT_MODE", "decide")
sys.path.insert(0, ".")
from biomedcat.config import settings
from biomedcat.runtime import generate
from biomedcat.stages.rag_engine import build_rag
from biomedcat.stages.ner_agent import NERAgentPipeline

REF = "../KG_nodes_eval/results/agent_decide_raw_s300/agent_per_query.parquet"

HEAD_V2 = ("Classify the mention into exactly one Biolink class from this list, or answer none if the mention "
           "denotes no entity of any kind in this list.\n"
           "The list contains the classes that best match the mention, plus their parents, children and siblings "
           "in the Biolink hierarchy; children are indented under their parent.\n")
TAIL_V2 = ("Think privately, do not write the analysis:\n"
           "1. Establish what the mention denotes. With a sentence, rely on the sentence only. If the mention is an "
           "abbreviation or symbol, first decide what it expands to, then classify the expansion.\n"
           "2. Classify what the mention denotes, not what it causes, does or is involved in.\n"
           "3. Take the two or three classes that fit best and contrast their definitions, in particular a class "
           "against its parent and its children.\n"
           "4. Choose the most specific class the evidence supports; if no child class holds, answer the parent; "
           "never a class more specific than the evidence supports.\n"
           "Reply with one line only: FINAL_VERDICT: <class name>")


def menu_grouped(agent, candidates):
    """Candidates in retrieval order; a candidate whose is_a parent is also a candidate is listed,
    indented, right under that parent (children keep their own order)."""
    flat = agent.rag_engine.flat_data
    meta = lambda c: flat.get(c, {}).get("metadata", {})
    line = lambda c, ind: f"{ind}- {c}: {agent._short_definition(meta(c).get('definition') or '')}"
    cset = set(candidates)
    lines, done = [], set()
    for c in candidates:
        if c in done or meta(c).get("parent") in cset:
            continue
        lines.append(line(c, "")); done.add(c)
        for k in candidates:
            if k not in done and meta(k).get("parent") == c:
                lines.append(line(k, "    ")); done.add(k)
    for c in candidates:            # grandchildren whose parent was itself indented, safety net
        if c not in done:
            lines.append(line(c, "    ")); done.add(c)
    return "\n".join(lines)


def prompt_v2(agent, term, candidates):
    return f"{HEAD_V2}{menu_grouped(agent, candidates)}\n\nMention: {term}\n{TAIL_V2}"


def parse(agent, response, candidates):
    ok, v, _ = agent._validate_output(response)
    if ok and (v == "NONE" or v in candidates):
        return v
    named = [c for c in candidates if c.lower() in response.lower()]
    return max(named, key=len) if named else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="all", choices=["all", "base"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    ref = pd.read_parquet(REF)
    if args.limit: ref = ref.head(args.limit)
    agent = NERAgentPipeline(build_rag(), settings.classification_model_id, settings.sanitization_model_id)
    rows, t0 = [], time.perf_counter()
    for i, r in enumerate(ref.itertuples(index=False)):
        term, gold = r.query, str(r.expected_class)
        cands = agent._decide_candidates(term)
        prompt = prompt_v2(agent, term, cands) if args.variant == "all" else agent._decide_prompt(term, term, cands)
        if i == 0: print(prompt, "\n" + "=" * 80, flush=True)
        resp = generate(settings.classification_model_id, [{"role": "user", "content": prompt}], agent.decide_num_predict, 0.0, think=True, raw=True)
        v = parse(agent, resp, cands)
        new_ok = str(v).lower() == gold.lower()
        ref_ok = r.agent_relation == "exact"
        rows.append({"term": term, "gold": gold, "ref_verdict": r.agent_verdict, "ref_ok": ref_ok, "new_verdict": v, "new_ok": new_ok, "n_candidates": len(cands)})
        print(f"{i+1:3d} {term[:30]:30s} gold={gold[:22]:22s} ref={str(r.agent_verdict)[:22]:22s}{'*' if ref_ok else ' '} new={str(v)[:22]:22s}{'*' if new_ok else ' '} ({time.perf_counter()-t0:.0f}s)", flush=True)
    df = pd.DataFrame(rows); out = f"output/decide_prompt_v2_{args.variant}_s300.csv"; df.to_csv(out, index=False, encoding="utf-8")
    ch = df[df.ref_verdict.astype(str).str.lower() != df.new_verdict.astype(str).str.lower()]
    print(f"\nvariant {args.variant}: reference {df.ref_ok.mean():.3f}  new {df.new_ok.mean():.3f}  (n={len(df)})")
    print(f"changed verdicts {len(ch)} | became correct {int((~ch.ref_ok & ch.new_ok).sum())} | became wrong {int((ch.ref_ok & ~ch.new_ok).sum())}")
    print(f"NONE: ref {int((df.ref_verdict.astype(str).str.upper()=='NONE').sum())} new {int((df.new_verdict.astype(str).str.upper()=='NONE').sum())} | no verdict new {int(df.new_verdict.isna().sum())}")
    print(f"grouping-class verdicts: ref {int(df.ref_verdict.astype(str).str.contains(' or ').sum())} new {int(df.new_verdict.astype(str).str.contains(' or ').sum())}")
    print(f"mean s/term {(time.perf_counter()-t0)/len(df):.1f} | csv {out}")


if __name__ == "__main__":
    main()
