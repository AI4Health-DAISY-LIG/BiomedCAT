"""Experiment (14 Sept 2026, no change to BiomedCAT code): "search, then decide".

The agent loop (search -> optional hierarchy lookups -> verdict, 3 steps, greedy) is replaced by:
  1. one retrieval of the term (top-10 candidates, same retriever as the agent);
  2. optional expansion with the CONCRETE neighbours of the top-3 candidates (their children and
     siblings, by name, with definitions) -- not their branch;
  3. ONE decision call with hidden reasoning (budget 1500) over that candidate list, with the
     agent's decision rule and the option 'none'.
Three modes on the 300 reference terms (public slice, seed 77, the ablation sample):
  D10   decide over the top-10                          (K6 mode B, now on 300 terms)
  D10N  decide over the top-10 + neighbours of the top-3 (cap 30 candidates)
  D20N  decide over the top-20 + neighbours of the top-3 (cap 40 candidates)
References on the same 300 terms: agent greedy 0.407, agent with reasoning in the loop 0.430.
Usage: uv run python output/search_then_decide_test.py --n 300
"""
import argparse, re, time, sys, json, requests
import pandas as pd
sys.path.insert(0, ".")
from biomedcat.config import settings
from biomedcat.stages.rag_engine import build_rag

URL = f"{settings.ollama_url.rstrip('/')}/api/generate"
MODEL = settings.classification_model_id

def call(prompt, num_predict=1500):
    r = requests.post(URL, json={"model": MODEL, "prompt": prompt, "stream": False, "think": True, "keep_alive": 0,
                                 "options": {"num_predict": num_predict, "temperature": 0, "seed": 0}}, timeout=900).json()
    return r.get("response", ""), int(r.get("eval_count", 0))

def parse(resp, options):
    m = re.search(r"FINAL_VERDICT:\s*(.+)", resp)
    ans = (m.group(1) if m else resp).strip().strip(".").strip("*").lower()
    if ans in ("none", "none.", "unknown"): return "none"
    for c in options:
        if c.lower() == ans: return c
    for c in sorted(options, key=len, reverse=True):
        if c.lower() in ans: return c
    return ans

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=300); ap.add_argument("--seed", type=int, default=77); ap.add_argument("--modes", default="", help='JSON {name: [k_top, n_neigh, cap, mixins]}'); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    df = pd.read_parquet("../KG_nodes_eval/release/v1/public/benchmark_v1_public.parquet").sample(n=a.n, random_state=a.seed).reset_index(drop=True)
    rag = build_rag(); flat = rag.flat_data
    meta = lambda c: flat.get(c, {}).get("metadata", {})
    defin = lambda c: (meta(c).get("definition") or "")[:150]
    ok_cls = lambda c: c in flat and rag._indexable(meta(c))
    groups = {}
    for c, v in flat.items():
        for mx in (v.get("metadata", {}).get("mixins") or []): groups.setdefault(mx, set()).add(c)
    def neighbours(c, mixins=False):
        # strict is_a neighbourhood: children, siblings (children of the is_a parent), parent;
        # with mixins=True the other members of the class's mixin groups are added too
        out = [x for x in (meta(c).get("children") or []) if ok_cls(x)]
        p = meta(c).get("parent")
        if p:
            out += [x for x in (meta(p).get("children") or []) if x != c and ok_cls(x)]
            if ok_cls(p): out.append(p)
        if mixins:
            for mx in (meta(c).get("mixins") or []):
                out += [x for x in sorted(groups.get(mx, ())) if x != c and ok_cls(x)]
        return out
    def candidates(top, k_top, k_neigh, cap, mixins=False):
        cands = list(top[:k_top])
        for c in top[:k_neigh]:
            for x in neighbours(c, mixins):
                if x not in cands: cands.append(x)
        return cands[:cap]
    modes = json.loads(a.modes) if a.modes else {"D10": [10, 0, 10, False], "D10N": [10, 3, 30, False], "D20N": [20, 3, 40, False]}
    modes = {k: tuple(v) for k, v in modes.items()}
    res = {m: {"ok": 0, "tok": 0, "sec": 0.0, "cover": 0, "none": 0} for m in modes}
    rows = []
    for i, r in df.iterrows():
        term, gold = r["query"], r["expected_class"]
        top = rag.search(term, top_k=20)
        row = {"term": term, "gold": gold, "rank": (top.index(gold) + 1) if gold in top else 0}
        for m, (kt, kn, cap, mix) in modes.items():
            cands = candidates(top, kt, kn, cap, mix)
            res[m]["cover"] += gold in cands
            menu = "\n".join(f"- {c}: {defin(c)}" for c in cands)
            prompt = (f"Classify the biomedical mention into exactly one Biolink class from this list, or answer none if it is "
                      f"not a biomedical entity.\n{menu}\n\nMention: {term}\n"
                      "Decision rule: choose the most specific class the evidence supports; if none of the child classes holds, "
                      "answer the parent class; never a more specific class than the evidence supports.\n"
                      "Reply with one line: FINAL_VERDICT: <class name>")
            t = time.perf_counter(); resp, tok = call(prompt); res[m]["sec"] += time.perf_counter() - t; res[m]["tok"] += tok
            p = parse(resp, cands); res[m]["ok"] += p == gold; res[m]["none"] += p == "none"; row[m] = p; row[m + "_n"] = len(cands)
        rows.append(row)
        print(f"{i+1:3d} {term[:28]:28s} gold={gold[:22]:22s} " + " ".join(f"{m}={'+' if row[m]==gold else '-'}" for m in modes), flush=True)
        if (i + 1) % 25 == 0:
            print("   running:", {m: round(res[m]["ok"] / (i + 1), 3) for m in modes}, flush=True)
    n = len(df)
    print(f"\nn={n}   references on these terms: agent greedy 0.407, agent with reasoning in loop 0.430")
    print(f"{'mode':6s} {'acc':>6s} {'gold in list':>13s} {'none':>6s} {'tok/term':>9s} {'s/term':>7s}")
    for m in modes:
        print(f"{m:6s} {res[m]['ok']/n:6.3f} {res[m]['cover']/n:13.3f} {res[m]['none']/n:6.3f} {res[m]['tok']/n:9.0f} {res[m]['sec']/n:7.1f}")
    pd.DataFrame(rows).to_csv(f"output/search_then_decide_test{a.tag}.csv", index=False)
    json.dump({m: {k: (v / n if k != "sec" else v / n) for k, v in res[m].items()} for m in modes}, open(f"output/search_then_decide_test{a.tag}.json", "w"), indent=1)

if __name__ == "__main__":
    main()
