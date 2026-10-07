#!/usr/bin/env python
"""Score every published document run against the harmonized gold (data/gold/gold_harmonized.json).

Identifiers are not scored: the gold serves the reading and typing evaluation.
Per gold term (included rows of kind positive or implicit) and per run:
  reading   is the term in the text produced by the reading stage (slide descriptions)?
              exact   = the term, its English query or the KG2c label of its CURIE is a substring
              lenient = every token (3+ characters, accents stripped) of one of those forms has a
                        match in the text tokens, equal or sharing a 5+ character prefix (a crude
                        stemmer: mitochondrie/mitochondria, hypoxie/hypoxia, macrophages/macrophage)
            This measures the reading stage alone: explicit terms are written on the slide,
            implicit terms are expected by the expert but not written.
  extraction is the term among the extracted mentions (reading + term extraction)? exact mention,
            or partial (containment / token subset as in scripts/eval_gold.py, with the same
            prefix tolerance so that an English mention can match a French gold term).
  typing    on found explicit terms with a gold class: predicted Biolink class equal to the gold
            class (exact), or at hierarchical distance <= 1 (parent or child) in the Biolink tree.
Adversarial terms (included) are checked for presence among the extracted mentions; the
dedicated adversarial suite (scripts/run_adversarial.py) remains the reference for that test.

Runs: with --runs-root, the publication runs (data/publication/runs/<doc>/, one per deck, from
scripts/chain_publication_runs.sh); without it, the legacy runs of September / 4 Oct 2026
(Output/_final_*, plus the 5 Oct full run of the glycolysis image).

Outputs (--out, default data/eval/gold_decks): per_term.csv, summary.csv, summary.json.
Usage:  uv run python scripts/eval_gold_decks.py [--runs-root data/publication/runs --out data/publication/gold_eval]
"""
from __future__ import annotations

import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval_gold as EG  # noqa: E402  (norm_text, norm_class, partial_match, same_curie, load_class_tree, hier_distance)

GOLD = ROOT / "data" / "gold" / "gold_harmonized.json"
OUT = ROOT / "data" / "eval" / "gold_decks"
OUTPUT = ROOT / "Output"
DECK_DOC = {"FSHD": "FSHD", "VoieA": "VoieA", "VoieB": "VoieB", "VoieC": "VoieC", "VoiesA-C": "VoiesA-C",
            "FSHD1multiscales": "FSHD1multiscales", "glycolysis": "chemicals"}


def legacy_runs() -> dict:
    """The runs of September / 4 Oct 2026 (Output/_final_*), used before the publication runs existed."""
    runs = {
        "FSHD": [("run1", OUTPUT / "_final_run1" / "FSHD_BiomedCAT.json", "pipeline"), ("run2", OUTPUT / "_final_run2" / "FSHD_BiomedCAT.json", "pipeline")],
        "VoieA": [("final", OUTPUT / "_final_VoieA" / "VoieA_BiomedCAT.json", "pipeline")],
        "VoieB": [("final", OUTPUT / "_final_VoieB" / "VoieB_BiomedCAT.json", "pipeline")],
        "VoieC": [("final", OUTPUT / "_final_VoieC" / "VoieC_BiomedCAT.json", "pipeline")],
        "VoiesA-C": [("final", OUTPUT / "_final_VoiesA-C" / "VoiesA-C_BiomedCAT.json", "pipeline")],
        "FSHD1multiscales": [("final", OUTPUT / "_final_FSHD1multiscales" / "FSHD1multiscales_BiomedCAT.json", "pipeline")],
    }
    full = OUTPUT / "resources" / "_run_chemicals" / "chemicals_BiomedCAT.json"
    runs["glycolysis"] = [("final", full, "pipeline")] if full.is_file() else [("ocr_only", OUTPUT / "ocr_only" / "chemicals_ocr.json", "ocr_only")]
    return runs


def publication_runs(root: Path) -> dict:
    """One run per deck under <root>/<doc>/<doc>_BiomedCAT.json (scripts/chain_publication_runs.sh)."""
    return {deck: [("publication", root / doc / f"{doc}_BiomedCAT.json", "pipeline")] for deck, doc in DECK_DOC.items()}




STOP = {"the", "and", "des", "les", "une", "pour", "par", "sur", "dans", "avec", "aux", "del", "von", "der"}


def fold(s: str) -> str:
    """Lowercase, strip accents and compatibility characters (H₂O₂ -> h2o2), collapse spaces."""
    s = unicodedata.normalize("NFKD", s or "")
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", s.strip().lower())


def toks(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9]+", fold(s)) if len(t) >= 3 and t not in STOP]


def tok_eq(a: str, b: str) -> bool:
    """Equal tokens, or a shared prefix of 5+ characters (mitochondrie/mitochondria, apoptose/apoptosis,
    hypoxie/hypoxia, macrophages/macrophage): a crude stemmer that tolerates French/English cognates."""
    if a == b:
        return True
    n = min(len(a), len(b))
    return n >= 5 and a[:5] == b[:5] and a[: n - 1] == b[: n - 1] if n < 7 else a[:5] == b[:5]


def tokens_covered(form: str, text_tokens: list[str]) -> bool:
    """Every token of the form has a tolerant match in the text tokens."""
    ft = toks(form)
    return bool(ft) and all(any(tok_eq(t, u) for u in text_tokens) for t in ft)


def mention_match(form: str, mention: str) -> bool:
    """Containment either way (eval_gold.partial_match) or tolerant token containment either way."""
    if EG.partial_match(form, mention) or fold(form) in fold(mention) or fold(mention) in fold(form):
        return True
    ft, mt = toks(form), toks(mention)
    return bool(ft) and bool(mt) and (all(any(tok_eq(t, u) for u in mt) for t in ft) or all(any(tok_eq(t, u) for u in ft) for t in mt))


def load_run(path: Path, kind: str) -> tuple[str, list[dict], dict]:
    r = json.loads(path.read_text(encoding="utf-8"))
    if kind == "ocr_only":
        text = "\n".join(p["text"] for p in r["pages"])
        return text, [], {"model_ocr": r.get("model"), "timestamp": r.get("started_utc")}
    text = "\n".join(s["text"] for s in r["slides"])
    ents = [{"text": fold(e.get("text")), "type": EG.norm_class(e.get("type")), "curie": EG.norm_curie(e.get("curie")),
             "page": e.get("page")} for e in r["entities"]]
    return text, ents, {"model_ocr": r["run"]["models"].get("ocr"), "model_ner": r["run"]["models"].get("ner"), "timestamp": r["run"].get("timestamp")}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default=None, help="Publication runs directory (data/publication/runs); default: the legacy Output/_final_* runs.")
    ap.add_argument("--out", default=str(OUT), help="Output directory (default data/eval/gold_decks).")
    args = ap.parse_args()
    runs_map = publication_runs(Path(args.runs_root).resolve()) if args.runs_root else legacy_runs()
    out_dir = Path(args.out)
    gold = pd.DataFrame(json.loads(GOLD.read_text(encoding="utf-8"))["terms"])
    gold = gold[gold["included"].astype(bool)]
    paths = EG.load_class_tree(ROOT / "data" / "biolink_classes_nested.json")
    rows, summaries = [], []
    for deck, runs in runs_map.items():
        g = gold[gold["deck"] == deck]
        for label, path, kind in runs:
            if not path.is_file():
                print(f"[!] missing run {deck}/{label}: {path}"); continue
            text, ents, meta = load_run(path, kind)
            ntext, ttext = fold(text), sorted(set(toks(text)))
            for _, t in g.iterrows():
                forms = [f for f in (t["term"], t["english_query"], t["derived_curie_label_kg2c"]) if isinstance(f, str) and f.strip()]
                row = {"deck": deck, "run": label, "gold_id": t["gold_id"], "term": t["term"], "row_kind": t["row_kind"], "vocabulary_type": t["vocabulary_type"],
                       "gold_class": t["biolink_class"], "forms": " | ".join(forms)}
                row["read_exact"] = any(fold(f) in ntext for f in forms)
                row["read_lenient"] = row["read_exact"] or any(tokens_covered(f, ttext) for f in forms)
                if kind == "pipeline":
                    exact = [e for e in ents if any(e["text"] == fold(f) for f in forms)]
                    partial = exact or [e for e in ents if any(mention_match(f, e["text"]) for f in forms)]
                    m = exact[0] if exact else (partial[0] if partial else None)
                    row["found"] = "exact" if exact else ("partial" if partial else "missed")
                    if m and t["row_kind"] == "positive":
                        gc = EG.norm_class(t["biolink_class"]) if isinstance(t["biolink_class"], str) else ""
                        row["pred_class"] = m["type"]
                        if gc:
                            d = EG.hier_distance(gc, m["type"], paths)
                            row["class_exact"] = gc == m["type"]
                            row["class_within_1"] = row["class_exact"] or (d is not None and d <= 1)
                            row["hier_distance"] = d
                rows.append(row)
            # adversarial terms seen among the mentions
            adv = gold[(gold["deck"] == deck) & (gold["row_kind"] == "adversarial")]
            n_adv_typed = sum(1 for _, a in adv.iterrows() if any(e["text"] == EG.norm_text(a["term"]) for e in ents)) if kind == "pipeline" else None

            df = pd.DataFrame([r for r in rows if r["deck"] == deck and r["run"] == label])
            ex, im = df[df["vocabulary_type"] == "explicit"], df[df["vocabulary_type"] == "implicit"]
            s = {"deck": deck, "run": label, "kind": kind, "result": str(path.relative_to(ROOT)).replace("\\", "/"), **meta,
                 "n_explicit": len(ex), "n_implicit": len(im),
                 "read_recall_explicit_exact": ex["read_exact"].mean() if len(ex) else None, "read_recall_explicit_lenient": ex["read_lenient"].mean() if len(ex) else None,
                 "read_recall_implicit_exact": im["read_exact"].mean() if len(im) else None, "read_recall_implicit_lenient": im["read_lenient"].mean() if len(im) else None,
                 "n_adversarial": len(adv), "n_adversarial_extracted": n_adv_typed}
            if kind == "pipeline":
                fe = ex[ex["found"] != "missed"]
                typed = fe[fe["class_exact"].notna()] if "class_exact" in fe else fe.iloc[0:0]
                s.update({"extraction_recall_explicit_exact": (ex["found"] == "exact").mean() if len(ex) else None,
                          "extraction_recall_explicit_partial": (ex["found"] != "missed").mean() if len(ex) else None,
                          "extraction_recall_implicit_partial": (im["found"] != "missed").mean() if len(im) else None,
                          "n_found_explicit": len(fe), "n_typed_explicit": len(typed),
                          "typing_accuracy_exact": typed["class_exact"].mean() if len(typed) else None,
                          "typing_accuracy_within_1": typed["class_within_1"].mean() if len(typed) else None,
                          "n_mentions": len(ents)})
            summaries.append(s)

    out_dir.mkdir(parents=True, exist_ok=True)
    per = pd.DataFrame(rows); per.to_csv(out_dir / "per_term.csv", index=False, encoding="utf-8")
    summ = pd.DataFrame(summaries); summ.to_csv(out_dir / "summary.csv", index=False, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps({"gold": str(GOLD.relative_to(ROOT)).replace("\\", "/"), "runs": json.loads(summ.to_json(orient="records"))}, indent=2), encoding="utf-8")
    cols = ["deck", "run", "n_explicit", "n_implicit", "read_recall_explicit_exact", "read_recall_explicit_lenient", "read_recall_implicit_lenient",
            "extraction_recall_explicit_partial", "n_typed_explicit", "typing_accuracy_exact", "typing_accuracy_within_1"]
    pd.set_option("display.width", 220)
    print(summ[[c for c in cols if c in summ.columns]].round(3).to_string(index=False))
    print(f"\nwritten: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
