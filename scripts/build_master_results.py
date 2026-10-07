"""Build the master results workbook: every result table of the BiomedCAT paper, one tab each, read
directly from the automated outputs (pipeline JSON and logs, evaluation reports, context-graph
summaries, presence and case files), with a first tab describing where each tab comes from.

Nothing in this file is typed in by hand: every cell is read from a source file listed in the
metadata tab, and tests/test_master_results.py rebuilds every tab and compares it cell by cell with
the workbook, plus direct regex checks of headline values against the raw logs.

Excluded on purpose (they belong to the benchmark-generation paper): the generator / verifier
statistics of the synthetic benchmark, its stratification design, the label-audit error
statistics and the audit sampling tables. The typing accuracy of the systems on that benchmark
(original and expert-audited labels) is a BiomedCAT result and is included.

    uv run python scripts/build_master_results.py [--out data/eval/master_results.xlsx] [--no-copy]

Layout of the repositories (relative paths in the metadata tab are relative to the folder that
contains them): BiomedCAT/ (pipeline, documents, graphs), KG_nodes_eval/ (typing benchmark
runs), manuscripts/biomedcat_manuscript/ (figures, material).
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import json
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

B = Path(__file__).resolve().parents[1]          # BiomedCAT
ROOT = B.parent                                  # GitHub/
K = ROOT / "KG_nodes_eval"
M = ROOT / "manuscripts" / "biomedcat_manuscript"
sys.path.insert(0, str(K))

PROFILES = ["uniform", "biochemical_actions", "clinical_mechanisms", "genetic_determinants",
            "pharmacological_intervention", "epidemiological_risk"]
DOCS = {"FSHD": "FSHD", "Maladies_musculaires": "Maladies musculaires"}

def discover_runs() -> list[tuple[str, Path, Path]]:
    """The published document runs (agent ReAct): every output/_final_*/ folder holding a pipeline
    JSON. The FSHD runs and the muscle-diseases run (chain I, 9-10 Sept 2026) come first; the
    decks of chain docs extra (3 Oct 2026) follow in alphabetical order. The log is the pipeline
    log next to the JSON, or the chain stdout for the muscle-diseases run, whose log was not kept."""
    fixed = [("FSHD run 1", B / "output/_final_run1/FSHD_BiomedCAT.json", B / "output/_final_run1/FSHD_BiomedCAT.log"),
             ("FSHD run 2", B / "output/_final_run2/FSHD_BiomedCAT.json", B / "output/_final_run2/FSHD_BiomedCAT.log"),
             ("Maladies musculaires", B / "output/_final_mm/Maladies musculaires_BiomedCAT.json", B / "output/chain_I/B4_mm.log")]
    runs = [r for r in fixed if r[1].is_file()]
    known = {r[1].parent for r in runs}
    for d in sorted((B / "output").glob("_final_*")):
        if d in known or not d.is_dir():
            continue
        for js in sorted(d.glob("*_BiomedCAT.json")):
            log = js.with_suffix(".log")
            runs.append((js.stem.replace("_BiomedCAT", ""), js, log if log.is_file() else js))
    return runs


RUNS = discover_runs()


# ----------------------------------------------------------------------------- helpers
def rel(p: Path | str) -> str:
    """Path relative to the GitHub folder, with the case the repositories use (no resolve(): on
    Windows it would return the case stored by the file system, e.g. 'Output')."""
    p = Path(p)
    try:
        return Path(os.path.relpath(str(p), str(ROOT))).as_posix()
    except ValueError:
        return p.as_posix()


def load_json(p: Path):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def mtime(p: Path) -> str:
    return datetime.fromtimestamp(Path(p).stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def stage_lines(log: Path) -> dict:
    """Stage summaries written by biomedcat.pipeline: OCR / NER / Norm / Context graph done lines."""
    txt = Path(log).read_text(encoding="utf-8", errors="ignore")
    out = {}
    m = re.search(r"OCR done: (\d+) slide\(s\) in ([\d.]+)s", txt)
    if m:
        out["ocr_slides"], out["ocr_seconds"] = int(m.group(1)), float(m.group(2))
    m = re.search(r"NER done: (\d+) entit\(y/ies\) in ([\d.]+) s", txt)
    if m:
        out["ner_entities"], out["ner_seconds"] = int(m.group(1)), float(m.group(2))
    m = re.search(r"Norm done: (\d+)/(\d+) linked in ([\d.]+)s", txt)
    if m:
        out["norm_linked"], out["norm_total"], out["norm_seconds"] = int(m.group(1)), int(m.group(2)), float(m.group(3))
    m = re.search(r"Context graph done: (\d+) nodes, (\d+) edges in ([\d.]+)s", txt)
    if m:
        out["graph_nodes"], out["graph_edges"], out["graph_seconds"] = int(m.group(1)), int(m.group(2)), float(m.group(3))
    # retriever / segmentation configuration as the run itself logged it (the published
    # configuration is: scispaCy not loaded, no BM25 leg, no lexical table)
    out["scispacy_loaded"] = "Successfully loaded scispaCy model" in txt
    out["bm25_leg"] = "Initializing BM25 index" in txt
    out["lexical_table"] = "Lexical table ready" in txt
    stamps = re.findall(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", txt, re.M)
    if stamps:
        out["log_first_timestamp"], out["log_last_timestamp"] = stamps[0], stamps[-1]
    return out


def flatten_report(d: dict, prefix: str = "") -> dict:
    """Scalars of a nested report dict (one level of nesting for 'relation'-like dicts)."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[prefix + k] = v
        elif isinstance(v, dict) and all(isinstance(x, (int, float, str, bool)) or x is None for x in v.values()):
            for kk, vv in v.items():
                out[f"{prefix}{k}.{kk}"] = vv
    return out


def report_row(dirname: str, kind_hint: str | None = None) -> dict | None:
    """One row per KG_nodes_eval result directory: the block that carries its headline metrics."""
    p = K / "results" / dirname / "report.json"
    if not p.is_file():
        return None
    r = load_json(p)
    row = {"result_dir": rel(p.parent), "report_file": rel(p), "report_date": mtime(p),
           "benchmark": r.get("benchmark", ""), "model": r.get("model", ""), "top_k": r.get("top_k", "")}
    if "agent" in r and r["agent"]:
        row["kind"] = "agent"; row.update(flatten_report(r["agent"], "agent."))
        if "rag" in r and isinstance(r["rag"], dict) and "all" in r["rag"]:
            row.update(flatten_report(r["rag"]["all"], "rag."))
    elif "rag" in r and isinstance(r["rag"], dict) and "all" in r["rag"]:
        row["kind"] = "retriever"; row.update(flatten_report(r["rag"]["all"], "rag."))
        row["rag.seconds"] = r["rag"].get("seconds", "")
    elif "all" in r:
        row["kind"] = kind_hint or "api"; row.update(flatten_report(r["all"], "all."))
        for k in ("seconds", "batch", "api_calls", "circular_with_benchmark", "service", "service_version", "limit"):
            if k in r:
                row[k] = r[k]
    elif "accuracy" in r:
        row["kind"] = kind_hint or "oneshot"; row.update(flatten_report(r))
    else:
        row["kind"] = kind_hint or "other"; row.update(flatten_report(r))
    return row


@contextlib.contextmanager
def cwd(path: Path):
    old = os.getcwd(); os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


# ----------------------------------------------------------------------------- tabs
def tab_pipeline_runs():
    rows = []
    for name, js, log in RUNS:
        d = load_json(js); run = d["run"]; ents = d["entities"]
        row = {"run": name, "document": run["file"], "run_timestamp_utc": run["timestamp"], "output_json": rel(js), "log": rel(log),
               "ocr_model": run["models"]["ocr"], "ner_model": run["models"]["ner"], "norm_model": run["models"]["norm"],
               "embedding_model": run["models"]["embeddings"], "resolvers_offline": run["resolvers"]["offline"],
               "resolver_api_limit": run["resolvers"]["api_limit"], "profile_at_run": Path(run["profile"]).stem,
               "elapsed_s": run["elapsed_s"], "n_slides": len(d["slides"]), "n_entities": len(ents),
               "n_distinct_mentions": len({(e["text"], e.get("page")) for e in ents}),
               "n_typed": sum(1 for e in ents if e.get("type") not in (None, "", "NONE")),
               "n_linked_kg2c": sum(1 for e in ents if e.get("kg2c_id")),
               "n_with_curie": sum(1 for e in ents if e.get("curie")),
               "context_graph_nodes_at_run": d["context_graph"].get("n_nodes"), "context_graph_edges_at_run": d["context_graph"].get("n_edges")}
        row.update(stage_lines(log))
        row.update(chain_env(name))
        rows.append(row)
    return pd.DataFrame(rows)


EXTRA_DECKS = {"VoieB": "output/chain_docs_extra.log", "VoieC": "output/chain_docs_extra.log", "VoiesA-C": "output/chain_docs_extra.log",
               "FSHD1multiscales": "output/chain_docs_extra.log", "VoieA": "output/chain_docs_redo.log"}


def chain_env(run_name: str) -> dict:
    """The environment the chain exported for the run: from the chain log's start line when the
    chain echoed it (chain docs extra), else from the export line of the chain script (chain docs
    redo, whose log does not repeat it). The lexical legs did not exist for the September runs."""
    if run_name not in EXTRA_DECKS:
        return {"chain_log": "", "chain_env": "", "chain_env_source": "", "lexical_leg_weight": "n/a (leg did not exist)"}
    log = B / EXTRA_DECKS[run_name]
    script = B / "scripts" / (Path(EXTRA_DECKS[run_name]).stem + ".sh")
    env, source = "", ""
    if log.is_file():
        m = re.search(r"^=== .* start .*?(AGENT_MODE=.*)$", log.read_text(encoding="utf-8", errors="ignore"), re.M)
        if m:
            env, source = m.group(1).strip(), rel(log)
    if not env and script.is_file():
        m = re.search(r"^export (AGENT_MODE=.*)$", script.read_text(encoding="utf-8", errors="ignore"), re.M)
        if m:
            env, source = m.group(1).strip(), rel(script)
    w = re.search(r"RAG_LEXICAL_WEIGHT=(\S+)", env)
    return {"chain_log": rel(log) if log.is_file() else "", "chain_env": env, "chain_env_source": source, "lexical_leg_weight": w.group(1) if w else ""}


def tab_ocr_reading():
    rows = []
    for name, js, log in RUNS:
        d = load_json(js); st = stage_lines(log)
        metrics = js.parent / "eval_v2" / "metrics.json"
        rec = load_json(metrics)["positives"] if metrics.is_file() else {}
        for s in d["slides"]:
            rows.append({"run": name, "document": d["run"]["file"], "page": s["page"], "reading_chars": len(s["text"]),
                         "reading_words": len(s["text"].split()), "ocr_model": d["run"]["models"]["ocr"],
                         "ocr_seconds_document": st.get("ocr_seconds"), "ocr_slides_document": st.get("ocr_slides"),
                         "ocr_seconds_per_slide": (st["ocr_seconds"] / st["ocr_slides"]) if st.get("ocr_slides") else None,
                         "gold_term_recall_exact (document)": rec.get("recall_exact"), "gold_term_recall_partial (document)": rec.get("recall_partial"),
                         "gold_file": load_json(metrics)["gold"] if metrics.is_file() else ""})
    return pd.DataFrame(rows)


def tab_document_gold_metrics():
    rows = []
    for name, js, log in RUNS:
        for sub, gold_version in (("eval", "v1"), ("eval_v2", "v2")):
            p = js.parent / sub / "metrics.json"
            if not p.is_file():
                continue
            m = load_json(p)
            row = {"run": name, "gold_version": gold_version, "gold_file": m.get("gold", ""), "metrics_file": rel(p), "metrics_date": mtime(p)}
            row.update(flatten_report(m["positives"], "positives."))
            if "creativity" in m:
                row.update({f"creativity.{k}": v for k, v in m["creativity"].items() if not isinstance(v, (list, dict))})
            if "adversarial" in m:
                row.update({f"adversarial.{k}": v for k, v in m["adversarial"].items() if not isinstance(v, (list, dict))})
            rows.append(row)
    return pd.DataFrame(rows)


def tab_adversarial_suite():
    p = B / "output/_final_run1/adversarial_run_v2.json"
    d = load_json(p); rows = []
    for group in ("positives", "adversarial"):
        for t in d[group]:
            rows.append({"group": group, "id": t["id"], "term": t["term"], "n_types": len(t.get("types", [])), "types": "; ".join(map(str, t.get("types", []))),
                         "n_curies": len(t.get("curies", [])), "curies": "; ".join(map(str, t.get("curies", []))),
                         "typed": bool(t.get("types")), "linked": bool(t.get("curies")),
                         "ner_model": d["models"]["ner"], "sanitizer_model": d["models"]["sanitizer"], "gold_file": d["gold"], "elapsed_s_suite": d["elapsed_s"], "source": rel(p)})
    return pd.DataFrame(rows)


LINKING_REPORTS = ["linking_predtypes_final", "linking_goldtypes_v2", "linking_goldtypes_nameres_es", "linking_predtypes"]


def tab_linking():
    rows = []
    for name in LINKING_REPORTS:
        p = B / "data/eval" / name / "report.json"
        if not p.is_file():
            continue
        r = load_json(p)
        for cond, v in r["conditions"].items():
            row = {"report": name, "report_file": rel(p), "report_date": mtime(p), "gold": r.get("gold", ""), "type_source": r.get("type_source", ""),
                   "n_terms": r.get("n_terms"), "n_gold_unknown_to_nodenorm": len(r.get("gold_unknown_to_nodenorm", [])), "condition": cond}
            row.update(flatten_report(v)); rows.append(row)
    return pd.DataFrame(rows)


def tab_linking_services():
    p = B / "data/eval/linking_services.json"; d = load_json(p)
    return pd.DataFrame([{"service": s, **v, "n_linkable": d["n_linkable"], "source": rel(p)} for s, v in d["summary"].items()])


def tab_existence_gate():
    p = B / "data/eval/gate_modes.json"; d = load_json(p); rows = []
    for mode, v in d["modes"].items():
        rows.append({"mode": mode, "gate_llm_model": d["model"], "es_url": d["es_url"], "n_positives": d["n_positives"], "n_adversarial": d["n_adversarial"],
                     "true_rejections": v["true_rejections"], "false_rejections": v["false_rejections"],
                     "true_rejection_rate": v["true_rejections"] / d["n_adversarial"], "false_rejection_rate": v["false_rejections"] / d["n_positives"], "source": rel(p)})
    return pd.DataFrame(rows)


PUBLIC_DIRS = ["rag_v1", "rag_v3_public", "rag_v4_exemplars_public", "rag_v4_bm25_public", "rag_v4_lexical_public", "rag_v4_lexical_kg2c_public",
               "rag_v4_sapbert_public", "rag_v5_kg2c_exemplars_public", "rag_embed_BAAI_bge-large-en-v1.5_public",
               "rag_embed_NeuML_pubmedbert-base-embeddings_public", "rag_embed__public", "rag_embed_pritamdeka_S-PubMedBert-MS-MARCO_public",
               "rag_embed_sentence-transformers_all-mpnet-base-v2_public", "nameres_v1", "nameres_es_v1",
               "agent_v4_all", "agent_final_all", "agent_final_all_timing", "agent_v4_top15_all", "agent_decide_all_run1", "agent_decide_all_run2",
               "claude_opus5_public", "claude_opus5_b1_public", "llm_deepseek-v4-flash_b1_public", "llm_gemma-4-31b-it_b1_public",
               "llm_mistral-small-3-2-24b-instruct-2506_b1_public", "llm_gpt-oss-120b_b1_public", "llm_gemma31b_v1", "llm_mistral_v1", "llm_gptoss_v1"]
REAL_DIRS = ["rag_v4_kg2c_real", "rag_ablation_base_dense_exemplars_kg2c_real", "rag_ablation_plus_bm25_kg2c_real", "rag_ablation_plus_lexical_classes_kg2c_real",
             "rag_ablation_bm25_plus_lexical_kg2c_real", "rag_ablation_leak_kg2c_names_kg2c_real", "agent_v4_kg2c_real", "agent_decide_kg2c_real"]
ISOLATION_DIRS = ["rag_v4_baseline_fshd_v2", "agent_v4_fshd_v2_terms", "agent_v4_lexical_fshd_v2", "agent_decide_fshd_v2", "agent_decide_raw_fshd_v2",
                  "claude_opus5_b1_fshd_v2", "llm_deepseek_b1_fshd_v2", "llm_gemma31b_b1_fshd_v2", "llm_mistral_b1_fshd_v2"]


def _tab_reports(dirs):
    rows = [r for r in (report_row(d, "nameres" if d.startswith("nameres") else None) for d in dirs) if r]
    return pd.DataFrame(rows)


def tab_typing_public_benchmark():
    df = _tab_reports(PUBLIC_DIRS)
    p = K / "results/desc_vs_term_v4/report.json"
    if p.is_file():
        r = load_json(p)
        extra = [{"result_dir": rel(p.parent), "report_file": rel(p), "report_date": mtime(p), "benchmark": r["benchmark"], "model": r["model"], "kind": "retriever (query variant)",
                  "query_variant": q, **{f"rag.{k}": v for k, v in r[q].items()}} for q in ("term", "description", "fused")]
        df = pd.concat([df, pd.DataFrame(extra)], ignore_index=True)
    return df


def tab_typing_real_kg2c():
    return _tab_reports(REAL_DIRS)


def tab_document_terms_isolation():
    return _tab_reports(ISOLATION_DIRS)


def tab_retriever_experiments():
    rows = []
    p = K / "results/branch_agg_v4/report.json"
    if p.is_file():
        r = load_json(p)
        for variant in ("plain", "branch_rrf", "branch_inv"):
            rows.append({"experiment": "branch aggregation", "variant": variant, "report_file": rel(p), "report_date": mtime(p), "n": r["n"], "top_k": r["top_k"], "branch_depth": r["branch_depth"], **r[variant]})
    for name in ("depth_rerank_v4", "depth_rerank_v4_inv"):
        p = K / "results" / name / "report.json"
        if p.is_file():
            r = load_json(p)
            for g in r["grid"]:
                rows.append({"experiment": name, "variant": f"base={g.get('base')}, power={g.get('power')}", "report_file": rel(p), "report_date": mtime(p), "n": r["n"], **g})
    p = K / "results/gaussian_nb_v4/report.json"
    if p.is_file():
        r = load_json(p)
        for block in ("baselines", "best_public"):
            for g in r[block]:
                rows.append({"experiment": f"gaussian_nb ({block})", "variant": f"{g.get('config')} {g.get('score')} {g.get('variant')} {g.get('restrict')}", "report_file": rel(p), "report_date": mtime(p), "n": r["n"], **{k: v for k, v in g.items() if k not in ("config", "score", "variant", "restrict")}})
    for name in ("oneshot_e4b_tree_s300", "oneshot_e4b_treedefs_s300"):
        row = report_row(name, "oneshot")
        if row:
            rows.append({"experiment": "one-shot prompting with the class tree", "variant": name, **row})
    return pd.DataFrame(rows)


def _gold_v11():
    """Gold v1.1 of the audited terms (rescore_audit_full.build_gold), evaluated with cwd = KG_nodes_eval."""
    import rescore_audit_full as raf
    from rescore_audit_s300 import NESTED, biolink_paths
    with cwd(K):
        paths = biolink_paths(NESTED)
        gold = raf.build_gold(paths)
    return gold


def _verdicts(key: str) -> pd.Series | None:
    from rescore_audit_s300 import load_verdicts
    with cwd(K):
        return load_verdicts(key)


def tab_typing_ablations_300():
    from rescore_audit_s300 import CONFIGS, canon
    gold = _gold_v11().set_index("query")
    ref = pd.read_parquet(K / "results/agent_decide_raw_s300/agent_per_query.parquet")[["query", "expected_class", "form"]]
    g300 = gold.reindex(ref["query"])
    rows = []
    for key, system, desc, params, testfile, resfile in CONFIGS:
        v = _verdicts(key)
        if v is None:
            continue
        vc = v.reindex(ref["query"]).map(canon).fillna("").values
        orig = ref.expected_class.map(canon).values
        acc = np.array([c in s if isinstance(s, set) else False for c, s in zip(vc, g300["_acc"].values)])
        b = g300.biomedical.fillna(False).values.astype(bool)
        row = {"system": system, "description": desc, "main_parameters": params, "test_file_or_log": testfile, "result_file": resfile,
               "n_terms_300": int(len(ref)), "accuracy_original_labels_300": float(np.mean(vc == orig)),
               "accuracy_audited_v1.1_all_300": float(acc.mean()), "accuracy_audited_v1.1_biomedical_300": float(acc[b].mean()), "n_biomedical_300": int(b.sum()),
               "n_missing_verdict_300": int((vc == "").sum())}
        kind, rest = key.split(":", 1)
        rp = K / "results" / rest / "report.json" if kind in ("agent", "rag", "llm") else None
        if rp and rp.is_file():
            # the report's own headline metric, on the sample the run was scored on (300 terms for the
            # *_s300 runs, 1,075 public terms for the retrievers and API baselines, 1,356 for the full runs)
            r = load_json(rp); blk = r.get("agent") or (r.get("rag", {}).get("all") if isinstance(r.get("rag"), dict) else None) or r.get("all") or {}
            row["report_metric_own_sample"] = blk.get("accuracy", blk.get("hit@1")); row["report_metric_n"] = blk.get("n"); row["report_seconds"] = blk.get("seconds", r.get("seconds")); row["report_date"] = mtime(rp)
        else:
            f = rest.split(":")[0]; row["report_date"] = mtime(B / "output" / f)
        rows.append(row)
    return pd.DataFrame(rows)


def tab_typing_audited_gold():
    from rescore_audit_full import FULL_SYSTEMS, wilson
    from rescore_audit_s300 import canon
    gold = _gold_v11(); n = len(gold)
    rows = []
    for key, label in FULL_SYSTEMS:
        v = _verdicts(key)
        if v is None:
            continue
        vc = v.reindex(gold["query"]).map(canon).fillna("").values
        ok = np.array([c in s for c, s in zip(vc, gold["_acc"].values)])
        def acc(mask):
            m = np.asarray(mask, bool); return float(ok[m].mean()) if m.sum() else None
        row = {"system": label, "result_key": key, "n_terms": n, "accuracy_all": acc(np.ones(n, bool)), "ci95_half_width_all": wilson(ok.mean(), n),
               "accuracy_biomedical": acc(gold.biomedical.values), "n_biomedical": int(gold.biomedical.sum()),
               "accuracy_non_biomedical": acc(~gold.biomedical.values),
               "accuracy_ambiguous_terms_any_reading": acc((gold.gold_kind == "ambiguous").values), "n_ambiguous": int((gold.gold_kind == "ambiguous").sum()),
               "accuracy_personal_terms_class_or_none": acc((gold.gold_kind == "personal").values), "n_personal": int((gold.gold_kind == "personal").sum()),
               "none_rate_on_personal_terms": float(np.mean(vc[(gold.gold_kind == "personal").values] == "none")) if (gold.gold_kind == "personal").any() else None,
               "accuracy_ok_terms": acc((gold.gold_kind == "ok").values), "accuracy_relabelled_terms": acc((gold.gold_kind == "relabelled").values),
               "accuracy_not_an_entity_terms": acc((gold.gold_kind == "not an entity").values)}
        strata = {s: acc((gold.coarse_stratum == s).values) for s in sorted(gold.coarse_stratum.unique())}
        row.update({f"accuracy_stratum_{s}": v for s, v in strata.items()}); row["accuracy_macro_average_strata"] = float(np.mean(list(strata.values())))
        for f in sorted(gold.form.unique()):
            row[f"accuracy_form_{f}"] = acc((gold.form == f).values)
        rows.append(row)
    return pd.DataFrame(rows)


def tab_context_graphs():
    rows = []
    for d in sorted((B / "data/context_graph").iterdir()):
        s = d / "summary.json"
        if not d.is_dir() or not s.is_file() or d.name.startswith("_"):
            continue
        j = load_json(s)
        doc = next((k for k in DOCS if d.name.startswith(k + "_")), d.name)
        row = {"graph_dir": rel(d), "document": DOCS.get(doc, doc), "profile": d.name[len(doc) + 1:] if d.name.startswith(doc + "_") else "", "profile_label": j.get("profile", ""),
               "summary_file": rel(s), "summary_date": mtime(s)}
        for k in ("n_seeds", "n_seeds_in_graph", "n_nodes", "n_edges", "n_reached", "vocabulary_expansion", "diameter", "n_components", "seed_pairs_connected", "seconds"):
            row[k] = j.get(k)
        for h in j.get("per_hop", []):
            row[f"hop{h['hop']}_nodes_reached"] = h.get("nodes_reached"); row[f"hop{h['hop']}_via_representation_edges"] = h.get("via_representation_edges")
        for k, v in (j.get("weighting") or {}).items():
            if isinstance(v, (int, float, str, bool)):
                row[f"weighting.{k}"] = v
        row["params"] = json.dumps(j.get("params", {}), ensure_ascii=False)
        rows.append(row)
    return pd.DataFrame(rows)


def tab_context_graph_categories():
    rows = []
    for d in sorted((B / "data/context_graph").iterdir()):
        s = d / "summary.json"
        if not d.is_dir() or not s.is_file() or d.name.startswith("_"):
            continue
        j = load_json(s)
        doc = next((k for k in DOCS if d.name.startswith(k + "_")), d.name)
        for cat, n in (j.get("categories") or {}).items():
            rows.append({"document": DOCS.get(doc, doc), "profile": d.name[len(doc) + 1:] if d.name.startswith(doc + "_") else "", "category": cat, "n_nodes": n, "summary_file": rel(s)})
    return pd.DataFrame(rows)


def tab_use_case_presence():
    p = B / "data/eval/fshd_presence_final.json"
    return pd.DataFrame([{**r, "source": rel(p)} for r in load_json(p)])


def tab_use_case_dux4():
    p = B / "data/eval/fshd_dux4_case.json"; d = load_json(p); rows = []
    for pr in d["profiles"]:
        base = {"document": d["doc"], "profile": pr["profile"], "category": d["category"], "category_size": pr["category_size"], "n_mechanism_nodes_found": pr["n_found"],
                "best_rank": pr["best_rank"], "worst_rank": pr["worst_rank"], "source": rel(p)}
        if pr["hits"]:
            for h in pr["hits"]:
                rows.append({**base, **{f"node_{k}": v for k, v in h.items()}})
        else:
            rows.append(base)
    return pd.DataFrame(rows)


def tab_use_case_drugs():
    p = B / "data/eval/fshd_drugs_presence.json"
    return pd.DataFrame([{**r, "source": rel(p)} for r in load_json(p)])


def tab_use_case_repurposing():
    p = B / "data/eval/fshd_repurposing_candidates.json"
    return pd.DataFrame([{**{k: ("; ".join(map(str, v)) if isinstance(v, list) else v) for k, v in r.items()}, "source": rel(p)} for r in load_json(p)])


def tab_corpus_comparison():
    runs = tab_pipeline_runs(); graphs = tab_context_graphs(); rows = []
    for _, r in runs.iterrows():
        d = load_json(ROOT / r["output_json"])
        ents = d["entities"]; cats = pd.Series([e.get("type") for e in ents if e.get("type")]).value_counts()
        row = {"run": r["run"], "document": r["document"], "n_slides": r["n_slides"], "reading_chars": sum(len(s["text"]) for s in d["slides"]),
               "n_entities": r["n_entities"], "n_distinct_mentions": r["n_distinct_mentions"], "n_typed": r["n_typed"], "n_linked_kg2c": r["n_linked_kg2c"],
               "linked_rate": r["n_linked_kg2c"] / r["n_entities"] if r["n_entities"] else None, "n_distinct_types": int(cats.size),
               "top_types": "; ".join(f"{k} ({v})" for k, v in cats.head(8).items()),
               "ocr_seconds": r.get("ocr_seconds"), "ner_seconds": r.get("ner_seconds"), "norm_seconds": r.get("norm_seconds"), "elapsed_s": r["elapsed_s"]}
        docname = "FSHD" if r["document"].startswith("FSHD") else "Maladies musculaires"
        for prof in PROFILES:
            g = graphs[(graphs.document == docname) & (graphs.profile == prof)]
            if len(g):
                row[f"graph_nodes_{prof}"] = int(g.n_nodes.iloc[0]); row[f"graph_edges_{prof}"] = int(g.n_edges.iloc[0]); row[f"seed_pairs_connected_{prof}"] = g.seed_pairs_connected.iloc[0]
        rows.append(row)
    return pd.DataFrame(rows)


def tab_reproducibility():
    from rescore_audit_s300 import canon
    rows = []
    def agent(d):
        p = K / "results" / d / "agent_per_query.parquet"; return pd.read_parquet(p).set_index("query")["agent_verdict"].map(canon) if p.is_file() else None
    pairs = [("_determinism_A", "_determinism_A2", "50 terms, seed 0, model unloaded per call"), ("_determinism_B", "_determinism_B2", "50 terms, seed 0, model resident (keep_alive 10 min)"),
             ("_determinism_A", "_determinism_C", "50 terms, seed 0 vs no seed"), ("agent_decide_all_run1", "agent_decide_all_run2", "1,356 terms, decide mode, unloaded per call"),
             ("agent_v4_all", "agent_final_all", "1,356 terms, ReAct, 5 Sept vs 10 Sept run (resident model, machine load)")]
    for a, b, desc in pairs:
        va, vb = agent(a), agent(b)
        if va is None or vb is None:
            continue
        common = va.index.intersection(vb.index)
        rows.append({"comparison": desc, "run_a": rel(K / "results" / a), "run_b": rel(K / "results" / b), "n_common_terms": int(len(common)),
                     "identical_verdicts": float((va.loc[common].values == vb.loc[common].values).mean()), "date_a": mtime(K / "results" / a / "agent_per_query.parquet"), "date_b": mtime(K / "results" / b / "agent_per_query.parquet")})
    a = load_json(RUNS[0][1])["entities"]; b = load_json(RUNS[1][1])["entities"]
    A = {(e["text"], e.get("type"), e.get("kg2c_id")) for e in a}; Bs = {(e["text"], e.get("type"), e.get("kg2c_id")) for e in b}
    rows.append({"comparison": "FSHD document, run 1 vs run 2 (entities: text, type, KG2c id)", "run_a": rel(RUNS[0][1]), "run_b": rel(RUNS[1][1]), "n_common_terms": len(A & Bs),
                 "identical_verdicts": 1.0 if A == Bs else len(A & Bs) / len(A | Bs), "date_a": mtime(RUNS[0][1]), "date_b": mtime(RUNS[1][1])})
    return pd.DataFrame(rows)


def tab_resources_timing():
    rows = []
    for _, r in tab_pipeline_runs().iterrows():
        for stage in ("ocr", "ner", "norm", "graph"):
            sec = r.get(f"{stage}_seconds")
            if sec is None or (isinstance(sec, float) and np.isnan(sec)):
                continue
            unit = {"ocr": r.get("ocr_slides"), "ner": r.get("ner_entities"), "norm": r.get("norm_total"), "graph": 1}[stage]
            rows.append({"scope": "document", "run": r["run"], "stage": stage, "seconds": sec, "units": unit, "unit_name": {"ocr": "slides", "ner": "entities", "norm": "entities", "graph": "graph"}[stage],
                         "seconds_per_unit": sec / unit if unit else None, "source": r["log"]})
        rows.append({"scope": "document", "run": r["run"], "stage": "total", "seconds": r["elapsed_s"], "units": r["n_slides"], "unit_name": "slides", "seconds_per_unit": r["elapsed_s"] / r["n_slides"], "source": r["output_json"]})
    for d, stage in (("agent_final_all_timing", "typing agent ReAct (gate llm, resolvers offline)"), ("agent_final_all", "typing agent ReAct (gate union, machine shared)"), ("agent_v4_all", "typing agent ReAct (5 Sept)"),
                     ("agent_decide_all_run1", "typing agent decide"), ("agent_v4_kg2c_real", "typing agent ReAct, real benchmark"), ("agent_decide_kg2c_real", "typing agent decide, real benchmark"),
                     ("rag_v4_exemplars_public", "retriever only"), ("claude_opus5_b1_public", "Claude Opus 5 API"), ("llm_deepseek-v4-flash_b1_public", "DeepSeek API")):
        p = K / "results" / d / "report.json"
        if not p.is_file():
            continue
        r = load_json(p); blk = r.get("agent") or (r.get("rag", {}).get("all") if isinstance(r.get("rag"), dict) else None) or r.get("all") or {}
        sec = blk.get("seconds", r.get("seconds", r.get("rag", {}).get("seconds") if isinstance(r.get("rag"), dict) else None)); n = blk.get("n")
        if sec is not None and n:
            rows.append({"scope": "benchmark", "run": d, "stage": stage, "seconds": sec, "units": n, "unit_name": "terms", "seconds_per_unit": sec / n, "source": rel(p)})
    return pd.DataFrame(rows)


def tab_resources_memory_kg():
    rows = []
    for name in ("memory.json", "stats.json"):
        p = B / "data/kg2c" / name
        if p.is_file():
            for k, v in flatten_report(load_json(p)).items():
                rows.append({"file": rel(p), "date": mtime(p), "measure": k, "value": v})
    return pd.DataFrame(rows)


def tab_resources_environment():
    p = B / "data/eval/environment.json"; e = load_json(p); rows = []
    for k, v in e["hardware"].items():
        rows.append({"section": "hardware", "item": k, "value": v, "source": rel(p)})
    rows.append({"section": "software", "item": "python", "value": e["software"]["python"], "source": rel(p)})
    rows.append({"section": "software", "item": "ollama", "value": e["software"]["ollama"], "source": rel(p)})
    for k, v in e["software"]["packages"].items():
        rows.append({"section": "package", "item": k, "value": v, "source": rel(p)})
    for m in e["ollama_models"]:
        rows.append({"section": "ollama model", "item": m["model"], "value": m["size"], "source": rel(p)})
    rows.append({"section": "recorded_at", "item": "utc", "value": e["recorded_at"], "source": rel(p)})
    return pd.DataFrame(rows)


def tab_resources_energy():
    e = load_json(B / "data/eval/environment.json")["energy_assumptions"]; rows = []
    t = tab_resources_timing()
    for _, r in t[(t.stage == "total") | (t.scope == "benchmark")].iterrows():
        for level, pw in e["power_w"].items():
            kwh = r["seconds"] * pw / 3.6e6
            row = {"scope": r["scope"], "run": r["run"], "stage": r["stage"], "seconds": r["seconds"], "power_assumption": level, "power_w": pw, "energy_kwh": kwh, "formula": e["formula"]}
            for region, g in e["carbon_intensity_g_per_kwh"].items():
                row[f"gCO2e ({region})"] = kwh * g
            rows.append(row)
    return pd.DataFrame(rows)


def tab_ocr_only_images():
    """Reading stage alone on image files (scripts/run_ocr_only.py): time, memory, full description."""
    rows = []
    for p in sorted((B / "output/ocr_only").glob("*_ocr.json")):
        r = load_json(p)
        for pg in r["pages"]:
            rows.append({"file": r["file"], "file_bytes": r["file_bytes"], "model": r["model"], "started_utc": r["started_utc"], "seconds": r["seconds"], "n_pages": r["n_pages"],
                         "page": pg["page"], "description_chars": pg["chars"], "description_words": pg["words"],
                         "process_rss_mb_before": r["process_rss_mb_before"], "process_rss_mb_after": r["process_rss_mb_after"],
                         "gpu_mib_before": r["gpu_mib_before"], "gpu_mib_peak_during": r["gpu_mib_peak_during"], "gpu_samples": r["gpu_samples"], "ollama_ps_during": r["ollama_ps_during"],
                         "temperature": r["temperature"], "max_new_tokens": r["max_new_tokens"], "keep_alive": r["keep_alive"], "seed": r["seed"],
                         "prompt_entity_scope": "; ".join(r["prompt_entity_scope"]), "prompt_reading_focus": r["prompt_reading_focus"], "profile": Path(r["profile"]).stem,
                         "description": pg["text"], "source": rel(p)})
    return pd.DataFrame(rows)


def tab_dataset_documents():
    """The author's inventory of the slide material (Dataset/documents_metadata.csv), verbatim."""
    p = B / "Dataset/documents_metadata.csv"
    df = pd.read_csv(p, dtype=str, keep_default_na=False)
    df["source"] = rel(p)
    return df


# ----------------------------------------------------------------------------- registry
REGISTRY = [
    # (tab, builder, content, generating scripts of the underlying runs, source files (relative), parameters)
    ("dataset_documents", tab_dataset_documents, "The slide material: one row per document with type, evaluation scope, pages, subject, biomedical scope, scale and component types (author's inventory).",
     "written by the author (Dataset/documents_metadata.csv)", [rel(B / "Dataset/documents_metadata.csv")], ""),
    ("pipeline_runs", tab_pipeline_runs, "The published document runs (agent ReAct, published retriever configuration: scispaCy not loaded, no BM25 leg, no lexical table; the three configuration columns are read from each run's log): models, stage durations and counts from the pipeline log, entity counts from the output JSON. FSHD x2 and Maladies musculaires from chain I (9-10 Sept); VoieA, VoieB, VoieC, VoiesA-C, FSHD1multiscales from chain docs extra (4 Oct, SCISPACY=0 RAG_BM25=0 RAG_LEXICAL_WEIGHT=0 RAG_KG2C_LEXICAL=0).",
     "biomedcat.pipeline (python -m biomedcat.pipeline <deck.pdf>), launched by archive/scripts/chain_I.sh (B2-B4) and scripts/chain_docs_extra.sh",
     [rel(p) for _, js, log in RUNS for p in (js, log)] + [rel(B / p) for p in sorted(set(EXTRA_DECKS.values())) if (B / p).is_file()] + [rel(B / "scripts" / (Path(p).stem + ".sh")) for p in sorted(set(EXTRA_DECKS.values())) if (B / "scripts" / (Path(p).stem + ".sh")).is_file()], "OCR qwen2.5vl:3b T=0 seed 0; NER gemma4:e4b-it-qat ReAct 3 steps top-k 10; linking NameRes+ARAX -> judge -> KG2c canonical -> NodeNorm (no conflation); gate union; profile biochemical_actions at run"),
    ("ocr_only_images", tab_ocr_only_images, "Reading stage alone on the two image examples (pathways.png, chemicals.png): time, process and GPU memory sampled during the call, Ollama residency, prompt scope, full description.",
     "scripts/run_ocr_only.py (scripts/chain_docs_extra.sh)", [rel(p) for p in sorted(glob.glob(str(B / "output/ocr_only/*_ocr.json")))] + ([rel(B / "output/ocr_only/ocr_only.log")] if (B / "output/ocr_only/ocr_only.log").is_file() else []),
     "qwen2.5vl:3b, description prompt with the profile's entity scope, T=0, seed 0, 8,192 new tokens, model unloaded after the call"),
    ("ocr_reading", tab_ocr_reading, "Reading stage per slide: characters and words of the description, OCR seconds per document and per slide, gold term recall of the document.",
     "biomedcat.stages.ocr via biomedcat.pipeline; recall from scripts/eval_gold.py score", [rel(js) for _, js, _ in RUNS] + [rel(log) for _, _, log in RUNS] + [rel(B / "output/_final_run1/eval_v2/metrics.json"), rel(B / "output/_final_run2/eval_v2/metrics.json")],
     "qwen2.5vl:3b, description prompt, T=0, seed 0, 4 pages rasterised per batch, one model call per image"),
    ("document_gold_metrics", tab_document_gold_metrics, "End-to-end scores of the FSHD runs against the expert gold (v1: 31 terms; v2: 32 terms + 34 adversarial): recall, precision, typing, linking, creativity, adversarial rates.",
     "scripts/eval_gold.py score --gold data/gold/fshd_slides_gold[_v2].json; scripts/run_adversarial.py", [rel(p) for p in glob.glob(str(B / "output/_final_run*/eval*/metrics.json"))] + [rel(B / "data/gold/fshd_slides_gold.json"), rel(B / "data/gold/fshd_slides_gold_v2.json")],
     "distinct (text, page) precision; linking on linkable terms = gold CURIE known to NodeNorm"),
    ("adversarial_suite", tab_adversarial_suite, "Adversarial suite (gold v2): every positive and invented term with the types and CURIEs the pipeline assigned in isolation.",
     "scripts/run_adversarial.py --gold data/gold/fshd_slides_gold_v2.json", [rel(B / "output/_final_run1/adversarial_run_v2.json")], "ReAct agent + linking on isolated terms; 33 positives, 34 adversarial"),
    ("linking", tab_linking, "Entity-linking ablation on the expert terms: every condition (NameRes / ARAX / typed / judge / canonical) with exact and clique accuracies, gold or predicted types.",
     "scripts/baseline_linking.py", [rel(B / "data/eval" / n / "report.json") for n in LINKING_REPORTS if (B / "data/eval" / n / "report.json").is_file()], "NodeNorm clique agreement without conflation; linkable = gold CURIE known to NodeNorm"),
    ("linking_services", tab_linking_services, "Name-resolution services compared on the linkable expert terms: top-1 / top-5 / top-20 / any hit.", "scripts/baseline_linking.py (services comparison)", [rel(B / "data/eval/linking_services.json")], ""),
    ("existence_gate", tab_existence_gate, "Existence gate modes (Elasticsearch, LLM, union) on the gold suite: true and false rejections.", "scripts/eval_gate_modes.py", [rel(B / "data/eval/gate_modes.json")], "gate LLM llama3:8b; ES NameRes lookup"),
    ("typing_public_benchmark", tab_typing_public_benchmark, "Typing systems on the synthetic benchmark (public 1,075 or all 1,356 terms, original labels): retrievers, name resolvers, BiomedCAT agent runs, API baselines. Report metrics as written by the evaluation scripts.",
     "KG_nodes_eval/evaluate_typing.py; baseline_nameres.py; baseline_llm_typing.py; baseline_claude_typing.py; measure_descriptions.py", [rel(K / "results" / d / "report.json") for d in PUBLIC_DIRS + ["desc_vs_term_v4"] if (K / "results" / d / "report.json").is_file()], "retriever RRF k=60; agent top-k 10; gate as recorded per run; seed 0, T=0 unless stated"),
    ("typing_real_kg2c", tab_typing_real_kg2c, "Typing systems on the real KG2c-mention benchmark (997 mentions, 52 classes): retriever ablations and agents.", "KG_nodes_eval/evaluate_typing.py --benchmark data/benchmark_kg2c_real.parquet (chain_J J1-J2, bm25_kg2c_real_test.sh, chain_V5)",
     [rel(K / "results" / d / "report.json") for d in REAL_DIRS if (K / "results" / d / "report.json").is_file()], "gate llm, resolvers offline for the agents"),
    ("document_terms_isolation", tab_document_terms_isolation, "The 32 FSHD gold-v2 terms typed in isolation by every system (retriever, agents, API models).", "KG_nodes_eval/evaluate_typing.py / baseline_*_typing.py --benchmark data/fshd_gold_v2_terms.parquet",
     [rel(K / "results" / d / "report.json") for d in ISOLATION_DIRS if (K / "results" / d / "report.json").is_file()], "gate off"),
    ("typing_ablations_300", tab_typing_ablations_300, "Every configuration measured on the 300 reference terms (public slice, seed 77): accuracy on the original labels and on the expert-audited labels (v1.1, any listed reading of an ambiguous term is correct).",
     "KG_nodes_eval/evaluate_typing.py --agent 300 --seed 77 (chains J-N, output/*_test.sh); standalone output/search_then_decide_test.py, output/decide_prompt_v2_test.py; KG_nodes_eval/rescore_audit_s300.py CONFIGS",
     [rel(K / "results/agent_decide_raw_s300/agent_per_query.parquet"), rel(K / "data/benchmark_audit_merged_v2.xlsx")], "reference ReAct control 0.410; threshold 5 points"),
    ("typing_audited_gold", tab_typing_audited_gold, "Systems with a verdict on every public term, scored on the 664 expert-audited terms (gold v1.1): all, biomedical, ambiguous, personal, strata and forms.",
     "KG_nodes_eval/rescore_audit_full.py (scoring rules) on the per-query outputs of the runs", [rel(K / "data/benchmark_audit_merged_v2.xlsx")] + [rel(K / "results" / k.split(":", 1)[1] / ("rag_per_query.parquet" if k.startswith("rag") else "agent_per_query.parquet" if k.startswith("agent") else "per_query.parquet")) for k, _ in __import__("rescore_audit_full").FULL_SYSTEMS],
     "ok -> original; wrong -> expert class (NONE = not an entity); ambiguous -> any listed reading; personal -> class or NONE"),
    ("retriever_experiments", tab_retriever_experiments, "Retriever-side experiments: branch aggregation, depth re-ranking grid, Gaussian naive-Bayes prior, one-shot prompting with the class tree.",
     "KG_nodes_eval/branch_aggregation.py, depth_weighted_rerank.py, gaussian_nb.py, baseline_oneshot_ollama.py", [rel(p) for p in [K / "results/branch_agg_v4/report.json", K / "results/depth_rerank_v4/report.json", K / "results/depth_rerank_v4_inv/report.json", K / "results/gaussian_nb_v4/report.json", K / "results/oneshot_e4b_tree_s300/report.json", K / "results/oneshot_e4b_treedefs_s300/report.json"] if p.is_file()], ""),
    ("context_graphs", tab_context_graphs, "The 12 context graphs (2 documents x 6 profiles) plus the reference 1-hop graphs: sizes, seeds, connectivity, expansion, hop counts, weighting.", "biomedcat.stages.context_graph via scripts/rebuild_graphs_final.sh", [rel(p) for p in sorted(glob.glob(str(B / "data/context_graph/*/summary.json"))) if "/_" not in rel(p)], "KG2c 2.10.1 filtered (concept genes removed); profile-derived predicate weights; representation edges identity-only; cap 300 per category; 2 hops"),
    ("context_graph_categories", tab_context_graph_categories, "Biolink category composition of every context graph (long format).", "biomedcat.stages.context_graph via scripts/rebuild_graphs_final.sh", [rel(p) for p in sorted(glob.glob(str(B / "data/context_graph/*/summary.json"))) if "/_" not in rel(p)], ""),
    ("use_case_presence", tab_use_case_presence, "Presence of the FSHD reference drugs and genes in every graph (in graph, reached, rank, coverage, hop).", "scripts/graph_presence.py --doc FSHD --extra <genes> --out data/eval/fshd_presence_final", [rel(B / "data/eval/fshd_presence_final.json")], ""),
    ("use_case_dux4", tab_use_case_dux4, "DUX4 target-activation mechanism (6 Reactome nodes): rank within the kept MolecularActivity category of each FSHD graph.", "scripts/fshd_dux4_mechanism_case.py --out data/eval/fshd_dux4_case.json", [rel(B / "data/eval/fshd_dux4_case.json")], ""),
    ("use_case_drugs", tab_use_case_drugs, "FSHD drugs (investigational and approved) and their presence in the profile graphs and the 1-hop references (Venn input).", "scripts/fshd_drugs_venn.py", [rel(B / "data/eval/fshd_drugs_presence.json")], ""),
    ("use_case_repurposing", tab_use_case_repurposing, "Repurposing candidates for FSHD and their presence in the graphs.", "data/eval builder (chain D/E, 5 Sept)", [rel(B / "data/eval/fshd_repurposing_candidates.json")], ""),
    ("corpus_comparison", tab_corpus_comparison, "The two slide decks side by side: reading size, entities, typing and linking counts, stage times, graph sizes per profile.", "derived from pipeline_runs and context_graphs", [rel(js) for _, js, _ in RUNS] + [rel(p) for p in sorted(glob.glob(str(B / "data/context_graph/*/summary.json"))) if "/_" not in rel(p)], ""),
    ("reproducibility", tab_reproducibility, "Run-to-run identity of verdicts: determinism tests (seed, resident model), the two decide runs, the two ReAct runs, the two FSHD document runs.", "KG_nodes_eval/evaluate_typing.py (archive/output/determinism_test*.sh, chain_N); biomedcat.pipeline",
     [rel(K / "results" / d / "agent_per_query.parquet") for d in ("_determinism_A", "_determinism_A2", "_determinism_B", "_determinism_B2", "_determinism_C", "agent_decide_all_run1", "agent_decide_all_run2", "agent_v4_all", "agent_final_all") if (K / "results" / d / "agent_per_query.parquet").is_file()] + [rel(RUNS[0][1]), rel(RUNS[1][1])], ""),
    ("resources_timing", tab_resources_timing, "Wall-clock time per stage and per unit (document runs) and per term (benchmark runs).", "pipeline logs; evaluation reports", [rel(log) for _, _, log in RUNS] + [rel(js) for _, js, _ in RUNS], ""),
    ("resources_memory_kg", tab_resources_memory_kg, "Memory cost of the filtered KG2c graph (igraph resident vs DuckDB out-of-core) and KG statistics.", "scripts/measure_kg_memory.py; scripts/build_kg2c_parquet.py", [rel(p) for p in (B / "data/kg2c/memory.json", B / "data/kg2c/stats.json") if p.is_file()], ""),
    ("resources_environment", tab_resources_environment, "Hardware, operating system, Python and package versions, Ollama version and local model sizes.", "scripts/record_environment.py", [rel(B / "data/eval/environment.json")], ""),
    ("resources_energy", tab_resources_energy, "Energy and carbon estimate per document run and per benchmark run: seconds x assumed power (low / high) / 3.6e6, x grid intensity.", "derived from resources_timing and data/eval/environment.json (assumptions documented there)", [rel(B / "data/eval/environment.json")] + [rel(log) for _, _, log in RUNS], "power low = 60 W, high = 200 W; intensity France 32, EU-27 242 gCO2e/kWh"),
]


def build_all() -> dict[str, pd.DataFrame]:
    tabs, meta = {}, []
    for name, fn, content, gen, sources, params in REGISTRY:
        try:
            df = fn()
        except Exception as e:  # a missing source: recorded in the metadata, the tab is skipped
            df = None; status = f"FAILED: {type(e).__name__}: {e}"
        else:
            status = "ok"
        existing = [s for s in sources if (ROOT / s).exists()]
        dates = [mtime(ROOT / s) for s in existing]
        meta.append({"tab": name, "status": status, "n_rows": 0 if df is None else len(df), "content": content, "generating_scripts": gen,
                     "source_files": "; ".join(sources), "n_sources": len(sources), "n_sources_present": len(existing),
                     "sources_earliest": min(dates) if dates else "", "sources_latest": max(dates) if dates else "", "parameters": params,
                     "builder": f"scripts/build_master_results.py::{fn.__name__}"})
        if df is not None:
            tabs[name] = df
    return {"metadata": pd.DataFrame(meta), **tabs}


def write_workbook(tabs: dict[str, pd.DataFrame], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        for name, df in tabs.items():
            df.to_excel(w, sheet_name=name[:31], index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(B / "data/eval/master_results.xlsx"))
    ap.add_argument("--no-copy", action="store_true", help="do not copy the workbook to the manuscript material folder")
    args = ap.parse_args()
    tabs = build_all()
    write_workbook(tabs, Path(args.out))
    meta = tabs["metadata"]
    pd.set_option("display.width", 200)
    print(meta[["tab", "status", "n_rows", "n_sources_present", "sources_latest"]].to_string(index=False))
    print(f"-> {args.out}")
    if not args.no_copy and (M / "material").is_dir():
        shutil.copy(args.out, M / "material" / Path(args.out).name); print(f"-> {M / 'material' / Path(args.out).name}")


if __name__ == "__main__":
    main()
