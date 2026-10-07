"""The master results workbook (data/eval/master_results.xlsx) shows exactly the data of the
automated outputs.

Two layers of verification:
  1. every tab is rebuilt from the source files by scripts/build_master_results.py and compared
     cell by cell with the workbook (file-date columns excluded: they describe the sources, they
     are not results);
  2. headline cells are re-read here directly from the raw logs, JSON reports and summaries with
     independent code (regular expressions on the log lines, plain json), so that a bug in the
     builder cannot hide behind its own rebuild.
Also: every source file listed in the metadata tab exists, and nothing from the benchmark-
generation paper (generator, verifier, stratification, audit statistics) enters the workbook.

    uv run python -m pytest tests/test_master_results.py -q
"""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

B = Path(__file__).resolve().parents[1]
ROOT = B.parent
WORKBOOK = B / "data/eval/master_results.xlsx"
DATE_COLUMNS = re.compile(r"(_date$|^sources_earliest$|^sources_latest$|^date_a$|^date_b$|^summary_date$|^report_date$|^metrics_date$)")


def _builder():
    spec = importlib.util.spec_from_file_location("build_master_results", B / "scripts/build_master_results.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def workbook() -> dict[str, pd.DataFrame]:
    if not WORKBOOK.is_file():
        pytest.skip(f"{WORKBOOK} missing: run scripts/build_master_results.py first")
    return pd.read_excel(WORKBOOK, sheet_name=None)


@pytest.fixture(scope="module")
def rebuilt(tmp_path_factory) -> dict[str, pd.DataFrame]:
    mod = _builder()
    tabs = mod.build_all()
    out = tmp_path_factory.mktemp("master") / "rebuilt.xlsx"
    mod.write_workbook(tabs, out)
    return pd.read_excel(out, sheet_name=None)


def _strip_dates(df: pd.DataFrame) -> pd.DataFrame:
    keep = [c for c in df.columns if not DATE_COLUMNS.search(str(c))]
    return df[keep].reset_index(drop=True)


# --------------------------------------------------------------------------- layer 1: rebuild
def test_same_tabs(workbook, rebuilt):
    assert list(workbook) == list(rebuilt)


def test_every_tab_identical_to_rebuild(workbook, rebuilt):
    for name in workbook:
        a, b = _strip_dates(workbook[name]), _strip_dates(rebuilt[name])
        pd.testing.assert_frame_equal(a, b, check_dtype=False, check_exact=True, obj=f"tab {name}")


def test_metadata_sources_exist(workbook):
    meta = workbook["metadata"]
    missing = []
    for _, r in meta.iterrows():
        for s in str(r["source_files"]).split("; "):
            if s and not (ROOT / s).exists():
                missing.append((r["tab"], s))
    assert not missing, f"sources listed in the metadata tab but absent: {missing}"
    assert (meta["status"] == "ok").all(), meta[meta.status != "ok"][["tab", "status"]].to_string()
    assert (meta["n_sources_present"] == meta["n_sources"]).all()


def test_generation_paper_results_excluded(workbook):
    banned_tab = re.compile(r"generat|verif|stratif|audit_sample|audit_stat|coverage_sampl", re.I)
    banned_col = re.compile(r"generator|verifier_choice|label wrong|relabelled rate|unusable", re.I)
    for name, df in workbook.items():
        assert not banned_tab.search(name), name
        for c in df.columns:
            assert not banned_col.search(str(c)), f"{name}: {c}"
    content = " ".join(workbook["metadata"]["content"].astype(str))
    assert "generator" not in content.lower() and "stratification" not in content.lower()


# --------------------------------------------------------------------------- layer 2: raw sources
def _log_float(log: Path, pattern: str) -> float:
    m = re.search(pattern, log.read_text(encoding="utf-8", errors="ignore"))
    assert m, f"{pattern!r} not found in {log}"
    return float(m.group(1))


def _eq(cell, value) -> bool:
    """Excel returns an integer column that has missing cells as floats (3 -> 3.0) and a boolean
    column with missing cells as 0.0 / 1.0: compare numerically when both sides are numbers."""
    if value == "" and (cell is None or (isinstance(cell, float) and np.isnan(cell))):
        return True                       # an empty JSON string is an empty Excel cell
    if isinstance(value, bool):
        return bool(cell) == value
    if isinstance(value, (int, float)) and isinstance(cell, (int, float, np.integer, np.floating)):
        return float(cell) == float(value)
    return str(cell) == str(value)


def _row(df: pd.DataFrame, **match) -> pd.Series:
    m = np.ones(len(df), bool)
    for k, v in match.items():
        m &= (df[k].astype(str) == str(v)).values
    assert m.sum() == 1, f"{match} matched {m.sum()} rows"
    return df[m].iloc[0]


def test_pipeline_stage_seconds_match_logs(workbook):
    runs = workbook["pipeline_runs"]
    for _, r in runs.iterrows():
        log = ROOT / r["log"]
        assert r["ocr_seconds"] == _log_float(log, r"OCR done: \d+ slide\(s\) in ([\d.]+)s")
        assert r["ner_seconds"] == _log_float(log, r"NER done: \d+ entit\(y/ies\) in ([\d.]+) s")
        assert r["norm_seconds"] == _log_float(log, r"Norm done: \d+/\d+ linked in ([\d.]+)s")
        assert r["graph_seconds"] == _log_float(log, r"Context graph done: \d+ nodes, \d+ edges in ([\d.]+)s")
        assert r["ner_entities"] == int(_log_float(log, r"NER done: (\d+) entit"))
        assert r["norm_linked"] == int(_log_float(log, r"Norm done: (\d+)/\d+ linked"))
        d = json.loads((ROOT / r["output_json"]).read_text(encoding="utf-8"))
        assert r["n_entities"] == len(d["entities"]) and r["n_slides"] == len(d["slides"]) and r["elapsed_s"] == d["run"]["elapsed_s"]
        assert r["n_linked_kg2c"] == sum(1 for e in d["entities"] if e.get("kg2c_id"))


def test_document_gold_metrics_match_metrics_json(workbook):
    df = workbook["document_gold_metrics"]
    for _, r in df.iterrows():
        m = json.loads((ROOT / r["metrics_file"]).read_text(encoding="utf-8"))["positives"]
        for k in ("recall_exact", "recall_partial", "precision_mentions_distinct", "typing_accuracy_on_found", "n_found_linkable"):
            if k in m:  # the gold-v1 metrics were written by an earlier eval_gold.py without the linkable count
                assert r[f"positives.{k}"] == m[k], (r["run"], r["gold_version"], k)


def test_typing_reports_match(workbook):
    for tab, key, block, metric in (("typing_public_benchmark", "agent_final_all", "agent", "accuracy"), ("typing_public_benchmark", "rag_v4_exemplars_public", "rag", "hit@1"),
                                    ("typing_public_benchmark", "claude_opus5_b1_public", "all", "accuracy"), ("typing_real_kg2c", "agent_v4_kg2c_real", "agent", "accuracy"),
                                    ("document_terms_isolation", "agent_v4_fshd_v2_terms", "agent", "accuracy")):
        df = workbook[tab]; r = _row(df, result_dir=f"KG_nodes_eval/results/{key}")
        rep = json.loads((ROOT / r["report_file"]).read_text(encoding="utf-8"))
        blk = rep["agent"] if block == "agent" else rep["rag"]["all"] if block == "rag" else rep["all"]
        assert r[f"{block}.{metric}"] == blk[metric], (tab, key, metric)


def test_context_graph_sizes_match_summaries(workbook):
    df = workbook["context_graphs"]
    for _, r in df.iterrows():
        s = json.loads((ROOT / r["summary_file"]).read_text(encoding="utf-8"))
        for k in ("n_nodes", "n_edges", "n_components", "seed_pairs_connected", "n_seeds_in_graph"):
            if k in s:  # the reference 1-hop graphs carry fewer fields
                assert _eq(r[k], s[k]), (r["graph_dir"], k, r[k], s[k])
    cats = workbook["context_graph_categories"]
    fshd_u = cats[(cats.document == "FSHD") & (cats.profile == "uniform")]
    s = json.loads((B / "data/context_graph/FSHD_uniform/summary.json").read_text(encoding="utf-8"))
    assert dict(zip(fshd_u.category, fshd_u.n_nodes)) == s["categories"]


def test_use_cases_match_sources(workbook):
    d = json.loads((B / "data/eval/fshd_dux4_case.json").read_text(encoding="utf-8"))
    dux = workbook["use_case_dux4"]
    for p in d["profiles"]:
        sub = dux[dux.profile == p["profile"]]
        assert sub.category_size.iloc[0] == p["category_size"] and sub.n_mechanism_nodes_found.iloc[0] == p["n_found"]
        if p["hits"]:
            assert sorted(sub.node_rank) == sorted(h["rank"] for h in p["hits"]) and sub.best_rank.iloc[0] == p["best_rank"]
    pres = json.loads((B / "data/eval/fshd_presence_final.json").read_text(encoding="utf-8"))
    wp = workbook["use_case_presence"]
    assert len(wp) == len(pres) and list(wp.label) == [r["label"] for r in pres]
    for r in pres:
        row = _row(wp, label=r["label"])
        for k, v in r.items():
            if isinstance(v, (bool, int, str)):
                assert _eq(row[k], v), (r["label"], k, row[k], v)


def test_gate_and_linking_match(workbook):
    g = json.loads((B / "data/eval/gate_modes.json").read_text(encoding="utf-8"))
    for _, r in workbook["existence_gate"].iterrows():
        assert r["true_rejections"] == g["modes"][r["mode"]]["true_rejections"] and r["false_rejections"] == g["modes"][r["mode"]]["false_rejections"]
    for _, r in workbook["linking"].iterrows():
        rep = json.loads((ROOT / r["report_file"]).read_text(encoding="utf-8"))["conditions"][r["condition"]]
        for k in ("accuracy_exact", "accuracy_clique", "n_linkable", "answered"):
            assert r[k] == rep[k], (r["report"], r["condition"], k)


def test_adversarial_rows_match(workbook):
    d = json.loads((B / "output/_final_run1/adversarial_run_v2.json").read_text(encoding="utf-8"))
    df = workbook["adversarial_suite"]
    assert len(df) == len(d["positives"]) + len(d["adversarial"])
    adv = df[df.group == "adversarial"]
    assert int(adv.typed.sum()) == sum(1 for t in d["adversarial"] if t.get("types"))
    m = json.loads((B / "output/_final_run1/eval_v2/metrics.json").read_text(encoding="utf-8"))["adversarial"]
    assert abs(adv.typed.mean() - m["typed_rate"]) < 5e-4


def test_energy_is_seconds_times_power(workbook):
    e = json.loads((B / "data/eval/environment.json").read_text(encoding="utf-8"))["energy_assumptions"]
    for _, r in workbook["resources_energy"].iterrows():
        pw = e["power_w"][r["power_assumption"]]
        assert r["power_w"] == pw
        assert abs(r["energy_kwh"] - r["seconds"] * pw / 3.6e6) < 1e-12
        for region, g in e["carbon_intensity_g_per_kwh"].items():
            assert abs(r[f"gCO2e ({region})"] - r["energy_kwh"] * g) < 1e-9
    t = workbook["resources_timing"]
    r1 = _row(t, run="FSHD run 1", stage="ocr")
    assert r1["seconds"] == _log_float(B / "output/_final_run1/FSHD_BiomedCAT.log", r"OCR done: \d+ slide\(s\) in ([\d.]+)s")


def test_audited_gold_scores_are_recomputable(workbook):
    """The published ReAct run on the audited terms: recomputed here from the audit file and the
    per-query parquet with the scoring rules of KG_nodes_eval/rescore_audit_full.py."""
    import sys
    sys.path.insert(0, str(ROOT / "KG_nodes_eval"))
    import os
    old = os.getcwd(); os.chdir(ROOT / "KG_nodes_eval")
    try:
        import rescore_audit_full as raf
        from rescore_audit_s300 import NESTED, biolink_paths, canon
        gold = raf.build_gold(biolink_paths(NESTED))
        v = pd.read_parquet("results/agent_final_all/agent_per_query.parquet").set_index("query")["agent_verdict"]
    finally:
        os.chdir(old)
    vc = v.reindex(gold["query"]).map(canon).fillna("").values
    acc = float(np.mean([c in s for c, s in zip(vc, gold["_acc"].values)]))
    row = _row(workbook["typing_audited_gold"], result_key="agent:agent_final_all")
    assert row["n_terms"] == len(gold) and abs(row["accuracy_all"] - acc) < 1e-12


def test_ocr_only_images_match_json(workbook):
    df = workbook["ocr_only_images"]
    files = sorted((B / "output/ocr_only").glob("*_ocr.json"))
    assert len(files) >= 1 and len(df) == sum(len(json.loads(f.read_text(encoding="utf-8"))["pages"]) for f in files)
    for f in files:
        r = json.loads(f.read_text(encoding="utf-8"))
        for pg in r["pages"]:
            row = _row(df, file=r["file"], page=pg["page"])
            assert _eq(row["seconds"], r["seconds"]) and _eq(row["description_chars"], pg["chars"]) and row["description"] == pg["text"]
            assert _eq(row["gpu_mib_peak_during"], r["gpu_mib_peak_during"]) if r["gpu_mib_peak_during"] is not None else True
    log = (B / "output/ocr_only/ocr_only.log").read_text(encoding="utf-8", errors="ignore")
    for f in files:
        r = json.loads(f.read_text(encoding="utf-8"))
        assert re.search(rf"OCR done: {r['n_pages']} page\(s\) in {r['seconds']:.1f}s", log), f


def test_dataset_documents_verbatim(workbook):
    src = pd.read_csv(B / "Dataset/documents_metadata.csv", dtype=str, keep_default_na=False)
    df = workbook["dataset_documents"].drop(columns=["source"]).fillna("").astype(str)
    assert list(df.columns) == list(src.columns) and len(df) == len(src)
    for c in src.columns:
        assert list(df[c]) == list(src[c]), c


def test_published_runs_share_the_published_retriever_configuration(workbook):
    """Every document run in the workbook ran the configuration of the benchmark runs (scispaCy not
    loaded, no BM25 leg, no lexical table), as its own log says; re-read here from the logs."""
    runs = workbook["pipeline_runs"]
    for _, r in runs.iterrows():
        txt = (ROOT / r["log"]).read_text(encoding="utf-8", errors="ignore")
        assert "Successfully loaded scispaCy model" not in txt and "Initializing BM25 index" not in txt, r["run"]
        assert not bool(r["scispacy_loaded"]) and not bool(r["bm25_leg"]), r["run"]
        # the lexical legs: absent in September (they did not exist), weight 0 in the October chains
        # (the table was still built and logged at weight 0 before the 4 Oct fix, never used)
        assert str(r["lexical_leg_weight"]) in ("0", "0.0", "n/a (leg did not exist)"), (r["run"], r["lexical_leg_weight"])
        if r["chain_env_source"] and str(r["chain_env_source"]) != "nan":
            assert "RAG_LEXICAL_WEIGHT=0" in (ROOT / r["chain_env_source"]).read_text(encoding="utf-8", errors="ignore"), r["run"]
