#!/usr/bin/env python
"""Build the figures of the BiomedCAT manuscript from the result files, one panel at a time.

Rewrite started 5 Oct 2026. The previous script is kept as scripts/make_figures_legacy.py until
every panel has been ported. Each panel reads primary result files (run logs, JSON reports), never
the master workbook, and writes next to the figure a CSV with the numbers it plotted.

Sources: data/publication/ (the deposit assembled by scripts/build_publication_folder.py: runs/,
gold_eval/, benchmark/) as soon as the publication runs of scripts/chain_publication_runs.sh exist;
until then the legacy runs under Output/ (September and 4-5 Oct 2026) are read, so the script can be
run at any time.

Panels
  panel2   Resources of a document run: (a) time per stage, (b) memory per stage.
           Inputs: Output/_final_*/<doc>_BiomedCAT.{log,json} (published runs) and
           Output/resources/<doc>_resources.json (scripts/measure_pipeline_resources.py).
  panel2_gold  fig2_gold_recall (reading recall of explicit / implicit expert terms per deck) and
           fig2_gold_accuracy (typing accuracy on explicit terms per deck).
           Inputs: data/eval/gold_decks/summary.csv (scripts/eval_gold_decks.py on the harmonized gold).
  panel2_benchmark  fig2_typing_benchmark_a (BiomedCAT vs hosted models on the audited synthetic benchmark,
           original vs expert labels) and _b (expert labels by scope) and fig2_linking_offline (FSHD linking, online vs local).
           Inputs: KG_nodes_eval/results/rescoring_v1.1_full.xlsx, Output/_fshd_online, Output/_fshd_offline.

Outputs: <name>.pdf, <name>.png (300 dpi), <name>_data.csv in --out (default: the manuscript's
figures/ directory) and figures_manifest.md listing each figure, its inputs and a draft caption.

Usage:
  uv run python scripts/make_figures.py                 # every panel
  uv run python scripts/make_figures.py --only panel2   # one panel
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger("make_figures")
B = Path(__file__).resolve().parents[1]
OUTPUT = B / "Output"
PUB = B / "data" / "publication"   # scripts/build_publication_folder.py; runs written by scripts/chain_publication_runs.sh
DEFAULT_OUT = B.parent / "manuscripts" / "biomedcat_manuscript" / "figures"


def use_publication() -> bool:
    """True when at least one publication run has finished (its pipeline JSON exists)."""
    return (PUB / "runs").is_dir() and any(d.is_dir() and list(d.glob("*_BiomedCAT.json")) for d in (PUB / "runs").iterdir())

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"], "font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 100})
STAGE_COL = {"ocr": "#4c72b0", "ner": "#55a868", "norm": "#dd8452", "graph": "#8172b2", "total": "#9e9e9e"}
STAGE_LABEL = {"ocr": "OCR (reading)", "ner": "NER (typing)", "norm": "linking", "graph": "context graph", "total": "total"}
RAM_TARGET_GB = 16  # design target for end users (README, Discussion)
MANIFEST: list[tuple[str, str, str]] = []
# Deck order shared by every figure of panel 2 (labels as displayed).
DECK_ORDER = ["FSHD", "VoieA", "VoieB", "VoieC", "VoiesA-C", "FSHD1multiscales", "Maladies musculaires", "glycolysis (image)"]


def order_decks(df: pd.DataFrame, col: str = "label") -> pd.DataFrame:
    """Rows in DECK_ORDER; decks not listed keep their relative order at the end."""
    rank = {d: i for i, d in enumerate(DECK_ORDER)}
    df = df.copy(); df["_rank"] = df[col].map(lambda v: rank.get(v, len(rank)))
    return df.sort_values("_rank", kind="stable").drop(columns="_rank").reset_index(drop=True)

# Stage markers written by biomedcat/pipeline.py; the same regexes as scripts/build_master_results.py.
STAGE_RX = {"ocr": re.compile(r"OCR done: (\d+) slide\(s\) in ([\d.]+)s"),
            "ner": re.compile(r"NER done: (\d+) entit\(y/ies\) in ([\d.]+) s"),
            "norm": re.compile(r"Norm done: (\d+)/(\d+) linked in ([\d.]+)s"),
            "graph": re.compile(r"Context graph done: (\d+) nodes, (\d+) edges in ([\d.]+)s")}


def save(fig, out: Path, name: str, inputs: list[str], caption: str, data: pd.DataFrame | None = None) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)
    if data is not None:
        data.to_csv(out / f"{name}_data.csv", index=False)
    MANIFEST.append((name, "; ".join(inputs), caption))
    logger.info("-> %s", name)


def rel(p: Path) -> str:
    return str(p.relative_to(B)).replace("\\", "/")


# ----------------------------------------------------------------------------------------------
# Panel 2: resources (time and memory per stage)
# ----------------------------------------------------------------------------------------------
def read_run_dir(d: Path, run_dir_name: str) -> dict | None:
    """Stage seconds from the log and the total from the JSON of one run directory."""
    jsons = list(d.glob("*_BiomedCAT.json"))
    if not jsons:
        return None
    stem = jsons[0].name[: -len("_BiomedCAT.json")]
    log = d / f"{stem}_BiomedCAT.log"
    if not log.is_file():
        return None
    row = {"run_dir": run_dir_name, "document": stem, "log": rel(log)}
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        for stage, rx in STAGE_RX.items():
            m = rx.search(line)
            if m and f"{stage}_s" not in row:
                g = m.groups(); row[f"{stage}_s"] = float(g[-1]); row[f"{stage}_units"] = int(g[0]) if stage != "norm" else int(g[1])
    run = json.loads(jsons[0].read_text(encoding="utf-8"))["run"]
    row.update({"total_s": run.get("elapsed_s"), "timestamp": run.get("timestamp"), "ocr_model": run["models"].get("ocr"), "ner_model": run["models"].get("ner"),
                "resolvers_offline": run.get("resolvers", {}).get("offline")})
    return row if "ocr_s" in row and "ner_s" in row else None


def published_runs() -> pd.DataFrame:
    """One row per document run: the publication runs (data/publication/runs/<name>/) when they exist,
    else the legacy runs (Output/_final_*/). Stage seconds from the log, total from the JSON."""
    if use_publication():
        rows = [r for d in sorted((PUB / "runs").iterdir()) if d.is_dir() and d.name != "FSHD_offline" for r in [read_run_dir(d, d.name)] if r]
        df = pd.DataFrame(rows)
        if df.empty:
            return df
        df["label"] = df["document"].replace({"chemicals": "glycolysis (image)"})
        return order_decks(df)
    return legacy_runs_df()


def legacy_runs_df() -> pd.DataFrame:
    rows = []
    for d in sorted(OUTPUT.glob("_final_*")):
        jsons = list(d.glob("*_BiomedCAT.json"))
        if not jsons:
            continue
        stem = jsons[0].name[: -len("_BiomedCAT.json")]
        log = d / f"{stem}_BiomedCAT.log"
        if not log.is_file():  # some published runs only kept the JSON: the chain log of the night holds the stage lines
            log = next((c for c in sorted(OUTPUT.glob("chain_*/*.log"))
                        if f"processing Dataset\\{stem}.pdf" in c.read_text(encoding="utf-8", errors="replace")), None)
            if log is None:
                logger.warning("published_runs: no log for %s", d.name); continue
        row = {"run_dir": d.name, "document": stem, "log": rel(log)}
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            for stage, rx in STAGE_RX.items():
                m = rx.search(line)
                if m and f"{stage}_s" not in row:
                    g = m.groups()
                    row[f"{stage}_s"] = float(g[-1])
                    row[f"{stage}_units"] = int(g[0]) if stage != "norm" else int(g[1])
        js = d / f"{stem}_BiomedCAT.json"
        if js.is_file():
            run = json.loads(js.read_text(encoding="utf-8"))["run"]
            row["total_s"] = run.get("elapsed_s")
            row["timestamp"] = run.get("timestamp")
            row["ocr_model"], row["ner_model"] = run["models"].get("ocr"), run["models"].get("ner")
        if "ocr_s" in row and "ner_s" in row:
            rows.append(row)
    full = OUTPUT / "resources" / "_run_chemicals"  # the glycolysis image, full run measured on 5 Oct 2026
    ocr_only = OUTPUT / "ocr_only" / "chemicals_ocr.json"  # reading-only run, used when the full run is absent
    if (full / "chemicals_BiomedCAT.json").is_file():
        row = {"run_dir": full.name, "document": "glycolysis (image)", "log": rel(full / "chemicals_BiomedCAT.log")}
        for line in (full / "chemicals_BiomedCAT.log").read_text(encoding="utf-8", errors="replace").splitlines():
            for stage, rx in STAGE_RX.items():
                m = rx.search(line)
                if m and f"{stage}_s" not in row:
                    g = m.groups(); row[f"{stage}_s"] = float(g[-1]); row[f"{stage}_units"] = int(g[0]) if stage != "norm" else int(g[1])
        run = json.loads((full / "chemicals_BiomedCAT.json").read_text(encoding="utf-8"))["run"]
        row.update({"total_s": run.get("elapsed_s"), "timestamp": run.get("timestamp"), "ocr_model": run["models"].get("ocr"), "ner_model": run["models"].get("ner")})
        rows.append(row)
    elif ocr_only.is_file():
        o = json.loads(ocr_only.read_text(encoding="utf-8"))
        rows.append({"run_dir": "ocr_only", "document": "glycolysis (image, reading only)", "log": rel(ocr_only), "ocr_s": o["seconds"],
                     "ocr_units": o["n_pages"], "total_s": o["seconds"], "timestamp": o.get("started_utc"), "ocr_model": o.get("model")})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    # Label: document name, plus the run suffix when the same document was run several times.
    counts = df["document"].value_counts()
    df["label"] = [f"{doc} ({d.replace('_final_', '')})" if counts[doc] > 1 else doc for doc, d in zip(df["document"], df["run_dir"])]
    return df.sort_values("total_s", ascending=True).reset_index(drop=True)


def resource_measurements() -> pd.DataFrame:
    """One row per (document, stage): data/publication/runs/<name>/resources.json when the publication
    runs exist, else Output/resources/<doc>_resources.json (measurements of 5 Oct 2026)."""
    rows = []
    files = sorted((PUB / "runs").glob("*/resources.json")) if use_publication() else sorted((OUTPUT / "resources").glob("*_resources.json"))
    files = [p for p in files if p.parent.name != "FSHD_offline"]
    for p in files:
        r = json.loads(p.read_text(encoding="utf-8"))
        for stage, s in r["stages"].items():
            rows.append({"document": {"chemicals": "glycolysis (image)"}.get(Path(r["file"]).stem, Path(r["file"]).stem), "stage": stage, "seconds_wall": s.get("seconds_wall"),
                         "cpu_ram_gb_peak": (s.get("cpu_ram_mb_peak") or 0) / 1024,
                         "system_ram_gb_delta_peak": (s.get("system_used_mb_delta_peak") or 0) / 1024,
                         "gpu_gb_peak": (s.get("gpu_mib_peak") or 0) / 1024,
                         "models": ", ".join(s.get("models_resident", [])), "source": rel(p)})
    return pd.DataFrame(rows)


def panel2(out: Path) -> None:
    runs = published_runs()
    if runs.empty:
        logger.warning("panel2: no published run under Output/_final_*"); return
    mem = resource_measurements()

    fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(8.2, 3.4), gridspec_kw={"width_ratios": [1.45, 1]})

    # (a) Time per stage, one horizontal stacked bar per run, in minutes; total annotated.
    y = np.arange(len(runs))[::-1]  # first deck of DECK_ORDER at the top
    left = np.zeros(len(runs))
    for stage in ("ocr", "ner", "norm", "graph"):
        vals = runs.get(f"{stage}_s", pd.Series(0, index=runs.index)).fillna(0).to_numpy() / 60
        ax_t.barh(y, vals, left=left, color=STAGE_COL[stage], label=STAGE_LABEL[stage], height=0.7)
        left += vals
    totals = runs["total_s"].fillna(pd.Series(left * 60, index=runs.index)) / 60
    for i, tot in enumerate(totals):
        ax_t.text(left[i] + 0.6, y[i], f"{tot:.0f} min", va="center", fontsize=7.5)
    ax_t.set_yticks(y); ax_t.set_yticklabels(runs["label"], fontsize=8)
    ax_t.set_xlabel("wall-clock time (min)"); ax_t.set_xlim(0, left.max() * 1.18)
    ax_t.set_title("(a) Time per stage, one slide deck per run", fontsize=9, loc="left")
    ax_t.legend(frameon=False, fontsize=7.5, loc="lower right")

    # (b) Memory per stage: CPU RAM (pipeline + model server, above idle) and GPU memory, peaks.
    stages = ["ocr", "ner", "norm", "graph", "total"]
    if mem.empty:
        ax_m.text(0.5, 0.5, "no memory measurement yet\nrun scripts/measure_pipeline_resources.py", ha="center", va="center",
                  transform=ax_m.transAxes, fontsize=8, color="#777")
        ax_m.set_axis_off()
        logger.warning("panel2: Output/resources/*_resources.json missing; memory panel left empty")
    else:
        agg = mem[mem["stage"].isin(stages)].groupby("stage").agg(ram=("system_ram_gb_delta_peak", "max"), gpu=("gpu_gb_peak", "max")).reindex(stages)
        x = np.arange(len(stages)); w = 0.38
        cols = [STAGE_COL[st] for st in stages]
        ax_m.bar(x - w / 2, agg["ram"], w, color=cols, alpha=1.0)
        ax_m.bar(x + w / 2, agg["gpu"], w, color=cols, alpha=0.4)
        from matplotlib.patches import Patch
        ax_m.legend(handles=[Patch(color="#555", alpha=1.0, label="system RAM in use, above idle"), Patch(color="#555", alpha=0.4, label="GPU memory in use")],
                    frameon=False, fontsize=7, loc="upper left", bbox_to_anchor=(0.0, 0.93))
        for doc, g in mem[mem["stage"].isin(stages)].groupby("document"):  # one marker per measured document
            g = g.set_index("stage").reindex(stages)
            ax_m.scatter(x - w / 2, g["system_ram_gb_delta_peak"], s=9, color="k", zorder=3)
            ax_m.scatter(x + w / 2, g["gpu_gb_peak"], s=9, color="k", zorder=3)
        ax_m.axhline(RAM_TARGET_GB, ls="--", lw=0.8, color="#777"); ax_m.text(len(stages) - 0.55, RAM_TARGET_GB * 1.02, f"{RAM_TARGET_GB} GB target", ha="right", fontsize=7, color="#555")
        ax_m.set_xticks(x); ax_m.set_xticklabels([STAGE_LABEL[s] for s in stages], rotation=25, ha="right", fontsize=8)
        ax_m.set_ylabel("peak memory (GB)"); ax_m.set_ylim(0, max(RAM_TARGET_GB * 1.1, float(np.nanmax(agg.to_numpy())) * 1.15))
        n_docs = mem["document"].nunique()
        ax_m.set_title(f"(b) Peak memory per stage ({n_docs} deck{'s' if n_docs > 1 else ''})", fontsize=9, loc="left")

    data = runs[["label", "document", "run_dir", "timestamp", "ocr_model", "ner_model", "ocr_s", "ocr_units", "ner_s", "ner_units",
                 "norm_s", "norm_units", "graph_s", "total_s", "log"]].copy()
    data["ner_s_per_entity"] = data["ner_s"] / data["ner_units"]
    data["ocr_s_per_slide"] = data["ocr_s"] / data["ocr_units"]
    if not mem.empty:
        data = pd.concat([data, mem], axis=0, ignore_index=True, sort=False)
    inputs = sorted(set(runs["log"])) + (sorted(set(mem["source"])) if not mem.empty else [])
    hw = "one laptop (Intel i9-13900H, 64 GB RAM, RTX 2000 Ada 8 GB; data/eval/environment.json)"
    save(fig, out, "fig2_resources", inputs,
         f"Resources of a document run on {hw}. (a) Wall-clock time of each stage for the published runs of the seven slide decks and the reading-only run of the glycolysis image "
         "(OCR with qwen2.5vl:3b, typing agent with gemma4:e4b-it-qat, linking with the same model against the public resolvers, "
         "context graph on RTX-KG2c); the typing stage dominates and scales with the number of candidate mentions. (b) Peak memory per "
         "stage, sampled every second while re-running a deck: system RAM in use above the idle level (pipeline process, Ollama model "
         "server and driver-backed buffers) and GPU memory in use. Ollama keeps the model weights in GPU memory on this machine; without a "
         "GPU the same weights occupy RAM. The RAM peak is the context-graph stage, which loads the RTX-KG2c edge table in memory; the "
         "typing-stage GPU peak includes the second model of the existence gate (llama3:8b). "
         f"The dashed line is the {RAM_TARGET_GB} GB design target.", data)


# ----------------------------------------------------------------------------------------------
# Panel 2, gold: reading recall (explicit / implicit) and typing accuracy per deck
# ----------------------------------------------------------------------------------------------
GOLD_SUMMARY = (PUB / "gold_eval" / "summary.csv") if (PUB / "gold_eval" / "summary.csv").is_file() else B / "data" / "eval" / "gold_decks" / "summary.csv"


def gold_runs() -> pd.DataFrame:
    if not GOLD_SUMMARY.is_file():
        return pd.DataFrame()
    df = pd.read_csv(GOLD_SUMMARY)
    df["label"] = [f"{d} ({r})" if r.startswith("run") else d for d, r in zip(df["deck"], df["run"])]
    df["label"] = df["label"].str.replace("glycolysis (publication)", "glycolysis", regex=False)
    df["label"] = [l.replace("glycolysis", "glycolysis\n(image, reading only)" if k == "ocr_only" else "glycolysis (image)") for l, k in zip(df["label"], df["kind"])]
    missing = [d for d in DECK_ORDER if d not in set(df["label"])]  # decks run but without expert gold (Maladies musculaires)
    if missing:
        df = pd.concat([df, pd.DataFrame({"label": missing, "deck": missing, "kind": "pipeline", "n_explicit": 0, "n_implicit": 0})], ignore_index=True)
    return order_decks(df)


def panel2_gold(out: Path) -> None:
    df = gold_runs()
    if df.empty:
        logger.warning("panel2_gold: %s missing; run scripts/eval_gold_decks.py", GOLD_SUMMARY); return
    n_terms = int(df.drop_duplicates("deck")[["n_explicit", "n_implicit"]].sum().sum())
    n_decks = int((df["n_explicit"] > 0).sum())

    # --- fig2_gold_recall: reading recall, explicit vs implicit terms, exact (dark) over lenient (light)
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    x = np.arange(len(df)); w = 0.38
    has_impl = df["n_implicit"].fillna(0) > 0
    ax.bar(x - w / 2, df["read_recall_explicit_lenient"], w, color="#aec7e8", label="explicit, lenient match")
    ax.bar(x - w / 2, df["read_recall_explicit_exact"], w, color="#1f77b4", label="explicit, exact match")
    ax.bar(x[has_impl] + w / 2, df.loc[has_impl, "read_recall_implicit_lenient"], w, color="#ffbb78", label="implicit, lenient match")
    ax.bar(x[has_impl] + w / 2, df.loc[has_impl, "read_recall_implicit_exact"], w, color="#ff7f0e", label="implicit, exact match")
    for i, r in df.reset_index().iterrows():
        if r["n_explicit"] == 0:
            ax.text(i, 0.03, "no expert\ngold", ha="center", va="bottom", fontsize=6.5, color="#777", style="italic"); continue
        ax.text(i - w / 2, r["read_recall_explicit_lenient"] + 0.02, f"n={int(r['n_explicit'])}", ha="center", fontsize=6.5, color="#333")
        if r["n_implicit"] > 0:
            ax.text(i + w / 2, r["read_recall_implicit_lenient"] + 0.02, f"n={int(r['n_implicit'])}", ha="center", fontsize=6.5, color="#333")
    ax.set_xticks(x); ax.set_xticklabels(df["label"], rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1.08); ax.set_ylabel("gold terms present in the reading")
    ax.set_title("Reading stage: recall of the expert terms per deck", fontsize=9, loc="left", pad=30)
    ax.legend(frameon=False, fontsize=7, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    save(fig, out, "fig2_gold_recall", ["data/eval/gold_decks/summary.csv", "data/gold/gold_harmonized.json"] + sorted(set(df["result"].dropna())),
         f"Recall of the expert gold terms ({n_terms} terms over {n_decks} decks) in the text produced by the reading stage "
         "(qwen2.5vl:3b, description mode). Explicit terms are written on the slide; implicit terms are expected by the expert but not "
         "written (FSHD: 7, FSHD1multiscales: 1, glycolysis image: 11). Exact: the term, its English query or the KG2c label of its CURIE "
         "is a substring of the reading; lenient: every token of one of those forms occurs in the reading, tolerating French/English "
         "cognates (shared 5-character prefix). French terms without an English cognate and without a KG2c label count as missed. "
         "The glycolysis slide is a structure-only image; the two FSHD runs are replicates of the same deck.",
         df[["deck", "run", "label", "n_explicit", "n_implicit", "read_recall_explicit_exact", "read_recall_explicit_lenient",
             "read_recall_implicit_exact", "read_recall_implicit_lenient", "result"]])

    # --- fig2_gold_accuracy: typing accuracy on found explicit terms, exact (dark) over within-1-hop (light)
    pr = df[df["kind"] == "pipeline"].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    x = np.arange(len(pr)); w = 0.5
    ax.bar(x, pr["typing_accuracy_within_1"], w, color="#b5dcb5", label="class within one hop (parent or child)")
    ax.bar(x, pr["typing_accuracy_exact"], w, color="#2ca02c", label="exact Biolink class")
    ax.plot(x, pr["extraction_recall_explicit_partial"], "o", color="#555", ms=4, label="explicit terms found by the extraction (recall)")
    for i, r in pr.iterrows():
        if r["n_explicit"] == 0:
            ax.text(i, 0.03, "no expert\ngold", ha="center", va="bottom", fontsize=6.5, color="#777", style="italic"); continue
        ax.text(i, r["typing_accuracy_within_1"] + 0.02, f"n={int(r['n_typed_explicit'])}", ha="center", fontsize=6.5, color="#333")
    ax.set_xticks(x); ax.set_xticklabels(pr["label"], rotation=25, ha="right", fontsize=8)
    ax.set_ylim(0, 1.08); ax.set_ylabel("typing accuracy on found explicit terms")
    ax.set_title("Typing stage: accuracy on the explicit expert terms per deck", fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    save(fig, out, "fig2_gold_accuracy", ["data/eval/gold_decks/summary.csv", "data/gold/gold_harmonized.json"] + sorted(set(pr["result"].dropna())),
         "Biolink typing accuracy of the pipeline (gemma4:e4b-it-qat, ReAct agent) on the explicit expert terms that the reading and "
         "extraction stages recovered (n above each bar): exact class, and class at hierarchical distance at most one in the Biolink tree. "
         "Dots: share of explicit terms recovered by the extraction (exact or partial mention match). Typing is scored on the mention "
         "matched to the gold term; terms without a gold class count for recall only.",
         pr[["deck", "run", "label", "n_explicit", "n_found_explicit", "n_typed_explicit", "extraction_recall_explicit_exact",
             "extraction_recall_explicit_partial", "typing_accuracy_exact", "typing_accuracy_within_1", "result"]])


# ----------------------------------------------------------------------------------------------
# Panel 2, benchmark: typing accuracy vs other models (original vs audited labels), linking on/offline
# ----------------------------------------------------------------------------------------------
RESCORING = (PUB / "benchmark" / "rescoring_v1.1_full.xlsx") if (PUB / "benchmark" / "rescoring_v1.1_full.xlsx").is_file() else B.parent / "KG_nodes_eval" / "results" / "rescoring_v1.1_full.xlsx"
SYSTEMS = [  # column in gold_v1.1_terms -> (label, colour); BiomedCAT from the retriever alone to the full agent, then hosted models
    ("retriever v4, top-1", "BiomedCAT retriever only", "#4c72b0"),
    ("agent decide (published)", "BiomedCAT single call", "#74c476"),
    ("agent ReAct (final run)", "BiomedCAT ReAct agent", "#2ca02c"),
    ("Claude Opus 5 (API, shown)", "Claude Opus 5", "#dd8452"),
    ("DeepSeek-V4-Flash (API, shown)", "DeepSeek-V4-Flash", "#dd8452"),
    ("gemma-4-31b (API, shown)", "gemma-4-31b", "#dd8452"),
    ("mistral-small-24b (API)", "mistral-small-24b", "#dd8452"),
]


def canon_class(x) -> str:
    """Same canonical form as KG_nodes_eval/rescore_audit_s300.canon: 'biolink:SmallMolecule' -> 'smallmolecule'."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    t = str(x).strip()
    if t.lower().startswith("biolink:"):
        t = re.sub(r"(?<!^)(?=[A-Z])", " ", t.split(":", 1)[1])
    return re.sub(r"[^a-z0-9]", "", t.lower())


def ci95(p: float, n: int) -> float:
    return 1.96 * np.sqrt(p * (1 - p) / n) if n else 0.0


def typing_scores() -> pd.DataFrame:
    """Per system: accuracy against the original labels and against the expert labels (gold v1.1),
    ambiguous terms excluded, over all terms and per scope (biomedical / non-biomedical / personal)."""
    g = pd.read_excel(RESCORING, sheet_name="gold_v1.1_terms")
    g["orig"] = g["original_class"].map(canon_class)
    g["accepted_set"] = g["accepted"].astype(str).map(lambda a: {canon_class(x) for x in a.split(";")})
    scored = g[g["gold_kind"] != "ambiguous"].copy()
    scopes = {"all": scored.index, "biomedical": scored.index[scored["biomedical"] == True], "non-biomedical": scored.index[scored["biomedical"] == False],
              "personal": scored.index[scored["gold_kind"] == "personal"]}
    rows = []
    for col, label, colour in SYSTEMS:
        pred = g[col].map(canon_class)
        for scope, idx in scopes.items():
            sub = scored.loc[idx]
            orig_ok = (pred.loc[idx] == sub["orig"]).mean()
            exp_ok = np.mean([p_ in a for p_, a in zip(pred.loc[idx], sub["accepted_set"])])
            rows.append({"system": label, "column": col, "colour": colour, "scope": scope, "n": len(sub),
                         "accuracy_original_labels": orig_ok, "accuracy_expert_labels": exp_ok, "ci95_expert": ci95(exp_ok, len(sub))})
        # check against the rescoring workbook: all 664 terms, any accepted reading (ambiguous included)
        rows[-4]["accuracy_expert_all664_check"] = np.mean([p_ in a for p_, a in zip(pred, g["accepted_set"])])
    return pd.DataFrame(rows)


def panel2_benchmark(out: Path) -> None:
    if not RESCORING.is_file():
        logger.warning("panel2_benchmark: %s missing", RESCORING); return
    df = typing_scores()
    summ = pd.read_excel(RESCORING, sheet_name="summary_systems").set_index("system")["all terms"]
    for _, r in df[df["scope"] == "all"].iterrows():
        ref = summ.get(r["column"])
        if ref is not None and abs(ref - r["accuracy_expert_all664_check"]) > 0.005:
            logger.warning("panel2_benchmark: %s recomputed %.3f vs workbook %.3f", r["column"], r["accuracy_expert_all664_check"], ref)
    from matplotlib.patches import Patch

    # --- fig2_typing_benchmark_a: original vs expert labels, all non-ambiguous audited terms
    a = df[df["scope"] == "all"].reset_index(drop=True)
    x = np.arange(len(a)); w = 0.38; n_all = int(a["n"].iloc[0])
    inputs = ["KG_nodes_eval/results/rescoring_v1.1_full.xlsx (gold_v1.1_terms)", "KG_nodes_eval/data/benchmark_audit_merged_v2.xlsx"]
    fig, ax1 = plt.subplots(figsize=(5.6, 5.6))
    ax1.bar(x - w / 2, a["accuracy_original_labels"], w, color=a["colour"], alpha=0.4)
    ax1.bar(x + w / 2, a["accuracy_expert_labels"], w, color=a["colour"], yerr=a["ci95_expert"], capsize=2, error_kw={"lw": 0.8})
    ax1.set_xticks(x); ax1.set_xticklabels(a["system"], fontsize=8, rotation=28, ha="right")
    ax1.set_ylim(0, 1.15); ax1.set_yticks(np.arange(0, 1.01, 0.2)); ax1.set_ylabel("typing accuracy")
    ax1.set_title(f"Typing on the synthetic benchmark\n({n_all} audited terms, ambiguous excluded)", fontsize=9, loc="left")
    ax1.legend(handles=[Patch(color="#777", alpha=0.4, label="original labels (LLM-generated)"), Patch(color="#777", label="expert labels, 95% CI")],
               frameon=False, fontsize=7, loc="upper left")
    save(fig, out, "fig2_typing_benchmark_a", inputs,
         f"Biolink typing on the synthetic benchmark: BiomedCAT (retriever alone, top-1; single-call variant; ReAct agent, all on gemma4:e4b-it-qat, "
         "4B parameters, local) against hosted models called one term at a time at temperature 0 (Claude Opus 5, DeepSeek-V4-Flash, gemma-4-31b, "
         f"mistral-small-24b). Accuracy on the {n_all} audited public terms whose label is not ambiguous, scored against the original LLM-generated "
         "label and against the expert label (any listed expert class accepted; NONE expected when the mention is not an entity); error bars, "
         "95% binomial interval. gpt-oss-120b, which generated the benchmark, is not shown.", a)

    # --- fig2_typing_benchmark_b: expert labels by scope of the term (personal-data terms are a subset of the biomedical ones)
    scopes = [("biomedical", "#4c72b0"), ("non-biomedical", "#dd8452"), ("personal", "#8172b2")]
    fig, ax2 = plt.subplots(figsize=(5.6, 5.6))
    w = 0.26
    for j, (scope, colour) in enumerate(scopes):
        b = df[df["scope"] == scope].reset_index(drop=True)
        n = int(b["n"].iloc[0])
        lo = np.minimum(b["ci95_expert"], b["accuracy_expert_labels"]); hi = np.minimum(b["ci95_expert"], 1 - b["accuracy_expert_labels"])
        ax2.bar(x + (j - 1) * w, b["accuracy_expert_labels"], w, color=colour, label=f"{scope} (n={n})", yerr=[lo, hi], capsize=1.5, error_kw={"lw": 0.6})
    ax2.set_xticks(x); ax2.set_xticklabels(a["system"], fontsize=8, rotation=28, ha="right"); ax2.set_ylim(0, 1.15); ax2.set_yticks(np.arange(0, 1.01, 0.2))
    ax2.set_ylabel("typing accuracy, expert labels")
    ax2.set_title("Typing on the synthetic benchmark, by scope of the term", fontsize=9, loc="left", pad=18)
    ax2.legend(frameon=False, fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3)
    save(fig, out, "fig2_typing_benchmark_b", inputs,
         f"Expert-label typing accuracy on the {n_all} non-ambiguous audited terms by scope: biomedical terms (n=411; class outside the information, "
         "administrative, attribute, activity, physical and planetary branches), non-biomedical terms (n=94), and the personal-data terms (n=5, "
         "named people, a subset of the biomedical terms), for which NONE is also accepted. Error bars, 95% binomial interval, clipped to [0, 1]. "
         "Systems and colours as in the previous figure.", df[df["scope"] != "all"])

    # --- fig2_linking_offline: FSHD deck, online resolvers vs fully local
    runs = {}
    pub_pair = {"online": PUB / "runs" / "FSHD", "offline": PUB / "runs" / "FSHD_offline"}
    use_pub = all((d / "eval_v2" / "metrics.json").is_file() for d in pub_pair.values())
    for tag in ("online", "offline"):
        base = pub_pair[tag] if use_pub else OUTPUT / f"_fshd_{tag}"
        mp = base / "eval_v2" / "metrics.json"; rp = base / "FSHD_BiomedCAT.json"
        if mp.is_file() and rp.is_file():
            m = json.loads(mp.read_text(encoding="utf-8"))["positives"]; r = json.loads(rp.read_text(encoding="utf-8"))
            ents = r["entities"]
            gold_v2 = PUB / "gold" / "fshd_slides_gold_v2.json" if (PUB / "gold" / "fshd_slides_gold_v2.json").is_file() else B / "data" / "gold" / "fshd_slides_gold_v2.json"
            linkable_ids = {g["id"] for g in json.loads(gold_v2.read_text(encoding="utf-8"))["positives"] if g.get("linkable", True)}
            found = [row for row in m["rows"] if row["match"] != "missed"]
            lf = [row for row in found if row["id"] in linkable_ids]
            not_linkable = [row["term"] for row in found if row["id"] not in linkable_ids]
            rate = lambda key: sum(bool(row.get(key) or row.get("curie_exact")) for row in lf) / len(lf) if lf else None
            runs[tag] = {"mode": tag, "n_mentions": len(ents), "linked_share": sum(1 for e in ents if e.get("curie")) / len(ents),
                         "n_found": int(round(m["recall_partial"] * m["n_gold"])), "n_found_linkable": m["n_found_linkable"],
                         "exact_all_found": m["linking_accuracy_exact_on_found"], "canonical_all_found": m["linking_accuracy_canonical_on_found"],
                         "clique_all_found": m["linking_accuracy_nodenorm_clique_on_found"],
                         "exact": rate("curie_exact"), "canonical": rate("curie_canonical"), "clique": rate("curie_same_clique"),
                         "n_linkable_found": len(lf), "found_not_linkable": "; ".join(not_linkable),
                         "recall_partial": m["recall_partial"], "typing_on_found": m["typing_accuracy_on_found"], "norm_seconds": None, "source": rel(mp)}
            for line in (base / "FSHD_BiomedCAT.log").read_text(encoding="utf-8", errors="replace").splitlines():
                mm = STAGE_RX["norm"].search(line)
                if mm:
                    runs[tag]["norm_seconds"] = float(mm.group(3)); runs[tag]["linked_count"] = int(mm.group(1)); break
    if len(runs) < 2:
        logger.warning("panel2_benchmark: offline/online FSHD runs missing under Output/_fshd_*"); return
    ld = pd.DataFrame([runs["online"], runs["offline"]])
    nl = runs["online"]["n_found_linkable"]
    nf = runs["online"]["n_found"]; nnl = len(runs["online"]["found_not_linkable"].split("; "))
    metrics = [("exact", "same CURIE"), ("canonical", "same canonical\nRTX-KG2c node"), ("clique", "same Node Normalizer\nclique")]
    fig, ax = plt.subplots(figsize=(5.6, 5.6))
    x = np.arange(len(metrics)); w = 0.36
    ax.bar(x - w / 2, [runs["online"][k] for k, _ in metrics], w, color="#dd8452", label="online: Name Resolver + ARAX, LLM judge")
    ax.bar(x + w / 2, [runs["offline"][k] for k, _ in metrics], w, color="#4c72b0", label="offline: local KG2c name table only")
    for i, (k, _) in enumerate(metrics):
        for dx, tag in ((-w / 2, "online"), (w / 2, "offline")):
            ax.text(i + dx, runs[tag][k] + 0.02, f"{runs[tag][k]:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels([l for _, l in metrics], fontsize=7.5); ax.set_ylim(0, 1.1); ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel(f"linking accuracy on the {nl} linkable expert terms found")
    ax.set_title("Linking on the FSHD deck: online resolvers vs fully local", fontsize=9, loc="left")
    ax.text(0.01, 0.99, f"{nf} expert terms found, {nnl} without any identifier in KG2c or Node Normalizer (excluded)\n"
            f"mentions that received an identifier: online {runs['online']['linked_share']:.0%} of {runs['online']['n_mentions']}, offline {runs['offline']['linked_share']:.0%}",
            transform=ax.transAxes, fontsize=7, color="#555", va="top")
    ax.legend(frameon=False, fontsize=7, loc="upper right", bbox_to_anchor=(1.0, 0.93))
    save(fig, out, "fig2_linking_offline", sorted(r_["source"] for r_ in runs.values()) + ["Output/_fshd_online/FSHD_BiomedCAT.json", "Output/_fshd_offline/FSHD_BiomedCAT.json"],
         "Entity linking on the FSHD deck in the two deployment modes. Reading and typing are identical in both (same 175 mentions, "
         f"recall {runs['online']['recall_partial']:.2f} and typing accuracy {runs['online']['typing_on_found']:.2f} on the 32 expert terms); only "
         "the linking stage differs. Online: candidate identifiers from the Name Resolver (unconstrained and constrained to the predicted "
         "Biolink class) and ARAX, canonicalised to RTX-KG2c, selected by the local judge model; offline: exact name and synonym match against "
         f"the local RTX-KG2c name table, no term leaves the machine. Accuracy on the {nl} found expert terms that have an identifier in RTX-KG2c "
         f"or Node Normalizer ({nnl} of the {nf} found terms have none and are excluded: {runs['online']['found_not_linkable']}), under three nested "
         "criteria: same CURIE string; same canonical RTX-KG2c node; same Node Normalizer clique (conflation off). The clique criterion is applied "
         "by the evaluation to both modes; the offline pipeline itself never calls Node Normalizer. Mentions that received any identifier, right "
         f"or wrong: online {runs['online']['linked_share']:.0%} of {runs['online']['n_mentions']}, offline {runs['offline']['linked_share']:.0%}. "
         f"Linking took {runs['online']['norm_seconds'] / 60:.1f} min online and {runs['offline']['norm_seconds'] / 60:.1f} min offline.", ld)


# ----------------------------------------------------------------------------------------------
PANELS = {"panel2": panel2, "panel2_gold": panel2_gold, "panel2_benchmark": panel2_benchmark}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--only", choices=sorted(PANELS), nargs="*", help="Build only these panels.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = Path(args.out)
    for name, fn in PANELS.items():
        if args.only and name not in args.only:
            continue
        try:
            fn(out)
        except Exception as e:  # one broken input must not stop the other panels
            logger.exception("%s failed: %s", name, e)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "figures_manifest.md"
    existing = manifest.read_text(encoding="utf-8").splitlines() if manifest.is_file() else []
    kept = [l for l in existing if l.startswith("| ") and not any(l.startswith(f"| {n} |") for n, _, _ in MANIFEST) and not l.startswith("| figure |") and not l.startswith("|---")]
    with manifest.open("w", encoding="utf-8") as f:
        f.write("# Figures manifest\n\n| figure | inputs | draft caption |\n|---|---|---|\n")
        for name, inputs, cap in MANIFEST:
            f.write(f"| {name} | {inputs} | {cap} |\n")
        for l in kept:  # rows of figures not rebuilt this time (legacy script)
            f.write(l + "\n")
    logger.info("%d figure(s) written to %s", len(MANIFEST), out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
