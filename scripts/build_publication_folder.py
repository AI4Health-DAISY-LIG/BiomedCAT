#!/usr/bin/env python
"""Assemble data/publication/: every source file behind the figures and tables of the article.

The folder is the deposit for the Data Availability Statement (one archive, replicable):
  runs/<name>/            one directory per document run, written directly by
                          scripts/chain_publication_runs.sh (pipeline JSON and log, context graph,
                          resources.json and resources_samples.csv from the memory/time sampler,
                          eval_v2/ for the two FSHD runs)
  gold/                   the harmonized expert gold (data/gold/gold_harmonized.*, verification
                          report) and the FSHD gold v2 files it was built from
  gold_eval/              scripts/eval_gold_decks.py on the publication runs (per_term, summary)
  benchmark/              the public slice of the synthetic typing benchmark (KG_nodes_eval
                          release/v1: Parquet, JSONL, CANARY.txt, split statistics, generation
                          statistics), the expert audit and the audited rescoring workbook
  environment.json        hardware, software, Ollama models and remote service versions recorded
                          at the end of the runs (environment_before.json: at their start)
  figures_data/           the <figure>_data.csv written by scripts/make_figures.py
  README.md, MANIFEST.json (size and SHA-256 of every file)

Run it after the chain and after make_figures.py; it copies the static inputs, never the runs.
Usage:  uv run python scripts/build_publication_folder.py
"""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from datetime import date
from pathlib import Path

B = Path(__file__).resolve().parents[1]
K = B.parent / "KG_nodes_eval"
PUB = B / "data" / "publication"
FIG = B.parent / "manuscripts" / "biomedcat_manuscript" / "figures"

STATIC = {  # destination (relative to PUB) -> source
    "gold/gold_harmonized.xlsx": B / "data/gold/gold_harmonized.xlsx",
    "gold/gold_harmonized.json": B / "data/gold/gold_harmonized.json",
    "gold/gold_harmonized_verification.md": B / "data/gold/gold_harmonized_verification.md",
    "gold/fshd_slides_gold_v2.json": B / "data/gold/fshd_slides_gold_v2.json",
    "gold/fshd_implicit_expected.json": B / "data/gold/fshd_implicit_expected.json",
    "benchmark/benchmark_v1_public.parquet": K / "release/v1/public/benchmark_v1_public.parquet",
    "benchmark/benchmark_v1_public.jsonl": K / "release/v1/public/benchmark_v1_public.jsonl",
    "benchmark/CANARY.txt": K / "release/v1/CANARY.txt",
    "benchmark/split_stats.json": K / "release/v1/split_stats.json",
    "benchmark/benchmark_v1.stats.json": K / "data/benchmark_v1.stats.json",
    "benchmark/benchmark_audit_merged_v2.xlsx": K / "data/benchmark_audit_merged_v2.xlsx",
    "benchmark/rescoring_v1.1_full.xlsx": K / "results/rescoring_v1.1_full.xlsx",
    "benchmark/biolink_classes_nested.json": B / "data/biolink_classes_nested.json",
    "gate_modes_v2_prod.json": B / "data/eval/gate_modes_v2_prod.json",
    "exemplars/biolink_exemplars.parquet": B / "data/biolink_exemplars.parquet",
    "exemplars/exemplars_v1.stats.json": K / "data/exemplars_v1.stats.json",
}
# Input documents of the article (slide decks and images), with their metadata sheet and attribution.
STATIC.update({f"input_documents/{p.name}": p for p in (B / "Dataset").iterdir()
               if p.suffix.lower() in (".pdf", ".png", ".jpg", ".jpeg", ".csv", ".md")})


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    PUB.mkdir(parents=True, exist_ok=True)
    missing = []
    for rel, src in STATIC.items():
        if not src.is_file():
            missing.append(str(src)); continue
        dst = PUB / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    fd = PUB / "figures_data"
    fd.mkdir(exist_ok=True)
    for p in sorted(FIG.glob("*_data.csv")) + [FIG / "figures_manifest.md"]:
        if p.is_file():
            shutil.copy2(p, fd / p.name)

    files = sorted(p for p in PUB.rglob("*") if p.is_file() and p.name not in ("MANIFEST.json", "README.md"))
    manifest = {"built": date.today().isoformat(), "n_files": len(files),
                "files": [{"path": str(p.relative_to(PUB)).replace("\\", "/"), "bytes": p.stat().st_size, "sha256": sha256(p)} for p in files]}
    (PUB / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    runs = sorted(d.name for d in (PUB / "runs").iterdir() if d.is_dir()) if (PUB / "runs").is_dir() else []
    readme = f"""# BiomedCAT: source data of the article (built {date.today().isoformat()})

Every number and figure of the article is computed from the files in this folder by the scripts of
the BiomedCAT repository (`scripts/`). Nothing here is edited by hand.

| folder | content | produced by |
|---|---|---|
| `runs/<name>/` | one pipeline run per document ({', '.join(runs) or 'pending'}): `<doc>_BiomedCAT.json` (reading, typed mentions, identifiers, context graph), `<doc>_BiomedCAT.log`, `<doc>_context_graph/`, `resources.json` and `resources_samples.csv` (per-second RAM, GPU and model residency), `eval_v2/` (FSHD: scores against the v2 gold) | `scripts/chain_publication_runs.sh` -> `scripts/measure_pipeline_resources.py`, `scripts/eval_gold.py` |
| `gold/` | the harmonized expert gold of the decks (terms, kinds, explicit/implicit, curation status) with its cell-by-cell verification against the curation sheets; the FSHD gold v2 files | `scripts/build_gold_harmonized.py` |
| `gold_eval/` | reading recall and typing accuracy of every run against the harmonized gold | `scripts/eval_gold_decks.py --runs-root data/publication/runs` |
| `benchmark/` | public slice of the synthetic Biolink typing benchmark (1,075 mentions, canary string), the expert audit (707 rows, 664 mentions) and the audited rescoring of every typing system | `KG_nodes_eval` (generate_benchmark.py, split_benchmark.py, rescore_audit_full.py) |
| `environment.json` | hardware, software, Ollama models, remote service versions (Name Resolver, Node Normalizer, ARAX) | `scripts/record_environment.py` |
| `exemplars/` | the retrieval exemplars indexed with the Biolink class definitions (3,712 mentions, generated with mistral-small and verified) | `scripts/build_biolink_exemplars.py` |
| `input_documents/` | the input documents of the article (slide decks and images, PDF/PNG) with `documents_metadata.csv`; `chemicals.png` is CC BY-SA 4.0 (Banushi B, Polito VA, via Wikimedia Commons, see `README.md` there) | authors |
| `figures_data/` | the data behind each figure, one CSV per figure, and the figure manifest with draft captions | `scripts/make_figures.py` |

Configuration of every run: `AGENT_MODE=react SCISPACY=0 RAG_BM25=0 RAG_LEXICAL_WEIGHT=0 RAG_KG2C_LEXICAL=0`,
existence gate `union` on the production Name Resolver, resolvers online except `runs/FSHD_offline`
(`RESOLVERS_OFFLINE=1`). The private slice of the benchmark (281 mentions) is deliberately absent.

Rebuild: `bash scripts/chain_publication_runs.sh`, then `uv run python scripts/make_figures.py`,
then `uv run python scripts/build_publication_folder.py`. `MANIFEST.json` lists every file with its SHA-256.
"""
    (PUB / "README.md").write_text(readme, encoding="utf-8")
    print(f"{len(files)} files in {PUB}; runs: {runs}")
    if missing:
        print("[!] missing sources:", *missing, sep="\n  ")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
