# BiomedCAT

Biomedical Context Analysis Tool: reads scientific slide decks (PDF or images), extracts the
biomedical entities they mention, types them against the Biolink model, links them to RTX-KG2c
identifiers and extracts a **user-defined contextual subgraph** of RTX-KG2c around them, one per
preference profile (biochemical actions, clinical mechanisms, uniform baseline).

Everything runs on one computer with 16 GB of RAM: a local vision model reads the slides, a local
language model types the entities, DuckDB extracts the subgraph. Only entity names are sent to the
Translator name resolvers, and that can be switched off.

## Quick start (researchers): the local console

Prerequisites: [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com/) with the models
`qwen2.5vl:3b`, `gemma4:e4b-it-qat`, `llama-guard3:1b`, `llama3:8b` (`ollama pull <model>`),
poppler (`pdfinfo`/`pdftoppm` on the PATH) and the filtered RTX-KG2c tables in `data/kg2c/`
(see *Building the knowledge graph* below).

```bash
uv sync
uv run python scripts/install_scispacy_model.py   # tokenizer of the sparse (BM25) retrieval leg, once
uv run python -m biomedcat.webapp
```

These defaults keep every retrieval leg on; the published results use `SCISPACY=0 RAG_BM25=0 RAG_LEXICAL_WEIGHT=0 RAG_KG2C_LEXICAL=0` (set them before the last command; the Docker image sets them). The second line installs the scispaCy `en_core_sci_sm` model, which cannot be declared in
`pyproject.toml` (its only release pins spaCy < 3.8); without it the retriever runs dense-only and
logs a warning. The console opens in your browser at http://127.0.0.1:8765 (localhost only). From there:

1. **Drop slide decks** (PDF, PNG, JPG), single files or whole folders, on the *New analysis* card.
   Choose the **profiles** to run (one context graph per profile, the slides are read once) and the
   options: human review of the entity list before anything is sent to the resolvers, offline
   linking, existence gate.
2. The job joins the **queue** on the left with its estimated duration, then its live progress and
   stage timings. Jobs run one at a time; keep the computer awake and the console open.
3. When a job is done, **Visualize my results** opens the Cytoscape viewer: nodes clustered by
   Biolink group and coloured by Biolink category, seeds and profile scope outlined, node names and
   edge types visible, a details panel with provenance (slides, seeds), RTX-KG2c descriptions and
   publications (when the extras tables are built), search, filters, PNG export, and a **GO
   enrichment** of the graph's genes (local hypergeometric test on RTX-KG2c annotations, or
   g:Profiler online on request).
4. **Download** packs the whole job: entities JSON, review sheet, per-profile graphs (CSV, GraphML,
   Cytoscape JSON, details tables with KG2c publications), enrichment results, the full log and a
   trace README.

Job directories live in `output/jobs/<job id>/`; uploaded documents are copied to `Dataset/<job id>/`.

### Optional: RTX-KG2c descriptions, publications and GO annotations

```bash
uv run python scripts/build_kg2c_extras.py --nodes <kg2c-nodes.jsonl.gz> --edges <kg2c-edges.jsonl.gz>
```

writes `node_details.parquet`, `edge_publications.parquet` and `gene_go.parquet` into `data/kg2c/`.
Without them the viewer still works, but shows no description or publication, and the local GO
enrichment only uses the annotations kept in the filtered graph.

### Offline viewer

The viewer loads Cytoscape.js from the CDNs; for a fully offline machine, place `cytoscape.min.js`
in `biomedcat/webapp/static/vendor/` (see the README there).

## Installation with Docker (end users)

For users who prefer not to install Python, uv, poppler and Ollama themselves. Requires
[Docker Desktop](https://www.docker.com/products/docker-desktop/) (Windows, macOS) or Docker Engine
with the compose plugin (Linux), 16 GB of RAM and about 25 GB of free disk space (models, image and knowledge-graph tables).

1. **Get the code and the knowledge graph.** Clone this repository, then place the filtered
   RTX-KG2c tables in `data/kg2c/` (see *Building the knowledge graph* below; the build needs the
   RTX-KG2c 2.10.1 dumps and runs once, with `uv`, outside Docker). The reading profiles
   (`data/profiles/`) and the retrieval exemplars (`data/biolink_exemplars.parquet`) are in the
   repository.

2. **Start the console and the model server.**

   ```bash
   docker compose up -d --build
   ```

3. **Download the four models** into the Ollama container (once, several GB):

   ```bash
   docker compose exec ollama ollama pull qwen2.5vl:3b
   docker compose exec ollama ollama pull gemma4:e4b-it-qat
   docker compose exec ollama ollama pull llama-guard3:1b
   docker compose exec ollama ollama pull llama3:8b
   ```

4. **Open http://localhost:8765** and use the console as described in *Quick start* above. The
   first job builds the Biolink index and downloads the embedding model (a few minutes, once).

The image runs the configuration of the published results (dense and exemplar retrieval, ReAct
agent, `union` existence gate). Documents, results and the Biolink index stay in the local folders
`Dataset/`, `output/` and `data/`; the console is reachable from this computer only. To link
without sending any entity name to the Translator services, tick *offline linking* in the console
or set `RESOLVERS_OFFLINE=1` in `docker-compose.yml`. With an NVIDIA GPU, uncomment the `deploy`
block of the `ollama` service; without one the models run on the CPU, more slowly.

```bash
docker compose logs -f biomedcat   # follow a job
docker compose down                # stop (models and results are kept)
```

## Command line

```bash
uv run python -m biomedcat.pipeline Dataset/slides.pdf            # one file, default profile
uv run python -m biomedcat.pipeline --review Dataset/slides.pdf   # stop after reading, write the review CSV
uv run python -m biomedcat.pipeline --resume output/slides_review.csv
uv run python -m biomedcat.stages.context_graph --result-json output/slides_BiomedCAT.json --profile data/profiles/clinical_mechanisms.json --out output/slides_clinical
```

Environment variables (see `biomedcat/config.py`): `OLLAMA_URL`, `OCR_MODEL_ID`, `CLASSIFICATION_MODEL_ID`,
`PREDICATE_PROFILE`, `RESOLVERS_OFFLINE`, `REVIEW_MODE`, `EXISTENCE_GATE` (`union`, `nameres`, `llm`, `off`; the resolver leg queries the production Name Resolver, `GATE_NAMERES_URL`),
`KG2C_DIR`, `RAG_EXEMPLARS` (`off` to disable the exemplar index), `AGENT_TREE_SKELETON`, `AGENT_SEARCH_TOP_K`,
`AGENT_MODE` (`react`, the default and the published configuration: the tool loop; `decide`: one retrieval of the
term, the top candidates plus their is_a neighbourhood, one decision call with hidden reasoning; better on the
original synthetic labels, not confirmed on the expert-audited ones), `RAG_BM25` (`0` to disable the sparse leg).
`BIOLINK_MODEL_SOURCE` pins the Biolink model (default: the v4.4.4 release on GitHub; a local YAML path works offline).
`SCISPACY=0` runs without the scispaCy model even when it is installed (rule-based sentencizer, basic tokenizer, no
BM25 leg). The configuration of every published number (benchmark runs and document runs) is
`SCISPACY=0 RAG_BM25=0 RAG_LEXICAL_WEIGHT=0 RAG_KG2C_LEXICAL=0`: dense + exemplar retrieval only, agent ReAct.

## Building the knowledge graph

```bash
uv run python scripts/build_kg2c_parquet.py --nodes <kg2c-nodes.jsonl.gz> --edges <kg2c-edges.jsonl.gz>
```

filters RTX-KG2c 2.10.1 to structural noise only -- aligned directly to `data/biolink-model.yaml`
(no predicate or category that isn't literally a slot or a class of that file). RTX-KG2c's
redundantly `biolink_`-doubled predicates (`biolink:biolink_treats`, ...) are normalized to their
real Biolink name and merged with the correctly-labeled edges rather than dropped -- for `treats`
the doubled form outnumbers the well-formed one ~180 to 1; a doubled predicate with no real name
even once normalized (`biolink:biolink_mentioned_in_trials_for`) is dropped like any other
off-model predicate. No SemMedDB, no equivalence edges, no overly generic predicate such as
`related_to` -- and writes
`data/kg2c/{edges,nodes,equivalents,degrees,predicates}.parquet`. This build knows nothing about
preference profiles: the same Parquet serves every profile, current and future, so creating a new
profile (or changing one) never requires rebuilding it. Predicate and knowledge-source weights,
Biolink-category scope and prefix restriction (e.g. `subclass_of` limited to disease/phenotype
ontologies) are all resolved at query time by the context-graph stage from the active profile
(`biomedcat.profiles`, `biomedcat.weights`).
The Biolink index of the typing agent is built on first run (`RAG_REINDEX=1` to rebuild); the
exemplar index reads `data/biolink_exemplars.parquet` when present.

## Evaluation

The typing benchmark, the baselines and the evaluation scripts live in the companion repository
`KG_nodes_eval`; the gold test suite of the FSHD deck and the document-level scripts are in
`scripts/`: `eval_gold.py` and `eval_gold_decks.py` (expert gold), `run_adversarial.py` and `eval_gate_modes.py`
(existence gate), `measure_pipeline_resources.py` (memory and time), `chain_publication_runs.sh` (the runs of the
article), `build_publication_folder.py` (the deposited archive), `make_figures.py`, `make_voie_panel.py`,
`make_fig4_dux4.py`, `voie_multiscale_case.py` and `build_fshd_profile_graphs.sh` (figures and tables).
Scripts and code not used by the tool or the article are kept locally in `archive/` (not distributed).

## Repository layout

- `biomedcat/` pipeline: `stages/ocr.py`, `stages/ner_agent.py` (typing agent), `stages/norm.py`
  (linking), `stages/context_graph.py`, `profiles.py`, `retrieval.py`, `pipeline.py`.
- `biomedcat/webapp/` the local console (FastAPI + static interface, job queue, runner, Cytoscape export, enrichment).
- `data/profiles/` preference profiles; `data/biolink_strata.json` reading scopes derived from the Biolink model.
- `scripts/` build and evaluation scripts; `tests/` pytest suites (`uv run python -m pytest tests`).
- `Dockerfile`, `docker-compose.yml`: the console and its Ollama model server for end users (see *Installation with Docker*).

## AI usage
Code development was assisted by generative AI models for testing, refactoring, and UI/Docker packaging.
Generative AI was used to create synthetic datasets for benchmarking.