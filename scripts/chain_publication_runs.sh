#!/usr/bin/env bash
# Publication runs (5 Oct 2026): every deck of the dataset through the full pipeline in the
# published configuration, with the existence gate on the production Name Resolver (same service
# as the linking) and per-second memory / GPU sampling (scripts/measure_pipeline_resources.py).
# FSHD is run twice: online (default) and fully offline (RESOLVERS_OFFLINE=1), for the linking
# comparison. Each run lands in data/publication/runs/<name>/ (pipeline JSON + log, context graph,
# resources.json, resources_samples.csv). The two FSHD runs are then scored against the v2 gold
# suite (scripts/eval_gold.py), every run against the harmonized gold (scripts/eval_gold_decks.py),
# and the environment (hardware, software, remote service versions) is recorded.
#   launched detached: bash scripts/chain_publication_runs.sh > Output/chain_publication_runs.log 2>&1
set -u
cd "$(dirname "$0")/.."
export AGENT_MODE=react SCISPACY=0 RAG_BM25=0 RAG_LEXICAL_WEIGHT=0 RAG_KG2C_LEXICAL=0 RAG_KG2C_LEXICAL_WEIGHT=0 PYTHONIOENCODING=utf-8
unset GATE_NAMERES_URL NAMERES_ES_URL RESOLVERS_OFFLINE
RUNS=data/publication/runs
mkdir -p "$RUNS"
step() { echo; echo "=== [$1] $(date '+%F %T')"; }
echo "=== publication runs start $(date '+%F %T')"
uv run python scripts/record_environment.py --out data/publication/environment_before.json

for DOC in VoieA VoieB VoieC VoiesA-C "Maladies musculaires" FSHD1multiscales FSHD chemicals; do
  EXT=pdf; [ "$DOC" = "chemicals" ] && EXT=png
  step "pipeline $DOC (online)"
  uv run python scripts/measure_pipeline_resources.py "Dataset/${DOC}.${EXT}" --out "$RUNS/${DOC}" || echo "!!! $DOC failed"
done

step "pipeline FSHD (offline, RESOLVERS_OFFLINE=1)"
RESOLVERS_OFFLINE=1 uv run python scripts/measure_pipeline_resources.py Dataset/FSHD.pdf --out "$RUNS/FSHD_offline" || echo "!!! FSHD offline failed"

for TAG in FSHD FSHD_offline; do
  step "score $TAG against the v2 gold suite"
  uv run python scripts/eval_gold.py score --gold data/gold/fshd_slides_gold_v2.json \
      --result "$RUNS/$TAG/FSHD_BiomedCAT.json" --out "$RUNS/$TAG/eval_v2" > "$RUNS/$TAG/eval_v2.stdout.log" 2>&1 || echo "!!! scoring $TAG failed"
  grep -E "recall|accuracy" "$RUNS/$TAG/eval_v2/metrics.md" 2>/dev/null
done

step "score every run against the harmonized gold"
uv run python scripts/eval_gold_decks.py --runs-root "$RUNS" --out data/publication/gold_eval || echo "!!! eval_gold_decks failed"
uv run python scripts/record_environment.py --out data/publication/environment.json
echo "=== publication runs done $(date '+%F %T')"
