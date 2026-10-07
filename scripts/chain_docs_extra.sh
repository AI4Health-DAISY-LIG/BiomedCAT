#!/usr/bin/env bash
# Chain "docs extra" (4 Oct 2026): the full pipeline, PUBLISHED configuration, on the four
# research-axis decks and the multi-scale deck. Published configuration = the one of the benchmark
# and FSHD runs of 5-10 Sept 2026: agent ReAct, retriever dense + exemplars only (scispaCy not
# loaded, hence no BM25 leg; no lexical legs, which did not exist yet). The OCR-only step on the two
# PNG examples ran on 3 Oct (output/ocr_only/, unaffected by these switches).
#   launched detached by output/run_chain_docs_extra.cmd -> output/chain_docs_extra.log
set -u
cd "$(dirname "$0")/.."
export AGENT_MODE=react SCISPACY=0 RAG_BM25=0 RAG_LEXICAL_WEIGHT=0 RAG_KG2C_LEXICAL=0 RAG_KG2C_LEXICAL_WEIGHT=0
step() { echo; echo "=== [$1] $(date '+%F %T')"; }
echo "=== chain docs extra start $(date '+%F %T')  AGENT_MODE=$AGENT_MODE SCISPACY=$SCISPACY RAG_BM25=$RAG_BM25 RAG_LEXICAL_WEIGHT=$RAG_LEXICAL_WEIGHT RAG_KG2C_LEXICAL=$RAG_KG2C_LEXICAL RAG_KG2C_LEXICAL_WEIGHT=$RAG_KG2C_LEXICAL_WEIGHT"
for NAME in VoieA VoieB VoieC VoiesA-C FSHD1multiscales; do
  step "pipeline: $NAME"
  rm -f "output/${NAME}_BiomedCAT.json" "output/${NAME}_review.csv"
  uv run python -m biomedcat.pipeline "Dataset/${NAME}.pdf" > "output/pipeline_${NAME}.stdout.log" 2>&1 || echo "!!! $NAME failed"
  mkdir -p "output/_final_${NAME}"
  cp "output/${NAME}_BiomedCAT.json" "output/_final_${NAME}/" 2>/dev/null
  cp "output/${NAME}_BiomedCAT.log" "output/_final_${NAME}/" 2>/dev/null
  grep -E "scispaCy|BM25|Lexical table|OCR done|NER done|Norm done|Context graph done" "output/_final_${NAME}/${NAME}_BiomedCAT.log" 2>/dev/null | sed 's/^[0-9-]* [0-9:,]* - //' | sort -u
done
echo "=== chain docs extra done $(date '+%F %T')"
