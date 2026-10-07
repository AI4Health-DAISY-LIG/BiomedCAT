#!/usr/bin/env bash
# Redo chain (4 Oct 2026): waits for scripts/chain_docs_extra.sh to finish, then reruns, in the
# published configuration and with a UTF-8 console, every deck whose JSON is missing (VoieA was
# lost to a UnicodeEncodeError in the console summary, fixed in biomedcat/pipeline.py since).
#   launched detached by output/run_chain_docs_redo.cmd -> output/chain_docs_redo.log
set -u
cd "$(dirname "$0")/.."
export AGENT_MODE=react SCISPACY=0 RAG_BM25=0 RAG_LEXICAL_WEIGHT=0 RAG_KG2C_LEXICAL=0 RAG_KG2C_LEXICAL_WEIGHT=0 PYTHONIOENCODING=utf-8
echo "=== redo chain start $(date '+%F %T'), waiting for chain docs extra"
until grep -q "=== chain docs extra done" output/chain_docs_extra.log 2>/dev/null; do sleep 60; done
echo "=== chain docs extra finished, $(date '+%F %T')"
for NAME in VoieA VoieB VoieC VoiesA-C FSHD1multiscales; do
  if [ -f "output/_final_${NAME}/${NAME}_BiomedCAT.json" ]; then echo "--- $NAME: JSON present, kept"; continue; fi
  echo; echo "=== [redo pipeline: $NAME] $(date '+%F %T')"
  rm -f "output/${NAME}_BiomedCAT.json" "output/${NAME}_review.csv"
  uv run python -m biomedcat.pipeline "Dataset/${NAME}.pdf" > "output/pipeline_${NAME}.stdout.log" 2>&1 || echo "!!! $NAME failed"
  mkdir -p "output/_final_${NAME}"
  cp "output/${NAME}_BiomedCAT.json" "output/_final_${NAME}/" 2>/dev/null
  cp "output/${NAME}_BiomedCAT.log" "output/_final_${NAME}/" 2>/dev/null
  grep -E "OCR done|NER done|Norm done|Context graph done" "output/_final_${NAME}/${NAME}_BiomedCAT.log" 2>/dev/null | sed 's/^[0-9-]* [0-9:,]* - //'
done
echo "=== redo chain done $(date '+%F %T')"
