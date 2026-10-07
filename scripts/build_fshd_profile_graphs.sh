#!/usr/bin/env bash
# Context graphs of the FSHD publication run (5 Oct 2026) under the six shipped profiles, then the
# presence of the FSHD drug list in each graph and its Venn figure. CPU only, a few minutes.
#
#   bash scripts/build_fshd_profile_graphs.sh 2>&1 | tee output/build_fshd_profile_graphs.log
set -u
cd "$(dirname "$0")/.."
RUN="data/publication/runs/FSHD/FSHD_BiomedCAT.json"
OUT="data/context_graph_pub"
FIG="$(pwd)/../manuscripts/biomedcat_manuscript/figures"
PROFILES="uniform biochemical_actions clinical_mechanisms genetic_determinants pharmacological_intervention epidemiological_risk"
FIND_GENES="DUX4=NCBIGene:100288687,SMCHD1=NCBIGene:23347,DNMT3B=NCBIGene:1789,LRIF1=NCBIGene:55791,MAPK14_p38a=NCBIGene:1432,MAPK11_p38b=NCBIGene:5600,MSTN=NCBIGene:2660,ADRB2=NCBIGene:154"
FIND_IDS="$(uv run python - <<'EOF'
import json
ids = [r["kg2c_id"] for r in json.load(open("data/eval/fshd_drugs_presence.json", encoding="utf-8")) if r.get("kg2c_id")]
ids += ["NCBIGene:100288687", "NCBIGene:23347", "NCBIGene:1789", "NCBIGene:55791", "NCBIGene:1432", "NCBIGene:5600", "NCBIGene:2660", "NCBIGene:154"]
print(",".join(dict.fromkeys(ids)))
EOF
)"
mkdir -p "$OUT" output
echo "=== start $(date '+%F %T')"
for P in $PROFILES; do
  echo "--- FSHD $P"
  uv run python -m biomedcat.stages.context_graph --result-json "$RUN" --out "$OUT/FSHD_$P" \
      --profile "data/profiles/$P.json" --find "$FIND_IDS" > "output/build_fshd_$P.log" 2>&1 || echo "!!! FSHD $P failed"
done
uv run python scripts/graph_presence.py --doc FSHD --graph-dir "$OUT" --extra "$FIND_GENES" --out data/eval/fshd_presence_pub
uv run python scripts/fshd_drugs_venn.py --presence data/eval/fshd_presence_pub.json --out "$FIG"
echo "=== done $(date '+%F %T')"
