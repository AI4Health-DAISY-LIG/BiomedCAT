#!/usr/bin/env bash
# Rebuild the 12 context graphs (FSHD + Maladies musculaires x 6 profiles) with the current
# context-graph stage, then the presence table, Venn, DUX4 case and figures. ~5 min, CPU only.
# Use after a change of the context-graph stage (e.g. representation_identity_only, 10 Sept 2026)
# so that every graph-derived number in the paper comes from the same code.
#
#   bash scripts/rebuild_graphs_final.sh 2>&1 | tee output/rebuild_graphs_final.log
set -u
cd "$(dirname "$0")/.."
FIG="$(pwd)/../manuscripts/biomedcat_manuscript/figures"
PROFILES="uniform biochemical_actions clinical_mechanisms genetic_determinants pharmacological_intervention epidemiological_risk"
FIND_GENES="DUX4=NCBIGene:100288687,SMCHD1=NCBIGene:23347,DNMT3B=NCBIGene:1789,LRIF1=NCBIGene:55791,FRG1=NCBIGene:2483,FRG2=NCBIGene:448831,FAT1=NCBIGene:2195,MAPK14_p38a=NCBIGene:1432,MAPK11_p38b=NCBIGene:5600,MSTN=NCBIGene:2660,TFRC=NCBIGene:7037,BRD4=NCBIGene:23476,ADRB2=NCBIGene:154"
FIND_IDS="$(uv run python - <<'EOF'
import json
ids=[r["kg2c_id"] for r in json.load(open("data/eval/fshd_drugs_presence.json")) if r.get("kg2c_id")]
ids += ["NCBIGene:100288687","NCBIGene:23347","NCBIGene:1789","NCBIGene:55791","NCBIGene:2483","NCBIGene:448831","NCBIGene:2195","NCBIGene:1432","NCBIGene:5600","NCBIGene:2660","NCBIGene:7037","NCBIGene:23476","NCBIGene:154"]
print(",".join(dict.fromkeys(ids)))
EOF
)"
EXTRA="${CONTEXT_GRAPH_EXTRA_ARGS:-}"   # e.g. --representation-any-endpoint to reproduce the pre-10-Sept behaviour
echo "=== rebuild start $(date '+%F %T')  extra args: '$EXTRA'"
for P in $PROFILES; do
  echo "--- FSHD $P"
  uv run python -m biomedcat.stages.context_graph --result-json output/_final_run1/FSHD_BiomedCAT.json \
      --out "data/context_graph/FSHD_$P" --profile "data/profiles/$P.json" --find "$FIND_IDS" $EXTRA > "output/chain_I/rebuild_FSHD_$P.log" 2>&1 || echo "!!! FSHD $P failed"
  echo "--- Maladies musculaires $P"
  uv run python -m biomedcat.stages.context_graph --result-json "output/_final_mm/Maladies musculaires_BiomedCAT.json" \
      --out "data/context_graph/Maladies_musculaires_$P" --profile "data/profiles/$P.json" $EXTRA > "output/chain_I/rebuild_MM_$P.log" 2>&1 || echo "!!! MM $P failed"
done
uv run python scripts/graph_presence.py --doc FSHD --extra "$FIND_GENES" --out data/eval/fshd_presence_final
uv run python scripts/fshd_drugs_venn.py --presence data/eval/fshd_presence_final.json --out "$FIG"
uv run python scripts/fshd_dux4_mechanism_case.py | tail -40
uv run python scripts/make_figures.py --out "$FIG" | tail -5
uv run python - <<'EOF'
import json
for p in ["uniform","biochemical_actions","clinical_mechanisms","genetic_determinants","pharmacological_intervention","epidemiological_risk"]:
    for d in ["FSHD","Maladies_musculaires"]:
        try:
            s=json.load(open(f"data/context_graph/{d}_{p}/summary.json"))
            print(f"{d:22s} {p:30s} nodes {s['n_nodes']:5d} edges {s['n_edges']:6d} comps {s['n_components']:2d} pairs {s['seed_pairs_connected']:>8s} expansion {s['vocabulary_expansion']:.0f}")
        except Exception as e: print(d,p,"missing",e)
EOF
echo "=== rebuild done $(date '+%F %T')"
