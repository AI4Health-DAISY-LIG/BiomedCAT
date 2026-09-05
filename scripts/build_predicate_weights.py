#!/usr/bin/env python
"""Derive (or re-derive) the class scope and predicate weights of a profile, and print the review.

The computation lives in biomedcat.weights (entity-branch priorities -> class weights ->
predicate weights, see that module's docstring); the context-graph stage calls it on the fly
and caches the result under data/profiles/derived/. This script exposes it on the command line
to inspect a profile, compare alternatives (alpha, scope rule) or force a recomputation.

Usage (from the BiomedCAT root)
  uv run python scripts/build_predicate_weights.py --profile data/profiles/biochemical_actions.json
  uv run python scripts/build_predicate_weights.py --profile data/profiles/biochemical_actions.json --alpha 1.0 --scope-rule min --dry-run
  uv run python scripts/build_predicate_weights.py --branches            # entity branches and their KG2c size
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from biomedcat import weights as W  # noqa: E402

logger = logging.getLogger("build_predicate_weights")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profile", help="Profile JSON with `branch_weights` (biomedcat.profiles format).")
    parser.add_argument("--strata", default=str(ROOT / "data/biolink_strata.json"))
    parser.add_argument("--kg-dir", default=str(ROOT / "data/kg2c"))
    parser.add_argument("--alpha", type=float, default=None, help="Override the profile's depth factor exp(alpha * depth).")
    parser.add_argument("--scope-rule", choices=["max", "min"], default=None, help="Override how domain and range weights combine.")
    parser.add_argument("--dry-run", action="store_true", help="Print the review without writing the derived files.")
    parser.add_argument("--branches", action="store_true", help="List the entity branches of the model and exit.")
    parser.add_argument("--stratum", help="Print the default branch priorities of a stratum of data/profiles/strata_config.json and exit.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    model = W.BiolinkModel.load(args.strata)
    if args.branches:
        for b in model.branch_info():
            print(f"{b['name']:40s} {b['n_classes']:4d} classes  group={b['group'] or '-'}")
        return 0
    if args.stratum:
        raw = json.loads(Path(args.strata).read_text(encoding="utf-8"))
        print(json.dumps(W.default_branch_weights_for_stratum(args.stratum, raw, model), indent=2))
        return 0
    if not args.profile:
        parser.error("--profile is required (or --branches / --stratum)")

    path = Path(args.profile)
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not raw.get("branch_weights"):
        parser.error(f"{path} has no `branch_weights`: nothing to derive (legacy explicit-predicate profile)")
    weighting = dict(raw.get("weighting") or {})
    if args.alpha is not None:
        weighting["alpha"] = args.alpha
    if args.scope_rule:
        weighting["scope_rule"] = args.scope_rule
    raw["weighting"] = weighting

    if args.dry_run:
        bw = W.normalise_branch_weights(raw["branch_weights"], model)
        d = W.derive(bw, model, alpha=float(weighting.get("alpha", W.DEFAULT_ALPHA)), edge_counts=W.kg_edge_counts(args.kg_dir),
                     scope_rule=str(weighting.get("scope_rule", "max")), observed_counts=W.kg_observed_counts(args.kg_dir))
        print(W.review_markdown(raw.get("name", path.stem), d, bw, float(weighting.get("alpha", W.DEFAULT_ALPHA)), W.directionality_of(raw)))
        return 0

    # Force a recomputation: drop the cache, then derive and write.
    json_path, md_path = W.derived_paths(path)
    for p in (json_path, md_path):
        if p.is_file():
            p.unlink()
    payload = W.derive_for_profile(raw, path, Path(args.strata), args.kg_dir)
    print(f"{payload['profile']}: {payload['n_predicates']} predicates, {payload['n_classes_in_scope']} classes in scope, "
          f"branch priorities {json.dumps(payload['branch_priorities'])}")
    if payload["absent_in_scope"]:
        print(f"in scope but filtered out of the KG2c build ({len(payload['absent_in_scope'])}): {', '.join(payload['absent_in_scope'])}")
    print(f"-> {json_path}\n-> {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
