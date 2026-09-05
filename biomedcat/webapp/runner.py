"""Execute one console job in its own process: documents x profiles, with progress in job.json.

    python -m biomedcat.webapp.runner --job output/jobs/<id> [--resume]

Per document: OCR + NER once (reading scope = union of the selected profiles, see
jobs.merge_profiles), optional human review stop, normalization once, then one context graph
per profile with its Cytoscape export. Outputs land in the job directory:

    documents/<stem>/<stem>_stage1.json      slides and typed entities (nothing sent out yet)
    documents/<stem>/<stem>_review.csv       review sheet (review option only)
    documents/<stem>/<stem>_BiomedCAT.json   linked entities, run metadata (schema 1.1)
    graphs/<stem>/<profile>/                 nodes.csv, edges.csv, context_graph.graphml, summary.json, graph.json

The worker (jobs.Worker) captures stdout/stderr into job.log, so everything logged here is the
trace shipped with the results. Settings come from the environment set by the worker
(PREDICATE_PROFILE = merged reading profile, RESOLVERS_OFFLINE, EXISTENCE_GATE).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from biomedcat.config import settings, ROOT_PATH
from biomedcat.webapp.jobs import JobStore, Job, now_iso, PROFILES_DIR

logger = logging.getLogger("biomedcat.webapp.runner")


class Progress:
    """Stage bookkeeping written to job.json after every step."""

    def __init__(self, store: JobStore, job: Job):
        self.store, self.job = store, job
        self.t_stage = time.perf_counter()

    def start(self, stage: str, document: str, detail: str = "") -> None:
        self.t_stage = time.perf_counter()
        p = self.store.get(self.job.id).progress
        p.update({"stage": stage, "document": document, "detail": detail})
        self.store.update(self.job.id, progress=p)
        logger.info("[%s] %s: %s %s", self.job.id, document, stage, detail)

    def done(self, stage: str, document: str, detail: str = "") -> float:
        seconds = round(time.perf_counter() - self.t_stage, 1)
        p = self.store.get(self.job.id).progress
        p["done_units"] = int(p.get("done_units", 0)) + 1
        p["detail"] = detail
        p.setdefault("history", []).append({"document": document, "stage": stage, "seconds": seconds, "detail": detail})
        self.store.update(self.job.id, progress=p)
        logger.info("[%s] %s: %s done in %.1f s (%s)", self.job.id, document, stage, seconds, detail)
        return seconds


def _load_stage1(stage1_path: Path):
    from biomedcat.pipeline import Stage1Result
    from biomedcat.types import Slide, Entity

    payload = json.loads(stage1_path.read_text(encoding="utf-8"))
    return Stage1Result(filename=payload["filename"], ocr=[Slide(**s) for s in payload["slides"]],
                        ner=[Entity(**e) for e in payload["entities"]], ner_model=payload.get("ner_model", ""))


def _save_stage1(stage1, stage1_path: Path) -> None:
    stage1_path.write_text(json.dumps({
        "filename": stage1.filename, "ner_model": stage1.ner_model,
        "slides": [asdict(s) for s in stage1.ocr], "entities": [asdict(e) for e in stage1.ner],
    }, indent=2, ensure_ascii=False), encoding="utf-8")


def run_job(job_dir: Path, resume: bool = False) -> int:
    from biomedcat.pipeline import run_stage1, write_review, apply_review, _write_json
    from biomedcat.stages.norm import run_norm
    from biomedcat.stages.context_graph import run_context_graph, seeds_from_normalized, ContextGraphParams
    from biomedcat.types import PipelineResult
    from biomedcat import retrieval
    from biomedcat.webapp.graphs import build_graph_json, write_details_tables
    from biomedcat.webapp.summary import write_graph_summary

    store = JobStore(root=job_dir.parent)
    job = store.get(job_dir.name)
    progress = Progress(store, job)
    model = settings.classification_model_id
    t_job = time.perf_counter()
    outputs = dict(job.outputs or {})
    review_requested = bool((job.options or {}).get("review"))
    logger.info("job %s: %d document(s), profiles %s, options %s, reading profile %s",
                job.id, len(job.documents), job.profiles, job.options, settings.predicate_profile)

    try:
        for doc in job.documents:
            stem = doc["stem"]
            path = Path(doc["file"])
            if not path.is_absolute():
                path = Path(ROOT_PATH) / path
            doc_dir = job_dir / "documents" / stem
            doc_dir.mkdir(parents=True, exist_ok=True)
            stage1_path = doc_dir / f"{stem}_stage1.json"
            review_path = doc_dir / f"{stem}_review.csv"
            result_path = doc_dir / f"{stem}_BiomedCAT.json"
            local_only: set[str] = set()
            doc_out = outputs.setdefault(stem, {})

            if result_path.is_file() and doc_out.get("result_json") and all(
                    (job_dir / "graphs" / stem / p / "graph.json").is_file() for p in job.profiles):
                logger.info("%s: already complete, skipped", stem)
                continue

            # ---- stage 1: OCR + NER (local), reused when present ------------------------------
            if review_path.is_file() and stage1_path.is_file():
                progress.start("Review", stem, "applying the reviewed entity list")
                stage1, local_only, _ = apply_review(review_path)
                progress.done("Review", stem, f"{len(stage1.ner)} entities kept, {len(local_only)} linked locally only")
            elif stage1_path.is_file():
                stage1 = _load_stage1(stage1_path)
                logger.info("%s: stage 1 reused from %s", stem, stage1_path.name)
            else:
                progress.start("OCR", stem, f"{doc.get('pages', '?')} page(s)")
                t0 = time.perf_counter()
                # run_stage1 does OCR then NER; the OCR/NER split in the timings comes from the log.
                stage1 = run_stage1(str(path), settings, model_id=model)
                progress.done("OCR+NER", stem, f"{len(stage1.ocr)} slide(s), {len(stage1.ner)} entities")
                _save_stage1(stage1, stage1_path)
                doc_out["stage1_json"] = str(stage1_path)
                doc_out["n_slides"] = len(stage1.ocr)
                doc_out["n_entities"] = len(stage1.ner)
                store.update(job.id, outputs=outputs)
                if review_requested:
                    write_review(stage1, review_path, stage1_path)
                    doc_out["review_csv"] = str(review_path)
                    store.update(job.id, outputs=outputs, status="review",
                                 progress={**store.get(job.id).progress, "stage": "review", "document": stem,
                                           "detail": "waiting for the entity review"})
                    logger.info("%s: stopped for review (%s)", stem, review_path)
                    return 0

            # ---- stage 2: normalization (network unless offline) ------------------------------
            progress.start("Norm", stem, f"{len(stage1.ner)} entities to link")
            original = retrieval.is_shareable
            if local_only:
                def gated(term: str, types: set[str], _orig=original, _local=local_only) -> bool:
                    return term not in _local and _orig(term, types)
                retrieval.is_shareable = gated
            try:
                normalized = run_norm(stage1.ner, model_id=stage1.ner_model or model)
            finally:
                retrieval.is_shareable = original
            linked = sum(1 for r in normalized if r.curie)
            in_kg = sum(1 for r in normalized if r.kg2c_id)
            progress.done("Norm", stem, f"{linked}/{len(normalized)} linked, {in_kg} in KG2c")
            result = PipelineResult(filename=stage1.filename, ocr=stage1.ocr, ner=stage1.ner, norm=normalized, context_graph=None)
            _write_json(result, out_path=result_path, elapsed=time.perf_counter() - t_job, model_id=stage1.ner_model or model)
            doc_out.update({"result_json": str(result_path), "n_slides": len(stage1.ocr), "n_entities": len(stage1.ner),
                            "n_linked": linked, "n_in_kg2c": in_kg, "graphs": doc_out.get("graphs", {})})
            store.update(job.id, outputs=outputs)

            # ---- stage 3: one context graph per profile (local) -------------------------------
            seeds = seeds_from_normalized(normalized)
            for profile_id in job.profiles:
                out_dir = job_dir / "graphs" / stem / profile_id
                progress.start(f"graph:{profile_id}", stem, f"{len(seeds)} seed(s)")
                if not seeds:
                    doc_out["graphs"][profile_id] = {"error": "no entity linked to RTX-KG2c: nothing to expand"}
                    progress.done(f"graph:{profile_id}", stem, "skipped, no seed")
                    store.update(job.id, outputs=outputs)
                    continue
                try:
                    cg = run_context_graph(seeds, out_dir, ContextGraphParams(profile=str(PROFILES_DIR / f"{profile_id}.json")))
                    summary = asdict(cg)
                    graph = build_graph_json(out_dir, profile_id)
                    (out_dir / "graph.json").write_text(json.dumps(graph, ensure_ascii=False), encoding="utf-8")
                    write_details_tables(out_dir)
                    write_graph_summary(out_dir, store.get(job.id).to_dict(), doc=stem)
                    doc_out["graphs"][profile_id] = {
                        "dir": str(out_dir), "n_nodes": summary["n_nodes"], "n_edges": summary["n_edges"],
                        "n_seeds": summary["n_seeds"], "n_seeds_in_graph": summary["n_seeds_in_graph"],
                        "diameter": summary["diameter"], "n_components": summary["n_components"],
                        "seed_pairs_connected": summary["seed_pairs_connected"], "categories": summary["categories"],
                        "vocabulary_expansion": summary["vocabulary_expansion"], "seconds": summary["seconds"],
                        "n_clusters": len(graph.get("clusters", [])),
                    }
                    progress.done(f"graph:{profile_id}", stem, f"{summary['n_nodes']} nodes, {summary['n_edges']} edges")
                except Exception as e:  # one failing profile must not lose the others
                    logger.exception("%s / %s: context graph failed", stem, profile_id)
                    doc_out["graphs"][profile_id] = {"error": f"{type(e).__name__}: {e}"}
                    progress.done(f"graph:{profile_id}", stem, "failed")
                store.update(job.id, outputs=outputs)

        elapsed = round(time.perf_counter() - t_job, 1)
        prev = store.get(job.id)
        total = round((prev.elapsed_s or 0) + elapsed, 1) if resume else elapsed
        store.update(job.id, status="done", finished_at=now_iso(), elapsed_s=total, outputs=outputs, resume=False,
                     progress={**prev.progress, "stage": "done", "detail": ""})
        logger.info("job %s done in %.1f s", job.id, elapsed)
        return 0
    except Exception as e:
        logger.error("job %s failed: %s\n%s", job.id, e, traceback.format_exc())
        prev = store.get(job.id)
        store.update(job.id, status="failed", error=f"{type(e).__name__}: {e}", finished_at=now_iso(),
                     elapsed_s=round(time.perf_counter() - t_job, 1), outputs=outputs,
                     progress={**prev.progress, "stage": "failed"})
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--job", required=True, help="Job directory (output/jobs/<id>).")
    parser.add_argument("--resume", action="store_true", help="Continue after a review or a failure (stage 1 is reused).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", stream=sys.stdout)
    for noisy in ("httpx", "urllib3", "sentence_transformers", "chromadb", "PIL"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return run_job(Path(args.job), resume=args.resume)


if __name__ == "__main__":
    sys.exit(main())
