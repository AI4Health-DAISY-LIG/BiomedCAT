"""FastAPI application of the BiomedCAT console (static interface + JSON API + worker).

    uv run python -m biomedcat.webapp             # http://127.0.0.1:8765

Environment: BIOMEDCAT_PORT (default 8765), BIOMEDCAT_HOST (default 127.0.0.1),
BIOMEDCAT_NO_WORKER=1 (serve without executing jobs, used by the tests),
BIOMEDCAT_NO_BROWSER=1 (do not open the browser).
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import threading
import time
import webbrowser
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from biomedcat.config import settings, ROOT_PATH
from biomedcat.webapp.jobs import JobStore, Worker, list_profiles, estimate_seconds, now_iso, ACTIVE
from biomedcat.webapp import graphs as graphs_mod
from biomedcat.webapp import enrichment
from biomedcat.webapp import summary as summary_mod

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"
SUPPORTED = {ext.lower() for ext in settings.supported_docs}

app = FastAPI(title="BiomedCAT console", version="1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

store: JobStore = JobStore()
worker: Optional[Worker] = None


def start_worker() -> None:
    global worker
    if worker is None and os.getenv("BIOMEDCAT_NO_WORKER", "0") != "1":
        worker = Worker(store)
        worker.start()


@app.on_event("startup")
def _startup() -> None:
    start_worker()


@app.exception_handler(FileNotFoundError)
async def _not_found(request, exc: FileNotFoundError) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


# ------------------------------------------------------------------------------------------
# Pages
# ------------------------------------------------------------------------------------------

@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/viewer")
def viewer() -> FileResponse:
    return FileResponse(STATIC_DIR / "viewer.html")


# ------------------------------------------------------------------------------------------
# Status, profiles
# ------------------------------------------------------------------------------------------

def _ollama_models() -> tuple[bool, list[str]]:
    try:
        import requests

        r = requests.get(f"{settings.ollama_url.rstrip('/')}/api/tags", timeout=2)
        r.raise_for_status()
        return True, sorted(m.get("name", "") for m in r.json().get("models", []))
    except Exception:
        return False, []


@app.get("/api/status")
def status() -> dict:
    ok, models = _ollama_models()
    required = {"ocr": settings.ocr_model_id, "typing": settings.classification_model_id,
                "screening": settings.sanitization_model_id, "existence": settings.existence_model_id}
    installed = {k: any(m.split(":")[0] == v.split(":")[0] and (v in models or v + ":latest" in models) for m in models) for k, v in required.items()}
    return {
        "ollama_reachable": ok, "models_required": required, "models_installed": installed,
        "kg2c": Path(settings.kg2c_dir, "edges.parquet").is_file(), "extras": graphs_mod.extras_available(),
        "offline_default": settings.resolvers_offline, "gate_default": settings.existence_gate,
        "sec_per_page": store.sec_per_page(), "worker": worker is not None and worker.is_alive(),
        "running": worker.current if worker else None, "dataset_dir": str(store.dataset_root), "jobs_dir": str(store.root),
    }


@app.get("/api/profiles")
def profiles() -> list[dict]:
    return list_profiles()


class EstimateRequest(BaseModel):
    pages: int
    n_profiles: int


@app.post("/api/estimate")
def estimate(req: EstimateRequest) -> dict:
    return {"seconds": estimate_seconds(req.pages, req.n_profiles, store.sec_per_page())}


# ------------------------------------------------------------------------------------------
# Jobs
# ------------------------------------------------------------------------------------------

def _safe_relative(path: str) -> Path:
    """Relative path from the browser (folder drops) reduced to safe components."""
    parts = []
    for part in re.split(r"[\\/]+", path or ""):
        part = re.sub(r"[^A-Za-z0-9 ._\-()]+", "_", part).strip()
        if part and part not in (".", ".."):
            parts.append(part)
    return Path(*parts) if parts else Path("document")


def _count_pages(path: Path) -> int:
    if path.suffix.lower() != ".pdf":
        return 1
    try:
        from pdf2image import pdfinfo_from_path

        return int(pdfinfo_from_path(str(path)).get("Pages", 1))
    except Exception as e:  # poppler missing or unreadable PDF: keep the job, estimate one page
        logger.warning("page count failed for %s: %s", path.name, e)
        return 1


def _job_view(job, with_log: bool = False) -> dict:
    d = job.to_dict()
    d["live_detail"] = store.live_detail(job)
    if job.status == "running" and job.started_at:
        try:
            from datetime import datetime

            started = datetime.fromisoformat(job.started_at)
            d["running_s"] = round((datetime.now(started.tzinfo) - started).total_seconds(), 0)
        except ValueError:
            d["running_s"] = None
    if with_log:
        d["log_tail"] = store.log_tail(job.id, 80)
    return d


@app.get("/api/jobs")
def list_jobs() -> list[dict]:
    return [_job_view(j) for j in store.list()]


@app.post("/api/jobs", status_code=201)
async def create_job(files: list[UploadFile] = File(...), paths: str = Form("[]"), name: str = Form(""),
                     profiles: str = Form("[]"), options: str = Form("{}")) -> dict:
    try:
        rel_paths = json.loads(paths or "[]")
        profile_ids = json.loads(profiles or "[]")
        opts = json.loads(options or "{}")
    except ValueError as e:
        raise HTTPException(400, f"bad form field: {e}")
    known = {p["id"] for p in list_profiles()}
    profile_ids = [p for p in profile_ids if p in known]
    if not profile_ids:
        raise HTTPException(400, "select at least one profile")
    incoming = store.dataset_root / f"_incoming_{int(time.time() * 1000)}"
    incoming.mkdir(parents=True, exist_ok=True)
    documents = []
    for i, up in enumerate(files):
        rel = _safe_relative(rel_paths[i] if i < len(rel_paths) and rel_paths[i] else (up.filename or f"file{i}"))
        if rel.suffix.lower() not in SUPPORTED:
            continue
        dest = incoming / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            shutil.copyfileobj(up.file, f)
        documents.append({"file": str(dest), "stem": dest.stem, "pages": _count_pages(dest), "bytes": dest.stat().st_size,
                          "relative": rel.as_posix()})
    if not documents:
        shutil.rmtree(incoming, ignore_errors=True)
        raise HTTPException(400, f"no supported document (accepted: {', '.join(sorted(SUPPORTED))})")
    # Distinct stems: two "slides.pdf" in different folders would otherwise share an output directory.
    seen: dict[str, int] = {}
    for d in documents:
        n = seen.get(d["stem"], 0)
        seen[d["stem"]] = n + 1
        if n:
            d["stem"] = f"{d['stem']}_{n + 1}"
    job_name = name.strip() or (Path(rel_paths[0]).parts[0] if rel_paths and rel_paths[0] and "/" in rel_paths[0] else documents[0]["stem"])
    job = store.create(job_name, documents, profile_ids, {"review": bool(opts.get("review")), "offline": bool(opts.get("offline")),
                                                          "gate": opts.get("gate") or settings.existence_gate})
    final_dir = store.dataset_root / job.id
    os.replace(incoming, final_dir)
    for d in job.documents:
        d["file"] = str(final_dir / Path(d["file"]).relative_to(incoming))
    store.save(job)
    return _job_view(job)


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    try:
        return _job_view(store.get(job_id), with_log=True)
    except FileNotFoundError:
        raise HTTPException(404, "job not found")


@app.get("/api/jobs/{job_id}/log")
def get_log(job_id: str, lines: int = 400) -> dict:
    return {"log": store.log_tail(job_id, lines)}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    if worker is None:
        store.update(job_id, status="cancelled", finished_at=now_iso())
        return {"ok": True}
    return {"ok": worker.cancel(job_id)}


@app.post("/api/jobs/{job_id}/requeue")
def requeue_job(job_id: str) -> dict:
    job = store.get(job_id)
    if job.status in ("running", "queued"):
        raise HTTPException(409, "job is already active")
    # Stage-1 outputs are reused by the runner, so a requeue after a failure never redoes the OCR.
    store.update(job_id, status="queued", error=None, finished_at=None, resume=True,
                 progress={**job.progress, "stage": "queued", "detail": ""})
    return {"ok": True}


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    job = store.get(job_id)
    if job.status == "running":
        raise HTTPException(409, "cancel the job before deleting it")
    store.delete(job_id)
    return {"ok": True}


# ------------------------------------------------------------------------------------------
# Review (human in the loop before anything is sent to the resolvers)
# ------------------------------------------------------------------------------------------

def _review_path(job, stem: str) -> Path:
    return store.job_dir(job.id) / "documents" / stem / f"{stem}_review.csv"


@app.get("/api/jobs/{job_id}/review")
def get_review(job_id: str) -> dict:
    import csv

    job = store.get(job_id)
    sheets = {}
    for d in job.documents:
        path = _review_path(job, d["stem"])
        if path.is_file():
            with open(path, newline="", encoding="utf-8") as f:
                sheets[d["stem"]] = list(csv.DictReader(f))
    return {"status": job.status, "sheets": sheets}


class ReviewDecision(BaseModel):
    document: str
    decisions: dict[str, str]   # "<text>||<type>" -> yes | local | drop


@app.post("/api/jobs/{job_id}/review")
def post_review(job_id: str, body: list[ReviewDecision]) -> dict:
    import csv

    job = store.get(job_id)
    if job.status != "review":
        raise HTTPException(409, "job is not waiting for a review")
    from biomedcat.pipeline import REVIEW_FIELDS, SEND_VALUES

    for item in body:
        path = _review_path(job, item.document)
        if not path.is_file():
            continue
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            decision = item.decisions.get(f"{row['text']}||{row['type']}")
            if decision in SEND_VALUES:
                row["send"] = decision
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
            w.writeheader()
            w.writerows(rows)
    store.update(job_id, status="queued", resume=True, progress={**job.progress, "stage": "queued", "detail": "review applied"})
    return {"ok": True}


# ------------------------------------------------------------------------------------------
# Results: graphs, enrichment, download
# ------------------------------------------------------------------------------------------

def _graph_dir(job, doc: str, profile: str) -> Path:
    path = store.job_dir(job.id) / "graphs" / doc / profile
    if not (path / "graph.json").is_file():
        raise HTTPException(404, "no graph for this document and profile")
    return path


@app.get("/api/jobs/{job_id}/graph")
def get_graph(job_id: str, doc: str, profile: str, max_nodes: int = 400, min_score: float = 0.0, hops: Optional[int] = None) -> JSONResponse:
    job = store.get(job_id)
    path = _graph_dir(job, doc, profile) / "graph.json"
    graph = json.loads(path.read_text(encoding="utf-8"))
    return JSONResponse(graphs_mod.filter_graph(graph, max_nodes=max_nodes or None, min_score=min_score, hops=hops))


class EnrichRequest(BaseModel):
    doc: str
    profile: str
    backend: str = "local"
    organism: str = "hsapiens"
    scope: str = "graph"        # graph: every gene node; seeds: seed genes only


@app.post("/api/jobs/{job_id}/enrich")
def enrich(job_id: str, req: EnrichRequest) -> dict:
    job = store.get(job_id)
    gdir = _graph_dir(job, req.doc, req.profile)
    # The cache name carries the annotation source, so building gene_go.parquet later invalidates it.
    source_tag = "kg2c-full" if graphs_mod.extras_available().get("gene_go") else "kg2c-filtered"
    cache = gdir / f"enrichment_{req.backend}_{req.scope}_{source_tag if req.backend == 'local' else req.organism}.json"
    if cache.is_file():
        return json.loads(cache.read_text(encoding="utf-8"))
    graph = json.loads((gdir / "graph.json").read_text(encoding="utf-8"))
    nodes = [e["data"] for e in graph["elements"] if "source" not in e["data"] and not e["data"].get("isCluster")]
    genes = [n["id"] for n in nodes if n["id"].startswith("NCBIGene:") and (req.scope != "seeds" or n["isSeed"])]
    if req.backend == "gprofiler":
        result = enrichment.enrich_gprofiler(genes, organism=req.organism)
    else:
        result = enrichment.enrich_local(genes)
    result.update({"doc": req.doc, "profile": req.profile, "scope": req.scope, "genes": genes, "computed_at": now_iso()})
    if not result.get("error"):
        cache.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


@app.get("/api/jobs/{job_id}/summary")
def graph_summary(job_id: str, doc: str, profile: str, format: str = "html"):
    """Narrative summary of one graph (every node name in bold), regenerated on request so a GO
    enrichment run from the viewer is included."""
    job = store.get(job_id)
    gdir = _graph_dir(job, doc, profile)
    md_path, html_path = summary_mod.write_graph_summary(gdir, job.to_dict(), doc=doc)
    if format == "md":
        return PlainTextResponse(md_path.read_text(encoding="utf-8"), media_type="text/markdown",
                                 headers={"Content-Disposition": f'attachment; filename="graph_summary_{doc}_{profile}.md"'})
    return HTMLResponse(html_path.read_text(encoding="utf-8"))


@app.get("/api/jobs/{job_id}/download")
def download(job_id: str) -> FileResponse:
    job = store.get(job_id)
    zip_path = graphs_mod.build_zip(store.job_dir(job_id), job.to_dict())
    return FileResponse(zip_path, media_type="application/zip", filename=f"biomedcat_{job_id}.zip")


@app.get("/api/jobs/{job_id}/file")
def get_file(job_id: str, path: str) -> FileResponse:
    """One file of the job directory (result JSON, CSV, GraphML), path relative to the job."""
    base = store.job_dir(job_id).resolve()
    target = (base / path).resolve()
    if base not in target.parents or not target.is_file():
        raise HTTPException(404, "file not found")
    return FileResponse(target)


# ------------------------------------------------------------------------------------------
# Entry point
# ------------------------------------------------------------------------------------------

def main() -> None:
    import uvicorn

    host = os.getenv("BIOMEDCAT_HOST", "127.0.0.1")
    port = int(os.getenv("BIOMEDCAT_PORT", "8765"))
    url = f"http://{host}:{port}/"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if os.getenv("BIOMEDCAT_NO_BROWSER", "0") != "1":
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    print(f"BiomedCAT console: {url}  (keep this window open while jobs run; Ctrl+C stops the console)")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    main()
