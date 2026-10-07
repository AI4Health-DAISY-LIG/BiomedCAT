"""Job model, on-disk job store and the single-slot worker that runs jobs in subprocesses.

A job directory (output/jobs/<id>/) is the only source of truth: the server and the runner
subprocess both read-modify-write job.json, so a restart of the console loses nothing. Writes
go through a temporary file and os.replace, which is atomic on Windows and POSIX.
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from biomedcat.config import settings, ROOT_PATH

logger = logging.getLogger(__name__)

STATUSES = ("queued", "running", "review", "done", "failed", "cancelled")
ACTIVE = ("queued", "running", "review")
PROFILES_DIR = Path(ROOT_PATH) / "data" / "profiles"
# Profiles offered in the console: every JSON of data/profiles with a `branch_weights` or a
# `predicates` block, except the strata configuration (legacy hand-written files live in
# data/profiles/legacy and are not listed).
EXCLUDED_PROFILE_FILES = {"strata_config.json", "biochemical_actions_probs.json"}
CUSTOM_PREFIX = "custom-"
MAX_READING_FOCUS = 400   # characters; keep in sync with the textarea of the custom-profile builder

# Default cost model until the console has seen jobs of its own (seconds).
DEFAULT_SEC_PER_PAGE = 720.0      # OCR (~2 min) + NER agent + linking, per slide, on a 16 GB laptop
SEC_PER_GRAPH = 90.0              # one context graph (DuckDB over the filtered KG2c)
SEC_STARTUP = 60.0                # model loading and index opening


def now_iso() -> str:
    # Millisecond resolution: jobs created within the same second must still order correctly.
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()
    return (slug or "job")[:max_len]


def atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------------------------------------------
# Profiles
# ------------------------------------------------------------------------------------------

def list_profiles(profiles_dir: Path = PROFILES_DIR) -> list[dict]:
    """Profiles selectable in the console, with their branch priorities and reading scope."""
    from biomedcat.profiles import load_profile

    out = []
    for path in sorted(profiles_dir.glob("*.json")):
        if path.name in EXCLUDED_PROFILE_FILES:
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if "predicates" not in raw and "branch_weights" not in raw:
            continue
        try:
            profile = load_profile(path)
            scope, n_pred = profile.entity_scope, len(profile.predicates)
            branch_weights, directionality = profile.branch_weights, profile.directionality
        except Exception:  # a malformed profile must not hide the others
            scope, n_pred = list(raw.get("entity_scope", [])), len(raw.get("predicates", {}))
            branch_weights, directionality = dict(raw.get("branch_weights", {})), raw.get("directionality", {})
        out.append({
            "id": path.stem,
            "name": raw.get("name", path.stem),
            "description": raw.get("description", ""),
            "stratum": raw.get("stratum", ""),
            "n_predicates": n_pred,
            "entity_scope": scope,
            "branch_weights": branch_weights,
            "directionality": directionality,
            "custom": False,
        })
    return out


def entity_branches() -> dict:
    """Entity branches of the Biolink model for the custom-profile builder, with KG2c node counts."""
    from biomedcat.weights import BiolinkModel, camel_curie

    strata = Path(settings.internal_data_path) / "biolink_strata.json"
    model = BiolinkModel.load(strata)
    counts: dict[str, int] = {}
    stats = Path(settings.kg2c_dir) / "stats.json"
    if stats.is_file():
        try:
            counts = json.loads(stats.read_text(encoding="utf-8")).get("nodes", {}).get("nodes_kept_by_category", {}) or {}
        except ValueError:
            counts = {}
    branches = []
    for b in model.branch_info():
        names = [b["name"]] + sorted(model.descendants.get(b["name"], ()))
        n_nodes = sum(int(counts.get(camel_curie(n), 0)) for n in names)
        branches.append({**b, "n_nodes_kg2c": n_nodes})
    return {"biolink_version": model.version, "branches": branches, "levels": [0, 0.5, 1]}


def custom_profile_id(name: str, taken: set[str]) -> str:
    base = CUSTOM_PREFIX + slugify(name or "profile", 24)
    pid, n = base, 1
    while pid in taken:
        n += 1
        pid = f"{base}-{n}"
    return pid


def normalise_custom_profile(spec: dict) -> dict:
    """Validate a custom profile from the console into a profile file (biomedcat.profiles format)."""
    from biomedcat.weights import BiolinkModel, directionality_of, normalise_branch_weights

    name = str(spec.get("name") or "custom profile").strip()[:80]
    model = BiolinkModel.load(Path(settings.internal_data_path) / "biolink_strata.json")
    weights = {b: w for b, w in normalise_branch_weights(spec.get("branch_weights") or {}, model).items() if w > 0}
    if not weights:
        raise ValueError(f"custom profile {name!r}: no known entity branch with a non-zero priority")
    profile = {
        "name": name,
        "description": str(spec.get("description") or "User-defined profile built in the console: entity-branch priorities "
                                                       "and directionality set by the user, weights derived automatically."),
        "branch_weights": weights,
        # Free text appended to the slide-reading prompt (biomedcat.prompts); whitespace collapsed, length capped.
        "reading_focus": " ".join(str(spec.get("reading_focus") or "").split())[:MAX_READING_FOCUS],
        "directionality": directionality_of(spec),
        "weighting": {"alpha": float((spec.get("weighting") or {}).get("alpha", 0.7)),
                      "scope_rule": str((spec.get("weighting") or {}).get("scope_rule", "max"))},
        "sources": dict(spec.get("sources") or {}),
        "knowledge_levels": dict(spec.get("knowledge_levels") or {}),
        "prefix_restricted_predicates": dict(spec.get("prefix_restricted_predicates") or {}),
        "custom": True,
    }
    return profile


def profile_path(profile_id: str, job_dir: Path | None = None, profiles_dir: Path = PROFILES_DIR) -> Path:
    """Profile file of a job profile: a custom profile lives in <job>/profiles/, a default one in data/profiles."""
    if job_dir is not None:
        candidate = Path(job_dir) / "profiles" / f"{profile_id}.json"
        if candidate.is_file():
            return candidate
    return profiles_dir / f"{profile_id}.json"


def merge_profiles(profile_ids: list[str], out_path: Path, profiles_dir: Path = PROFILES_DIR,
                   job_dir: Path | None = None) -> Path:
    """Write the reading profile of a job: the union of the selected profiles.

    OCR and NER run once per document, so the slide-reading prompt must ask for every entity
    kind any selected profile cares about (union of entity scopes, concatenated reading focus).
    Predicate, source and knowledge-level weights take the maximum across profiles; this merged
    profile is only used for reading, each context graph is built with its own profile.
    """
    from biomedcat.profiles import load_profile

    merged: dict[str, Any] = {"name": "merged: " + " + ".join(profile_ids), "predicates": {}, "sources": {},
                              "knowledge_levels": {}, "prefix_restricted_predicates": {}, "entity_scope": [],
                              "reading_focus": "", "members": profile_ids}
    focus_parts: list[str] = []
    for pid in profile_ids:
        p = load_profile(profile_path(pid, job_dir, profiles_dir))
        for block, values in (("predicates", p.predicates), ("sources", p.sources), ("knowledge_levels", p.knowledge_levels)):
            for k, v in values.items():
                merged[block][k] = max(float(v), merged[block].get(k, 0.0))
        for pred, prefixes in p.prefix_restricted_predicates.items():
            merged["prefix_restricted_predicates"].setdefault(pred, [])
            merged["prefix_restricted_predicates"][pred] = sorted(set(merged["prefix_restricted_predicates"][pred]) | set(prefixes))
        for kind in p.entity_scope:
            if kind not in merged["entity_scope"]:
                merged["entity_scope"].append(kind)
        if p.reading_focus and p.reading_focus not in focus_parts:
            focus_parts.append(p.reading_focus.strip())
    merged["reading_focus"] = " ".join(focus_parts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out_path, merged)
    return out_path


# ------------------------------------------------------------------------------------------
# Job
# ------------------------------------------------------------------------------------------

@dataclass
class Job:
    id: str
    name: str
    created_at: str
    documents: list[dict]                  # {file, stem, pages, bytes}
    profiles: list[str]
    options: dict = field(default_factory=dict)   # review, offline, gate
    custom_profiles: dict = field(default_factory=dict)   # custom profile id -> its profile file content
    status: str = "queued"
    estimate_s: float = 0.0
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    elapsed_s: Optional[float] = None
    error: Optional[str] = None
    progress: dict = field(default_factory=dict)  # stage, document, detail, done_units, total_units, history
    outputs: dict = field(default_factory=dict)   # per document stem: result_json, graphs {profile: summary}
    resume: bool = False                          # set after a review: the runner continues from stage 1

    @property
    def pages(self) -> int:
        return sum(int(d.get("pages") or 1) for d in self.documents)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Job":
        known = {k: d.get(k) for k in cls.__dataclass_fields__ if k in d}
        return cls(**known)


def estimate_seconds(pages: int, n_profiles: int, sec_per_page: float = DEFAULT_SEC_PER_PAGE) -> float:
    return SEC_STARTUP + pages * sec_per_page + pages * 0 + n_profiles * SEC_PER_GRAPH


# ------------------------------------------------------------------------------------------
# Store
# ------------------------------------------------------------------------------------------

class JobStore:
    """Jobs as directories under <root>/jobs; the dataset copies live under <dataset>/<job id>."""

    def __init__(self, root: Path | None = None, dataset_root: Path | None = None):
        self.root = Path(root or Path(settings.output_path) / "jobs")
        self.dataset_root = Path(dataset_root or settings.dataset_path)
        self.root.mkdir(parents=True, exist_ok=True)
        self.dataset_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # -- paths ---------------------------------------------------------------------------
    def job_dir(self, job_id: str) -> Path:
        return self.root / job_id

    def job_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.json"

    def log_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.log"

    # -- CRUD ----------------------------------------------------------------------------
    def new_id(self, name: str) -> str:
        base = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{slugify(name, 24)}"
        job_id, n = base, 1
        while self.job_dir(job_id).exists():
            n += 1
            job_id = f"{base}-{n}"
        return job_id

    def create(self, name: str, documents: list[dict], profiles: list[str], options: dict | None = None,
               custom_profiles: list[dict] | None = None) -> Job:
        """Create a queued job. `custom_profiles` are console-built profiles (see normalise_custom_profile):
        each is written to <job>/profiles/<id>.json, its derived weights land next to it for the trace."""
        with self._lock:
            job_id = self.new_id(name)
            job_dir = self.job_dir(job_id)
            job_dir.mkdir(parents=True)
            profiles = list(profiles)
            customs: dict[str, dict] = {}
            for spec in custom_profiles or []:
                profile = normalise_custom_profile(spec)
                pid = custom_profile_id(profile["name"], set(profiles) | set(customs))
                (job_dir / "profiles").mkdir(exist_ok=True)
                atomic_write_json(job_dir / "profiles" / f"{pid}.json", profile)
                customs[pid] = profile
                profiles.append(pid)
            merge_profiles(profiles, job_dir / "profile_merged.json", job_dir=job_dir)
            job = Job(id=job_id, name=name, created_at=now_iso(), documents=documents, profiles=profiles,
                      options=dict(options or {}), custom_profiles=customs)
            job.estimate_s = round(estimate_seconds(job.pages, len(profiles), self.sec_per_page()), 0)
            job.progress = {"stage": "queued", "document": "", "detail": "", "done_units": 0,
                            "total_units": len(documents) * (3 + len(profiles)), "history": []}
            self.save(job)
            return job

    def save(self, job: Job) -> None:
        with self._lock:
            atomic_write_json(self.job_path(job.id), job.to_dict())

    def update(self, job_id: str, **fields) -> Job:
        """Read-modify-write of a few fields, so two writers never clobber each other's fields."""
        with self._lock:
            job = self.get(job_id)
            for k, v in fields.items():
                setattr(job, k, v)
            self.save(job)
            return job

    def get(self, job_id: str) -> Job:
        path = self.job_path(job_id)
        for attempt in range(5):  # the runner may be replacing the file right now
            try:
                return Job.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                time.sleep(0.05 * (attempt + 1))
        raise FileNotFoundError(f"job {job_id} not found")

    def list(self) -> list[Job]:
        jobs = []
        for path in self.root.glob("*/job.json"):
            try:
                jobs.append(Job.from_dict(json.loads(path.read_text(encoding="utf-8"))))
            except (OSError, ValueError):
                continue
        return sorted(jobs, key=lambda j: (j.created_at, j.id), reverse=True)

    def delete(self, job_id: str) -> None:
        with self._lock:
            shutil.rmtree(self.job_dir(job_id), ignore_errors=True)
            shutil.rmtree(self.dataset_root / job_id, ignore_errors=True)

    def next_queued(self) -> Optional[Job]:
        queued = [j for j in self.list() if j.status == "queued"]
        return min(queued, key=lambda j: (j.created_at, j.id)) if queued else None

    # -- cost model ----------------------------------------------------------------------
    def stats_path(self) -> Path:
        return self.root / "_stats.json"

    def sec_per_page(self) -> float:
        try:
            hist = json.loads(self.stats_path().read_text(encoding="utf-8")).get("sec_per_page", [])
            return statistics.median(hist[-10:]) if hist else DEFAULT_SEC_PER_PAGE
        except (OSError, ValueError):
            return DEFAULT_SEC_PER_PAGE

    def record_timing(self, job: Job) -> None:
        """Feed the cost model with a finished job (graph time excluded, it is per profile)."""
        if not job.elapsed_s or job.pages <= 0:
            return
        graph_s = sum(h.get("seconds", 0) for h in job.progress.get("history", []) if str(h.get("stage", "")).startswith("graph"))
        per_page = max(30.0, (job.elapsed_s - graph_s - SEC_STARTUP) / job.pages)
        try:
            data = json.loads(self.stats_path().read_text(encoding="utf-8")) if self.stats_path().is_file() else {}
        except ValueError:
            data = {}
        data.setdefault("sec_per_page", []).append(round(per_page, 1))
        data["sec_per_page"] = data["sec_per_page"][-50:]
        atomic_write_json(self.stats_path(), data)

    # -- log helpers ---------------------------------------------------------------------
    def log_tail(self, job_id: str, n_lines: int = 60) -> str:
        path = self.log_path(job_id)
        if not path.is_file():
            return ""
        try:
            with open(path, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                f.seek(max(0, size - 200_000))
                text = f.read().decode("utf-8", errors="replace")
        except OSError:
            return ""
        lines = [l for l in text.splitlines() if "Batches:" not in l]
        return "\n".join(lines[-n_lines:])

    def live_detail(self, job: Job) -> str:
        """Progress detail derived from the trace: terms typed so far, slides read."""
        if job.status != "running":
            return job.progress.get("detail", "")
        text = self.log_tail(job.id, 4000)
        stage = job.progress.get("stage", "")
        if stage == "NER":
            n_terms = text.count("[Agent Step 1]")
            return f"{n_terms} term(s) examined by the typing agent"
        if stage == "Norm":
            n_judge = text.count("[Judge]") or text.count("candidates for")
            return f"linking entities to RTX-KG2c ({n_judge} judged)" if n_judge else "linking entities to RTX-KG2c"
        return job.progress.get("detail", "")


# ------------------------------------------------------------------------------------------
# Worker
# ------------------------------------------------------------------------------------------

class Worker(threading.Thread):
    """Runs queued jobs one at a time, each in a `biomedcat.webapp.runner` subprocess."""

    def __init__(self, store: JobStore, poll_s: float = 2.0, python: str | None = None):
        super().__init__(daemon=True, name="biomedcat-worker")
        self.store = store
        self.poll_s = poll_s
        self.python = python or sys.executable
        self.current: Optional[str] = None
        self.process: Optional[subprocess.Popen] = None
        self._stop = threading.Event()
        self._cancel = threading.Event()

    def stop(self) -> None:
        self._stop.set()

    def cancel(self, job_id: str) -> bool:
        """Cancel a queued job, or kill the running one."""
        job = self.store.get(job_id)
        if job.status == "queued":
            self.store.update(job_id, status="cancelled", finished_at=now_iso())
            return True
        if job.status == "running" and self.current == job_id and self.process:
            self._cancel.set()
            try:
                self.process.kill()
            except OSError:
                pass
            return True
        return False

    def env_for(self, job: Job) -> dict:
        env = dict(os.environ)
        env["PREDICATE_PROFILE"] = str(self.store.job_dir(job.id) / "profile_merged.json")
        env["PYTHONUNBUFFERED"] = "1"
        opts = job.options or {}
        env["RESOLVERS_OFFLINE"] = "1" if opts.get("offline") else "0"
        if opts.get("gate"):
            env["EXISTENCE_GATE"] = str(opts["gate"])
        env["OLLAMA_KEEP_ALIVE"] = env.get("OLLAMA_KEEP_ALIVE", "15m")
        return env

    def run(self) -> None:
        # Jobs left "running" by a previous console process were interrupted.
        for job in self.store.list():
            if job.status == "running":
                self.store.update(job.id, status="failed", error="interrupted: the console was stopped during this job (requeue to run it again)",
                                  finished_at=now_iso())
        while not self._stop.is_set():
            job = self.store.next_queued()
            if job is None:
                self._stop.wait(self.poll_s)
                continue
            self._run_one(job)

    def _run_one(self, job: Job) -> None:
        self.current = job.id
        self._cancel.clear()
        job = self.store.update(job.id, status="running", started_at=now_iso(), error=None)
        cmd = [self.python, "-m", "biomedcat.webapp.runner", "--job", str(self.store.job_dir(job.id))]
        if job.resume:
            cmd.append("--resume")
        logger.info("starting job %s: %s", job.id, " ".join(cmd))
        t0 = time.perf_counter()
        try:
            with open(self.store.log_path(job.id), "a", encoding="utf-8") as log:
                self.process = subprocess.Popen(cmd, cwd=str(ROOT_PATH), env=self.env_for(job), stdout=log, stderr=subprocess.STDOUT)
                code = self.process.wait()
        except OSError as e:
            code = -1
            self.store.update(job.id, error=f"could not start the runner: {e}")
        finally:
            self.process = None
        elapsed = round(time.perf_counter() - t0, 1)
        job = self.store.get(job.id)
        if self._cancel.is_set():
            self.store.update(job.id, status="cancelled", finished_at=now_iso(), elapsed_s=elapsed)
        elif job.status in ("done", "review"):
            if job.status == "done":
                self.store.record_timing(job)
        else:
            tail = self.store.log_tail(job.id, 15)
            self.store.update(job.id, status="failed", finished_at=now_iso(), elapsed_s=elapsed,
                              error=job.error or f"runner exited with code {code}\n{tail[-1500:]}")
        self.current = None
