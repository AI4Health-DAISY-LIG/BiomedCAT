#!/usr/bin/env python
"""Run the pipeline on one document and record memory and time per stage.

Why: the run logs give the time of each stage, but nothing in the pipeline records memory. This
wrapper launches `python -m biomedcat.pipeline <file>` in the published configuration and, once a
second, samples:
  - the RSS of the pipeline process tree (python + children),
  - the RSS of the Ollama processes (the model server is a separate process),
  - the system RAM in use (psutil.virtual_memory().used) and the GPU memory in use (nvidia-smi),
  - the `ollama ps` line, to know which model was resident.
Stage boundaries are read from the pipeline's own log file (`<stem>_BiomedCAT.log`: "OCR done",
"NER done", "Norm done", "Context graph done"), so the samples are attributed to OCR, NER, Norm,
Graph and the whole run. Peaks are reported relative to the idle baseline sampled before launch.

Memory is measured for the configuration of the machine that runs it. Here Ollama holds the model
weights in GPU memory when a GPU is present; `gpu_mib_peak` is then the figure to compare with the
"no GPU" case, where the same weights would sit in RAM.

Outputs, all in one run directory (--out, default Output/resources/<stem>/): the pipeline's own
files (<stem>_BiomedCAT.json, .log, context graph), resources.json (per-stage summary) and
resources_samples.csv (the time series). The published runs are never overwritten.

Usage:
  uv run python scripts/measure_pipeline_resources.py Dataset/VoieA.pdf
  uv run python scripts/measure_pipeline_resources.py Dataset/FSHD.pdf --out data/publication/runs/FSHD
  RESOLVERS_OFFLINE=1 uv run python scripts/measure_pipeline_resources.py Dataset/FSHD.pdf --out data/publication/runs/FSHD_offline
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Output" / "resources"
# The configuration of every published number (README, "Environment variables").
PUBLISHED_ENV = {"AGENT_MODE": "react", "SCISPACY": "0", "RAG_BM25": "0", "RAG_LEXICAL_WEIGHT": "0",
                 "RAG_KG2C_LEXICAL": "0", "RAG_KG2C_LEXICAL_WEIGHT": "0", "PYTHONIOENCODING": "utf-8"}
STAGE_MARKERS = [("ocr", re.compile(r"OCR done: (\d+) slide\(s\) in ([\d.]+)s")),
                 ("ner", re.compile(r"NER done: (\d+) entit\(y/ies\) in ([\d.]+) s")),
                 ("norm", re.compile(r"Norm done: (\d+)/(\d+) linked in ([\d.]+)s")),
                 ("graph", re.compile(r"Context graph done: (\d+) nodes, (\d+) edges in ([\d.]+)s"))]
STAGE_ORDER = ["ocr", "ner", "norm", "graph"]


def gpu_used_mib() -> int | None:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.strip()
        return int(out.splitlines()[0])
    except Exception:
        return None


def ollama_ps() -> str:
    try:
        out = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        return " | ".join(" ".join(l.split()) for l in out[1:]) if len(out) > 1 else ""
    except Exception:
        return ""


def ollama_rss_mb() -> float:
    total = 0
    for p in psutil.process_iter(["name", "memory_info"]):
        try:
            if p.info["name"] and p.info["name"].lower().startswith("ollama"):
                total += p.info["memory_info"].rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return round(total / 2**20, 1)


def tree_rss_mb(proc: psutil.Process) -> float:
    total = 0
    try:
        for p in [proc, *proc.children(recursive=True)]:
            try:
                total += p.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    except psutil.NoSuchProcess:
        pass
    return round(total / 2**20, 1)


def sample(proc: psutil.Process | None) -> dict:
    vm = psutil.virtual_memory()
    return {"t": time.time(), "pipeline_rss_mb": tree_rss_mb(proc) if proc else 0.0, "ollama_rss_mb": ollama_rss_mb(),
            "system_used_mb": round(vm.used / 2**20, 1), "gpu_mib": gpu_used_mib(), "ollama_ps": ollama_ps()}


def read_markers(log_path: Path, seen: dict[str, dict]) -> None:
    if not log_path.is_file():
        return
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        for stage, rx in STAGE_MARKERS:
            if stage in seen:
                continue
            m = rx.search(line)
            if m:
                ts = datetime.strptime(line[:23], "%Y-%m-%d %H:%M:%S,%f").timestamp()
                groups = [float(g) for g in m.groups()]
                seen[stage] = {"t_done": ts, "seconds_logged": groups[-1], "units": groups[0], "extra": groups[:-1]}


def summarize(samples: list[dict], baseline: dict, markers: dict[str, dict], t_start: float, t_end: float) -> dict:
    def window(t0: float, t1: float) -> list[dict]:
        return [s for s in samples if t0 <= s["t"] <= t1]

    def peaks(rows: list[dict]) -> dict:
        if not rows:
            return {}
        g = [r["gpu_mib"] for r in rows if r["gpu_mib"] is not None]
        return {"n_samples": len(rows),
                "pipeline_rss_mb_peak": max(r["pipeline_rss_mb"] for r in rows),
                "ollama_rss_mb_peak": max(r["ollama_rss_mb"] for r in rows),
                "ollama_rss_mb_delta_peak": round(max(r["ollama_rss_mb"] for r in rows) - baseline["ollama_rss_mb"], 1),
                "system_used_mb_delta_peak": round(max(r["system_used_mb"] for r in rows) - baseline["system_used_mb"], 1),
                "gpu_mib_peak": max(g) if g else None,
                "gpu_mib_delta_peak": (max(g) - (baseline["gpu_mib"] or 0)) if g else None,
                "cpu_ram_mb_peak": round(max(r["pipeline_rss_mb"] + r["ollama_rss_mb"] for r in rows) - baseline["ollama_rss_mb"], 1),
                "models_resident": sorted({r["ollama_ps"].split()[0] for r in rows if r["ollama_ps"]})}

    stages, t_prev = {}, t_start
    for stage in STAGE_ORDER:
        if stage not in markers:
            continue
        m = markers[stage]
        stages[stage] = {"t_start": t_prev, "t_done": m["t_done"], "seconds_wall": round(m["t_done"] - t_prev, 1),
                         "seconds_logged": m["seconds_logged"], "units": m["units"], **peaks(window(t_prev, m["t_done"]))}
        t_prev = m["t_done"]
    stages["total"] = {"t_start": t_start, "t_done": t_end, "seconds_wall": round(t_end - t_start, 1), **peaks(window(t_start, t_end))}
    return stages


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("file", help="Document to process (e.g. Dataset/VoieA.pdf).")
    ap.add_argument("--interval", type=float, default=1.0, help="Sampling interval in seconds.")
    ap.add_argument("--baseline-seconds", type=float, default=5.0, help="Idle sampling before launch.")
    ap.add_argument("--out", default=None, help="Run directory (default Output/resources/<stem>/).")
    args = ap.parse_args()

    doc = Path(args.file)
    stem = doc.stem
    run_dir = Path(args.out) if args.out else OUT / stem
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / f"{stem}_BiomedCAT.log"
    if log_path.exists():
        log_path.unlink()

    idle = []
    for _ in range(max(1, int(args.baseline_seconds / args.interval))):
        idle.append(sample(None))
        time.sleep(args.interval)
    baseline = {k: (max(r[k] for r in idle) if k != "gpu_mib" else max((r[k] or 0) for r in idle)) for k in ("ollama_rss_mb", "system_used_mb", "gpu_mib")}
    print(f"baseline: ollama {baseline['ollama_rss_mb']} MB RSS, system {baseline['system_used_mb']} MB used, gpu {baseline['gpu_mib']} MiB")

    env = {**os.environ, **PUBLISHED_ENV, "OUTPUT_PATH": str(run_dir)}
    cmd = [sys.executable, "-m", "biomedcat.pipeline", str(doc)]
    t_start = time.time()
    stdout = (run_dir / f"pipeline_{stem}.stdout.log").open("w", encoding="utf-8")
    child = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=stdout, stderr=subprocess.STDOUT)
    proc = psutil.Process(child.pid)
    print(f"started pid {child.pid}: {' '.join(cmd)}")

    samples, markers = [], {}
    try:
        while child.poll() is None:
            samples.append(sample(proc))
            read_markers(log_path, markers)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        child.terminate()
        print("interrupted; partial results are written")
    t_end = time.time()
    samples.append(sample(None))
    read_markers(log_path, markers)
    stdout.close()

    with (run_dir / "resources_samples.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(samples[0].keys()))
        w.writeheader()
        w.writerows(samples)

    stages = summarize(samples, baseline, markers, t_start, t_end)
    hw = {"cpu": psutil.cpu_count(logical=True), "ram_total_mb": round(psutil.virtual_memory().total / 2**20)}
    try:
        hw["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        hw["gpu"] = None
    result = {"file": str(doc), "started_utc": datetime.fromtimestamp(t_start, timezone.utc).isoformat(timespec="seconds"),
              "exit_code": child.returncode, "interval_s": args.interval, "env": PUBLISHED_ENV, "hardware": hw,
              "resolvers_offline": os.environ.get("RESOLVERS_OFFLINE", "0"), "gate_nameres_url": os.environ.get("GATE_NAMERES_URL", "production (default)"),
              "baseline": baseline, "stages": stages, "log": str(log_path.resolve().relative_to(ROOT)),
              "samples_csv": str((run_dir / "resources_samples.csv").resolve().relative_to(ROOT))}
    (run_dir / "resources.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"\nexit code {child.returncode}; {len(samples)} samples")
    print(f"{'stage':6s} {'wall s':>8s} {'logged s':>9s} {'CPU RAM peak MB':>16s} {'GPU peak MiB':>13s}  models")
    for name, s in stages.items():
        print(f"{name:6s} {s['seconds_wall']:8.1f} {str(s.get('seconds_logged', '')):>9s} {s.get('cpu_ram_mb_peak', 0):16.1f} {str(s.get('gpu_mib_peak')):>13s}  {', '.join(s.get('models_resident', []))}")
    print(f"\nwritten: {run_dir / 'resources.json'}")
    return child.returncode or 0


if __name__ == "__main__":
    sys.exit(main())
