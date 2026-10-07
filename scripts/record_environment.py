"""Record the execution environment of the reported runs into data/eval/environment.json.

The master results workbook (scripts/build_master_results.py) reads this file for its resources
tab: hardware, operating system, Ollama version and the size of every local model, Python and
key package versions, and the power assumptions used for the energy estimate. Everything is
queried from the machine at the time of the call; nothing is typed in by hand except the power
assumptions, which are documented in the file itself.

    uv run python scripts/record_environment.py [--out data/eval/environment.json]
"""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

B = Path(__file__).resolve().parents[1]

# Power assumptions for the energy estimate (W). The laptop GPU (RTX 2000 Ada, 35-140 W
# configurable TDP) does not expose its power limit through nvidia-smi on this machine, so the
# estimate is given as a range: low = GPU at its minimum TDP plus an idle platform, high = GPU at
# its maximum TDP plus a busy CPU. energy_kWh = seconds * P_W / 3.6e6.
POWER_ASSUMPTIONS_W = {"low": 35 + 25, "high": 140 + 60}
CARBON_INTENSITY_G_PER_KWH = {"France (RTE 2024 average)": 32, "EU-27 (2023 average)": 242}


def run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # tool absent: recorded as such, never fatal
        return f"unavailable ({type(e).__name__})"


def ps(expr: str) -> str:
    return run(["powershell", "-NoProfile", "-Command", expr])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(B / "data/eval/environment.json"))
    args = ap.parse_args()
    gpu = run(["nvidia-smi", "--query-gpu=name,memory.total,power.limit,driver_version", "--format=csv,noheader"])
    models = []
    for line in run(["ollama", "list"]).splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 4:
            models.append({"model": parts[0], "id": parts[1], "size": f"{parts[2]} {parts[3]}"})
    pkgs = {}
    for p in ["torch", "transformers", "sentence-transformers", "chromadb", "duckdb", "spacy", "scispacy", "rank-bm25", "requests", "pandas", "fastapi"]:
        try:
            pkgs[p] = md.version(p)
        except md.PackageNotFoundError:
            pkgs[p] = "not installed"
    services = {}
    try:
        import requests
        from biomedcat.config import settings
        for name, url in (("name_resolver_gate", settings.gate_nameres_url), ("name_resolver_linking", settings.renci_url)):
            base = url.rsplit("/", 1)[0]
            st = requests.get(base + "/status", timeout=20).json()
            services[name] = {"url": url, **{k: st.get(k) for k in ("nameres_version", "babel_version", "biolink_model", "lastModified", "numDocs")}}
        base = settings.nodenorm_url.rsplit("/", 1)[0]
        oa = requests.get(base + "/openapi.json", timeout=20).json()
        services["node_normalizer"] = {"url": settings.nodenorm_url, "version": oa.get("info", {}).get("version"), "title": oa.get("info", {}).get("title")}
        services["arax"] = {"url": settings.arax_url}
    except Exception as e:  # the record must be written even when a service is down
        services["error"] = str(e)
    env = {
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "services": services,
        "hardware": {
            "cpu": ps("(Get-CimInstance Win32_Processor).Name"),
            "ram_gb": ps("[math]::Round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory/1GB,1)"),
            "gpu": gpu,
            "os": platform.platform(),
        },
        "software": {"python": sys.version.split()[0], "ollama": run(["ollama", "--version"]), "packages": pkgs},
        "ollama_models": models,
        "energy_assumptions": {"power_w": POWER_ASSUMPTIONS_W, "formula": "energy_kWh = seconds * power_W / 3.6e6",
                               "carbon_intensity_g_per_kwh": CARBON_INTENSITY_G_PER_KWH},
    }
    Path(args.out).write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
