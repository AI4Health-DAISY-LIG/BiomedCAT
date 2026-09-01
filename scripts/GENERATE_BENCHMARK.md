# 📄 **`scripts/GENERATE_BENCHMARK.md`**  
**How to generate the *Biomed‑Stratified* QA benchmark**  

---  

## Table of Contents
1. [Overview](#overview)  
2. [Prerequisites](#prerequisites)  
3. [Repository layout (relevant files)](#repo‑layout)  
4. [Step‑by‑step generation workflow](#workflow)  
   - 4.1 [Create class‑frequency statistics](#step‑1‑frequencies)  
   - 4.2 [Configure the benchmark (YAML)](#step‑2‑config)  
   - 4.3 [Generate the three reproducible splits](#step‑3‑splits)  
   - 4.4 [Validate the output](#step‑4‑validation)  
5. [Advanced options](#advanced‑options)  
6. [Common pitfalls & troubleshooting](#troubleshooting)  
7. [Publishing the benchmark](#publishing)  
8. [Citation & acknowledgements](#citation)  

---  

<a name="overview"></a>
## 1️⃣ Overview  

The **Biomed‑Stratified QA benchmark** is a synthetic, stratified dataset of short biomedical queries (1‑3 words) together with the target Biolink class.  
Key properties:

| Property | Value (default) |
|----------|-----------------|
| **Total size** | 40 000 samples (`dataset.target_size`) |
| **Leaf‑class minimum** | 30 samples per leaf (`leaf_class.min`) |
| **Style controls** | Ambiguity (70 % unambiguous / 30 % ambiguous) <br> Surface form (50 % common / 50 % uncommon) |
| **Frequency balancing** | Inverse‑frequency weighting (`inverse_freq_exponent = 0.7`) |
| **Reproducibility** | Three deterministic splits (seeds 77, 78, 79) |
| **Enriched context** | Each row contains a `context` column = *definition + first child example* (useful for RAG evaluation) |
| **Output format** | Parquet (`data/large_splits/<seed>/qa_dataset.parquet`) + a global CSV summary |

All heavy‑lifting is performed by the Python scripts in `scripts/` and the core library in `biomedcat/`.  

---  

<a name="prerequisites"></a>
## 2️⃣ Prerequisites  

| Requirement | How to install / configure |
|-------------|----------------------------|
| **Python** | `>=3.10` (tested on 3.11). Use a virtual environment: <br>`python -m venv .venv && source .venv/bin/activate` |
| **Package dependencies** | `pip install -r requirements.txt` (includes `biopython`, `tqdm`, `chromadb`, `rank_bm25`, `spacy`, `torch`, `pandas`, `pyarrow`, `openai`, `requests`, `PyYAML`). |
| **SciSpaCy model** | `pip install https://github.com/allenai/scispacy/releases/download/v0.5.0/en_core_sci_sm-0.5.0.tar.gz` |
| **Ollama** (local LLM server) | Install from <https://ollama.com> and start it: `ollama serve`. The default URL is `http://localhost:11434`. |
| **OpenAI (optional)** | If you want to use an OpenAI‑compatible backend, set `OLLAMA_URL` to the OpenAI endpoint and provide `OPENAI_API_KEY` (the script reads it via `openai.OpenAI`). |
| **NCBI Entrez e‑mail** | The script `build_class_frequencies.py` requires a valid e‑mail address (already set to ``). Change it in the script if you need a different address. |
| **Git LFS (optional)** | The Biolink YAML is fetched via HTTP; no LFS needed. |
| **Disk space** | ~ 2 GB for the ChromaDB index + the generated Parquet files. |

> **Tip** – Keep the virtual environment activated while running the scripts; otherwise the library imports will fail.

---  

<a name="repo‑layout"></a>
## 3️⃣ Repository layout (relevant files)

