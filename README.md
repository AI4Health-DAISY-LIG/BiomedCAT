# BiomedCAT

BiomedCAT converts biomedical presentation slides into knowledge-base-grounded entities: it reads the text off each slide, extracts the typed biomedical entities, and links each one to a standard identifier (CURIE) in a biomedical knowledge base.

Knowledge-graph construction methods assume clean plain text such as PubMed abstracts, but biomedical researchers communicate through slides that mix text, figures, tables, and charts, where a single corrupted character can invalidate a gene symbol and break downstream extraction. BiomedCAT treats input modality as a first-class problem. The working domain is facioscapulohumeral muscular dystrophy (FSHD).

![pipeline](docs/pipeline.png)

The pipeline runs in three stages, each an independent module with a single entry point that manages its own model lifecycle:

1. **OCR** — slide images to text (GLM-OCR).
2. **NER** — text to typed biomedical entities (Llama-3.1-8B, zero-shot).
3. **Normalization** — entities to knowledge-base identifiers (retrieval plus a language-model judge).

## Requirements

**Hardware.** A CUDA-capable **NVIDIA GPU with at least 8 GB of VRAM**. The language models run in 4-bit CUDA quantization, so an NVIDIA GPU is required. The pipeline therefore runs on **Linux or Windows** machines that have such a GPU. **macOS is not supported**, because it does not provide NVIDIA CUDA.

**Disk.** Roughly 25 GB free: about 5 GB for the Python environment and about 20 GB for the model weights, which are downloaded once on the first run and cached in `~/.cache/huggingface`.

**Software.** [uv](https://docs.astral.sh/uv/) as the dependency manager, plus poppler and LibreOffice as system tools. Python itself is provisioned by uv and does not need to be installed separately.

## Setup

### Step 0. Request model access first

Access to `meta-llama/Llama-3.1-8B-Instruct` is gated: request it on [its Hugging Face page](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) and accept the license. Approval is not immediate and can take hours, so start this before anything else. While waiting, create a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens); it is needed in step 4.

### Step 1. Install uv

| Platform | Command |
|---|---|
| Linux | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Windows (PowerShell) | `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 \| iex"` |
| Any (via pip) | `pip install uv` |

### Step 2. Install poppler and LibreOffice

| Platform | Command |
|---|---|
| Linux (Debian / Ubuntu) | `sudo apt install poppler-utils libreoffice` |
| Windows | Install LibreOffice from [libreoffice.org](https://www.libreoffice.org/), then install poppler (for example `conda install -c conda-forge poppler`, or download the prebuilt binaries). |

On Windows, **both** tools must be on `PATH`. LibreOffice is located by searching `PATH` for `soffice`, and its installer does not add itself, so add its `program` directory (typically `C:\Program Files\LibreOffice\program`) manually. Poppler's `bin` directory needs the same treatment.

### Step 3. Clone the repository and build the environment

```
git clone https://github.com/AI4Health-DAISY-LIG/BiomedCAT.git
cd BiomedCAT
uv sync
```

`uv sync` reproduces the exact dependency versions from `pyproject.toml` and `uv.lock`. The lockfile pins the CUDA 11.8 build of PyTorch; if your GPU needs a different CUDA version, adjust the `pytorch-cu118` index in `pyproject.toml` before running it.

### Step 4. Add your Hugging Face token

Create a `.env` file in the repository root, next to `pyproject.toml`:

```
BIOMEDCAT_HF_TOKEN=hf_your_token_here
```

This file is read relative to the directory the pipeline is launched from, so **run every command below from the repository root**. Launching from elsewhere leaves the token unset and the run fails at model download.

### Step 5. Add your slides

Input files are not distributed with the code. Create a `Dataset/` directory in the repository root and place the presentations to process inside it:

```
mkdir Dataset
```

Supported formats: `.pptx`, `.pdf`, `.png`, `.jpg`, `.jpeg`. A single file elsewhere on disk can also be passed directly, in which case `Dataset/` is not needed.

### Step 6. Verify the setup

Each check is fast and isolates one failure that would otherwise appear minutes into a run:

```
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available())"
uv run python -c "import shutil; print('LibreOffice:', shutil.which('soffice') or shutil.which('libreoffice'))"
uv run python -c "import shutil; print('poppler:', shutil.which('pdftoppm'))"
uv run python -c "from biomedcat.config import settings; print('token loaded:', bool(settings.hf_token))"
```

Expect `True`, two paths, and `True`. Any `None` or `False` points at the corresponding step above.

## Usage

Process every file in `Dataset/` that has no output yet:

```
uv run python -m biomedcat.pipeline
```

Each file produces two artefacts in `Output/`, which is created if absent: `<name>_BiomedCAT.json` holding the results, and `<name>_BiomedCAT.log` holding the full per-sentence trace of the run. A file whose JSON output already exists is skipped, so the batch is resumable across sessions; delete an output file to process that presentation again.

To run on a single file anywhere on disk instead:

```
uv run python -m biomedcat.pipeline path/to/your/slides.pptx
```

**The first run downloads roughly 20 GB of model weights** before any processing starts, and produces no output while doing so. This is expected and happens once. Later runs start immediately from the cache.

After the download, each file takes several minutes: three models load in sequence (GLM-OCR, then Llama for NER, then Llama for normalization) and two public resolvers are queried, with the NER stage dominating. Progress is logged per stage.

The terminal shows stage-level progress and any warnings. The JSON holds the transcribed text per slide, the typed entities, and the identifier each entity was linked to, together with the model identifiers, resolver endpoints, and elapsed time behind the run, so a later run that differs can be distinguished from a code change. The log holds the complete trace that the terminal omits: the sentences produced by segmentation, the candidate terms proposed and the subset surviving grounding, the type assigned to each term, and the candidate the judge selected for each entity. It is written even when a file fails, since that is when the trace is most needed.

### Example output

```
--- OCR (2 slide(s)) ---
[slide 1]
Facioscapulohumeral muscular dystrophy (FSHD)
- FSHD is the third most common type of muscular dystrophy...

--- NER (31 entit(y/ies)) ---
  DISEASE                  FSHD
  GENE                     DUX4 protein
  EPIGENETIC_MODIFICATION  methylation
  ...

--- Normalization (28/31 linked) ---
  DISEASE   FSHD           -> MONDO:0008030
  GENE      DUX4 protein   -> NCBIGene:100288687
  ...

Wrote /path/to/BiomedCAT/Output/FSHD1_BiomedCAT.json
```

The final `Wrote` line confirms the run completed and names the file produced. In the JSON, an entity the judge could not link has `"curie": null` rather than being omitted.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `No Dataset/ directory` | no input supplied | step 5, or pass a file path |
| `torch.cuda.OutOfMemoryError` | another process holds VRAM | close other GPU processes and check `nvidia-smi`; in VS Code, reload the window to clear stale kernels |
| `RuntimeError: LibreOffice (soffice) not found` | LibreOffice not on `PATH` | step 2 |
| poppler or `pdftoppm` errors | poppler not on `PATH` | step 2 |
| HTTP 401 or 403 from Hugging Face | token missing, or access not approved | steps 0 and 4; confirm the run is launched from the repository root |
| Long silence at the start | first-run model download | expected once; watch `~/.cache/huggingface` grow |

## How it works

Three design choices, all shaped by the 8 GB VRAM budget, are shared across the stages:

- **4-bit quantization** shrinks Llama-3.1-8B to roughly 5.7 GB so it fits the card.
- **Scale-to-zero lifecycle**: each stage loads its model, processes the input, and frees it, so the three models share one small GPU.
- **Determinism**: greedy decoding plus rule-based text steps (segmentation, grounding, deduplication) make runs reproducible; non-determinism is confined to the model calls.

**OCR.** GLM-OCR, a vision-language model, transcribes each slide. Decks and PDFs are rendered to page images, which preserves the layout that plain-text extraction would lose. Pages are transcribed in batches and streamed one batch at a time, so large files stay within memory.

**NER.** Following ZeroTuneBio, each sentence passes through three steps over Llama-3.1-8B: extract all professional terms to maximize recall, keep only the terms grounded in the sentence to drop hallucinations, and classify each grounded term into one of ten types or none to recover precision.

### Entity type system

The ten entity types are derived from the Biolink Model (version 4.4.3), the schema underlying the NCATS Biomedical Data Translator, so that every type the extractor can assign corresponds to a class the resolvers can be queried against. A class was admitted only if it satisfies three conditions: it is not a mixin, it declares `id_prefixes` and can therefore be the category of a resolved identifier (58 of the model's 334 classes qualify), and a type-constrained query against the RENCI name resolver returns results for it. The third condition is not redundant: an unrecognised type is answered with HTTP 200 and an empty result list rather than an error, so an invalid mapping is otherwise indistinguishable from a term that has no candidates.

| Entity type | Biolink class |
|---|---|
| `GENE` | `biolink:Gene` |
| `PROTEIN` | `biolink:Protein` |
| `DISEASE` | `biolink:Disease` |
| `PHENOTYPIC_FEATURE` | `biolink:PhenotypicFeature` |
| `CHEMICAL` | `biolink:ChemicalEntity` |
| `CELL_TYPE` | `biolink:Cell` |
| `CELLULAR_COMPONENT` | `biolink:CellularComponent` |
| `ANATOMY` | `biolink:GrossAnatomicalStructure` |
| `BIOLOGICAL_PROCESS` | `biolink:BiologicalProcessOrActivity` |
| `SEQUENCE_VARIANT` | `biolink:SequenceVariant` |

Two mappings warrant comment. `ANATOMY` targets `GrossAnatomicalStructure` rather than the broader `AnatomicalEntity`, which the model itself documents as a grouping class whose concrete subclasses should be preferred; this also keeps `ANATOMY`, `CELL_TYPE`, and `CELLULAR_COMPONENT` disjoint, since all three descend from `AnatomicalEntity`. `BIOLOGICAL_PROCESS` targets `BiologicalProcessOrActivity` because `MolecularActivity` and `BiologicalProcess` are siblings beneath it, so the narrower class would exclude molecular activities. Type filtering at the resolver is hierarchical, so a mapping to a parent class also retrieves its descendants.

Several categories occurring in slide material have no admissible target, for two distinct reasons. Genomic regions such as chromosomal loci and repeat arrays, and sub-protein regions such as domains and termini, are expressible in the Biolink Model but declare no `id_prefixes`, and the related `genomic entity` and `epigenomic entity` classes are mixins; no identifier can be assigned to any of them. Transcripts and macromolecular complexes are, by contrast, fully specified in the model but are not indexed by the RENCI resolver. The first is a modelling limitation that no change of resolver would address; the second reflects the coverage of one service and may be met by another. Entities of these kinds are still extracted by the first module and are typed into the nearest available category, which is a known source of error.

**Normalization.** For each entity, candidates are retrieved concurrently from two resolvers (the RENCI name resolver and the ARAX entity normalizer), supplemented by a pass constrained to the Biolink class the entity's type maps to, which recovers correctly typed candidates that surface matching misses. The three result sets are unioned by CURIE and given a single ranking, with candidates that several resolvers agree on placed first. A language-model judge then selects the candidate whose type and meaning both match the entity, or abstains.

## Limitations and future work

- **Gold annotation.** No gold set exists yet for NER or Normalization, so those heuristics are unvalidated. This is the highest-priority item.
- **Hardware.** The pipeline requires a CUDA-capable NVIDIA GPU with 8 GB of VRAM and does not run on CPU or macOS. On the 8 GB target the language-model stages are at their computational floor.
- **Retrieval inputs.** Normalization depends on two public resolvers, so retrieval recall can drift as those services change, and responses are not cached between runs. A lookup that times out is logged and treated as returning no candidates, which is indistinguishable downstream from a term genuinely having none; a run containing such failures is therefore not exactly reproducible.
- **Type coverage.** Genomic regions and sub-protein regions cannot be normalized, for the reasons given under the entity type system. Terms of these kinds are still extracted and assigned the nearest available type, so they may receive a confident but incorrect identifier rather than an abstention.
- **Taxon.** Retrieval is not constrained by organism, so a term may be linked to a non-human orthologue.
- **Modality coverage.** OCR handles slide decks, PDFs, and images; slide-deck support depends on a system LibreOffice installation.

## Repository layout

The core is an installable Python package, `biomedcat`, organized one module per concern:

- `config.py`: deployment configuration (model identifiers, resolver endpoints, the Hugging Face token), read from the environment or a local `.env` file.
- `types.py`: the typed data model passed between stages (Slide, Entity, Candidate, NormalizedEntity) and the entity-type vocabulary.
- `runtime.py`: the shared runtime layer (CUDA and determinism setup, the 4-bit language-model factory, the greedy generation helper).
- `prompts.py`: every model-facing prompt, isolated so the research-tuned wording is versioned separately from the pipeline logic.
- `stages/ocr.py`, `stages/ner.py`, `stages/norm.py`: the three stages, each with a single entry point.
- `retrieval.py`: the RENCI and ARAX resolvers behind a common interface, and the mapping from entity type to Biolink class that drives the type-constrained pass.
- `pipeline.py`: the orchestrator, which runs one file through the three stages and drives the batch over `Dataset/`.
