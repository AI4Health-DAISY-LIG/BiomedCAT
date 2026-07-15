# BiomedCAT

BiomedCAT is a three-stage pipeline that converts biomedical presentation slides into knowledge-base-grounded entities. It reads the text off each slide, extracts the typed biomedical entities, and links each entity to a standard identifier (CURIE) in a biomedical knowledge base.

The motivation is that knowledge-graph construction literature assumes clean plain text such as PubMed abstracts, whereas biomedical researchers communicate through slides that combine text blocks, figures, tables, and charts. The corruption of a single character can alter a gene symbol and invalidate downstream extraction, so input modality is treated as a first-class problem. The working domain is facioscapulohumeral muscular dystrophy (FSHD).

## Pipeline


![pipeline](docs/pipeline.png)


Each stage is an independent module, exposes a single entry point, and manages its own model lifecycle. The design target is a single consumer GPU (NVIDIA RTX 4060 Laptop, 8 GB VRAM), which constrains every modelling and systems decision.

## Repository layout

The core is an installable Python package, `biomedcat`, organized one module per concern:

- `config.py`: deployment configuration (model identifiers, resolver endpoints, and the Hugging Face token), read from the environment or a local `.env` file.
- `types.py`: the typed data model passed between stages (Slide, Entity, Candidate, NormalizedEntity), together with the biomedical entity-type vocabulary.
- `runtime.py`: the shared runtime layer, comprising the CUDA and determinism setup, the 4-bit language-model factory, and the greedy generation helper.
- `prompts.py`: every model-facing prompt, isolated so the research-tuned wording is versioned separately from the pipeline logic.
- `stages/ocr.py`, `stages/ner.py`, `stages/norm.py`: the three stages, each exposing a single entry point and managing its own model lifecycle.
- `retrieval.py`: the RENCI and ARAX resolvers behind a common interface, supplying candidates to the normalization stage.
- `pipeline.py`: the orchestrator that runs one file through the three stages in sequence.

## Requirements and setup

The pipeline targets a single consumer GPU (NVIDIA RTX 4060 Laptop, 8 GB VRAM) with CUDA. Beyond the Python dependencies, two system tools are required: `poppler-utils` for PDF rasterization, and LibreOffice for the slide-deck to PDF conversion.

Python dependencies are pinned with `uv`. From the project root:

    sudo apt install poppler-utils libreoffice
    uv sync

`uv sync` builds the environment from `pyproject.toml` and the committed `uv.lock`, reproducing the exact dependency versions the pipeline was developed against. The gated Llama-3.1-8B repository requires a Hugging Face token, supplied as `BIOMEDCAT_HF_TOKEN` in a local `.env` file.

## Engineering substrate

The three stages share a common set of constraints and conventions:

- **4-bit quantization.** Llama-3.1-8B is loaded in 4-bit, reducing the 8B model to roughly 5.7 GB so it fits an 8 GB card. GLM-OCR is small enough (about 2.3 GB) to load in its native precision.
- **Scale-to-zero memory lifecycle.** Each stage loads its model, processes the batch, and frees the model afterwards, so GPU memory returns to baseline after every call. This fits a deployment where GPU workers are spun up on demand.
- **Greedy decoding.** All language-model stages decode deterministically (temperature 0), so a given input maps to a single output in principle.
- **Deterministic non-model steps.** Sentence segmentation, text normalization, term grounding, and deduplication are rule-based and reproducible by construction; non-determinism is confined to the quantized model calls.

## Stage 1: OCR

Transcribes each slide image to text using GLM-OCR, a vision-language model. Slide decks and PDFs are first rasterized to images: a slide deck is converted to PDF and then to page images, which preserves the layout the model reads, whereas plain text extraction would lose figures, tables, and positioning. Slide decks are the primary target modality and are supported end to end.

OCR is not memory-bound: the small model leaves the GPU underutilized between images, so processing several images together (static batching) gives up to roughly 2x throughput with identical output. Two heavier optimizations were evaluated and rejected, one because the vision model cannot be compiled cleanly, the other because its advantage (serving many concurrent requests) does not apply to this offline single-deck workload and conflicts with the memory budget.

Stage 1 is the only stage with a completed quantitative evaluation. Against a manually transcribed FSHD slide set, GLM-OCR is compared to PaddleOCR on character and word error rate:

| Model | CER | WER | Time (s) |
|---|---|---|---|
| GLM-OCR | 0.050 | 0.101 | 1.69 |
| PaddleOCR 3.0 | 0.194 | 0.303 | 26.28 |

The evaluation includes a pre-registered analysis protocol with a smallest-effect-size-of-interest threshold. The sample is small and FSHD-specific, so the result is a defensible pilot rather than a general claim.

## Stage 2: NER

Extracts typed biomedical entities from the transcribed text. The method follows ZeroTuneBio (zero-shot, zero-tuning extraction) over Llama-3.1-8B. Seven entity types are assigned: gene, disease, chemical, cell type, anatomy, chromosomal locus, and epigenetic modification.

Preprocessing is fully deterministic (normalization, repair of hyphenated line breaks, and sentence segmentation), and each sentence becomes the context unit. Extraction then proceeds in three conceptual steps per sentence:

- **Extract.** A single model call pulls out all professional biomedical terms without filtering, to maximize recall.
- **Ground.** Only terms that actually occur in the sentence are kept, removing hallucinated spans before typing.
- **Classify.** For each grounded term, the model explains the term in context and assigns a single type or none, recovering the precision lost by the recall-maximizing first step. A final pass removes only assignments it explicitly flags as wrong, so a malformed reply can never silently delete a valid entity.

The per-term classification loop dominates runtime. Batching it was evaluated and rejected: it ran several times slower (long, variable model outputs force every call in a batch to wait for the slowest) and it changed a verdict (a gene was reclassified as none), so the stage is at its computational floor on this hardware.

**Status:** no gold set yet, so Stage 2 is not quantitatively evaluated. The planned protocol uses exact and partial-match precision, recall, and F1.

## Stage 3: Normalization

Maps each typed entity to a standard identifier in a biomedical knowledge base, following the retrieve-and-rerank approach with a language model as judge (BeLink).

- **Retrieval.** For each unique mention, candidates are fetched concurrently from two resolvers (the RENCI name resolver and the ARAX entity normalizer). A type-constrained pass supplements this: for entity types with a clean ontology category, an additional category-filtered lookup is merged in. This recovers correctly typed candidates that surface-form matching misses (for example, "skeletal muscle" returns the correct tissue concept rather than only muscle proteins). The pooled candidates are then ranked, favouring agreement between the two resolvers.
- **Selection.** The model is shown the entity, its type and a type definition, the sentence context, and a numbered menu of every candidate with its label and type. It selects the best candidate whose type is consistent with the entity, or abstains when nothing matches both meaning and type.

This stage is reproducible: across separate runs the smoke set produced identical identifiers. Batching the judge was evaluated and rejected (several times slower, and out of memory at full batch) because the long, variable inputs waste computation on padding; the selections themselves stay identical, so the only cost is speed.

**Status:** no gold set yet. The planned protocol scores linking precision, recall, and F1 with "no link" as an explicit class, alongside retrieval recall versus judge accuracy when the correct candidate is present.

## Cross-cutting finding: batching is hardware-gated

The three stages share an identical loop structure (one model call per item) yet respond oppositely to batching:

| Stage | Model footprint | GPU at batch size 1 | Batching result |
|---|---|---|---|
| OCR | 2.3 GB | underutilized | ~2x faster, output identical (adopted) |
| NER | 5.7 GB | saturated | several times slower, changes a verdict (rejected) |
| Normalization | 5.7 GB | saturated | several times slower, out of memory at full batch (rejected) |

Static batching helps only when the GPU is underutilized at a batch size of one. The small OCR model leaves the card idle between images, so batching fills it; the larger quantized model saturates the same card on its own, so batching adds overhead with no spare compute. This result is hardware-specific: on a larger GPU the language-model stages could batch safely.

## Evaluation status

| Stage | Gold set | Baseline comparison | Metric | Status |
|---|---|---|---|---|
| OCR | available | GLM-OCR vs PaddleOCR | character / word error rate | complete |
| NER | pending | ZeroTuneBio vs direct query | precision / recall / F1 | apparatus ready |
| Normalization | pending | top-hit vs judge | linking F1, recall at k | apparatus ready |
| End-to-end | pending | not yet defined | entity survival through the chain | not started |

Stage 1 has crossed the research-evaluation bar. Stages 2 and 3 have working, instrumented apparatus and specified metrics, but the gold annotation is still in preparation, so their outputs are verified qualitatively on small smoke sets. The end-to-end question, whether error compounds through the OCR to NER to Normalization chain, is the intended novel contribution and requires its own gold set and an entity-survival metric.

## Limitations and future work

- **Gold annotation.** No gold set exists for NER or Normalization, so every heuristic is currently unvalidated. This is the highest-priority item.
- **Reproducibility.** Determinism is enforced pipeline-wide through a fixed seed, deterministic algorithm selection, and a pinned cuBLAS workspace. The non-model steps are reproducible by construction, and the greedy model steps are deterministic up to the best-effort limits of the available deterministic GPU kernels.
- **Modality coverage.** OCR processes slide decks, PDFs, and images. Slide-deck support depends on a system office suite installation for the conversion step.
- **Computational ceiling.** On the 8 GB target GPU the language-model stages are at their computational floor; further gains would require different hardware.
- **Definition-grounded typing (planned).** NER typing currently relies on a handful of hand-written type definitions. A planned extension retrieves authoritative definitions from a standard biomedical ontology and supplies them to the classifier, grounding typing in the ontology rather than ad hoc definitions and aligning it with the categories used during normalization.
