import os
from pathlib import Path

ROOT_PATH = Path(__file__).parent.parent

class Settings:
    """
    BiomedCAT application configuration.
    Variables can be overridden via Docker environment variables.
    """
    def __init__(self):
        # URL of the Ollama instance (default to host via Docker bridge)
        # self.ollama_url = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434") 
        # Biolink model source, pinned to the release used by every result of the paper (4.4.4).
        # `master` moved on 17 Sept 2026 without changing the 159 classes in use, but a later
        # commit could: the release tag makes the class index reproducible. Override with
        # BIOLINK_MODEL_SOURCE (a GitHub blob URL or a local YAML path).
        self.biolink_model_data = os.getenv("BIOLINK_MODEL_SOURCE", "https://github.com/biolink/biolink-model/blob/v4.4.4/biolink-model.yaml")

        # INPUTS. PPTX is not accepted: export slides to PDF first (LibreOffice is no longer a dependency).
        self.supported_docs = {".pdf", ".png", ".jpg", ".jpeg"}

        # Ollama keep_alive sent with every request. Default 0: the model is unloaded after every
        # call. Measured on 11 Sept 2026: a resident model ("10m") saves only ~15 % of the agent's
        # time (generation dominates, not loading) but breaks reproducibility -- two back-to-back
        # runs of the same 50 terms agreed on 82 % of verdicts with a resident model against 100 %
        # with unload-per-call (prompt/KV cache reuse changes the arithmetic of greedy decoding).
        # Set "10m" for interactive use where speed matters more than bit-identical runs; the
        # stage-boundary unload_models() calls keep the 16 GB budget in both modes.
        self.ollama_keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "0")
        # Hidden "thinking" of reasoning models (gemma4): off, otherwise the token budget of short
        # answers is consumed by reasoning and the visible reply comes back empty.
        self.ollama_think = os.getenv("OLLAMA_THINK", "0") in ("1", "true", "yes")
        # Sampling seed sent to Ollama with every call. Temperature 0 alone is not enough:
        # Ollama re-samples on ties and its kernels are not bit-reproducible, so two runs of
        # the same document could differ. A fixed seed makes a run reproducible; set
        # OLLAMA_SEED to an empty string to let the server choose (non-reproducible).
        self.ollama_seed = os.getenv("OLLAMA_SEED", "0")
        # Sampling overrides for experiments (empty = use the temperature each call passes, greedy
        # by default, and Ollama's own top-k / top-p). Gemma's recommended sampling is
        # OLLAMA_TEMPERATURE=1.0 OLLAMA_TOP_K=64 OLLAMA_TOP_P=0.95; with a fixed seed the run stays
        # reproducible, so temperature governs exploration and the seed governs reproducibility.
        self.ollama_temperature = os.getenv("OLLAMA_TEMPERATURE", "")
        self.ollama_top_k = os.getenv("OLLAMA_TOP_K", "")
        self.ollama_top_p = os.getenv("OLLAMA_TOP_P", "")
        
        # Model used for inference
        # self.model_name = os.getenv("OLLAMA_MODEL", "zai-org/GLM-OCR") # "gemma4:e4b-it-qat"

        # Working paths (mounted via Docker volumes)
        self.internal_data_path = os.getenv("INTERNAL_DATA_PATH", Path.joinpath(ROOT_PATH, "data"))
        base_dir = Path(self.internal_data_path)
        base_dir.mkdir(parents=True, exist_ok=True)

        self.chroma_db_path = os.getenv("CHROMADB_PATH",Path.joinpath(ROOT_PATH, "data/biolink_chromaDB"))
        base_dir = Path(self.chroma_db_path)
        base_dir.mkdir(parents=True, exist_ok=True)

        self.dataset_path = os.getenv("DATASET_PATH", Path.joinpath(ROOT_PATH, "dataset"))
        self.output_path = os.getenv("OUTPUT_PATH", Path.joinpath(ROOT_PATH, "output"))
        base_dir = Path(self.output_path)
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Pipeline configuration
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.timeout = int(os.getenv("TIMEOUT", "300"))

        # --- Attributes required by the pipeline (to avoid AttributeError) ---
        
        # Map task models to the default main model
        self.ocr_model_id = os.getenv("OCR_MODEL_ID","qwen2.5vl:3b")
        self.RAG_embedding_model = os.getenv("EMBEDDINGS_MODEL_ID", "NeuML/bioclinical-modernbert-base-embeddings")
        self.classification_model_id = os.getenv("CLASSIFICATION_MODEL_ID", "gemma4:e4b-it-qat")
        self.sanitization_model_id = os.getenv("SANITIZATION_MODEL_ID", "llama-guard3:1b")
        # Content screening of each slide's text with the sanitization model, once per slide
        # (not once per term). "0" disables it. It is a content-safety check, not an injection
        # detector; injection is handled structurally in the NER agent.
        self.document_screening = os.getenv("DOCUMENT_SCREENING", "1") not in ("0", "false", "no")
        # Resolver service URLs (configure via environment if needed).
        # RENCI Name Resolver: GET /lookup?string=...&limit=N[&biolink_type=...] -> list of candidates.
        # ARAX entity endpoint: GET /entity?q=term -> {term: {"id": {...}, "knowledge_graph": {...}}}.
        self.renci_url = os.getenv("RENCI_URL", "https://name-resolution-sri.renci.org/lookup")
        self.arax_url = os.getenv("ARAX_URL", "https://arax.ncats.io/beta/api/arax/v1.4/entity")

        # Existence gate applied once per document to the agent's typed terms, against the
        # invention of entities by the classifier: "nameres" rejects terms without any candidate
        # in the Name Resolver (production instance, the same service as the linking; "es" is
        # accepted as a legacy alias from the time the gate used the Elasticsearch test instance),
        # "llm" those a model of another family does not recognise, "union" (default) either,
        # "off" disables the gate. Since 5 Oct 2026 the gate queries the production Name Resolver.
        self.existence_gate = os.getenv("EXISTENCE_GATE", "union")
        self.existence_model_id = os.getenv("EXISTENCE_MODEL_ID", "llama3:8b")
        self.gate_nameres_url = os.getenv("GATE_NAMERES_URL", os.getenv("NAMERES_ES_URL", self.renci_url))
        self.nameres_es_url = self.gate_nameres_url   # legacy name, kept for the evaluation scripts
        # self.llm_model_id = os.getenv("LLM_MODEL_ID", self.model_name)
        # Translator Node Normalizer: clique-preferred id for CURIEs absent from the local table.
        self.nodenorm_url = os.getenv("NODENORM_URL", "https://nodenorm.transltr.io/1.5/get_normalized_nodes")

        # API call limit and concurrency for the resolver lookups
        self.api_limit = int(os.getenv("API_LIMIT", "10"))
        # RESOLVERS_OFFLINE=1: no text leaves the machine; entities are linked by exact name
        # against the local KG2c node table only (lower recall, full privacy).
        self.resolvers_offline = os.getenv("RESOLVERS_OFFLINE", "0") in ("1", "true", "yes")
        # REVIEW_MODE=1: stop after OCR+NER and wait for the user's review of the entity list
        # before any term is sent to the resolvers (human in the loop for clinical material).
        self.review_mode = os.getenv("REVIEW_MODE", "0") in ("1", "true", "yes")
        self.max_concurrent_requests = int(os.getenv("MAX_CONCURRENT_REQUESTS", "4"))

        # Filtered RTX-KG2c build (scripts/build_kg2c_parquet.py): edges/nodes/equivalents Parquet.
        # equivalents.parquet maps any known CURIE to the canonical KG2c id, used as an offline
        # canonicalization step after name resolution and by the context-graph stage.
        self.kg2c_dir = os.getenv("KG2C_DIR", str(Path.joinpath(ROOT_PATH, "data/kg2c")))
        # Default user preference profile (entity-branch format, biomedcat.profiles); the console
        # overrides it per job with the merged reading profile.
        self.predicate_profile = os.getenv(
            "PREDICATE_PROFILE", str(Path.joinpath(ROOT_PATH, "data/profiles/biochemical_actions.json"))
        )

settings = Settings()
