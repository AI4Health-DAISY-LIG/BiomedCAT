import os

class Settings:
    """
    BiomedCAT application configuration.
    Variables can be overridden via Docker environment variables.
    """
    def __init__(self):
        # URL of the Ollama instance (default to host via Docker bridge)
        # self.ollama_url = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
        self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434") 

        self.biolink_model_data = "https://github.com/biolink/biolink-model/blob/master/biolink-model.yaml"
        
        # Model used for inference
        # self.model_name = os.getenv("OLLAMA_MODEL", "zai-org/GLM-OCR") # "gemma4:e4b-it-qat"

        # Working paths (mounted via Docker volumes)
        self.dataset_path = os.getenv("DATASET_PATH", "/app/data")
        self.output_path = os.getenv("OUTPUT_PATH", "/app/output")
        
        # Pipeline configuration
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.timeout = int(os.getenv("TIMEOUT", "300"))

        # --- Attributes required by the pipeline (to avoid AttributeError) ---
        
        # Map task models to the default main model
        self.glm_model_id = os.getenv("GLM_MODEL_ID", "zai-org/GLM-OCR")
        self.RAG_embedding_model = os.getenv("GLM_MODEL_ID", "nomic-embed-text")
        self.extraction_model_id = os.getenv("EXTRACTION_MODEL_ID", "llama3:8b")
        self.classification_model_id = os.getenv("CLASSIFICATION_MODEL_ID", "gemma4:e4b-it-qat")
        # self.llm_model_id = os.getenv("LLM_MODEL_ID", self.model_name)

        # Resolver service URLs (configure via environment if needed)
        self.renci_url = os.getenv("RENCI_URL", "https://renci.org/api")
        self.arax_url = os.getenv("ARAX_URL", "https://arax.ebi.ac.uk/services/api")

        # API call limit
        self.api_limit = int(os.getenv("API_LIMIT", "10"))

settings = Settings()
