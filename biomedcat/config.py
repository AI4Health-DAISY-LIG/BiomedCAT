import os

class Settings:
    """
    Configuration de l'application BiomedCAT.
    Les variables peuvent être surchargées via les variables d'environnement Docker.
    """
    def __init__(self):
        # URL de l'instance Ollama (par défaut vers l'hôte via le pont Docker)
        self.ollama_url = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
        
        # Modèle utilisé pour l'inférence
        # self.model_name = os.getenv("OLLAMA_MODEL", "llama3.1")
        
        # Chemins de travail (montés via Docker volumes)
        self.dataset_path = os.getenv("DATASET_PATH", "/app/data")
        self.output_path = os.getenv("OUTPUT_PATH", "/app/output")
        
        # Configuration du pipeline
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.timeout = int(os.getenv("TIMEOUT", "300"))

        # --- Attributs requis par le pipeline (pour éviter les AttributeError) ---
        
        # On mappe les modèles de tâches sur le modèle principal par défaut
        self.glm_model_id = os.getenv("GLM_MODEL_ID", self.model_name)
        self.llm_model_id = os.getenv("LLM_MODEL_ID", self.model_name)

        # URLs des services de résolution (à configurer via l'environnement si besoin)
        self.renci_url = os.getenv("RENCI_URL", "https://renci.org/api")
        self.arax_url = os.getenv("ARAX_URL", "https://arax.ebi.ac.uk/services/api")

        # Limite d'appels API
        self.api_limit = int(os.getenv("API_LIMIT", "10"))

settings = Settings()
