import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """
    Configuration de l'application BiomedCAT.
    Les variables peuvent être surchargées via les variables d'environnement Docker.
    """
    # URL de l'instance Ollama (par défaut vers l'hôte via le pont Docker)
    ollama_url: str = os.getenv("OLLAMA_URL", "http://host.docker.internal:11434")
    
    # Modèle utilisé pour l'inférence
    model_name: str = os.getenv("OLLAMA_MODEL", "llama3.1")
    
    # Chemins de travail (montés via Docker volumes)
    dataset_path: str = "/app/data"
    output_path: str = "/app/output"
    
    # Configuration du pipeline
    max_retries: int = 3
    timeout: int = 300  # 5 minutes

    class Config:
        env_file = ".env"

settings = Settings()
