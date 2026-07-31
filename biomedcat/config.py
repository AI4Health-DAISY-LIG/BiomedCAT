"""Runtime configuration for the BiomedCAT core.

Holds only what varies per deployment or must stay secret: the model
identifiers, the even external resolver endpoints, and the Hugging Face token.
Fixed implementation details (generation lengths, CUDA flags, batch size) live
as constants beside the code that uses them, not here.

Any field can be overridden by a BIOMEDCAT_<NAME> environment variable or a line
in a local .env file, so the service is retuned without editing source.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="BIOMEDCAT_", env_file=".env", extra="ignore")

    # Hugging Face token for the gated Llama repo (set BI_HF_TOKEN in .env).
    hf_token: str | None = None

    # Ollama Configuration.
    # In Docker, we use 'host.docker.internal' to reach the host machine's Ollama service.
    ollama_url: str = "http://host.docker.internal:11434"

    # Models.
    glm_model_id: str = "zai-org/GLM-OCR"                    # OCR (stage 1)
    llm_model_id: str = "meta-llama/Llama-3.1-8B-Instruct"   # NER + normalization (stages 2-3)

    # Normalization resolvers.
    renci_url: str = "https://name-resolution-sri.renci.org/lookup"
    arax_url: str = "https://arax.ncats.io/api/arax/v1.4/entity"
    api_limit: int = 10                # max candidates pulled per resolver
    max_concurrent_requests: int = 10  # cap on in-flight retrieval requests


settings = Settings()
