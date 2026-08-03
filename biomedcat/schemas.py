from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# --- Modèles de Réponse (Mirroring the JSON output of the pipeline) ---

class SlideSchema(BaseModel):
    """Représente le contenu textuel d'une slide."""
    page: int
    text: str

class EntitySchema(BaseModel):
    """Représente une entité extraite lors de l'étape NER."""
    text: str
    type: str
    segment: str

class NormalizedEntitySchema(BaseModel):
    """Représente une entité après l'étape de normalisation (avec CURIE)."""
    text: str
    type: str
    segment: str
    curie: Optional[str] = None

class RunInfoSchema(BaseModel):
    """Métadonnées relatives à l'exécution du pipeline."""
    file: str
    timestamp: str
    models: Dict[str, str]
    resolvers: Dict[str, Any]
    elapsed_s: float

class PipelineResultResponse(BaseModel):
    """
    Représente le payload JSON complet écrit sur le disque par le pipeline.
    Ce modèle est utilisé pour renvoyer les résultats finaux via l'API.
    """
    schema_version: str = "1.0"
    run: RunInfoSchema
    slides: List[SlideSchema]
    entities: List[NormalizedEntitySchema]

# --- Modèles de Requête (Input) ---

class ProcessingParams(BaseModel):
    """Paramètres optionnels pour configurer le processus de traitement."""
    model_id: Optional[str] = Field(
        None, 
        description="L'identifiant du modèle LLM à utiliser pour les étapes NER et Normalization."
    )
