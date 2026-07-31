import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

def free_gpu():
    """
    Fonction factice pour maintenir la compatibilité avec l'ancien code.
    Comme nous utilisons Ollama (service externe) pour le NER/Norm, 
    la gestion de la VRAM est déléguée au service. Pour l'OCR local, 
    le nettoyage est déjà géré par 'del' dans le stage.
    """
    pass

class InferenceEngine:
    """
    Moteur d'inférence léger utilisant l'API HTTP d'Ollama.
    Remplace l'ancienne implémentation lourde (torch/transformers).
    """
    def __init__(self, model, settings):
        self.model = model
        # On suppose que logger et settings sont définis dans le contexte du module
        logger.info(f"InferenceEngine initialisé avec le modèle: {self.model} via {settings.ollama_url}")

    async def generate(self, messages: List[Dict[str, str]]) -> str:
        """
        Génère une réponse à partir des messages fournis.
        """
        # L'implémentation n'est pas fournie dans l'extrait original.
        pass
