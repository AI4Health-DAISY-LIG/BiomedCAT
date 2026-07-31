class InferenceEngine:
    """
    Moteur d'inférence léger utilisant l'API HTTP d'Ollama.
    Remplace l'ancienne implémentation lourde (torch/transformers).
    """
    def __init__(self):
        self.api_url = f"{settings.ollama_url}/api/chat"
        self.model = settings.model_name
        logger.info(f"InferenceEngine initialisé avec le modèle: {self.model} via {settings.ollama_url}")

    async def generate(self, messages: List[Dict[str, str]]) -> str:
        """
