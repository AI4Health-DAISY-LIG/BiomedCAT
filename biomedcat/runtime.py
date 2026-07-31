import httpx
import logging
from typing import List, Dict, Any
from .config import settings

logger = logging.getLogger(__name__)

class InferenceEngine:
    """
    Moteur d'inférence léger utilisant l'API HTTP d'Ollama.
    Remplace l'ancienne implémentation lourde (torch/transformers).
    """
    
    def __init__(self):
        self.api_url = f"{settings.ollama_url}/api/chat"
        self.model = settings.model_name
        logger.info(t"InferenceEngine initialisé avec le modèle: {self.model} via {settings.ollama_url}")

    async def generate(self, messages: List[Dict[str, str]]) -> str:
        """
        Envoie une requête de chat à Ollama et récupère la réponse textuelle.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.9
            }
        }

        try:
            async with httpx.AsyncClient(timeout=settings.timeout) as client:
                logger.info("Envoi de la requête à Ollama...")
                response = await client.post(self.api_url, json=payload)
                response.raise_for_status()
                
                data = response.json()
                content = data.get("message", {}).get("content", "")
                
                if not content:
                    logger.error("Réponse vide reçue d'Ollama.")
                    return "Erreur: Aucune réponse générée par le modèle."
                
                return content

        except httpx.HTTPStatusError as e:
            logger.error(f"Erreur HTTP lors de l'appel à Ollama: {e.response.status_code} - {e.response.text}")
            return f"Erreur de communication avec le serveur d'inférence (Status: {e.response.status_code})"
        except httpx.ConnectError:
            logger.error(f"Impossible de contacter Ollama à l'adresse {settings.ollama_url}. Vérifiez que Ollama est lancé sur l'hôte.")
            return "Erreur: Impossible de contacter Ollama. Assurez-vous qu'Ollama est installé et actif sur votre machine."
        except Exception as e:
            logger.error(f"Erreur inattendue lors de la génération: {str(e)}")
            return f"Erreur interne du moteur d'inférence: {str(e)}"

    async def check_connection(self) -> bool:
        """Vérifie si le service Ollama est accessible."""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{settings.ollama_url}/api/tags")
                return response.status_code == 200
        except Exception:
            return False
