import logging
import requests
from biomedcat.config import settings


def generate(model_id: str, messages: list[dict[str, str]], max_new_tokens: int, temperature: float = 0.0) -> str:
    """
    Sends a request to the Ollama API to generate a response.

    Args:
        model_id: The identifier of the model to use.
        messages: A list of message dictionaries (role and content).
        max_new_tokens: Maximum number of tokens to predict.
        temperature: Sampling temperature.

    Returns:
        The generated text response from the model, or an empty string if an error occurs.
    """
    url = f"{settings.ollama_url.rstrip('/')}/api/chat"
    payload = {
        "model": model_id,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_new_tokens
        }
    }

    try:
        response = requests.post(url, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data["message"]["content"].strip()
    except requests.exceptions.Timeout:
        logger = logging.getLogger(__name__)
        logger.error(f"Timeout communicating with Ollama API for model {model_id}")
        return ""
    except requests.exceptions.ConnectionError:
        logger = logging.getLogger(__name__)
        logger.error(f"Connection error communicating with Ollama API for model {model_id}")
        return ""
    except requests.exceptions.HTTPError as e:
        logger = logging.getLogger(__name__)
        if e.response.status_code == 404:
            logger.error(f"Model {model_id} not found on Ollama")
        else:
            logger.error(f"HTTP error communicating with Ollama API: {e}")
        return ""
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Unexpected error communicating with Ollama API: {e}")
        return ""
