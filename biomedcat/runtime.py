import logging
import requests
import time
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
    prompt = ""
    for msg in messages:
        role = msg.get("role", "user").capitalize()
        content = msg["content"]
        prompt += f"{role}: {content}\n"

    url = f"{settings.ollama_url.rstrip('/')}/api/generate"
    payload = {
        "model": model_id,
        "prompt": prompt.strip(),
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_new_tokens,
            "do_sample":False
        }
    }

    max_retries = 3
    base_delay = 1
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            return data["response"].strip()
        except requests.exceptions.Timeout:
            logger = logging.getLogger(__name__)
            logger.error(f"Timeout communicating with Ollama API for model {model_id} (attempt {attempt + 1})")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
                continue
            return ""
        except requests.exceptions.ConnectionError:
            logger = logging.getLogger(__name__)
            logger.error(f"Connection error communicating with Ollama API for model {model_id} (attempt {attempt + 1})")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
                continue
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
    
    return ""


def chat(model_id: str, messages: list[dict[str, str]], max_new_tokens: int, temperature: float = 0.0) -> str:
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

    max_retries = 3
    base_delay = 1
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()
            return data["message"]["content"].strip()
        except requests.exceptions.Timeout:
            logger = logging.getLogger(__name__)
            logger.error(f"Timeout communicating with Ollama API for model {model_id} (attempt {attempt + 1})")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
                continue
            return ""
        except requests.exceptions.ConnectionError:
            logger = logging.getLogger(__name__)
            logger.error(f"Connection error communicating with Ollama API for model {model_id} (attempt {attempt + 1})")
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
                continue
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
    
    return ""
