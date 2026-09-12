import logging
import requests
import time
from biomedcat.config import settings

logger = logging.getLogger(__name__)


def unload_models(*model_ids: str) -> None:
    """Ask Ollama to unload the given models now (keep_alive 0 with no prompt).

    Called at stage boundaries by the pipeline so that a model kept resident during its stage
    (settings.ollama_keep_alive) releases its memory before the next model loads. Unknown or
    already-unloaded models are ignored; a server error is logged, never raised.
    """
    url = f"{settings.ollama_url.rstrip('/')}/api/generate"
    for model_id in dict.fromkeys(m for m in model_ids if m):
        try:
            requests.post(url, json={"model": model_id, "keep_alive": 0}, timeout=60)
            logger.info("[Ollama] unloaded %s", model_id)
        except requests.RequestException as e:
            logger.warning("[Ollama] could not unload %s: %s", model_id, e)


def stage_models() -> tuple[str, ...]:
    """Every Ollama model the pipeline may have loaded, for an unconditional release."""
    return (settings.ocr_model_id, settings.classification_model_id, settings.sanitization_model_id,
            settings.existence_model_id)


def _sampling_options(temperature: float) -> dict:
    """Temperature (and top-k / top-p when set) for one Ollama call: the caller's temperature unless
    OLLAMA_TEMPERATURE overrides it globally (see config.py; used by the T=1 sampling experiments)."""
    opts: dict = {"temperature": float(settings.ollama_temperature) if str(settings.ollama_temperature).strip() else temperature}
    if str(settings.ollama_top_k).strip():
        opts["top_k"] = int(settings.ollama_top_k)
    if str(settings.ollama_top_p).strip():
        opts["top_p"] = float(settings.ollama_top_p)
    return opts


def generate(model_id: str, messages: list[dict[str, str]], max_new_tokens: int, temperature: float = 0.0,
             think: bool | None = None) -> str:
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
        # Resident for the stage (see config.ollama_keep_alive); released by unload_models().
        "keep_alive": settings.ollama_keep_alive,
        # Thinking models (gemma4) otherwise spend the whole num_predict budget on hidden
        # reasoning and return an empty response; every BiomedCAT prompt is a direct answer.
        # Hidden reasoning off by default (config.ollama_think); a caller may enable it for one
        # call (second pass of the typing agent) and must then raise max_new_tokens accordingly.
        "think": settings.ollama_think if think is None else think,
        "options": {
            "num_predict": max_new_tokens,
            # Reproducibility: a fixed seed plus temperature 0 makes two runs of the same document
            # comparable. Without it Ollama re-samples on ties and the run-to-run variance is real.
            **({"seed": int(settings.ollama_seed)} if str(settings.ollama_seed).strip() else {}),
            **_sampling_options(temperature),
        }
    }

    max_retries = 3
    base_delay = 1
    
    for attempt in range(max_retries):
        try:
            response = requests.post(url, json=payload, timeout=300)
            response.raise_for_status()
            data = response.json()
            return data["response"].strip()
        except requests.exceptions.Timeout:
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
        # Resident for the stage (see config.ollama_keep_alive); released by unload_models().
        "keep_alive": settings.ollama_keep_alive,
        # Thinking models (gemma4) otherwise spend the whole num_predict budget on hidden
        # reasoning and return an empty response; every BiomedCAT prompt is a direct answer.
        "think": settings.ollama_think,
        "options": {
            "num_predict": max_new_tokens,
            # Reproducibility: a fixed seed plus temperature 0 makes two runs of the same document
            # comparable. Without it Ollama re-samples on ties and the run-to-run variance is real.
            **({"seed": int(settings.ollama_seed)} if str(settings.ollama_seed).strip() else {}),
            **_sampling_options(temperature),
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
