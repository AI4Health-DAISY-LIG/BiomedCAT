import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import gc
import logging
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
from biomedcat.config import settings

# Ensure deterministic behavior for reproducibility
torch.use_deterministic_algorithms(True, warn_only=True)
torch.manual_seed(0)


def free_gpu() -> None:
    """Release Python garbage, then return PyTorch's cached VRAM to the driver if available."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def hf_login() -> None:
    """Authenticate to Hugging Face for gated models (e.g., Llama-3) using HF_TOKEN or settings."""
    token = os.environ.get("HF_TOKEN") or settings.hf_token
    if token:
        login(token=token)

def build_4bit_config() -> BitsAndBytesConfig:
    """Configure 4-bit NF4 quantization with double quantization for VRAM efficiency."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

def build_llm(model_id: str | None = None):
    """
    Load the LLM (tokenizer + model) using a hybrid strategy based on hardware availability.

    Decision Logic:
    1. If CUDA is available: Use high-performance 4-bit quantization (BitsAndBytes).
       This is optimized for NVIDIA GPUs to minimize VRAM usage.
    2. If CUDA is NOT available: Use standard CPU loading (Float32/Bfloat16).
       Note: For true production portability on low-end hardware, integrating 
       llama-cpp-python with GGUF models would be the recommended next step.

    Returns:
        tuple: (tokenizer, model)
    """
    hf_login()   # Authenticate if token is provided
    free_gpu()   # Clear VRAM before loading new model
    
    target_model_id = model_id or settings.llm_model_id
    tokenizer = AutoTokenizer.from_pretrained(target_model_id)
    
    if torch.cuda.is_available():
        # --- GPU PATH: Optimized for NVIDIA GPUs using 4-bit quantization ---
        logger = logging.getLogger(__name__)
        logger.info("Hardware detected: NVIDIA GPU. Loading quantized 4-bit model...")
        
        quantization_config = build_4bit_config()
        model = AutoModelForCausalLM.from_pretrained(
            target_model_id,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ).eval()
    else:
        # --- CPU PATH: Standard loading for maximum compatibility ---
        logger = logging.getLogger(__name__)
        logger.info("Hardware detected: CPU only. Loading standard model (this may be slow)...")
        
        model = AutoModelForCausalLM.from_pretrained(
            target_model_id,
            torch_dtype=torch.float32, # Use float32 for stability on CPU
            device_map="cpu",
        ).eval()
        
    return tokenizer, model

def generate(model, tokenizer, messages: list[dict[str, str]], max_new_tokens: int) -> str:
    """Greedily decode a chat message list and return the generated text.

    Uses greedy decoding (do_sample=False) to ensure deterministic output.
    Works for both GPU-based and CPU-based models loaded via build_llm.
    """
    # Prepare inputs using the model's specific chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    
    input_len = inputs["input_ids"].shape[1]  # Track prompt length to slice output

    # Use inference_mode for reduced memory footprint and faster execution
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,                      # Greedy decoding
            pad_token_id=tokenizer.eos_token_id,  # Use EOS as padding if PAD is missing
        )

    # Extract only the newly generated tokens (exclude the prompt)
    generated = output_ids[0, input_len:]
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    
    # Clean up tensors from memory immediately
    del inputs, output_ids
    return text
