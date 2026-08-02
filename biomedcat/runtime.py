import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import gc
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login
from biomedcat.config import settings

torch.use_deterministic_algorithms(True, warn_only=True)
torch.manual_seed(0)


def free_gpu() -> None:
    """Release Python garbage, then return PyTorch's cached VRAM to the driver."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def hf_login() -> None:
    """Authenticate to Hugging Face for the gated Llama repo (no-op if no token)."""
    token = os.environ.get("HF_TOKEN") or settings.hf_token
    if token:
        login(token=token)

def build_4bit_config() -> BitsAndBytesConfig:
    """4-bit NF4 with double quantization: shrinks Llama-3.1-8B to ~5.7 GB for the 8 GB card."""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

def build_llm(model_id: str | None = None):
    """Load the 4-bit NF4 LLM (tokenizer + model), shared by NER and normalization.

    Handles auth and a pre-load VRAM clear, then loads quantized and in eval mode.
    Uses provided model_id if available, otherwise falls back to settings.llm_model_id.
    Returns (tokenizer, model).
    """
    hf_login()   # gated repo; no-op if no token
    free_gpu()   # clear cached VRAM before the load
    
    target_model_id = model_id or settings.llm_model_id
    
    tokenizer = AutoTokenizer.from_pretrained(target_model_id)
    model = AutoModelForCausalLM.from_pretrained(
        target_model_id,
        quantization_config=build_4bit_config(),
        dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    return tokenizer, model

def generate(model, tokenizer, messages: list[dict[str, str]], max_new_tokens: int) -> str:
    """Greedily decode a chat message list and return the generated text.

    Greedy (do_sample=False) argmax decoding, deterministic in exact arithmetic.
    Llama has no pad token, so EOS is reused for padding.
    """
    # Render chat turns into Llama's token format, on the model's device.
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device)
    input_len = inputs["input_ids"].shape[1]  # prompt length, to slice it off later

    # inference_mode: no autograd graph, lighter on VRAM.
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,                      # greedy
            pad_token_id=tokenizer.eos_token_id,  # Llama has no pad token; reuse EOS
        )

    generated = output_ids[0, input_len:]  # keep only the new tokens
    text = tokenizer.decode(generated, skip_special_tokens=True).strip()
    del inputs, output_ids  # free GPU tensors promptly
    return text
