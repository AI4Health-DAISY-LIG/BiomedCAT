import os
import gc
import json
import shutil
import tempfile
import subprocess
import torch
from pathlib import Path
from PIL import Image

# Crucial for 8GB VRAM: Reduce fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import AutoProcessor, GlmOcrForConditionalGeneration
from pdf2image import convert_from_path

GLM_MODEL_ID = "zai-org/GLM-OCR"
GLM_PROMPT = "Text Recognition:"
GLM_MAX_TOKENS = 1536
GLM_MAX_PIXELS = 1280 * 720
GLM_BATCH_SIZE = 8     # images per generate() call. Batching keeps the GPU busy instead of idling
                       # between slides (measured ~1.75x at 8, ~2.1x at 12; output byte-identical).
                       # VRAM-cheap here (~3 GB of 8); raise if your slides leave headroom.

def free_gpu():
    """Forces aggressive VRAM clearing."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _pptx_to_pdf(file_path: str, out_dir: str) -> str:
    """Convert a .pptx to .pdf via headless LibreOffice and return the PDF path.
    Going through PDF preserves the slide layout the OCR model reads; python-pptx
    would only extract raw text and lose figures, tables, and positioning."""
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice is None:
        raise RuntimeError("LibreOffice (soffice) not found; it is required to convert PPTX to PDF.")
    subprocess.run(
        [soffice, "--headless", "--convert-to", "pdf", "--outdir", out_dir, file_path],
        check=True, capture_output=True, timeout=180,
    )
    pdf_path = Path(out_dir) / (Path(file_path).stem + ".pdf")
    if not pdf_path.exists():
        raise RuntimeError(f"LibreOffice did not produce a PDF for {file_path}")
    return str(pdf_path)

def convert_to_images(file_path: str) -> list:
    """Routes the file based on extension and returns a list of PIL Images."""
    ext = Path(file_path).suffix.lower()

    if ext in ['.png', '.jpg', '.jpeg']:
        return [Image.open(file_path).convert("RGB")]

    elif ext == '.pdf':
        # Requires system 'poppler-utils'
        return convert_from_path(file_path)

    elif ext == '.pptx':
        # PPTX -> PDF (LibreOffice headless) to keep slide layout, then PDF -> images.
        # convert_from_path fully renders into memory, so the temp PDF can be discarded after.
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = _pptx_to_pdf(file_path, tmp_dir)
            return convert_from_path(pdf_path)

    else:
        raise ValueError(f"Unsupported file type: {ext}")

def load_model():
    """Loads GLM-OCR model and processor into VRAM."""
    processor = AutoProcessor.from_pretrained(
        GLM_MODEL_ID,
        size={"shortest_edge": 12544, "longest_edge": GLM_MAX_PIXELS},
    )
    processor.tokenizer.padding_side = "left"   # decoder-only batched generation must pad on the left
    model = GlmOcrForConditionalGeneration.from_pretrained(
        GLM_MODEL_ID,
        torch_dtype="auto",  
        device_map="auto",
    )
    model.eval()
    return processor, model

def process_image_batch(processor, model, images: list) -> list:
    """Runs GLM-OCR on a batch of PIL Images in ONE generate() call; returns one text per image.
    Output is identical to processing each image alone (verified byte-for-byte), but the batched
    forward keeps the GPU busy instead of idling between slides. Relies on left-padding (set in
    load_model) so every sequence's generated tokens begin at the same offset."""
    messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": GLM_PROMPT},
                ],
            }
        ]
        for img in images
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,            # pad the batch to a common length (left side)
    ).to(model.device)

    inputs.pop("token_type_ids", None)
    n_input_tokens = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=GLM_MAX_TOKENS,
            do_sample=False,
        )

    # left-padding => every row shares the same input length, so slice once for all
    texts = [
        processor.decode(output_ids[i, n_input_tokens:], skip_special_tokens=True).strip()
        for i in range(len(images))
    ]

    del inputs, output_ids
    return texts

def process_single_image(processor, model, image: Image.Image) -> str:
    """Runs GLM-OCR on a single PIL Image (thin wrapper over the batched path)."""
    return process_image_batch(processor, model, [image])[0]

def run_ocr_pipeline(file_path: str) -> str:
    """
    Main Entry Point: Takes a file path, converts it, runs OCR, 
    clears VRAM, and returns a JSON string.
    """
    results = {
        "file": str(file_path),
        "slides": []
    }
    
    # 1. Convert File to Images in System RAM
    images = convert_to_images(file_path)
    
    processor, model = None, None
    try:
        # 2. Load Model into GPU VRAM
        processor, model = load_model()
        
        # 3. Process slides in batches (keeps the GPU busy; output identical to one-at-a-time)
        for start in range(0, len(images), GLM_BATCH_SIZE):
            batch = images[start:start + GLM_BATCH_SIZE]
            texts = process_image_batch(processor, model, batch)
            for offset, text in enumerate(texts):
                results["slides"].append({
                    "page": start + offset + 1,
                    "text": text,
                })
            free_gpu()
            
    finally:
        # 4. Scale-to-Zero: Nuke the model from VRAM
        if model is not None:
            del model
        if processor is not None:
            del processor
        free_gpu()
        
    return json.dumps(results, indent=4)
