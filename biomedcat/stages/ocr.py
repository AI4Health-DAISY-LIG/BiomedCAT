"""Stage 1 (OCR): transcribe slide / PDF / image files to per-slide text with GLM-OCR.

Pages are rasterized and OCR'd one batch at a time, so only a single batch of images is
ever held in memory -- large decks never load fully into RAM.
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

from biomedcat.config import settings
# from biomedcat.runtime import free_gpu   # importing runtime bootstraps the CUDA env before torch
from biomedcat.types import Slide

# Tentative d'importation des dépendances lourdes avec un message d'erreur explicite
try:
    import torch
    from PIL import Image
    from transformers import AutoProcessor, GlmOcrForConditionalGeneration
    from pdf2image import convert_from_path, pdfinfo_from_path
except ImportError as e:
    raise ImportError(
        f"\n\n[ERREUR DÉPENDANCE MANQUANTE] : {e}\n"
        "L'OCR local nécessite des bibliothèques spécifiques pour fonctionner.\n"
        "Veuillez exécuter la commande suivante dans votre terminal pour réparer l'environnement :\n"
        "pip install torch torchvision torchaudio transformers pillow pdf2image\n"
    ) from None

GLM_PROMPT = "Text Recognition:"   # the instruction GLM-OCR was trained to transcribe under
GLM_MAX_TOKENS = 1536              # cap on tokens generated per image
GLM_BATCH_SIZE = 8                 # images per generate() call (batching keeps the small model from idling the GPU)

def free_gpu() -> None:
    """Release Python garbage, then return PyTorch's cached VRAM to the driver if available."""
    gc.collect() #### TO CHECK
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _pptx_to_pdf(file_path: str, out_dir: str) -> str:
    """Convert a .pptx to .pdf via headless LibreOffice and return the PDF path.

    Going through PDF preserves the slide layout the OCR model reads; python-pptx would
    extract only raw text and lose figures, tables, and positioning.
    """
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if soffice is None:
        raise RuntimeError("LibreOffice (soffice) not found; required to convert PPTX to PDF.")

    subprocess.run(
        [soffice, "--headless", "--convert-to", "pdf", "--outdir", out_dir, file_path],
        check=True, capture_output=True, timeout=180,
    )

    pdf_path = Path(out_dir) / (Path(file_path).stem + ".pdf")
    if not pdf_path.exists():
        raise RuntimeError(f"LibreOffice did not produce a PDF for {file_path}")
    return str(pdf_path)


def _to_pdf_or_image(file_path: str, tmp_dir: str) -> tuple[str, object]:
    """Normalise the input to a single page source: ('image', PIL.Image) or ('pdf', path)."""
    ext = Path(file_path).suffix.lower()

    if ext in (".png", ".jpg", ".jpeg"):
        return "image", Image.open(file_path).convert("RGB")
    if ext == ".pdf":
        return "pdf", file_path
    if ext == ".pptx":
        return "pdf", _pptx_to_pdf(file_path, tmp_dir)

    raise ValueError(f"Unsupported file type: {ext}")


def _iter_batches(kind: str, source):
    """Yield lists of up: GLM_BATCH_SIZE PIL images, rasterizing PDF pages on demand.

    Rasterizing page-by-page rather than the whole PDF at once keeps only one batch of
    images in memory, so a 300-page deck costs the same RAM as an 8-page one.
    """
    if kind == "image":
        yield [source]
        return

    n_pages = pdfinfo_from_path(source)["Pages"]
    for start in range(1, n_pages + 1, GLM_BATCH_SIZE):   # PDF pages are 1-indexed
        end = min(start + GLM_BATCH_SIZE - 1, n_pages)
        yield convert_from_path(source, first_page=start, last_page=end)


def load_model():
    """Load the GLM-OCR processor and model into VRAM."""
    processor = AutoProcessor.from_pretrained(settings.ocr_model_id)
    processor.tokenizer.padding_side = "left"   # decoder-only batched generation must pad on the left

    model = GlmOcrForConditionalGeneration.from_pretrained(
        settings.ocr_model_id,
        dtype="auto",
        device_map="auto",
    )
    model.eval()
    return processor, model


def _ocr_message(image: Image.Image) -> list[dict]:
    """Build the one chat message that asks GLM-OCR to transcribe a single image."""
    return [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": GLM_PROMPT},
        ],
    }]


def process_image_batch(processor, model, images: list[Image.Image]) -> list[str]:
    """Transcribe a batch of images in one generate() call; return one text per image.

    Output is identical to processing each image alone (verified byte-for-byte); batching
    just keeps the GPU busy instead of idling between slides.
    """
    messages = []
    for image in images:
        messages.append(_ocr_message(image))

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        padding=True,             # pad the batch to a common length (on the left)
    ).to(model.device)
    inputs.pop("token_type_ids", None)   # GLM's processor emits this; generate() does not accept it

    # Left-padding means every row shares the same input length, so the prompt is sliced
    # off at one shared offset for the whole batch.
    n_input_tokens = inputs["input_ids"].shape[1]

    with torch.inference_mode():         # no autograd graph: faster and lighter on VRAM
        output_ids = model.generate(**inputs, max_new_tokens=GLM_MAX_TOKENS, do_sample=False)

    texts = []
    for i in range(len(images)):
        text = processor.decode(output_ids[i, n_input_tokens:], skip_special_tokens=True).strip()
        texts.append(text)

    del inputs, output_ids               # release the batch's GPU tensors promptly
    return texts


def run_ocr(file_path: str) -> list[Slide]:
    """Transcribe a file to per-slide text.

    Loads GLM-OCR once, streams the pages through it in batches, and always frees the
    model afterwards (scale-to-zero) so a mid-run error cannot leak VRAM and the next
    stage inherits a clean GPU.
    """
    processor, model = None, None
    texts = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        # CPU prep (including any PPTX -> PDF) happens before the model touches the GPU.
        kind, source = _to_pdf_or_image(file_path, tmp_dir)

        try:
            processor, model = load_model()
            for batch in _iter_batches(kind, source):
                texts.extend(process_image_batch(processor, model, batch))
        finally:
            del processor, model   # drop the references, then hand the VRAM back
            free_gpu()

    slides = []
    for page, text in enumerate(texts, start=1):
        slides.append(Slide(page=page, text=text))
    return slides
