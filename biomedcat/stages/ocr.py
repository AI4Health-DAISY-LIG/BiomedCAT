"""Stage 1 (OCR): transcribe slide / PDF / image files to per-slide text using Ollama API.

Pages are rasterized and sent to the external OCR service one batch at a time, so only a single batch of images is
ever held in memory -- large decks never load fully into RAM.
"""
import shutil
import subprocess
import tempfile
import gc
import requests
from pathlib import Path
from PIL import Image

from biomedcat.config import settings
from biomedcat.types import Slide

GLM_PROMPT = "Text Recognition:"   # The instruction used for the OCR prompt
GLM_BATCH_SIZE = 4  # Batch size for processing pages/images
OLLAMA_URL = "http://localhost:11434/api/generate" # Default Ollama endpoint

def free_gpu() -> None:
    """Release Python garbage."""
    gc.collect()


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


def _to_pdf_or_image(file_path: str, tmp_dir: str) -> tuple[str, Image.Image | str]:
    """Normalise the input to a single page source: ('image', PIL.Image) or ('pdf', path)."""
    ext = Path(file_path).suffix.lower()

    if ext in (".png", ".jpg", ".jpeg"):
        return "image", Image.open(file_path).convert("RGB")
    if ext == ".pdf":
        try:
            from pdf2image import convert_from_path, pdfinfo_from_path
            # We need to ensure these are imported if they aren't globally available in the scope where this function is called.
            return "pdf", file_path
        except ImportError:
             raise ImportError("Missing required libraries (pdf2image) for PDF handling.")

    if ext == ".pptx":
        return "pdf", _pptx_to_pdf(file_path, tmp_dir)

    raise ValueError(f"Unsupported file type: {ext}")


def _iter_batches(kind: str, source):
    """Yield lists of up to GLM_BATCH_SIZE PIL images, rasterizing PDF pages on demand.

    Rasterizing page-by-page rather than the whole PDF at once keeps only one batch of
    images in memory, so a 300-page deck costs the same RAM as an 8-page one.
    """
    if kind == "image":
        yield [source]
        return

    try:
        from pdf2image import convert_from_path, pdfinfo_from_path
        n_pages = pdfinfo_from_path(source)["Pages"]
    except NameError:
         raise ImportError("pdfinfo_from_path not defined. Ensure dependencies are installed.")

    for start in range(1, n_pages + 1, GLM_BATCH_SIZE):   # PDF pages are 1-indexed
        end = min(start + GLM_BATCH_SIZE - 1, n_pages)
        yield convert_from_path(source, first_page=start, last_page=end)


def load_model() -> str:
    """Returns the model ID used for external OCR calls."""
    return settings.ocr_model_id

def _call_ollama_ocr(image: Image.Image, model_id: str) -> str:
    """Sends an image and prompt to the Ollama service for transcription."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".png") as tmp_img:
            image.save(tmp_img.name)
            files = {'file': (f'{Path(tmp_img.name).name}', 'image/png', open(tmp_img.name, 'rb'))}
            data = {
                "model": model_id,
                "prompt": GLM_PROMPT,
                "stream": False
            }

            response = requests.post(OLLAMA_URL, files=files, data=data)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
    except Exception as e:
        print(f"Error calling Ollama OCR service: {e}")
        return ""


def process_image_batch(model_id: str, images: list[Image.Image]) -> list[str]:
    """Transcribe a batch of images by calling the external Ollama OCR service."""
    texts = []
    for image in images:
        # Call the external API for each image (or implement batched API calls if supported)
        text = _call_ollama_ocr(image, model_id)
        texts.append(text)

    return texts


def run_ocr(file_path: str) -> list[Slide]:
    """Transcribe a file to per-slide text using Ollama OCR."""
    model_id = None # Will be set by load_model()
    texts = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        # CPU prep (including any PPTX -> PDF) happens before the model touches the GPU/API call.
        kind, source = _to_pdf_or_image(file_path, tmp_dir)

        try:
            model_id = load_model() # Get the configured OCR model ID
            for batch in _iter_batches(kind, source):
                # Pass the model_id instead of processor/model objects
                texts.extend(process_image_batch(model_id, batch))
        finally:
            pass

    slides = []
    for page, text in enumerate(texts, start=1):
        slides.append(Slide(page=page, text=text))
    return slides
