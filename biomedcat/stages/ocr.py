"""Stage 1 (OCR): transcribe slide / PDF / image files to per-slide text using Ollama API.

Pages are rasterized and sent to the external OCR service one batch at a time, so only a single batch of images is
ever held in memory -- large decks never load fully into RAM.
"""
import tempfile
import gc
import requests
from pathlib import Path
from PIL import Image
import os
import io
import base64

from biomedcat.config import settings
from biomedcat.types import Slide
from biomedcat.runtime import chat
from biomedcat.prompts import ocr_description

GLM_BATCH_SIZE = 4  # Batch size for processing pages/images
CACHE_OCR_FOLDER_PATH = Path.joinpath(Path(__file__).parent.parent.parent,'data/cache_ocr')
Path(CACHE_OCR_FOLDER_PATH).mkdir(parents=True, exist_ok=True)

def _to_pdf_or_image(file_path: str) -> tuple[str, Image.Image | str]:
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
        # Slide decks must be exported to PDF by the user: it preserves the layout the vision
        # model reads and removes the LibreOffice dependency from the deployment.
        raise ValueError("PPTX input is not supported: export the slides to PDF first.")

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


def _reading_scope() -> tuple[list[str], str]:
    """Entity kinds and focus text for the reading prompt, from the active profile (empty if none)."""
    try:
        from biomedcat.profiles import load_profile

        profile = load_profile(settings.predicate_profile)
        return profile.entity_scope, profile.reading_focus
    except (OSError, ValueError, KeyError):
        return [], ""


def load_model() -> str:
    """Returns the model ID used for external OCR calls."""
    return settings.ocr_model_id

def _call_ollama_ocr(image: Image.Image, model_id: str) -> str:
    """Sends an image and prompt to the Ollama service for transcription."""
    try:

        image_buffer = io.BytesIO()
        image.save(image_buffer, format="PNG")
        image_bytes = image_buffer.getvalue()
        image_base64_string = base64.b64encode(image_bytes).decode('utf-8')
        image_base64_string = image_base64_string.replace("\n", "").replace("\r", "").strip()
        messages = [{"role": "user", "content": ocr_description(*_reading_scope()), "images": [image_base64_string]}]
        # Temperature 0: the reading stage drove the whole run-to-run variance (a different
        # transcription yields different mentions); recall comes from the prompt, not from sampling.
        result = chat(model_id, messages, 8192, temperature=0.0)

        return result
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


def run_ocr(file_path: str, model_id) -> list[Slide]:
    """Transcribe a file to per-slide text using Ollama OCR."""
    texts = []

    kind, source = _to_pdf_or_image(file_path)

    try:
        for batch in _iter_batches(kind, source):
            texts.extend(process_image_batch(model_id, batch))
    finally:
        pass

    slides = []
    for page, text in enumerate(texts, start=1):
        slides.append(Slide(page=page, text=text))
    return slides


if __name__ == "__main__":

    TEST_FILE = "Dataset/chemicals.pdf"

    try:
        print(f"Starting OCR test on {TEST_FILE}...")
        slides = run_ocr(TEST_FILE,settings.ocr_model_id)

        for slide in slides:
            print("-" * 20)
            print(f"Page {slide.page}:")
            print(slide.text[:150] + "..." if len(slide.text) > 150 else slide.text)

    except FileNotFoundError:
        print(f"\nERROR: Test file not found at '{TEST_FILE}'. Please update TEST_FILE to a valid path.")

    except Exception as e:
        print(f"An error occurred during OCR test: {e}")
