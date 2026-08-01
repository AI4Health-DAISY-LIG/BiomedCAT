# Use a lightweight Python image
FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required for document processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libomp-dev \
    python3-dev \
    unzip \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install 'uv' for ultra-fast Python dependency management
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Install Python dependencies
# Added langchain-core, spacy and scispacy for NER and Normalization stages
# The scispaCy model is installed via the official S3 release URL to ensure stability
RUN uv pip install --system \
    httpx \
    pyd  \
    pydantic-settings \
    pydantic \
    torch \
    torchvision \
    torchaudio \
    transformers \
    pillow \
    pdf2image \
    langchain-core \
    spacy \
    scispacy \
    https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

# Copy the application source code
COPY ./biomedcat /app/biomedcat

# Creation of data folders for volume mounting
RUN mkdir -p /app/data/Dataset /app/data/Output

# Default command (will be overridden by docker-compose)
CMD ["python", "-m", "biomedcat.pipeline"]
