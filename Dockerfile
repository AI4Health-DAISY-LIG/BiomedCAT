# Utilisation d'une image Python légère
FROM python:3.12-slim

# Éviter la création de fichiers .pyc et permettre l'affichage immédiat des logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Installation des dépendances système nécessaires au traitement de documents
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

# Installation de 'uv' pour une gestion ultra-rapide des dépendances Python
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Installation des dépendances Python
# Added langchain-core, spacy and scispacy for NER and Normalization stages
# The scispaCy model is installed directly via URL to avoid 'spacy download' registry errors
RUN uv pip install --system \
    httpx \
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
    https://github.com/allenai/scispacy/releases/download/v0.5.4/en_core_sci_sm-0.5.4.tar.gz

# Copy the application source code
COPY ./biomedcat /app/biomedcat

# Creation of data folders for volume mounting
RUN mkdir -p /app/data/Dataset /app/data/Output

# Default command (will be overridden by docker-compose)
CMD ["python", "-m", "biomedcat.pipeline"]
