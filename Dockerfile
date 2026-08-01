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
    unzip \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Installation de 'uv' pour une gestion ultra-rapide des dépendances Python
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

WORKDIR /app

# Copie des fichiers de dépendances (si présents) ou installation directe
# Ici, nous installons les dépendances nécessaires au pipeline uniquement
RUN uv pip install --system httpx pydantic-settings pydantic torch torchvision torchaudio transformers pillow pdf2image

# Copie du code source de l'application
COPY ./biomedcat /app/biomedcat

# Création des dossiers de données pour le montage des volumes
RUN mkdir -p /app/data/Dataset /app/data/Output

# Commande par défaut (sera surchargée par docker-compose)
CMD ["python", "-m", "biomedcat.pipeline"]
