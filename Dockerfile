# BiomedCAT local console (biomedcat.webapp) for end users.
# The language and vision models are served by Ollama in a separate container (see docker-compose.yml).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

# poppler-utils: pdfinfo/pdftoppm used to rasterise the slide decks
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies first, so that code changes do not reinstall them
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project

COPY biomedcat ./biomedcat
COPY scripts ./scripts
RUN uv sync --frozen --no-dev

# Published configuration of the article (dense + exemplar retrieval legs, ReAct agent, union gate)
ENV AGENT_MODE=react \
    SCISPACY=0 \
    RAG_BM25=0 \
    RAG_LEXICAL_WEIGHT=0 \
    RAG_KG2C_LEXICAL=0 \
    RAG_KG2C_LEXICAL_WEIGHT=0 \
    OLLAMA_KEEP_ALIVE=0 \
    OLLAMA_URL=http://ollama:11434 \
    BIOMEDCAT_HOST=0.0.0.0 \
    BIOMEDCAT_PORT=8765 \
    BIOMEDCAT_NO_BROWSER=1

EXPOSE 8765

CMD ["uv", "run", "--no-sync", "python", "-m", "biomedcat.webapp"]
