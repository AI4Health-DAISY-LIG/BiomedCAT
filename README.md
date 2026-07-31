# BiomedCAT

Biomedical Entity Extraction and Normalization Pipeline. This system performs OCR on presentation files (PPTX, PDF) and uses LLMs to extract biomedical entities and link them to standard ontologies (CURIEs).

## 🚀 Quick Start

The easiest way to run the entire ecosystem (Pipeline + Frontend) is using the provided shell script.

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running.
- [Ollama](https://ollama.com/) installed on your host machine with the required models pulled.

### Running the application
1. Clone the repository.
2. Place your input files (PPTX, PDF, etc.) in the `data/` folder.
3. Run the launcher:
   ```bash
   chmod +x run.sh
   ./run.sh
   ```

Once started:
- **Frontend**: [http://localhost:3000](http://localhost:3000)
- **Pipeline**: Running in the background, processing files from `data/` and saving results to `output/`.

## 🛠 Architecture

- **Pipeline (Python)**: Uses `uv` for dependency management. It performs OCR $\rightarrow$ NER $\rightarrow$ Normalization.
- **Frontend (Next.js)**: A web interface to visualize the extracted entities and links.
- **Docker**: Orchestrates both services using `docker-compose`.

## 📂 Folder Structure
- `data/`: Place your source files here.
- `output/`: JSON results and logs are generated here.
