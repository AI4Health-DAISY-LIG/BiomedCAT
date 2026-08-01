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

## For developpers
### mode simple batch                                                                                                                  
Si vous avez déjà des fichiers dans votre dossier data/ (ou Dataset/ selon votre montage Docker), lancez simplement la commande suivante depuis la racine de votre projet :
   ```python         
   python -m biomedcat.pipeline
   ``` 

Le script va scanner le dossier, traiter chaque fichier supporté, et vous afficher les résultats dans la console.                      

### mode batch with a specific file path: 
    ```python                                                                                                                                      python -m biomedcat.pipeline chemin/vers/votre/document pptx
    ```                                                                                                

### Outputs:
 • La création du fichier JSON : Allez vérifier dans votre dossier Output/. Un fichier .json est généré 
 • Data quality : Ouvrez le fichier .json généré.                                                                              
    • entités extraites (entities) correspondent bien au texte du document                                                                 
    • CURIE field corresponding to the normalized ID in any knowledge graph aligned to the biolink model                                                             

Un point de vigilance important (Docker)                                                                                                                     

Comme votre code utilise host.docker.internal pour contacter Ollama, si vous lancez le test directement sur votre machine hôte (hors Docker), assurez_vous   
que l'URL d'Ollama est accessible localement.                                                                                                                

Si vous voulez tester dans les conditions réelles de production, utilisez Docker :                                                                           

                                                                                                                                                             
docker compose run pipeline python -m biomedcat.pipeline                                                                                                     
                                                                                                                                                             

Une fois que vous aurez confirmé que le pipeline produit des résultats JSON corrects et complets, nous pourrons passer à l'étape de création de l'API en     
toute confiance.                                                                                                                                         