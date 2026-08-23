import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from biomedcat.rag_engine import BiomedRAG
from biomedcat.config import Settings
import chromadb

@pytest.fixture
def mock_settings(tmp_path):
    """Fournit un objet Settings pointant vers le dossier temporaire de test."""
    s = Settings()
    s.internal_data_path = str(tmp_path)
    return s

@pytest.fixture
def dummy_biolink_data(tmp_path):
    """Crée un fichier JSON factice pour simuler la Phase 1."""
    data = {
        "Gene": {
            "doc_text": "Class: Gene. Definition: A unit of heredity. Examples: BRCA1, TP53.",
            "metadata": {
                "definition": "A unit of heredity.",
                "examples": ["BRCA1", "TP53"],
                "aliases": ["gen"]
            }
        },
        "Disease": {
            "doc_text": "Class: Disease. Definition: An abnormal condition. Examples: Cancer, Diabetes.",
            "metadata": {
                "definition": "An abnormal condition.",
                "examples": ["Cancer", "Diabetes"],
                "aliases": ["dis"]
            }
        }
    }
    file_path = tmp_path / "biolink_classes_flat.json"
    print(f"Test written at {file_path}")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return file_path

@pytest.fixture
def rag_engine(mock_settings, dummy_biolink_data):                                                                         
    # 1. Instantiate DB                                       
    client = chromadb.EphemeralClient()                                                                                    
    collection_name = "test_collection"                                                                                    
                                                                                                                           
    # 2. Verify if collection exists                                                                      
    try:                                                                                                                   
        # On tente de récupérer la collection existante                                                                    
        collection = client.get_collection(name=collection_name)                                                           
        print(f"[!] Collection '{collection_name}' déjà présente, réutilisation.")                                         
    except Exception:                                                                                                      
        # Si elle n'existe pas, on la crée                                                                                 
        print(  f"[*] Création de la collection '{collection_name}'.")                                                     
        collection = client.create_collection(name=collection_name)                                                        
                                                                                                                           
    with patch("chromadb.Client", return_value=client):                                                                    
        engine = BiomedRAG(mock_settings)                                                                                  
        engine.collection = collection                                                                                     
                                                                                                                                                             
        if collection.count() == 0:                                                                                        
            engine.build_indices()                                                                                         
        else:                                                                                                              
            print("[!] Index déjà chargé, saut de build_indices.")                                                         
                                                                                                                           
        yield engine, collection

# @pytest.fixture
# def rag_engine(mock_settings,dummy_biolink_data):
#     client = chromadb.EphemeralClient() 
#     collection = client.create_collection(name="test_collection")
    
#     with patch("chromadb.Client", return_value=client):
#         engine = BiomedRAG(mock_settings)
#         engine.collection = collection # Ensure it uses our in-memory one
#         engine.build_indices() # file read in

#         yield engine, collection

def test_search_hybrid_logic(rag_engine):
    """Test documents ranking."""
    engine, mock_collection = rag_engine
    
    # 1. Test de la recherche sémantique (Dense)
    # La requête "heredity" doit retourner Gene via le mock Chroma
    results = engine.search("heredity")
    assert "Gene" == results[0]

    # 2. Search example (as key-word) (Sparse - BM25)
    results = engine.search("Cancer")
    assert "Disease" == results[0]

    # 3. General search:
    results = engine.search("Alzheimer")
    assert "Disease" == results[0] 

def test_get_context_formatting(rag_engine):
    """Verify generated context is correctly formatted for NER agent."""
    engine, _ = rag_engine
    
    class_names = ["Gene", "Disease"]
    context = engine.get_context(class_names)
    
    # Vérification de la structure attendue
    assert "Class: Gene" in context
    assert "Definition: A unit of heredity" in context
    assert "Examples: [BRCA1, TP53]" in context
    assert "Class: Disease" in context
    assert "Examples: [Cancer, Diabetes]" in context

# def test_search_empty_results(rag_engine):
#     """Verify that no entity is retrieved."""
#     engine, mock_collection = rag_engine
    
#     results = engine.search("non-existent-term")
#     assert isinstance(results, list)

#     assert len(results) == 0

if __name__ == "__main__":
    pytest.main([__file__])