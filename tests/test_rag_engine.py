import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from biomedcat.rag_engine import BiomedRAG
from biomedcat.config import Settings

@pytest.fixture
def mock_settings(tmp_path):
    """Fournit un objet Settings pointant vers le dossier temporaire de test."""
    s = Settings()
    # On redirige l'output_path vers le dossier temporaire de pytest
    s.output_path = str(tmp_path)
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
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return file_path

@pytest.fixture
def rag_engine(mock_settings, dummy_biolink_data):
    """Initialise le moteur RAG avec les mocks nécessaires."""
    # On mock ChromaDB pour éviter de créer une vraie DB sur le disque
    with patch("chromadb.Client") as mock_client_class:
        mock_instance = mock_client_class.return_value
        mock_collection = MagicMock()
        mock_instance.create_collection.return_value = mock_collection
        
        # On initialise le moteur
        engine = BiomedRAG(mock_settings)
        
        # On injecte le mock de la collection dans l'instance du moteur
        engine.collection = mock_collection
        
        # On construit les index (BM25 sera réel, ChromaDB sera mocké)
        engine.build_indices()
        
        yield engine, mock_collection

def test_search_hybrid_logic(rag_engine):
    """Teste la fusion RRF entre le moteur Dense et Sparse."""
    engine, mock_collection = rag_engine
    
    # 1. Simulation de la recherche DENSE (ChromaDB)
    # On simule que ChromaDB trouve 'Gene'
    mock_collection.query.return_value = {
        "ids": [["Gene"]]
    }

    # 2. Test de la recherche sémantique (Dense)
    # La requête "heredity" doit retourner Gene via le mock Chroma
    results_dense = engine.search("heredity")
    assert "Gene" in results_dense

    # 3. Test de la recherche par mot-clé (Sparse - BM25)
    # On cherche "Cancer", qui est un mot-clé dans l'exemple de 'Disease'
    # Le moteur doit trouver 'Disease' via l'index BM25 réel
    results_sparse = engine.search("Cancer")
    assert "Disease" in results_sparse

def test_get_context_formatting(rag_engine):
    """Vérifie que le contexte généré est correctement formaté pour l'Agent."""
    engine, _ = rag_engine
    
    class_names = ["Gene", "Disease"]
    context = engine.get_context(class_names)
    
    # Vérification de la structure attendue
    assert "Class: Gene" in context
    assert "Definition: A unit of heredity" in context
    assert "Examples: [BRCA1, TP53]" in context
    assert "Class: Disease" in context
    assert "Examples: [Cancer, Diabetes]" in context

def test_search_empty_results(rag_engine):
    """Vérifie le comportement quand aucun résultat n'est trouvé."""
    engine, mock_collection = rag_engine
    
    # On simule une recherche vide dans ChromaDB
    mock_collection.query.return_value = {"ids": [[]]}
    
    results = engine.search("non-existent-term")
    assert isinstance(results, list)
    # Le résultat peut être vide si BM25 ne trouve rien non plus
    assert len(results) == 0
