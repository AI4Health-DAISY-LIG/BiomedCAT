#!/bin/bash

# Script de lancement simplifié pour les utilisateurs finaux
# Ce script automatise la vérification et le démarrage de l'infrastructure

set -e

echo "-------------------------------------------------------"
echo "🚀 Bienvenue dans le lanceur BiomedCAT"
echo "-------------------------------------------------------"

# 1. Vérification de Docker
if ! command -v docker &> /dev/app/dev/null; then
    echo "❌ Erreur: Docker n'est pas installé sur votre machine."
    echo "Veuillez installer Docker Desktop : https://www.docker.com/products/docker-desktop"
    exit 1
fi

# 2. Vérification d'Ollama
if ! command -v ollama &> /dev/null; then
    echo "❌ Erreur: Ollama n'est pas installé."
    echo "Veuillez installer Ollama : https://ollama.com/"
    exit 1
fi

# 3. Préparation du modèle (Téléchargement automatique si nécessaire)
echo "🔍 Vérification de la présence du modèle gemma4:12b-it-qat..."
if ! ollama list | grep -q "gemma4:12b-it-qat"; then
    echo "📥 Modèle non trouvé. Téléchargement en cours (cela peut prendre quelques minutes)..."
    ollama pull gemma4:12b-it-qat
    echo "✅ Modèle téléchargé avec succès."
else
    echo "✅ Modèle gemma4:12b-it-qat est prêt."
fi

# 4. Lancement de l'application via Docker Compose
echo "🐳 Démarrage des conteneurs BiomedCAT..."
docker compose up

echo "-------------------------------------------------------"
echo "🎉 Application lancée !"
echo "👉 Accédez à l'interface ici : http://localhost:3000"
echo "-------------------------------------------------------"
