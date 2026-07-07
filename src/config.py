"""Configuration centrale du RAG.

C'est le SEUL endroit où apparaissent les noms de modèles et les chemins :
pour changer de modèle, on ne modifie que ce fichier.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Ce fichier vit dans src/ ; la racine du projet est un niveau au-dessus.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

# --- Modèles ---
# Modèle d'embedding : multilingue, léger, suffisant pour ce mini-TP.
EMBEDDING_MODEL = "distiluse-base-multilingual-cased-v2"
# LLM de génération.
LLM_MODEL = "llama-3.3-70b-versatile"
# Modèle de modération (famille « safeguard » de Groq).
MODERATION_MODEL = "openai/gpt-oss-safeguard-20b"

# --- Génération ---
# Proche de zéro : on veut une restitution fidèle du corpus, pas de créativité.
LLM_TEMPERATURE = 0.0
# Nombre de chunks récupérés et injectés dans le prompt système.
N_CHUNKS = 3

# --- Chemins & noms ---
CHROMA_PATH = PROJECT_ROOT / "chroma_db"
COLLECTION_NAME = "rag_corpus"
CORPUS_PATH = PROJECT_ROOT / "data" / "05_corpus_rag.csv"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
MODERATOR_PROMPT_PATH = PROMPTS_DIR / "moderator_system.txt"
RAG_PROMPT_PATH = PROMPTS_DIR / "rag_system.txt"

# --- Secret ---
try:
    GROQ_API_KEY = os.environ["GROQ_API_KEY"]
except KeyError:
    raise Exception("The groq api key does not exist")
