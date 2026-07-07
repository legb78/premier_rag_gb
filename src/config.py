"""Configuration centrale du RAG.

C'est le SEUL endroit où apparaissent les noms de modèles et les chemins :
pour changer de modèle, on ne modifie que ce fichier.

La configuration est représentée par une dataclass immuable `Config`. On la
construit une seule fois via `Config.load()` (qui charge le `.env` et vérifie
la présence de la clé API), puis on passe l'instance obtenue aux autres classes
du projet (VectorDB, Moderator, RAG).
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Ce fichier vit dans src/ ; la racine du projet est un niveau au-dessus.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Config:
    """Regroupe, de façon immuable, tout ce qui pilote le comportement du RAG."""

    # --- Secret ---
    groq_api_key: str

    # --- Modèles ---
    # Modèle d'embedding : multilingue, léger, suffisant pour ce mini-TP.
    embedding_model: str = "distiluse-base-multilingual-cased-v2"
    # LLM de génération.
    llm_model: str = "llama-3.3-70b-versatile"
    # Modèle de modération (famille « safeguard » de Groq).
    moderation_model: str = "openai/gpt-oss-safeguard-20b"

    # --- Génération ---
    # Proche de zéro : on veut une restitution fidèle du corpus, pas de créativité.
    llm_temperature: float = 0.0
    # Nombre de chunks récupérés et injectés dans le prompt système.
    n_chunks: int = 3

    # --- Chemins & noms ---
    chroma_path: Path = field(default=PROJECT_ROOT / "chroma_db")
    collection_name: str = "rag_corpus"
    prompts_dir: Path = field(default=PROJECT_ROOT / "prompts")

    @property
    def moderator_prompt_path(self) -> Path:
        """Chemin du prompt système du modérateur."""
        return self.prompts_dir / "moderator_system.txt"

    @property
    def rag_prompt_path(self) -> Path:
        """Chemin du prompt système du RAG (contient le marqueur {{Chunks}})."""
        return self.prompts_dir / "rag_system.txt"

    @classmethod
    def load(cls) -> "Config":
        """Charge le `.env`, vérifie la clé API et retourne une instance figée.

        Échoue avec un message explicite si `GROQ_API_KEY` est absente : mieux
        vaut planter tout de suite au démarrage qu'au premier appel réseau.
        """
        load_dotenv(PROJECT_ROOT / ".env")
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            sys.exit(
                "Erreur de configuration : la variable GROQ_API_KEY est absente ou vide.\n"
                f"  1. Copiez .env.example vers .env à la racine du projet ({PROJECT_ROOT})\n"
                "  2. Renseignez-y votre clé API Groq (à créer sur https://console.groq.com/keys)"
            )

        return cls(groq_api_key=api_key)
