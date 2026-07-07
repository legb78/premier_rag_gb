"""API web (FastAPI) pour interroger le RAG depuis un navigateur.

Cette couche HTTP ne contient aucune logique RAG : elle se contente d'appeler
`RAG.answer_question()`. Le RAG est instancié UNE seule fois au démarrage
(chargement de la base et du modèle d'embedding), pas à chaque requête.

Lancement en local :
    uvicorn src.api:app --reload
puis ouvrir http://localhost:8000
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.rag import RAG

# Dossier des fichiers statiques (index.html, style.css…).
STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

app = FastAPI(title="Mon premier RAG")

# Chargé une fois au démarrage : base vectorielle + modèle d'embedding.
rag = RAG()


class Question(BaseModel):
    """Corps attendu par /ask : {"question": "..."}."""

    question: str


@app.get("/health")
def health():
    """Endpoint léger pour vérifier que le service est vivant (ne charge rien)."""
    return {"status": "ok"}


@app.post("/ask")
def ask(q: Question):
    """Interroge le RAG et renvoie sa réponse (ou le message de refus)."""
    return {"answer": rag.answer_question(q.question)}


# Sert la page web : GET / renvoie static/index.html (html=True).
# Monté en dernier pour ne pas masquer les routes /health et /ask.
app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
