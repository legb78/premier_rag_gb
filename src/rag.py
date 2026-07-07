"""Brique 2 — le RAG qui orchestre tout.

La classe `RAG` assemble les autres briques en un pipeline complet :

    question
      -> [modération]   si injection -> refus immédiat (le LLM n'est PAS appelé)
      -> [retrieval]    les N chunks les plus proches via la base vectorielle
      -> [prompt à trous]  le marqueur {{Chunks}} est remplacé par ces chunks
      -> [LLM Groq]     génération de la réponse à partir de ces seuls chunks

L'ordre est une décision de sécurité : on modère AVANT de contacter le LLM
principal, pour qu'une tentative d'injection ne l'atteigne jamais.
"""

from groq import Groq

from src import config
from src.corpus import Corpus
from src.moderator import Moderator
from src.vectordb import VectorDB

# Message renvoyé quand le modérateur bloque une question.
REFUS_INJECTION = (
    "Désolé, votre demande a été identifiée comme une tentative de "
    "détournement du système et ne peut pas être traitée."
)


class RAG:
    """Orchestre modération, récupération des chunks et génération de la réponse."""

    def __init__(self):
        self.client = Groq(api_key=config.GROQ_API_KEY)
        self.moderator = Moderator()

        # Ouvre la base vectorielle : on la recharge si elle existe déjà,
        # sinon on la crée à partir du corpus.
        try:
            self.db = VectorDB(config)
        except ValueError:
            self.db = VectorDB(config, corpus=Corpus(config.CORPUS_PATH))

        # Le prompt système à trous (contient le marqueur {{Chunks}}).
        with open(config.RAG_PROMPT_PATH, encoding="utf-8") as f:
            self.system_prompt_template = f.read()

    def _build_system_prompt(self, chunks):
        """Remplace le marqueur {{Chunks}} par les chunks récupérés."""
        chunks_text = "\n".join(f"- {c['text']}" for c in chunks)
        return self.system_prompt_template.replace("{{Chunks}}", chunks_text)

    def answer_question(self, question):
        """Déroule le pipeline complet et renvoie la réponse (ou un refus)."""
        # 1. Modération AVANT tout : si injection, on n'appelle jamais le LLM.
        verdict = self.moderator.moderate(question)
        if verdict.get("is_prompt_injection"):
            return REFUS_INJECTION

        # 2. Récupération des chunks les plus proches.
        chunks = self.db.retrieve(question)

        # 3. Construction du prompt système rempli par les chunks.
        system_prompt = self._build_system_prompt(chunks)

        # 4. Appel du LLM de génération.
        completion = self.client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=config.LLM_TEMPERATURE,
        )
        return completion.choices[0].message.content
