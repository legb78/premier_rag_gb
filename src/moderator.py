"""Brique 3 — l'agent modérateur (détection de prompt injection).

Avant que le RAG ne traite une question, le modérateur demande à un modèle de
sécurité dédié (famille « safeguard » de Groq) si la question est une tentative
de détournement (prompt injection). Il renvoie sa décision sous forme de dict.

Pourquoi un modèle dédié plutôt qu'une consigne « refuse les injections » dans
le prompt du RAG ? Parce que la sécurité ne doit pas dépendre de la bonne
volonté du LLM générateur — lequel peut justement être manipulé par l'injection.
C'est une barrière séparée, franchie AVANT tout appel au LLM principal.
"""

import json

from groq import Groq

from src import config


class Moderator:
    """Classe qui détecte les tentatives de prompt injection via un modèle Groq."""

    def __init__(self):
        self.client = Groq(api_key=config.GROQ_API_KEY)
        # Le prompt système vit dans un fichier : on peut le retravailler sans
        # toucher au code.
        with open(config.MODERATOR_PROMPT_PATH, encoding="utf-8") as f:
            self.system_prompt = f.read()

    def moderate(self, question):
        """Analyse `question` et renvoie {"is_prompt_injection": True/False}.

        On force une sortie strictement JSON (`response_format`) et une
        température de 0 : la classification doit être déterministe, pas créative.
        """
        completion = self.client.chat.completions.create(
            model=config.MODERATION_MODEL,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": question},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        # La réponse est une chaîne JSON : on la transforme en dict Python.
        return json.loads(completion.choices[0].message.content)
