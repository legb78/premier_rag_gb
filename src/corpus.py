"""Chargement du corpus depuis le CSV.

Le corpus vit dans un fichier CSV (`data/05_corpus_rag.csv`) plutôt que codé en
dur : les données sont séparées du code, et on peut changer de corpus sans
toucher au programme.

Le CSV a quatre colonnes : `id`, `text`, `source`, `categorie`. La classe
`Corpus` les lit et les expose sous une forme directement consommable par
ChromaDB : trois listes parallèles `ids`, `documents`, `metadatas`.
"""

import csv


class Corpus:
    """Le corpus chargé en mémoire, prêt à être indexé par ChromaDB.

    - `ids`        : identifiants uniques (colonne `id` du CSV, ex. "chunk_001").
    - `documents`  : le texte de chaque chunk (colonne `text`).
    - `metadatas`  : un dict {"source": ..., "categorie": ...} par chunk.

    Les trois listes sont alignées : l'élément i de chacune décrit le même chunk.
    """

    def __init__(self, path):
        """Lit le CSV à l'emplacement `path` et remplit les trois listes.

        Le fichier est ouvert en UTF-8 (le corpus contient des accents) ;
        `csv.DictReader` transforme chaque ligne en dict dont les clés sont
        l'en-tête (`id`, `text`, `source`, `categorie`).
        """
        self.ids = []
        self.documents = []
        self.metadatas = []

        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.ids.append(row["id"])
                self.documents.append(row["text"])
                self.metadatas.append(
                    {"source": row["source"], "categorie": row["categorie"]}
                )

        if not self.ids:
            raise ValueError(f"Corpus vide ou illisible : {path}")

    def __len__(self):
        return len(self.ids)
