"""Brique 1 — la base vectorielle persistante (ChromaDB + sentence-transformers).

La classe `VectorDB` a trois comportements, choisis dans le constructeur :

  - si une base existe déjà sur le disque -> elle la RECHARGE (sans réencoder) ;
  - sinon, si on lui fournit un `corpus` -> elle la CRÉE et l'indexe ;
  - sinon -> elle refuse de démarrer avec une erreur explicite.

Ce simple aiguillage évite de réindexer les 200 chunks à chaque lancement.

Détail important : le nom du modèle d'embedding est stocké dans les métadonnées
de la collection elle-même. Au rechargement, on relit cette métadonnée et on
recharge CE modèle-là — jamais celui, peut-être différent, de la config du jour.
Sans cela, on pourrait encoder les documents avec un modèle A puis les questions
avec un modèle B : les vecteurs seraient incomparables et le retrieval
renverrait des chunks aberrants, sans la moindre erreur pour le signaler.
"""

import chromadb
from sentence_transformers import SentenceTransformer


class VectorDB:
    """Encode un corpus, le persiste dans ChromaDB, et retrouve les k plus proches."""

    def __init__(self, config, corpus=None):
        self.config = config

        # Client ChromaDB persistant : les données survivent à l'arrêt du programme.
        self.client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))

        # La collection existe-t-elle déjà sur le disque ?
        existing = [c.name for c in self.client.list_collections()]

        if config.COLLECTION_NAME in existing:
            # --- Cas 1 : RECHARGER une base déjà indexée ---
            self.collection = self.client.get_collection(config.COLLECTION_NAME)
            # On recharge le modèle enregistré à la création, pas celui de la config.
            model_name = self.collection.metadata["embedding_model"]
            self.model = SentenceTransformer(model_name)

        elif corpus is not None:
            # --- Cas 2 : CRÉER la base à partir du corpus fourni ---
            self.model = SentenceTransformer(config.EMBEDDING_MODEL)
            self.collection = self.client.create_collection(
                name=config.COLLECTION_NAME,
                # On grave le nom du modèle dans les métadonnées de la collection.
                metadata={"embedding_model": config.EMBEDDING_MODEL},
            )
            self._index(corpus)

        else:
            # --- Cas 3 : ni base existante, ni corpus -> impossible de démarrer ---
            raise ValueError(
                "Aucune base à l'emplacement "
                f"{config.CHROMA_PATH} et aucun corpus fourni : "
                "impossible de créer ou recharger la base vectorielle."
            )

    def _encode(self, textes):
        """Encode une liste de textes en vecteurs normalisés.

        La normalisation (`normalize_embeddings=True`) rend le produit scalaire
        équivalent à la similarité cosinus — c'est ce dont ChromaDB a besoin ici.
        On passe TOUJOURS par cette méthode, pour documents comme pour questions,
        afin que les deux vivent dans le même espace vectoriel.
        """
        return self.model.encode(
            textes,
            batch_size=32,
            normalize_embeddings=True,
            show_progress_bar=True,
        ).tolist()

    def _index(self, corpus):
        """Encode tous les chunks du corpus et les insère dans la collection."""
        embeddings = self._encode(corpus.documents)
        self.collection.add(
            ids=corpus.ids,
            documents=corpus.documents,
            embeddings=embeddings,
            metadatas=corpus.metadatas,
        )

    def retrieve(self, question, n=None):
        """Retourne les `n` chunks les plus proches de `question`.

        La question est encodée avec le même modèle que les documents. Le résultat
        est une liste de dicts triés du plus au moins pertinent, chacun contenant
        le texte, la métadonnée de source et la distance.
        """
        if n is None:
            n = self.config.N_CHUNKS

        query_embedding = self._encode([question])
        results = self.collection.query(query_embeddings=query_embedding, n_results=n)

        # ChromaDB renvoie des listes de listes (une par requête) : on prend [0].
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        return [
            {"text": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(documents, metadatas, distances)
        ]
