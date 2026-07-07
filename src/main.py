"""Point d'entrée pour interroger le RAG depuis le terminal.

Usage :
    python -m src.main "Quelle est la couleur du chat de Bob ?"   # une question directe
    python -m src.main                                            # mode interactif
"""

import sys

from src.rag import RAG


def interactive_loop(rag):
    """Boucle interactive : l'utilisateur pose ses questions une par une."""
    print("Pose ta question (tape 'quit' ou 'exit' pour sortir).")
    while True:
        question = input("\n> ").strip()
        if question.lower() in ("quit", "exit", ""):
            break
        print(rag.answer_question(question))


def main():
    rag = RAG()

    # Si une question est passée en argument, on y répond directement et on sort.
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(rag.answer_question(question))
    else:
        interactive_loop(rag)


if __name__ == "__main__":
    main()
