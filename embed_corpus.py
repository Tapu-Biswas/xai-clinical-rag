"""
(Re)build the local ChromaDB vector database from corpus_raw.json.

You normally DON'T need to run this: app.py and precompute_answers.py build
the database automatically if chroma_db/ is missing. Run it yourself when
you've pulled new papers (python pull_papers.py) and want a fresh database.

  python embed_corpus.py

The first run downloads a small (~90MB) embedding model -- that only happens once.
After rebuilding, re-run precompute_answers.py so the ready-made answers match.
"""

from config import DB_PATH, COLLECTION_NAME, build_collection


def main():
    collection = build_collection()
    print(f"\nDone. Vector database saved to {DB_PATH}/")
    print(f"Collection '{COLLECTION_NAME}' now has {collection.count()} entries.")


if __name__ == "__main__":
    main()
