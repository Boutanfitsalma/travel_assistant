"""
rag.py
Pipeline RAG : indexation des documents de la base de connaissances
et recherche par similarité pour répondre aux questions générales.
"""

import os
import logging
import hashlib
import json
from pathlib import Path

# Disable optional Chroma telemetry in this local demo to avoid noisy logs.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

logging.getLogger("chromadb.telemetry.product.posthog").disabled = True

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "travel_knowledge_base"
INDEX_MANIFEST_PATH = CHROMA_DIR / "index_manifest.json"

# Fonction d'embedding par défaut de Chroma (modèle local, pas besoin de clé API)
embedding_fn = embedding_functions.DefaultEmbeddingFunction()

_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR),
    settings=Settings(anonymized_telemetry=False),
)


def _chunk_text(text: str, max_chunk_size: int = 500) -> list[str]:
    """
    Découpe un texte par paragraphe (séparés par une ligne vide), afin de
    préserver le sens complet de chaque règle/information. Un chunk fixe en
    caractères risquerait de couper une phrase (ex: séparer une condition
    de son montant en euros).

    Si un paragraphe dépasse max_chunk_size, on le garde quand même entier
    (préférer un chunk un peu long à une info tronquée).
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return [text.strip()]

    # Le premier paragraphe est souvent un titre (ex: "POLITIQUE DE MODIFICATION DE VOL").
    # On le préfixe à chaque chunk pour garder le contexte du document lors de la recherche.
    title = paragraphs[0] if paragraphs[0].isupper() else None
    body = paragraphs[1:] if title else paragraphs

    chunks = [f"{title} — {p}" if title else p for p in body]
    return chunks if chunks else [text.strip()]


def is_index_empty() -> bool:
    """
    Vérifie si l'index ChromaDB est vide (jamais construit).
    Utilisé au démarrage de l'API pour construire l'index automatiquement
    si besoin, sans obliger l'utilisateur à lancer `python rag.py` manuellement.
    """
    collection = _client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=embedding_fn
    )
    return len(collection.get()["ids"]) == 0


def _knowledge_base_fingerprint() -> str:
    """Return a stable signature for the documents currently indexed."""
    digest = hashlib.sha256()
    for filepath in sorted(KNOWLEDGE_BASE_DIR.glob("*.txt")):
        digest.update(filepath.name.encode("utf-8"))
        digest.update(filepath.read_bytes())
    return digest.hexdigest()


def is_index_stale() -> bool:
    """Detect a changed knowledge base and trigger a safe rebuild at startup."""
    if is_index_empty() or not INDEX_MANIFEST_PATH.exists():
        return True
    try:
        manifest = json.loads(INDEX_MANIFEST_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return True
    return manifest.get("fingerprint") != _knowledge_base_fingerprint()


def build_index() -> None:
    """
    Charge tous les fichiers .txt de knowledge_base/, les découpe en chunks,
    et les indexe dans ChromaDB. À exécuter une fois (ou si les documents changent).
    """
    collection = _client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=embedding_fn
    )

    # On repart de zéro pour éviter les doublons si le script est relancé
    existing_ids = collection.get()["ids"]
    if existing_ids:
        collection.delete(ids=existing_ids)

    documents, ids, metadatas = [], [], []
    idx = 0

    for filename in os.listdir(KNOWLEDGE_BASE_DIR):
        if not filename.endswith(".txt"):
            continue
        filepath = KNOWLEDGE_BASE_DIR / filename
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()

        topic = filename.replace(".txt", "")
        for chunk in _chunk_text(text):
            documents.append(chunk)
            ids.append(f"{topic}_{idx}")
            metadatas.append({"source": topic})
            idx += 1

    collection.add(documents=documents, ids=ids, metadatas=metadatas)
    INDEX_MANIFEST_PATH.write_text(
        json.dumps({"fingerprint": _knowledge_base_fingerprint(), "chunks": len(documents)}),
        encoding="utf-8",
    )
    print(f"Index créé : {len(documents)} chunks indexés.")


def search_knowledge_base(query: str, n_results: int = 3) -> str:
    """
    Recherche les passages les plus pertinents dans la base de connaissances
    pour une question donnée. Retourne un texte concaténé prêt à être injecté
    dans le prompt du LLM.
    """
    collection = _client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=embedding_fn
    )

    results = collection.query(query_texts=[query], n_results=n_results)

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    if not documents:
        return "Aucune information trouvée dans la base de connaissances."

    formatted = []
    for doc, meta in zip(documents, metadatas):
        source = meta.get("source", "inconnu")
        formatted.append(f"[Source: {source}]\n{doc.strip()}")

    return "\n\n".join(formatted)


if __name__ == "__main__":
    # Permet de construire l'index manuellement : python rag.py
    build_index()
