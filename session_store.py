"""
session_store.py
Gestion de l'historique de conversation côté serveur, par conversation_id.

Choix : stockage en mémoire (dict), thread-safe via un verrou.
Limite assumée : l'historique est perdu si le serveur redémarre. En
production, on utiliserait un store externe (Redis avec TTL) pour
supporter plusieurs instances et la persistance. Pour ce challenge/démo,
un stockage en mémoire est suffisant et évite une dépendance supplémentaire.
"""

import threading
import time
import uuid

MAX_TURNS_KEPT = 6  # nombre d'échanges (user+assistant) conservés par session
SESSION_TTL_SECONDS = 60 * 30  # 30 min d'inactivité avant expiration

_lock = threading.Lock()
_sessions: dict[str, dict] = {}
# _sessions[conversation_id] = {"messages": [...], "last_seen": timestamp}


def create_session() -> str:
    """Crée une nouvelle session et retourne son identifiant."""
    conversation_id = str(uuid.uuid4())
    with _lock:
        _sessions[conversation_id] = {"messages": [], "last_seen": time.time()}
    return conversation_id


def get_history(conversation_id: str) -> list[dict]:
    """
    Retourne l'historique (liste de {"role", "content"}) d'une session.
    Retourne une liste vide si l'ID est inconnu ou expiré (dégradation
    silencieuse : mieux vaut repartir d'une conversation vide que planter).
    """
    with _lock:
        session = _sessions.get(conversation_id)
        if session is None:
            return []
        if time.time() - session["last_seen"] > SESSION_TTL_SECONDS:
            del _sessions[conversation_id]
            return []
        return list(session["messages"])


def append_turn(conversation_id: str, user_message: str, assistant_answer: str) -> None:
    """
    Ajoute un échange (question + réponse) à l'historique de la session,
    et tronque pour ne garder que les MAX_TURNS_KEPT derniers échanges.
    """
    with _lock:
        session = _sessions.setdefault(
            conversation_id, {"messages": [], "last_seen": time.time()}
        )
        session["messages"].append({"role": "user", "content": user_message})
        session["messages"].append({"role": "assistant", "content": assistant_answer})
        session["messages"] = session["messages"][-(MAX_TURNS_KEPT * 2):]
        session["last_seen"] = time.time()


def cleanup_expired_sessions() -> None:
    """Supprime les sessions inactives depuis plus de SESSION_TTL_SECONDS."""
    now = time.time()
    with _lock:
        expired = [
            cid for cid, s in _sessions.items()
            if now - s["last_seen"] > SESSION_TTL_SECONDS
        ]
        for cid in expired:
            del _sessions[cid]
