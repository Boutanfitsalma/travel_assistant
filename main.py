"""
main.py
API FastAPI exposant l'endpoint POST /chat de l'assistant de voyage.
"""

import os
import logging
import time
import uuid
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from llm import AssistantServiceError, ask_assistant
from rag import build_index, is_index_stale
from session_store import create_session, get_history, append_turn, cleanup_expired_sessions

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialise l'index documentaire avant de servir des requêtes."""
    if is_index_stale():
        build_index()
    yield


allowed_origins = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://127.0.0.1:8000,http://localhost:8000,null",
).split(",")
environment = os.environ.get("APP_ENV", "development").lower()
cors_origins = ["*"] if environment == "development" else allowed_origins

app = FastAPI(title="AI Travel Assistant", lifespan=lifespan)


@app.middleware("http")
async def request_observability(request, call_next):
    """Add a correlation id and a minimal, privacy-preserving request log."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    started_at = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("Unhandled request error request_id=%s path=%s", request_id, request.url.path)
        raise
    response.headers["X-Request-ID"] = request_id
    logger.info(
        "request_id=%s method=%s path=%s status=%s duration_ms=%.1f",
        request_id,
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - started_at) * 1000,
    )
    return response

# Autorise demo.html (ouvert en local, origine "null" ou différente) à appeler l'API.
# En production, remplacer "*" par le(s) domaine(s) exact(s) du frontend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000, description="Le message de l'utilisateur")
    conversation_id: str | None = Field(
        default=None,
        description="ID de conversation retourné par un précédent appel, pour garder le contexte. "
                     "Omettre pour démarrer une nouvelle conversation.",
    )


class ChatResponse(BaseModel):
    answer: str
    source: str
    conversation_id: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Reçoit un message utilisateur (et optionnellement un conversation_id pour
    garder le contexte des échanges précédents), l'envoie à l'assistant
    (LLM + RAG + tools), et retourne la réponse générée avec la source
    utilisée et l'ID de conversation à réutiliser pour le message suivant.
    """
    message = request.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Le message ne peut pas être vide.")

    conversation_id = request.conversation_id or create_session()
    history = get_history(conversation_id)

    try:
        result = ask_assistant(message, history=history)
    except AssistantServiceError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        logger.exception("Assistant execution failed conversation_id=%s", conversation_id)
        raise HTTPException(status_code=500, detail="Erreur interne du serveur.")

    append_turn(conversation_id, message, result["answer"])
    cleanup_expired_sessions()

    return ChatResponse(
        answer=result["answer"],
        source=result["source"],
        conversation_id=conversation_id,
    )


@app.get("/")
def root():
    return {"status": "AI Travel Assistant API is running"}


@app.get("/health")
def health():
    """Lightweight readiness endpoint for local checks and deployment probes."""
    return {"status": "ok", "rag_index_stale": is_index_stale()}
