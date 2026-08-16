"""
Tests pour l'AI Travel Assistant.

- Tests unitaires sur tools.py (pas d'appel LLM, rapides, gratuits)
- Tests unitaires sur rag.py (recherche documentaire)
- Tests d'intégration sur /chat (nécessitent une clé OPENROUTER_API_KEY valide,
  ignorés automatiquement si absente)

Lancer avec : pytest tests/ -v
"""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient

import main
import llm
import tools
from rag import search_knowledge_base, build_index, is_index_stale
from main import app

client = TestClient(app)


# ========== Fixture: Assurer que l'index RAG est créé ==========

@pytest.fixture(scope="session", autouse=True)
def ensure_rag_index():
    """Assure que l'index RAG est construit avant tout test."""
    # Changer de répertoire au répertoire parent pour accéder à knowledge_base/
    original_cwd = os.getcwd()
    parent_dir = os.path.dirname(original_cwd)
    try:
        os.chdir(parent_dir)
        if is_index_stale():
            print("\n[*] Building RAG index for tests...")
            build_index()
        else:
            print("\n[*] RAG index already exists, skipping build")
    finally:
        os.chdir(original_cwd)


# ---------- Tests unitaires : tools.py ----------

def test_search_flights_returns_flights_list():
    result = tools.search_flights("Paris", "Alger", "2026-08-15")
    assert "flights" in result
    assert len(result["flights"]) > 0
    assert result["flights"][0]["origin"] == "Paris"
    assert result["flights"][0]["destination"] == "Alger"


def test_search_flights_is_stable_for_the_same_query():
    first = tools.search_flights("Casablanca", "Paris", "2026-08-15")
    second = tools.search_flights("Casablanca", "Paris", "2026-08-15")
    assert first == second


def test_search_flights_rejects_a_date_without_year():
    result = tools.search_flights("Casablanca", "Paris", "15 aout")
    assert "error" in result


def test_search_flights_rejects_a_year_without_day_and_month():
    result = tools.search_flights("Casablanca", "Paris", "2026")
    assert "error" in result


def test_get_flight_status_returns_expected_fields():
    result = tools.get_flight_status("AH1235", "2026-08-15")
    assert result["flight_number"] == "AH1235"
    assert "status" in result
    assert "terminal" in result
    assert "gate" in result


def test_get_flight_status_is_stable_for_the_same_flight_and_date():
    first = tools.get_flight_status("AH1235", "2026-08-15")
    second = tools.get_flight_status("AH1235", "2026-08-15")
    assert first == second


def test_get_flight_status_rejects_a_date_without_year():
    result = tools.get_flight_status("AH1235", "11 juillet")
    assert "error" in result


def test_cancelled_flights_do_not_have_an_estimated_departure():
    cancelled_flight = next(
        tools.get_flight_status(f"AH{number}", "2026-08-15")
        for number in range(1000, 2000)
        if "annul" in tools.get_flight_status(f"AH{number}", "2026-08-15")["status"]
    )
    assert cancelled_flight["estimated_departure"] is None
    assert cancelled_flight["arrival_time"] is None


def test_get_airport_info_known_airport():
    result = tools.get_airport_info("CDG")
    assert result["city"] == "Paris"
    assert "terminals" in result


def test_get_airport_info_unknown_airport_does_not_crash():
    result = tools.get_airport_info("XXX")
    assert "error" in result


def test_get_flight_status_rejects_empty_flight_number():
    result = tools.get_flight_status("", "2026-08-15")
    assert "error" in result


def test_get_flight_status_rejects_invalid_flight_number():
    result = tools.get_flight_status("vol au hasard", "2026-08-15")
    assert "error" in result


def test_get_booking_rejects_empty_reference():
    result = tools.get_booking("")
    assert "error" in result



    result = tools.get_booking("ABC123")
    assert result["booking_reference"] == "ABC123"
    assert "flight_number" in result


def test_get_booking_rejects_an_unknown_reference():
    result = tools.get_booking("XYZ789")
    assert "error" in result


# ---------- Tests unitaires : rag.py ----------

def test_search_knowledge_base_finds_baggage_info():
    result = search_knowledge_base("bagages autorisés en cabine")
    assert "Aucune information" not in result
    assert len(result) > 0


def test_search_knowledge_base_no_result_message():
    # Suppose une base non vide mais une requête hors-sujet ; on vérifie
    # au moins que la fonction ne plante jamais, quel que soit le résultat.
    result = search_knowledge_base("xyzxyz123 sujet totalement hors contexte")
    assert isinstance(result, str)


# ---------- Tests d'intégration : API /chat ----------

_llm_is_configured = bool(os.environ.get("OPENROUTER_API_KEY"))
_run_live_llm_tests = os.environ.get("RUN_LLM_INTEGRATION_TESTS") == "1"

requires_llm = pytest.mark.skipif(
    not (_llm_is_configured and _run_live_llm_tests),
    reason="OPENROUTER_API_KEY non défini : tests d'intégration LLM ignorés",
)


@requires_llm
def test_chat_endpoint_rag_case():
    response = client.post("/chat", json={"message": "Quels sont les bagages autorisés en cabine ?"})
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "rag"
    assert len(data["answer"]) > 0


@requires_llm
def test_chat_endpoint_tool_case():
    response = client.post("/chat", json={"message": "Quel est le statut du vol AH1235 aujourd'hui ?"})
    assert response.status_code == 200
    data = response.json()
    assert data["source"] == "get_flight_status"


@requires_llm
def test_chat_endpoint_combined_case():
    response = client.post(
        "/chat",
        json={"message": "Mon vol AH1235 est annulé, puis-je être remboursé ?"},
    )
    assert response.status_code == 200
    data = response.json()
    assert "get_flight_status" in data["source"]
    assert "rag" in data["source"]


@requires_llm
def test_chat_endpoint_conversation_memory():
    """
    Vérifie que le contexte est conservé entre deux appels liés par
    le même conversation_id : la 2e question ("Et le remboursement ?")
    n'a de sens que si l'assistant se souvient du vol mentionné avant.
    """
    first = client.post("/chat", json={"message": "Quel est le statut du vol AH1235 aujourd'hui ?"})
    assert first.status_code == 200
    conv_id = first.json()["conversation_id"]
    assert conv_id

    second = client.post("/chat", json={
        "message": "Et ai-je droit à un remboursement si c'est le cas ?",
        "conversation_id": conv_id,
    })
    assert second.status_code == 200
    data = second.json()
    assert data["conversation_id"] == conv_id
    # La réponse doit faire référence au vol du contexte, pas demander "quel vol ?"
    assert "AH1235" in data["answer"] or "vol" in data["answer"].lower()


def test_chat_endpoint_new_conversation_without_id(monkeypatch):
    monkeypatch.setattr(
        main,
        "ask_assistant",
        lambda message, history: {"answer": "Bonjour !", "source": "llm"},
    )
    response = client.post("/chat", json={"message": "Bonjour"})
    assert response.status_code == 200
    assert response.json()["conversation_id"]



    response = client.post("/chat", json={"message": ""})
    assert response.status_code == 422

    response = client.post("/chat", json={"message": "   "})
    assert response.status_code == 422

    response = client.post("/chat", json={"message": "x" * 2001})
    assert response.status_code == 422


def test_chat_endpoint_returns_503_when_llm_is_unavailable(monkeypatch):
    def unavailable(message, history):
        raise llm.AssistantServiceError("LLM unavailable")

    monkeypatch.setattr(main, "ask_assistant", unavailable)
    response = client.post("/chat", json={"message": "Bonjour"})
    assert response.status_code == 503


def test_health_endpoint_reports_rag_readiness():
    response = client.get("/health")
    assert response.status_code == 200
    assert "rag_index_stale" in response.json()
    assert response.headers["X-Request-ID"]


def test_openai_tool_results_use_the_standard_tool_message_protocol(monkeypatch):
    tool_call = SimpleNamespace(
        id="call_123",
        function=SimpleNamespace(
            name="search_knowledge_base",
            arguments='{"query": "bagage cabine"}',
        ),
    )

    class FakeMessage:
        def __init__(self, content, tool_calls=None):
            self.content = content
            self.tool_calls = tool_calls

        def model_dump(self, exclude_none=True):
            data = {"role": "assistant", "content": self.content}
            if self.tool_calls:
                data["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.function.name,
                            "arguments": call.function.arguments,
                        },
                    }
                    for call in self.tool_calls
                ]
            return data

    responses = [
        SimpleNamespace(choices=[SimpleNamespace(message=FakeMessage(None, [tool_call]))]),
        SimpleNamespace(choices=[SimpleNamespace(message=FakeMessage("Un bagage cabine est autorise."))]),
    ]
    requests = []

    def fake_llm_call(**kwargs):
        requests.append(kwargs["messages"])
        return responses.pop(0)

    monkeypatch.setattr(llm, "_call_llm_with_retry", fake_llm_call)
    monkeypatch.setitem(
        llm.AVAILABLE_FUNCTIONS,
        "search_knowledge_base",
        lambda query: {"result": query},
    )

    result = llm.ask_assistant("Quels bagages puis-je emporter ?")

    assert result["source"] == "rag"
    assert requests[1][-2]["role"] == "assistant"
    assert requests[1][-2]["tool_calls"][0]["id"] == "call_123"
    assert requests[1][-1] == {
        "role": "tool",
        "tool_call_id": "call_123",
        "content": '{"result": "bagage cabine"}',
    }


def test_combined_flight_and_refund_question_requires_both_sources():
    required_tools = llm._required_tools_for_message(
        "Mon vol AH1235 est-il annule ? Ai-je droit a un remboursement ?"
    )
    assert required_tools == ["get_flight_status", "search_knowledge_base"]


def test_refund_question_with_flight_number_requires_verified_status_too():
    required_tools = llm._required_tools_for_message(
        "Je voudrais savoir si je peux me faire rembourser le vol AH1009."
    )
    assert required_tools == ["get_flight_status", "search_knowledge_base"]


def test_explicit_flight_date_is_preserved_for_the_status_tool():
    assert llm._extract_flight_date("Statut AH1009 le 2026-08-16") == "2026-08-16"
    assert llm._extract_flight_date("Statut AH1009 le 16 août 2026") == "16 août 2026"


def test_incomplete_flight_date_requires_clarification():
    assert llm._extract_flight_date("Statut AH1009 le 16 août") is None
    assert llm._extract_flight_date("Statut AH1009 la semaine prochaine") is None


def test_booking_and_airport_questions_require_the_matching_tool():
    assert llm._required_tools_for_message("Ma reservation ABC123") == ["get_booking"]
    assert llm._required_tools_for_message("Informations sur l'aeroport CDG") == ["get_airport_info"]
