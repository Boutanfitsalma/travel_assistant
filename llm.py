"""LLM orchestration for the travel assistant."""

import json
import os
import re
import time
from datetime import date

from openai import OpenAI

import tools
from rag import search_knowledge_base

LLM_REQUEST_TIMEOUT_SECONDS = float(os.environ.get("LLM_REQUEST_TIMEOUT_SECONDS", "30"))
MODEL = os.environ.get("OPENROUTER_MODEL", "openrouter/free")
client = OpenAI(
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    timeout=LLM_REQUEST_TIMEOUT_SECONDS,
    default_headers={"HTTP-Referer": "https://localhost", "X-Title": "Travel Assistant"},
)


SYSTEM_PROMPT = """Tu es un assistant de voyage pour une compagnie aerienne.

Regles :
- Utilise search_knowledge_base pour les regles et procedures generales : bagages,
  remboursement, enregistrement, documents, changement de vol, aeroport et assistance.
- Utilise les outils pour les donnees dynamiques : statut de vol, recherche de vols,
  aeroport et reservation.
- Une question sur un vol annule et un remboursement exige get_flight_status ET
  search_knowledge_base.
- Si la date d'un statut de vol est absente, considere explicitement la date du
  jour fournie dans les donnees verifiees et annonce cette convention. N'ecris
  jamais une autre date.
- Pour rechercher un vol ou consulter un statut a une date calendrier, demande une
  annee explicite. Ne transforme jamais "15 aout" en une annee inventee.
- Ne deduis jamais la cause d'une annulation. Si elle n'est pas fournie par l'outil,
  dis que cette information n'est pas disponible.
- Ne deduis jamais le tarif du passager ni le delai restant avant le depart a partir
  du seul statut de vol. Sans reference de reservation, explique que les conditions
  d'annulation volontaire dependent du tarif et demande cette reference; ne conclus
  jamais que le passager est a moins (ou a plus) de 24 heures du depart.
- Pour une question de remboursement, ne cite que les regles de remboursement
  pertinentes. Ne melange pas les frais de modification de vol avec un remboursement.
  Si le vol est a l'heure ou simplement retarde, ne parle pas de raison d'annulation.
- Si un utilisateur demande le prochain vol disponible sans date, demande une date
  relative (par exemple demain) ou une date complete avant d'appeler search_flights.
- Ne devine jamais un numero de vol, une reference de reservation ou un code aeroport.
- Si une information est indisponible, dis-le clairement. N'invente jamais de reponse.
- Pour les questions hors voyage, explique poliment que tu es specialise dans les voyages.
- Reponds toujours en francais, avec des paragraphes courts et sans syntaxe Markdown.
"""

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_flights",
            "description": "Search available flights between two cities. The date must include a year or be relative, such as tomorrow.",
            "parameters": {
                "type": "object",
                "properties": {
                    "origin": {"type": "string"},
                    "destination": {"type": "string"},
                    "departure_date": {"type": "string"},
                },
                "required": ["origin", "destination", "departure_date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_flight_status",
            "description": "Get live flight status, schedule, terminal and gate. Never invent a flight number, date or cancellation cause.",
            "parameters": {
                "type": "object",
                "properties": {"flight_number": {"type": "string"}, "date": {"type": "string"}},
                "required": ["flight_number", "date"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_airport_info",
            "description": "Get airport name, city, terminals and timezone.",
            "parameters": {
                "type": "object",
                "properties": {"airport_code": {"type": "string"}},
                "required": ["airport_code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_booking",
            "description": "Get a booking from its exact user-provided reference.",
            "parameters": {
                "type": "object",
                "properties": {"booking_reference": {"type": "string"}},
                "required": ["booking_reference"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search travel rules and procedures in the knowledge base.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]

SOURCE_LABELS = {
    "search_knowledge_base": "rag",
    "search_flights": "search_flights",
    "get_flight_status": "get_flight_status",
    "get_airport_info": "get_airport_info",
    "get_booking": "get_booking",
}
AVAILABLE_FUNCTIONS = {
    "search_flights": tools.search_flights,
    "get_flight_status": tools.get_flight_status,
    "get_airport_info": tools.get_airport_info,
    "get_booking": tools.get_booking,
    "search_knowledge_base": search_knowledge_base,
}

MAX_TOOL_ROUNDS = 4
FLIGHT_NUMBER_IN_MESSAGE = re.compile(r"\b[A-Z]{2}\d{3,4}\b", re.IGNORECASE)
BOOKING_REFERENCE_IN_MESSAGE = re.compile(r"\b(?=[A-Z0-9]{5,8}\b)[A-Z0-9]*\d[A-Z0-9]*\b", re.IGNORECASE)
AIRPORT_CODE_IN_MESSAGE = re.compile(r"\b[A-Z]{3}\b")
ISO_DATE_IN_MESSAGE = re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b")
NUMERIC_DATE_IN_MESSAGE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-](?:19|20)\d{2}\b")
MONTH_DATE_IN_MESSAGE = re.compile(
    r"\b\d{1,2}\s+(?:janvier|fevrier|février|mars|avril|mai|juin|juillet|"
    r"aout|août|septembre|octobre|novembre|decembre|décembre)\s+(?:19|20)\d{2}\b",
    re.IGNORECASE,
)
INCOMPLETE_DATE_IN_MESSAGE = re.compile(
    r"\b\d{1,2}\s+(?:janvier|fevrier|février|mars|avril|mai|juin|juillet|"
    r"aout|août|septembre|octobre|novembre|decembre|décembre)\b",
    re.IGNORECASE,
)
REFUND_KEYWORDS = ("remboursement", "rembours", "rembourser")
FLIGHT_STATUS_KEYWORDS = (
    "statut", "annul", "retard", "a l'heure", "à l'heure", "depart",
    "arrive", "terminal", "porte", "embarquement", "horaire", "heure",
)
BOOKING_KEYWORDS = ("reservation", "réservation", "booking")
AIRPORT_KEYWORDS = ("aeroport", "aéroport", "terminal", "fuseau", "timezone")


class AssistantServiceError(RuntimeError):
    """Raised when the configured LLM provider cannot answer."""


def _call_llm_with_retry(max_retries: int = 3, **kwargs):
    last_error = None
    for attempt in range(max_retries):
        try:
            data = {key: value for key, value in kwargs.items() if value is not None}
            return client.chat.completions.create(**data)
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
    raise last_error


def _required_tools_for_message(user_message: str) -> list[str]:
    message = user_message.lower()
    has_flight_number = bool(FLIGHT_NUMBER_IN_MESSAGE.search(user_message))
    required = []
    # A refund request tied to a flight number is a combined case: the policy
    # alone is insufficient because eligibility depends on the actual status.
    if has_flight_number and (
        any(word in message for word in FLIGHT_STATUS_KEYWORDS)
        or any(word in message for word in REFUND_KEYWORDS)
    ):
        required.append("get_flight_status")
    if any(word in message for word in BOOKING_KEYWORDS) and BOOKING_REFERENCE_IN_MESSAGE.search(user_message):
        required.append("get_booking")
    if any(word in message for word in AIRPORT_KEYWORDS) and AIRPORT_CODE_IN_MESSAGE.search(user_message):
        required.append("get_airport_info")
    if any(word in message for word in REFUND_KEYWORDS):
        required.append("search_knowledge_base")
    return required


def _extract_flight_date(user_message: str) -> str | None:
    """Keep an explicit date; return None when the user supplied an incomplete one."""
    for pattern in (ISO_DATE_IN_MESSAGE, NUMERIC_DATE_IN_MESSAGE, MONTH_DATE_IN_MESSAGE):
        match = pattern.search(user_message)
        if match:
            return match.group(0)

    message = user_message.lower()
    if "aujourd'hui" in message or "aujourd’hui" in message:
        return "aujourd'hui"
    if "demain" in message:
        return "demain"
    if INCOMPLETE_DATE_IN_MESSAGE.search(user_message) or "semaine prochaine" in message:
        return None
    # Use an unambiguous value in the verified context. Passing the literal
    # "aujourd'hui" left the model room to turn it into a date from history.
    return date.today().isoformat()


def _get_guardrail_context(user_message: str, required_tools: list[str]) -> list[tuple[str, object]]:
    context = []
    flight_match = FLIGHT_NUMBER_IN_MESSAGE.search(user_message)
    booking_match = BOOKING_REFERENCE_IN_MESSAGE.search(user_message)
    airport_match = AIRPORT_CODE_IN_MESSAGE.search(user_message)
    for tool_name in required_tools:
        if tool_name == "get_flight_status" and flight_match:
            requested_date = _extract_flight_date(user_message)
            if requested_date is None:
                result = {
                    "error": (
                        "Date incomplète. Indique une date avec l'année, par exemple "
                        "11 juillet 2026, pour vérifier le statut du vol."
                    )
                }
            else:
                result = tools.get_flight_status(flight_match.group(0).upper(), requested_date)
        elif tool_name == "get_booking" and booking_match:
            result = tools.get_booking(booking_match.group(0).upper())
        elif tool_name == "get_airport_info" and airport_match:
            result = tools.get_airport_info(airport_match.group(0).upper())
        elif tool_name == "search_knowledge_base":
            result = search_knowledge_base(user_message)
        else:
            continue
        context.append((tool_name, result))
    return context


def _extract_tool_calls(response):
    return response.choices[0].message.tool_calls or []


def _get_message_content(response):
    return response.choices[0].message.content


def _message_with_tool_calls(response) -> dict:
    message = response.choices[0].message
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    return message.dict(exclude_none=True)


def _tool_result_message(call, result: object) -> dict:
    message = {"role": "tool", "content": json.dumps(result, ensure_ascii=False)}
    message["tool_call_id"] = call.id
    return message


def ask_assistant(user_message: str, history: list | None = None) -> dict:
    messages = [{
        "role": "system",
        "content": (
            SYSTEM_PROMPT
            + f"\nDate serveur de reference : {date.today().isoformat()}."
        ),
    }]
    if history:
        messages.extend(history)

    sources_used = []
    required_tools = _required_tools_for_message(user_message)
    guardrail_context = _get_guardrail_context(user_message, required_tools)
    if guardrail_context:
        verified_data = "\n\n".join(
            f"[{tool_name}]\n{json.dumps(result, ensure_ascii=False)}"
            for tool_name, result in guardrail_context
        )
        messages.append({
            "role": "system",
            "content": "Verified travel data. Use these facts and do not contradict them:\n" + verified_data,
        })
        sources_used.extend(SOURCE_LABELS[tool_name] for tool_name, _ in guardrail_context)

    # If the model requests a source that was just obtained by a guardrail,
    # return that exact result again. This prevents a second tool call with an
    # invented date from contradicting the verified data.
    verified_results = dict(guardrail_context)

    messages.append({"role": "user", "content": user_message})

    for round_number in range(MAX_TOOL_ROUNDS):
        try:
            response = _call_llm_with_retry(
                model=MODEL,
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.3,
            )
        except Exception as exc:
            print(f"[LLM ERROR - round {round_number}] {exc}")
            raise AssistantServiceError("Le service LLM est momentanement indisponible.") from exc

        tool_calls = _extract_tool_calls(response)
        if not tool_calls:
            answer = _get_message_content(response)
            return {"answer": answer, "source": " + ".join(dict.fromkeys(sources_used or ["llm"]))}

        messages.append(_message_with_tool_calls(response))
        for call in tool_calls:
            function_name = call.function.name
            try:
                function_args = json.loads(call.function.arguments)
            except (json.JSONDecodeError, AttributeError, TypeError):
                function_args = {}

            function = AVAILABLE_FUNCTIONS.get(function_name)
            if function_name in verified_results:
                result = verified_results[function_name]
            elif function is None:
                result = {"error": f"Unknown tool: {function_name}"}
            else:
                try:
                    result = function(**function_args)
                except Exception as exc:
                    result = {"error": f"Tool execution failed: {exc}"}

            sources_used.append(SOURCE_LABELS.get(function_name, function_name))
            messages.append(_tool_result_message(call, result))

    try:
        final_response = _call_llm_with_retry(model=MODEL, messages=messages, temperature=0.3)
        answer = _get_message_content(final_response)
    except Exception as exc:
        print(f"[LLM ERROR - final] {exc}")
        raise AssistantServiceError("Le service LLM est momentanement indisponible.") from exc

    return {"answer": answer, "source": " + ".join(dict.fromkeys(sources_used))}
