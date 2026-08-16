"""
tools.py
Fonctions "outils" simulant des APIs réelles (vols, aéroports, réservations).
Dans un vrai projet, ces fonctions appelleraient des APIs externes.
Ici, elles retournent des données mockées pour la démo.
"""

import hashlib
import re

# Format attendu : 2 lettres + 3-4 chiffres (ex: AH1235). Sert de garde-fou
# côté code : si le LLM invente ou copie un exemple au lieu de demander le
# vrai numéro à l'utilisateur, on ne peut pas s'appuyer uniquement sur le
# prompt pour l'en empêcher (un modèle open-source n'obéit pas à 100%).
FLIGHT_NUMBER_PATTERN = re.compile(r"^[A-Z]{2}\d{3,4}$")
BOOKING_REFERENCE_PATTERN = re.compile(r"^[A-Z0-9]{5,8}$")
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
MONTH_PATTERN = re.compile(
    r"\b(?:janvier|fevrier|février|mars|avril|mai|juin|juillet|aout|août|"
    r"septembre|octobre|novembre|decembre|décembre|january|february|march|"
    r"april|may|june|july|august|september|october|november|december)\b",
    re.IGNORECASE,
)
ISO_DATE_PATTERN = re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b")
NUMERIC_DATE_PATTERN = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-](?:19|20)\d{2}\b")
RELATIVE_DATES = {"aujourd'hui", "demain", "today", "tomorrow"}

BOOKINGS_DB = {
    "ABC123": {
        "booking_reference": "ABC123",
        "flight_number": "AH1235",
        "date": "2026-08-19",
        "passengers": ["John Doe"],
        "class": "Economy",
        "origin": "CDG",
        "destination": "ALG",
        "baggage": "1 bagage cabine + 1 bagage soute 23kg",
    },
}


def _has_complete_date(date: str) -> bool:
    """Accept a relative date or a calendar date that includes its year."""
    normalized = (date or "").strip().lower()
    return normalized in RELATIVE_DATES or bool(
        ISO_DATE_PATTERN.search(normalized)
        or NUMERIC_DATE_PATTERN.search(normalized)
        or (YEAR_PATTERN.search(normalized) and MONTH_PATTERN.search(normalized))
    )


def search_flights(origin: str, destination: str, departure_date: str) -> dict:
    """
    Recherche des vols disponibles entre deux villes à une date donnée.
    """
    if not origin or not destination:
        return {"error": "Les villes de départ et d'arrivée sont obligatoires."}
    if not _has_complete_date(departure_date):
        return {
            "error": (
                "Date incomplète. Indique une date avec l'année (ex. 15 août 2026) "
                "ou une date relative comme 'demain'."
            )
        }

    seed = f"{origin.strip().upper()}|{destination.strip().upper()}|{departure_date.strip().lower()}"
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    flights = []
    for i in range(2):
        offset = i * 3
        flights.append({
            "flight_number": f"AH{1000 + (digest[offset] * 4 + digest[offset + 1]) % 9000}",
            "origin": origin,
            "destination": destination,
            "departure_date": departure_date,
            "departure_time": f"{6 + digest[offset + 2] % 17:02d}:00",
            "price_eur": [120, 150, 180, 210][digest[offset + 3] % 4],
        })
    return {"flights": flights}


def get_flight_status(flight_number: str, date: str) -> dict:
    """
    Retourne le statut d'un vol à une date donnée.
    Le statut est déterministe (basé sur le numéro de vol + la date) plutôt
    que purement aléatoire, afin que des questions répétées sur le même vol
    dans une même conversation renvoient toujours la même réponse.
    """
    if not flight_number or not FLIGHT_NUMBER_PATTERN.match(flight_number.strip().upper()):
        return {
            "error": (
                "Numéro de vol manquant ou invalide. Ne pas inventer de numéro : "
                "demande explicitement le numéro de vol exact à l'utilisateur "
                "(format attendu : 2 lettres + 3-4 chiffres, ex. XX1234)."
            )
        }
    if not _has_complete_date(date):
        return {
            "error": (
                "Date incomplète. Indique une date avec l'année (ex. 11 juillet 2026) "
                "ou une date relative comme 'aujourd\'hui'."
            )
        }

    statuses = ["à l'heure", "retardé", "annulé"]
    seed = f"{flight_number}_{date}"
    # hashlib remains stable across Python processes, unlike hash(), which is
    # deliberately salted at each interpreter start.
    status_index = int(hashlib.sha256(seed.encode("utf-8")).hexdigest(), 16) % len(statuses)
    status = statuses[status_index]

    if status == "à l'heure":
        estimated_departure = "10:30"
    elif status == "retardé":
        estimated_departure = "11:15"
    else:  # annulé
        estimated_departure = None

    result = {
        "flight_number": flight_number,
        "date": date,
        "status": status,
        "scheduled_departure": "10:30",
        "estimated_departure": estimated_departure,
        "arrival_time": "13:45" if status != "annulé" else None,
        "terminal": "2A",
        "gate": "B12",
        "cancellation_reason": "non disponible" if status == "annulé" else None,
    }
    return result


def get_airport_info(airport_code: str) -> dict:
    """
    Retourne des informations sur un aéroport à partir de son code IATA.
    """
    airports_db = {
        "CDG": {
            "name": "Aéroport Paris-Charles de Gaulle",
            "city": "Paris",
            "terminals": ["1", "2A", "2B", "2C", "2D", "2E", "2F", "3"],
            "timezone": "Europe/Paris",
            "useful_info": "Navette gratuite entre terminaux, wifi gratuit, plusieurs lounges disponibles.",
        },
        "ALG": {
            "name": "Aéroport Houari Boumediene",
            "city": "Alger",
            "terminals": ["1", "2"],
            "timezone": "Africa/Algiers",
            "useful_info": "Parking longue durée disponible, comptoirs de change ouverts 24h/24.",
        },
    }

    code = airport_code.strip().upper()
    info = airports_db.get(code)
    if info is None:
        return {"error": f"Aucune information disponible pour l'aéroport {code}."}
    return info


def get_booking(booking_reference: str) -> dict:
    """
    Retourne les informations d'une réservation à partir de sa référence.
    """
    if not booking_reference or not BOOKING_REFERENCE_PATTERN.match(booking_reference.strip().upper()):
        return {
            "error": (
                "Référence de réservation manquante ou invalide. Ne pas inventer de "
                "référence : demande explicitement la référence exacte à l'utilisateur."
            )
        }

    reference = booking_reference.strip().upper()
    booking = BOOKINGS_DB.get(reference)
    if booking is None:
        return {"error": f"Réservation {reference} introuvable."}
    return booking.copy()
