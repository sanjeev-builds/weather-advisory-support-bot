"""
Geocoding service — resolves city names to (latitude, longitude) using
the Open-Meteo Geocoding API.

Design decision: This is a thin, deterministic wrapper around the API.
No LLM involvement. If the API returns nothing or errors, we return None
and let the graph handle it (honest failure, not a guess).
"""

import httpx
from typing import Optional

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
TIMEOUT_SECONDS = 10


def geocode_city(city_name: str) -> Optional[dict]:
    """
    Resolve a city name to geographic coordinates.

    Returns dict with keys: name, latitude, longitude, country, admin1 (state/region)
    or None if the city cannot be resolved.

    We take the first result — Open-Meteo returns results ranked by population,
    so the first hit is the most prominent city with that name. This is a
    reasonable default (Bhopal → Bhopal, India; not some tiny village).
    """
    try:
        response = httpx.get(
            GEOCODING_URL,
            params={"name": city_name, "count": 5, "language": "en", "format": "json"},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()

        results = data.get("results")
        if not results:
            return None

        top = results[0]
        return {
            "name": top.get("name", city_name),
            "latitude": top["latitude"],
            "longitude": top["longitude"],
            "country": top.get("country", "Unknown"),
            "admin1": top.get("admin1", ""),  # state / region
        }

    except (httpx.HTTPError, httpx.TimeoutException, KeyError, IndexError):
        return None
