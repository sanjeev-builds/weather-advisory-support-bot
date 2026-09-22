"""
Weather service — fetches live conditions from the Open-Meteo forecast API.

Design decision: This is a thin, deterministic wrapper. No LLM involvement.
The numbers that come back from this function are the ONLY numbers allowed
to appear in a user-facing reply — everything downstream (SOP matching,
reply composition) must read weather values from this dict, never from
the model's own "knowledge" of weather.

Returns a dict with:
  - "current": flat dict of current weather values
  - "hourly": dict of hourly arrays (for time-specific queries)

The "current" dict is the primary data source for SOP matching.
"""

import httpx
from typing import Optional

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_SECONDS = 10

# All fields needed by any SOP. Add new fields here if a new SOP needs them.
CURRENT_FIELDS = ",".join([
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation",
    "rain",
    "weather_code",
    "cloud_cover",
    "wind_speed_10m",
    "wind_gusts_10m",
])

HOURLY_FIELDS = ",".join([
    "temperature_2m",
    "precipitation_probability",
    "precipitation",
    "weather_code",
    "wind_speed_10m",
    "wind_gusts_10m",
    "uv_index",
    "cloud_cover",
])


def fetch_weather(latitude: float, longitude: float) -> Optional[dict]:
    """
    Fetch current weather and today's hourly forecast for given coordinates.

    Returns a dict with:
        "current" - flat dict of current weather values + uv_index and
                    precipitation_probability pulled from hourly data
        "hourly"  - dict of hourly arrays (24 entries)

    Returns None if the API is unreachable or returns malformed data.
    """
    try:
        response = httpx.get(
            FORECAST_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": CURRENT_FIELDS,
                "hourly": HOURLY_FIELDS,
                "timezone": "auto",
                "forecast_days": 1,
            },
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()

        if "current" not in data:
            return None

        current = dict(data["current"])  # copy to avoid mutating API response
        hourly = data.get("hourly", {})

        # uv_index and precipitation_probability are not available in
        # Open-Meteo's `current` block — pull from hourly for current hour
        current_hour_key = current.get("time", "")[:13]  # "YYYY-MM-DDTHH"
        hourly_times = hourly.get("time", [])
        idx = 0  # default to first hour
        for i, t in enumerate(hourly_times):
            if t[:13] == current_hour_key:
                idx = i
                break

        # Merge hourly values into current for easy access
        if "precipitation_probability" in hourly and idx < len(hourly["precipitation_probability"]):
            current["precipitation_probability"] = hourly["precipitation_probability"][idx]
        else:
            current["precipitation_probability"] = 0

        if "uv_index" in hourly and idx < len(hourly["uv_index"]):
            current["uv_index"] = hourly["uv_index"][idx]
        else:
            current["uv_index"] = 0.0

        return {
            "current": current,
            "hourly": hourly,
        }

    except (httpx.HTTPError, httpx.TimeoutException, KeyError, IndexError, TypeError):
        return None
