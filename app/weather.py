"""City lookup, live weather (Open-Meteo, free, no API key) and spray-safety check."""
import requests

GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

MAX_WIND_KMH = 15
MIN_WIND_KMH = 2
MAX_RAIN_PROB = 70
MAX_TEMP_C = 35


def get_coordinates(city_name: str) -> dict:
    """Find latitude/longitude for a city. 'success' is a real bool now."""
    try:
        r = requests.get(
            GEO_URL,
            params={"name": city_name, "count": 1, "language": "en", "format": "json"},
            timeout=8,
        )
        r.raise_for_status()
        results = r.json().get("results") or []
        if not results:
            return {"success": False, "error": f"City '{city_name}' not found"}
        top = results[0]
        return {
            "success": True,
            "city": top["name"],
            "country": top.get("country", ""),
            "latitude": top["latitude"],
            "longitude": top["longitude"],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def parse_weather(data: dict, lookahead_hours: int = 6) -> dict:
    """Pull wind, temperature and rain probability out of an Open-Meteo reply.

    Rain probability = the highest value over the next `lookahead_hours` hours.
    """
    current = data["current"]
    times = data["hourly"]["time"]
    probs = data["hourly"]["precipitation_probability"]


    now_hour = current["time"][:13] + ":00"
    start = next((i for i, t in enumerate(times) if t >= now_hour), 0)
    window = [p for p in probs[start:start + lookahead_hours] if p is not None]

    return {
        "temp_c": current["temperature_2m"],
        "wind_kmh": current["wind_speed_10m"],
        "rain_prob": max(window) if window else 0,
    }


def get_weather(latitude: float, longitude: float) -> dict:
    """Fetch current weather. Returns {'success': bool, ...}."""
    try:
        r = requests.get(
            FORECAST_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "current": "temperature_2m,wind_speed_10m",
                "hourly": "precipitation_probability",
                "forecast_days": 2,
                "timezone": "auto",
                "wind_speed_unit": "kmh",
            },
            timeout=8,
        )
        r.raise_for_status()
        return {"success": True, **parse_weather(r.json())}
    except Exception as e:
        return {"success": False, "error": str(e)}


def check_spray_safety(wind_speed: float, rain_probability: float, temp: float) -> dict:
    """Decide whether spraying is OK right now."""
    warnings = []

    if wind_speed > MAX_WIND_KMH:
        warnings.append(f"Wind {wind_speed} km/h is too high: spray will drift. Do not spray.")
    if wind_speed < MIN_WIND_KMH:
        warnings.append(f"Wind {wind_speed} km/h is too low: inversion risk, spray may hang in the air.")
    if rain_probability > MAX_RAIN_PROB:
        warnings.append(f"Rain chance {rain_probability}% is high: rain will wash the treatment off.")
    if temp > MAX_TEMP_C:
        warnings.append(f"Temperature {temp} C is too high: chemical may evaporate or burn leaves.")

    is_safe = not warnings
    return {
        "status": "APPROVED" if is_safe else "BLOCKED/DELAYED",
        "is_safe": is_safe,
        "warnings": warnings or ["Optimal spraying conditions."],
    }
