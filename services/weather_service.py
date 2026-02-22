import os
import logging
import httpx
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()  # ← Ensure .env is loaded before os.getenv calls

logger = logging.getLogger(__name__)

WARNING_CONDITIONS = ["Rain", "Thunderstorm", "Drizzle", "Snow", "Extreme"]


async def get_weather_forecast(city: str, days: int = 3) -> List[Dict]:
    """
    Fetch weather forecast from OpenWeatherMap.
    Returns empty list gracefully if OPENWEATHER_API_KEY is not set.
    """
    api_key = os.getenv("OPENWEATHER_API_KEY")

    if not api_key:
        logger.warning("OPENWEATHER_API_KEY not set — weather skipped")
        return []

    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "q": city,
        "cnt": days * 8,
        "units": "metric",
        "appid": api_key
    }

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            res = await client.get(url, params=params)
            data = res.json()
    except Exception as e:
        logger.warning(f"Weather API request failed: {e}")
        return []

    if data.get("cod") != "200":
        logger.warning(f"Weather API error for '{city}': {data.get('message', 'unknown')}")
        return []

    daily = {}
    for item in data.get("list", []):
        date = item["dt_txt"].split(" ")[0]
        if date not in daily:
            daily[date] = {
                "date": date,
                "temp_min": item["main"]["temp_min"],
                "temp_max": item["main"]["temp_max"],
                "conditions": [],
                "warning": None
            }
        condition = item["weather"][0]["main"]
        if condition not in daily[date]["conditions"]:
            daily[date]["conditions"].append(condition)
        if condition in WARNING_CONDITIONS and not daily[date]["warning"]:
            daily[date]["warning"] = (
                f"⚠️ {condition} expected on {date} — "
                "carry umbrella or reschedule outdoor visits"
            )

    result = list(daily.values())[:days]
    logger.info(f"Weather: {len(result)} days fetched for {city}")
    return result
