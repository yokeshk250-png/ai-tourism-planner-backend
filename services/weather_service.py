import os
import httpx
from typing import List, Dict

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

WARNING_CONDITIONS = ["Rain", "Thunderstorm", "Drizzle", "Snow", "Extreme"]

async def get_weather_forecast(city: str, days: int = 3) -> List[Dict]:
    """
    Fetch weather forecast and flag bad-weather days for trip planning.
    """
    url = "https://api.openweathermap.org/data/2.5/forecast"
    params = {
        "q": city,
        "cnt": days * 8,  # 8 readings per day (every 3hrs)
        "units": "metric",
        "appid": OPENWEATHER_API_KEY
    }

    async with httpx.AsyncClient() as client:
        res = await client.get(url, params=params, timeout=10)
        data = res.json()

    if data.get("cod") != "200":
        return [{"error": "Could not fetch weather"}]

    # Summarize per day
    daily = {}
    for item in data["list"]:
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
        if condition in WARNING_CONDITIONS:
            daily[date]["warning"] = f"⚠️ {condition} expected — carry umbrella or reschedule outdoor visits"

    return list(daily.values())[:days]
