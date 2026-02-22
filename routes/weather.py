from fastapi import APIRouter, Query
from services.weather_service import get_weather_forecast

router = APIRouter()

@router.get("/forecast")
async def weather_forecast(
    city: str = Query(..., description="City name e.g. Chennai"),
    days: int = Query(3, ge=1, le=7)
):
    """
    Get weather forecast for trip planning.
    Warns about rain or extreme heat on travel days.
    """
    forecast = await get_weather_forecast(city, days)
    return {"success": True, "forecast": forecast}
