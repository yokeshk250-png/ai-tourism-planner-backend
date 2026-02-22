from fastapi import APIRouter, Query, HTTPException
from services.weather_service import get_weather_forecast

router = APIRouter()


# ─────────────────────────────────────────────────────────────
# GET /api/weather/forecast
# Get weather forecast with trip-planning warnings
# ─────────────────────────────────────────────────────────────
@router.get("/forecast")
async def weather_forecast(
    city: str = Query(..., description="City name e.g. Chennai"),
    days: int = Query(3, ge=1, le=7)
):
    try:
        forecast = await get_weather_forecast(city, days)
        warnings = [f["warning"] for f in forecast if f.get("warning")]
        return {
            "success": True,
            "city": city,
            "days": days,
            "forecast": forecast,
            "has_warnings": len(warnings) > 0,
            "warnings": warnings
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
