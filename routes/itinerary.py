from fastapi import APIRouter, HTTPException, Query
from models.schemas import (
    TripRequest, ItineraryResponse, ItineraryMeta,
    UserRating, PlaceEnrichRequest, PlaceEnrichResponse
)
from services.llm_service import generate_itinerary_llm, enrich_place_with_perplexity
from services.places_service import enrich_itinerary_with_places
from services.weather_service import get_weather_forecast
from services.firebase_service import (
    save_itinerary, get_itinerary,
    save_place_rating, get_user_itineraries
)
import asyncio

router = APIRouter()


# ─────────────────────────────────────────────────────────────
# POST /api/itinerary/generate
# Full AI itinerary generation pipeline:
# 1. Perplexity sonar-pro generates day-wise plan
# 2. Enrich each stop with live POI data
# 3. Check weather warnings for travel dates
# ─────────────────────────────────────────────────────────────
@router.post("/generate", response_model=ItineraryResponse)
async def generate_itinerary(req: TripRequest):
    try:
        # Step 1: Generate raw itinerary via Perplexity sonar-pro
        raw_itinerary = await generate_itinerary_llm(req)

        # Step 2: Enrich stops with coordinates via Geoapify
        enriched = await enrich_itinerary_with_places(raw_itinerary)

        # Step 3: Fetch weather warnings in parallel (if travel date given)
        weather_warnings = []
        if req.travel_dates:
            forecast = await get_weather_forecast(req.destination, req.days)
            weather_warnings = [
                f["warning"] for f in forecast if f.get("warning")
            ]

        # Step 4: Build response metadata
        meta = ItineraryMeta(
            destination=req.destination,
            days=req.days,
            travel_type=req.travel_type,
            budget=req.budget,
            mood=req.mood
        )

        return ItineraryResponse(
            success=True,
            meta=meta,
            itinerary=enriched,
            weather_warnings=weather_warnings if weather_warnings else None
        )

    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Invalid input: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


# ─────────────────────────────────────────────────────────────
# POST /api/itinerary/enrich
# Get real-time details for a specific place via Perplexity
# ─────────────────────────────────────────────────────────────
@router.post("/enrich", response_model=PlaceEnrichResponse)
async def enrich_place(req: PlaceEnrichRequest):
    try:
        data = await enrich_place_with_perplexity(req.place_name, req.city)
        return PlaceEnrichResponse(
            place_name=req.place_name,
            city=req.city,
            **data
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# POST /api/itinerary/save
# Save generated itinerary to Firebase Firestore
# ─────────────────────────────────────────────────────────────
@router.post("/save")
async def save_user_itinerary(user_id: str, itinerary: dict):
    try:
        doc_id = await save_itinerary(user_id, itinerary)
        return {"success": True, "itinerary_id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# GET /api/itinerary/user/{user_id}
# Get all saved itineraries for a user
# ─────────────────────────────────────────────────────────────
@router.get("/user/{user_id}")
async def get_user_saved_itineraries(user_id: str):
    try:
        itineraries = await get_user_itineraries(user_id)
        return {"success": True, "itineraries": itineraries, "count": len(itineraries)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# GET /api/itinerary/{itinerary_id}
# Fetch a saved itinerary by ID
# ─────────────────────────────────────────────────────────────
@router.get("/{itinerary_id}")
async def get_saved_itinerary(itinerary_id: str):
    data = await get_itinerary(itinerary_id)
    if not data:
        raise HTTPException(status_code=404, detail="Itinerary not found")
    return {"success": True, "itinerary": data}


# ─────────────────────────────────────────────────────────────
# POST /api/itinerary/rate
# Submit user rating for a visited place
# ─────────────────────────────────────────────────────────────
@router.post("/rate")
async def rate_place(rating: UserRating):
    try:
        await save_place_rating(
            rating.user_id,
            rating.place_id,
            rating.rating,
            rating.review or ""
        )
        return {"success": True, "message": "Rating saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
