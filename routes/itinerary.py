import logging
from fastapi import APIRouter, HTTPException
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

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/generate", response_model=ItineraryResponse)
async def generate_itinerary(req: TripRequest):
    try:
        logger.info(f"Generating itinerary: {req.destination} | {req.days}d | {req.travel_type} | {req.interests}")

        # Step 1: Perplexity LLM generates raw itinerary
        raw_itinerary = await generate_itinerary_llm(req)
        logger.info(f"LLM returned {len(raw_itinerary)} stops")

        # Step 2: Enrich with coordinates
        enriched = await enrich_itinerary_with_places(raw_itinerary)

        # Step 3: Weather warnings (only if travel_dates provided)
        weather_warnings = []
        if req.travel_dates:
            try:
                forecast = await get_weather_forecast(req.destination, req.days)
                weather_warnings = [f["warning"] for f in forecast if f.get("warning")]
            except Exception as we:
                logger.warning(f"Weather fetch failed (non-critical): {we}")

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
            weather_warnings=weather_warnings or None
        )

    except Exception as e:
        logger.error(f"[/generate] Error: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


@router.post("/enrich", response_model=PlaceEnrichResponse)
async def enrich_place(req: PlaceEnrichRequest):
    try:
        data = await enrich_place_with_perplexity(req.place_name, req.city)
        return PlaceEnrichResponse(place_name=req.place_name, city=req.city, **data)
    except Exception as e:
        logger.error(f"[/enrich] Error: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


@router.post("/save")
async def save_user_itinerary(user_id: str, itinerary: dict):
    try:
        doc_id = await save_itinerary(user_id, itinerary)
        return {"success": True, "itinerary_id": doc_id}
    except Exception as e:
        logger.error(f"[/save] Error: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


@router.get("/user/{user_id}")
async def get_user_saved_itineraries(user_id: str):
    try:
        itineraries = await get_user_itineraries(user_id)
        return {"success": True, "itineraries": itineraries, "count": len(itineraries)}
    except Exception as e:
        logger.error(f"[/user] Error: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


@router.get("/{itinerary_id}")
async def get_saved_itinerary(itinerary_id: str):
    data = await get_itinerary(itinerary_id)
    if not data:
        raise HTTPException(status_code=404, detail="Itinerary not found")
    return {"success": True, "itinerary": data}


@router.post("/rate")
async def rate_place(rating: UserRating):
    try:
        await save_place_rating(
            rating.user_id, rating.place_id,
            rating.rating, rating.review or ""
        )
        return {"success": True, "message": "Rating saved successfully"}
    except Exception as e:
        logger.error(f"[/rate] Error: {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")
