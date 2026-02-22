from fastapi import APIRouter, HTTPException
from models.schemas import TripRequest, ItineraryResponse
from services.llm_service import generate_itinerary_llm
from services.places_service import enrich_itinerary_with_places
from services.firebase_service import save_itinerary, get_itinerary

router = APIRouter()

@router.post("/generate", response_model=ItineraryResponse)
async def generate_itinerary(req: TripRequest):
    """
    Generate a day-wise AI itinerary based on user preferences.
    """
    try:
        # Step 1: Generate raw itinerary via LLM
        raw_itinerary = await generate_itinerary_llm(req)

        # Step 2: Enrich with real-time POI data (Foursquare/Geoapify)
        enriched = await enrich_itinerary_with_places(raw_itinerary)

        return ItineraryResponse(success=True, itinerary=enriched)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/save")
async def save_user_itinerary(user_id: str, itinerary: dict):
    """
    Save a generated itinerary to Firebase Firestore.
    """
    doc_id = await save_itinerary(user_id, itinerary)
    return {"success": True, "itinerary_id": doc_id}

@router.get("/{itinerary_id}")
async def get_saved_itinerary(itinerary_id: str):
    """
    Fetch a saved itinerary by ID.
    """
    data = await get_itinerary(itinerary_id)
    if not data:
        raise HTTPException(status_code=404, detail="Itinerary not found")
    return data
