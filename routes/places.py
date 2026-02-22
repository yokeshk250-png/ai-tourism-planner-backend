from fastapi import APIRouter, Query, HTTPException
from models.schemas import PlaceSearchResponse, PlaceEnrichRequest, PlaceEnrichResponse
from services.places_service import search_places, get_place_details
from services.llm_service import enrich_place_with_perplexity

router = APIRouter()


# ─────────────────────────────────────────────────────────────
# GET /api/places/search
# Search tourist places via Foursquare → Geoapify → OpenTripMap
# ─────────────────────────────────────────────────────────────
@router.get("/search", response_model=PlaceSearchResponse)
async def search_tourist_places(
    destination: str = Query(..., description="City name e.g. Chennai"),
    category: str = Query("tourist_attraction", description="Category: temple, beach, museum, park"),
    limit: int = Query(10, ge=1, le=30)
):
    try:
        places = await search_places(destination, category, limit)
        return PlaceSearchResponse(
            success=True,
            destination=destination,
            category=category,
            count=len(places),
            places=places
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# GET /api/places/{place_id}
# Get full place details from Foursquare
# ─────────────────────────────────────────────────────────────
@router.get("/{place_id}")
async def get_place(place_id: str):
    try:
        place = await get_place_details(place_id)
        if not place:
            raise HTTPException(status_code=404, detail="Place not found")
        return {"success": True, "place": place}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─────────────────────────────────────────────────────────────
# POST /api/places/enrich
# Use Perplexity to get real-time place info
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
