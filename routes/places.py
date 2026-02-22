from fastapi import APIRouter, Query
from services.places_service import search_places, get_place_details

router = APIRouter()

@router.get("/search")
async def search_tourist_places(
    destination: str = Query(..., description="City name e.g. Chennai"),
    category: str = Query("tourist_attraction", description="Category: temple, beach, museum, park"),
    limit: int = Query(10, ge=1, le=30)
):
    """
    Search tourist places by destination and category.
    Uses Foursquare → Geoapify → OpenTripMap fallback chain.
    """
    places = await search_places(destination, category, limit)
    return {"success": True, "places": places, "count": len(places)}

@router.get("/{place_id}")
async def get_place(place_id: str):
    """
    Get full details of a specific place including hours, photos, and tips.
    """
    place = await get_place_details(place_id)
    return {"success": True, "place": place}
