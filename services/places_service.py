import os
import logging
import httpx
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

FOURSQUARE_API_KEY = os.getenv("FOURSQUARE_API_KEY", "")
GEOAPAFY_API_KEY   = os.getenv("GEOAPIFY_API_KEY", "")
OPENTRIPMAP_API_KEY = os.getenv("OPENTRIPMAP_API_KEY", "")


# ─────────────────────────────────────────
async def geocode_city(city: str) -> Optional[Dict]:
    if not GEOAPIFY_API_KEY:
        logger.warning("GEOAPIFY_API_KEY not set — geocoding skipped")
        return None
    url = f"https://api.geoapify.com/v1/geocode/search?text={city}&apiKey={GEOAPIFY_API_KEY}"
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            res = await client.get(url)
            data = res.json()
            if data.get("features"):
                props = data["features"][0]["properties"]
                return {"lat": props["lat"], "lon": props["lon"]}
    except Exception as e:
        logger.warning(f"Geocoding failed for '{city}': {e}")
    return None


async def search_places(destination: str, category: str, limit: int = 10) -> List[Dict]:
    coords = await geocode_city(destination)
    if not coords:
        return []
    lat, lon = coords["lat"], coords["lon"]

    places = await _foursquare_search(lat, lon, category, limit)
    if places:
        return places

    places = await _geoapify_search(lat, lon, category, limit)
    if places:
        return places

    return await _opentripmap_search(lat, lon, category, limit)


async def _foursquare_search(lat, lon, category, limit) -> List[Dict]:
    if not FOURSQUARE_API_KEY:
        logger.warning("FOURSQUARE_API_KEY not set — skipping Foursquare")
        return []
    headers = {"Authorization": FOURSQUARE_API_KEY, "Accept": "application/json"}
    params  = {
        "ll": f"{lat},{lon}", "query": category, "limit": limit,
        "radius": 15000, "fields": "name,location,categories,hours,rating,photos"
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            res = await client.get("https://api.foursquare.com/v3/places/search",
                                   headers=headers, params=params)
            results = res.json().get("results", [])
            return [{
                "source": "foursquare",
                "place_id": p.get("fsq_id"),
                "place_name": p.get("name"),
                "lat": p.get("geocodes", {}).get("main", {}).get("latitude"),
                "lon": p.get("geocodes", {}).get("main", {}).get("longitude"),
                "address": p.get("location", {}).get("formatted_address"),
                "rating": p.get("rating"),
                "opening_hours": p.get("hours", {}).get("display"),
                "tags": [c["name"].lower() for c in p.get("categories", [])]
            } for p in results]
    except Exception as e:
        logger.warning(f"Foursquare search failed: {e}")
        return []


async def _geoapify_search(lat, lon, category, limit) -> List[Dict]:
    if not GEOAPIFY_API_KEY:
        return []
    params = {
        "categories": "tourism",
        "filter": f"circle:{lon},{lat},15000",
        "limit": limit,
        "apiKey": GEOAPIFY_API_KEY
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            res = await client.get("https://api.geoapify.com/v2/places", params=params)
            features = res.json().get("features", [])
            return [{
                "source": "geoapify",
                "place_id": f.get("properties", {}).get("place_id"),
                "place_name": f.get("properties", {}).get("name"),
                "lat": f.get("properties", {}).get("lat"),
                "lon": f.get("properties", {}).get("lon"),
                "address": f.get("properties", {}).get("formatted"),
                "opening_hours": f.get("properties", {}).get("opening_hours"),
                "tags": f.get("properties", {}).get("categories", [])
            } for f in features if f.get("properties", {}).get("name")]
    except Exception as e:
        logger.warning(f"Geoapify search failed: {e}")
        return []


async def _opentripmap_search(lat, lon, category, limit) -> List[Dict]:
    if not OPENTRIPMAP_API_KEY:
        return []
    params = {
        "radius": 15000, "lon": lon, "lat": lat,
        "kinds": "cultural,religion,natural,historic",
        "limit": limit, "apikey": OPENTRIPMAP_API_KEY
    }
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            res = await client.get("https://api.opentripmap.com/0.1/en/places/radius", params=params)
            data = res.json()
            return [{
                "source": "opentripmap",
                "place_id": str(p.get("xid")),
                "place_name": p.get("name"),
                "lat": p.get("point", {}).get("lat"),
                "lon": p.get("point", {}).get("lon"),
                "tags": p.get("kinds", "").split(",")
            } for p in data.get("features", []) if p.get("name")]
    except Exception as e:
        logger.warning(f"OpenTripMap search failed: {e}")
        return []


async def get_place_details(place_id: str) -> Optional[Dict]:
    if not FOURSQUARE_API_KEY:
        return None
    headers = {"Authorization": FOURSQUARE_API_KEY, "Accept": "application/json"}
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            res = await client.get(
                f"https://api.foursquare.com/v3/places/{place_id}",
                headers=headers,
                params={"fields": "name,location,categories,hours,rating,photos,tips,description"}
            )
            return res.json()
    except Exception as e:
        logger.warning(f"Place details fetch failed: {e}")
        return None


async def enrich_itinerary_with_places(raw_itinerary: list) -> list:
    for stop in raw_itinerary:
        try:
            coords = await geocode_city(stop.get("place_name", ""))
            if coords:
                stop["lat"] = coords["lat"]
                stop["lon"] = coords["lon"]
        except Exception as e:
            logger.warning(f"Geocoding stop '{stop.get('place_name')}' failed: {e}")
    return raw_itinerary
