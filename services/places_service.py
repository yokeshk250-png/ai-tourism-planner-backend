import os
import httpx
from typing import List, Dict, Optional
from datetime import datetime, timedelta

FOURSQUARE_API_KEY = os.getenv("FOURSQUARE_API_KEY")
GEOAPAFY_API_KEY = os.getenv("GEOAPIFY_API_KEY")
OPENTRIPMAP_API_KEY = os.getenv("OPENTRIPMAP_API_KEY")

# ─────────────────────────────────────────
# Geocode city name to lat/lon
# ─────────────────────────────────────────
async def geocode_city(city: str) -> Optional[Dict]:
    url = f"https://api.geoapify.com/v1/geocode/search?text={city}&apiKey={GEOAPIFY_API_KEY}"
    async with httpx.AsyncClient() as client:
        res = await client.get(url, timeout=10)
        data = res.json()
        if data.get("features"):
            props = data["features"][0]["properties"]
            return {"lat": props["lat"], "lon": props["lon"]}
    return None

# ─────────────────────────────────────────
# Search places using fallback chain
# ─────────────────────────────────────────
async def search_places(destination: str, category: str, limit: int = 10) -> List[Dict]:
    coords = await geocode_city(destination)
    if not coords:
        return []

    lat, lon = coords["lat"], coords["lon"]

    # 1st: Try Foursquare
    places = await _foursquare_search(lat, lon, category, limit)
    if places:
        return places

    # 2nd: Fallback to Geoapify
    places = await _geoapify_search(lat, lon, category, limit)
    if places:
        return places

    # 3rd: Fallback to OpenTripMap
    return await _opentripmap_search(lat, lon, category, limit)

# ─────────────────────────────────────────
# Foursquare Search
# ─────────────────────────────────────────
async def _foursquare_search(lat: float, lon: float, category: str, limit: int) -> List[Dict]:
    headers = {
        "Authorization": FOURSQUARE_API_KEY,
        "Accept": "application/json"
    }
    params = {
        "ll": f"{lat},{lon}",
        "query": category,
        "limit": limit,
        "radius": 15000,
        "fields": "name,location,categories,hours,rating,photos,tips"
    }
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(
                "https://api.foursquare.com/v3/places/search",
                headers=headers,
                params=params,
                timeout=10
            )
            data = res.json()
            results = data.get("results", [])
            return [
                {
                    "source": "foursquare",
                    "place_id": p.get("fsq_id"),
                    "place_name": p.get("name"),
                    "lat": p["geocodes"]["main"]["latitude"],
                    "lon": p["geocodes"]["main"]["longitude"],
                    "address": p.get("location", {}).get("formatted_address"),
                    "rating": p.get("rating"),
                    "opening_hours": p.get("hours", {}).get("display"),
                    "tags": [c["name"].lower() for c in p.get("categories", [])]
                }
                for p in results
            ]
    except Exception:
        return []

# ─────────────────────────────────────────
# Geoapify Search
# ─────────────────────────────────────────
async def _geoapify_search(lat: float, lon: float, category: str, limit: int) -> List[Dict]:
    params = {
        "categories": "tourism",
        "filter": f"circle:{lon},{lat},15000",
        "limit": limit,
        "apiKey": GEOAPIFY_API_KEY
    }
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(
                "https://api.geoapify.com/v2/places",
                params=params,
                timeout=10
            )
            data = res.json()
            features = data.get("features", [])
            return [
                {
                    "source": "geoapify",
                    "place_id": f.get("properties", {}).get("place_id"),
                    "place_name": f.get("properties", {}).get("name"),
                    "lat": f.get("properties", {}).get("lat"),
                    "lon": f.get("properties", {}).get("lon"),
                    "address": f.get("properties", {}).get("formatted"),
                    "opening_hours": f.get("properties", {}).get("opening_hours"),
                    "tags": f.get("properties", {}).get("categories", [])
                }
                for f in features if f.get("properties", {}).get("name")
            ]
    except Exception:
        return []

# ─────────────────────────────────────────
# OpenTripMap Search
# ─────────────────────────────────────────
async def _opentripmap_search(lat: float, lon: float, category: str, limit: int) -> List[Dict]:
    params = {
        "radius": 15000,
        "lon": lon,
        "lat": lat,
        "kinds": "cultural,religion,natural,historic",
        "limit": limit,
        "apikey": OPENTRIPMAP_API_KEY
    }
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(
                "https://api.opentripmap.com/0.1/en/places/radius",
                params=params,
                timeout=10
            )
            data = res.json()
            return [
                {
                    "source": "opentripmap",
                    "place_id": str(p.get("xid")),
                    "place_name": p.get("name"),
                    "lat": p.get("point", {}).get("lat"),
                    "lon": p.get("point", {}).get("lon"),
                    "tags": p.get("kinds", "").split(",")
                }
                for p in data.get("features", []) if p.get("name")
            ]
    except Exception:
        return []

# ─────────────────────────────────────────
# Get full place details
# ─────────────────────────────────────────
async def get_place_details(place_id: str) -> Optional[Dict]:
    headers = {
        "Authorization": FOURSQUARE_API_KEY,
        "Accept": "application/json"
    }
    try:
        async with httpx.AsyncClient() as client:
            res = await client.get(
                f"https://api.foursquare.com/v3/places/{place_id}",
                headers=headers,
                params={"fields": "name,location,categories,hours,rating,photos,tips,description"},
                timeout=10
            )
            return res.json()
    except Exception:
        return None

# ─────────────────────────────────────────
# Enrich LLM itinerary with real POI data
# ─────────────────────────────────────────
async def enrich_itinerary_with_places(raw_itinerary: list) -> list:
    for stop in raw_itinerary:
        try:
            coords = await geocode_city(stop.get("place_name", ""))
            if coords:
                stop["lat"] = coords["lat"]
                stop["lon"] = coords["lon"]
        except Exception:
            pass
    return raw_itinerary
