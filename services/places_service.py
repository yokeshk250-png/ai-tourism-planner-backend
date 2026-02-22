import os
import logging
import httpx
from typing import List, Dict, Optional
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

FOURSQUARE_API_KEY  = os.getenv("FOURSQUARE_API_KEY", "")
GEOAPIFY_API_KEY    = os.getenv("GEOAPIFY_API_KEY", "")
OPENTRIPMAP_API_KEY = os.getenv("OPENTRIPMAP_API_KEY", "")

# Geoapify result_type values that represent a city/admin area, not a named place.
# If Geoapify returns one of these types it means it couldn't find the specific
# place and fell back to the city centroid — we discard such results to prevent
# 8+ places clustering at the same city-centre coordinates.
_GEOAPIFY_SKIP_TYPES = frozenset({
    "city", "county", "state", "country", "region",
    "district", "postcode", "unknown",
})


# ─────────────────────────────────────────
async def geocode_city(city: str) -> Optional[Dict]:
    """Geocode a city/destination name (no place-specific context)."""
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


async def geocode_place(
    place_name: str,
    city: str,
    state: str = "Tamil Nadu",
    country: str = "India",
) -> Optional[Dict]:
    """
    Geocode a specific place name anchored to its city/state/country.

    Reject results whose Geoapify result_type is a city/admin area — this
    prevents the well-known clustering bug where Geoapify returns the city
    centroid (e.g. 9.9261, 78.1141 for Madurai) when it cannot find the
    specific place, causing 8+ places to share the same coordinates.

    Query order:
      1. "<place>, <city>, <state>, <country>"
      2. "<place>, <state>, <country>"           (broader fallback)
    """
    if not GEOAPIFY_API_KEY:
        logger.warning("GEOAPIFY_API_KEY not set — geocoding skipped")
        return None

    queries = [
        f"{place_name}, {city}, {state}, {country}",
        f"{place_name}, {state}, {country}",
    ]

    async with httpx.AsyncClient(timeout=10) as client:
        for query in queries:
            try:
                url = (
                    f"https://api.geoapify.com/v1/geocode/search"
                    f"?text={query}&apiKey={GEOAPIFY_API_KEY}"
                )
                res  = await client.get(url)
                data = res.json()
                if not data.get("features"):
                    continue

                props       = data["features"][0]["properties"]
                result_lat  = props["lat"]
                result_lon  = props["lon"]
                result_type = props.get("result_type", "").lower()

                # ── Reject city/admin-area fallbacks ──────────────────────────────
                if result_type in _GEOAPIFY_SKIP_TYPES:
                    logger.warning(
                        f"[geocode] '{place_name}': Geoapify returned result_type='{result_type}' "
                        f"(city/admin fallback) — discarding [{query}]"
                    )
                    continue

                # ── Distance sanity-check: result must be within ~100 km of city ──
                city_coords = await geocode_city(city)
                if city_coords:
                    import math
                    dlat = math.radians(result_lat - city_coords["lat"])
                    dlon = math.radians(result_lon - city_coords["lon"])
                    a = (
                        math.sin(dlat / 2) ** 2
                        + math.cos(math.radians(city_coords["lat"]))
                        * math.cos(math.radians(result_lat))
                        * math.sin(dlon / 2) ** 2
                    )
                    dist_km = 6371 * 2 * math.asin(math.sqrt(a))
                    if dist_km > 100:
                        logger.warning(
                            f"[geocode] '{place_name}': result ({result_lat:.4f},{result_lon:.4f}) "
                            f"is {dist_km:.0f} km from {city} — discarding"
                        )
                        continue

                logger.debug(
                    f"[geocode] '{place_name}' → ({result_lat:.5f}, {result_lon:.5f})"
                    f"  type='{result_type}'  [query='{query}']"
                )
                return {"lat": result_lat, "lon": result_lon}

            except Exception as e:
                logger.warning(f"Geocoding failed for '{place_name}' query='{query}': {e}")

    logger.warning(f"[geocode] '{place_name}': no valid result within 100 km of '{city}'")
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
    params = {
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
        logger.warning("GEOAPIFY_API_KEY not set — skipping Geoapify")
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
        logger.warning("OPENTRIPMAP_API_KEY not set — skipping OpenTripMap")
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


async def enrich_itinerary_with_places(
    raw_itinerary: list,
    destination: str = "",
    state: str = "Tamil Nadu",
) -> list:
    """
    Geocode every stop in the itinerary, anchoring each place name to the
    trip destination so results can't drift to wrong cities/countries.
    """
    for stop in raw_itinerary:
        place_name = stop.get("place_name", "")
        try:
            coords = await geocode_place(
                place_name,
                city=destination or place_name,
                state=state,
            )
            if coords:
                stop["lat"] = coords["lat"]
                stop["lon"] = coords["lon"]
            else:
                logger.warning(
                    f"[enrich] '{place_name}': geocode returned None — coords unchanged"
                )
        except Exception as e:
            logger.warning(f"Geocoding stop '{place_name}' failed: {e}")
    return raw_itinerary
