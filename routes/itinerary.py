import asyncio
import logging
import math
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException
from models.schemas import (
    TripRequest, ItineraryResponse, ItineraryMeta,
    ScheduledStop, PlaceCandidate, TimeSlot,
    PlaceEnrichRequest, PlaceEnrichResponse, UserRating
)
from services.llm_service import (
    generate_place_candidates_llm,
    suggest_alternates_llm,
    enrich_place_with_perplexity,
)
from services.places_service import geocode_city, geocode_place
from services.scheduler_service import build_day_slots, schedule_candidates
from services.weather_service import get_weather_forecast
from services.firebase_service import (
    save_itinerary, get_itinerary,
    save_place_rating, get_user_itineraries
)

logger = logging.getLogger(__name__)
router  = APIRouter()

# Enrich only the top N candidates (by priority) to stay within Groq rate limits.
MAX_ENRICH = 15

# Maximum distance (km) a place may be from the destination city centre.
# Candidates that geocode further away are discarded as wrong-city matches.
MAX_PLACE_DISTANCE_KM = 60

# Map enrich's best_time_to_visit string → scheduler slot name
_BEST_TIME_TO_SLOT: dict = {
    "morning":        "morning",
    "early morning":  "morning",
    "morning hours":  "morning",
    "sunrise":        "morning",
    "forenoon":       "morning",
    "afternoon":      "afternoon",
    "midday":         "afternoon",
    "noon":           "afternoon",
    "late morning":   "afternoon",
    "evening":        "evening",
    "sunset":         "evening",
    "dusk":           "evening",
    "late afternoon": "evening",
    "twilight":       "evening",
    "night":          "night",
    "nighttime":      "night",
    "after dark":     "night",
    "after sunset":   "night",
}


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────
def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return R * 2 * math.asin(math.sqrt(a))


def _merge_enrich_data(candidate: dict, enrich_data: dict) -> dict:
    """
    Merge Groq enrich response into a candidate dict.
    Enrich values override the LLM's initial guesses because they come
    from a focused, per-place prompt — more accurate for scheduling.
    """
    c = dict(candidate)

    if enrich_data.get("opening_hours"):
        c["opening_hours"] = enrich_data["opening_hours"]

    if enrich_data.get("closed_on") is not None:
        c["closed_on"] = enrich_data["closed_on"]

    if enrich_data.get("avg_visit_duration_hrs"):
        c["duration_hrs"] = float(enrich_data["avg_visit_duration_hrs"])

    if enrich_data.get("best_time_to_visit"):
        btt  = enrich_data["best_time_to_visit"].lower().strip()
        slot = _BEST_TIME_TO_SLOT.get(btt) or next(
            (v for k, v in _BEST_TIME_TO_SLOT.items() if k in btt), None
        )
        if slot:
            c["best_slot"] = slot

    if enrich_data.get("entry_fee_indian") is not None:
        c["entry_fee"] = enrich_data["entry_fee_indian"]

    if enrich_data.get("entry_fee_foreign") is not None:
        c["entry_fee_foreign"] = enrich_data["entry_fee_foreign"]

    if enrich_data.get("local_tip"):
        c["tip"] = enrich_data["local_tip"]

    if enrich_data.get("nearby_food"):
        c["nearby_food"] = enrich_data["nearby_food"]

    return c


async def _enrich_all_candidates(candidates: list, destination: str) -> list:
    """
    Enrich candidates using per-place Groq calls (concurrent, semaphore-limited).
    Only enriches the top MAX_ENRICH by priority.
    destination is passed into each enrich call to ground Groq to the right city.
    """
    sem = asyncio.Semaphore(5)

    async def _enrich_one(c: dict) -> dict:
        async with sem:
            try:
                data = await enrich_place_with_perplexity(
                    c.get("place_name", ""), destination   # ← always pass destination
                )
                return _merge_enrich_data(c, data) if data else c
            except Exception as e:
                logger.warning(f"[enrich] skip '{c.get('place_name')}': {e}")
                return c

    sorted_cands = sorted(candidates, key=lambda x: -int(x.get("priority", 3)))
    to_enrich    = sorted_cands[:MAX_ENRICH]
    rest         = sorted_cands[MAX_ENRICH:]

    enriched = list(await asyncio.gather(*[_enrich_one(c) for c in to_enrich]))
    logger.info(f"[enrich] {len(enriched)} enriched, {len(rest)} kept as-is")
    return enriched + rest


def _compute_day_dates(travel_dates: str, days: int) -> dict:
    try:
        start = datetime.strptime(travel_dates, "%Y-%m-%d").date()
        return {i: start + timedelta(days=i - 1) for i in range(1, days + 1)}
    except ValueError as e:
        logger.warning(f"[gen] Could not parse travel_dates '{travel_dates}': {e}")
        return {}


async def _geocode_candidates(
    candidates: list,
    destination: str,
    city_lat: float,
    city_lon: float,
) -> list:
    """
    Resolve lat/lon for every candidate via geocode_place() (city-anchored).
    After geocoding, drop any place whose resolved coords are more than
    MAX_PLACE_DISTANCE_KM from the destination city centre — this is the
    hard guard against Canada/Australia/wrong-city results slipping through.

    Places that fail geocoding (no coords returned) are KEPT in the list so
    the scheduler can still place them using the default travel estimate; they
    are just flagged with lat=None/lon=None.
    """
    out      = []
    dropped  = 0

    for raw in candidates:
        c = dict(raw) if isinstance(raw, dict) else raw.model_dump()
        name = c.get("place_name", "?")

        # Always re-geocode using the anchored helper (even if LLM provided coords,
        # those may be wrong-city guesses from the model).
        try:
            coords = await geocode_place(
                place_name=name,
                city=destination,
            )
        except Exception as ge:
            logger.debug(f"[geocode] skip '{name}': {ge}")
            coords = None

        if coords:
            lat, lon = coords["lat"], coords["lon"]
            dist = _haversine_km(city_lat, city_lon, lat, lon)

            if dist > MAX_PLACE_DISTANCE_KM:
                # geocode_place already does a 100 km check internally,
                # but we apply our stricter 60 km policy here.
                logger.warning(
                    f"[geocode] DROPPING '{name}': geocoded {dist:.0f} km from "
                    f"{destination} — likely wrong-city match"
                )
                dropped += 1
                continue   # skip this candidate entirely

            c["lat"] = lat
            c["lon"] = lon
            logger.debug(f"[geocode] '{name}' → ({lat:.5f}, {lon:.5f})  [{dist:.1f} km]")
        else:
            # No coords found: keep with None so scheduler uses default travel time
            c["lat"] = None
            c["lon"] = None
            logger.debug(f"[geocode] '{name}': no result, coords=None")

        out.append(c)

    if dropped:
        logger.warning(
            f"[geocode] {dropped} candidate(s) dropped (>{MAX_PLACE_DISTANCE_KM} km from {destination})"
        )
    return out


# ─────────────────────────────────────────────────────────────
# POST /api/itinerary/generate
#
# 6-stage pipeline:
#   S1  — Groq: top must-visit candidates (no meals)
#   S2  — Geocode: city-anchored lat/lon + distance filter (drops wrong-city results)
#   S2.5— Enrich: opening hrs, real duration, fees, slot (city-anchored)
#   S3  — Build day slots
#   S4  — Schedule (opening-hours enforced)
#   S5  — Conflict resolver (Groq alternates, geocoded + filtered)
#   S6  — Weather warnings
# ─────────────────────────────────────────────────────────────
@router.post("/generate", response_model=ItineraryResponse)
async def generate_itinerary(req: TripRequest):
    try:
        logger.info(
            f"[gen] {req.destination} | {req.days}d | {req.travel_type.value} | "
            f"{req.mood.value} | dates={req.travel_dates}"
        )

        # ─ Resolve city centre coords once (used for distance filter in every geocode step) ─
        city_coords = await geocode_city(req.destination)
        if not city_coords:
            logger.warning(f"[gen] Could not geocode city '{req.destination}' — distance filter disabled")
            city_lat, city_lon = None, None
        else:
            city_lat = city_coords["lat"]
            city_lon = city_coords["lon"]
            logger.info(f"[gen] City centre: ({city_lat:.4f}, {city_lon:.4f})")

        # ─ S1: Groq candidates ─
        raw_candidates = await generate_place_candidates_llm(req)
        logger.info(f"[gen] S1: {len(raw_candidates)} candidates")

        # ─ S2: Geocode — city-anchored + distance filter ─
        if city_lat is not None:
            candidates = await _geocode_candidates(
                raw_candidates, req.destination, city_lat, city_lon
            )
        else:
            # Fallback: old behaviour (no distance filter) if city geocode failed
            candidates = []
            for raw in raw_candidates:
                c = dict(raw) if isinstance(raw, dict) else raw.model_dump()
                candidates.append(c)
        logger.info(f"[gen] S2: {len(candidates)} candidates after geocode+filter")

        # ─ S2.5: Enrich — city-anchored ─
        candidates = await _enrich_all_candidates(candidates, req.destination)
        logger.info(f"[gen] S2.5: enriched {len(candidates)} candidates")

        # ─ S3: Build slots + day dates ─
        slots     = build_day_slots(req.days)
        day_dates = _compute_day_dates(req.travel_dates, req.days) if req.travel_dates else {}
        logger.info(
            f"[gen] S3: {len(slots)} slots | "
            f"day_dates={'set' if day_dates else 'not set (weekday checks skipped)'}"
        )

        # ─ S4: Schedule ─
        scheduled, unscheduled = schedule_candidates(
            candidates, slots,
            avoid_crowded=req.avoid_crowded,
            accessibility_needs=req.accessibility_needs,
            day_dates=day_dates or None,
        )
        logger.info(f"[gen] S4: {len(scheduled)} placed | {len(unscheduled)} unscheduled")

        # ─ S5: Conflict resolution — Groq alternates ─
        if unscheduled:
            logger.info(f"[gen] S5: resolving {len(unscheduled)} conflicts")
            still_unscheduled = []

            for failed in unscheduled:
                slot_name = failed.get("best_slot", "morning")
                alts = await suggest_alternates_llm(
                    destination=req.destination,
                    failed_place=failed,
                    scheduled_places=scheduled,
                    slot_name=slot_name,
                )
                if not alts:
                    still_unscheduled.append(failed)
                    continue

                # geocode + distance-filter + enrich alternates the same way as main candidates
                if city_lat is not None:
                    enriched_alts = await _geocode_candidates(
                        alts, req.destination, city_lat, city_lon
                    )
                else:
                    enriched_alts = [dict(a) if isinstance(a, dict) else a.model_dump() for a in alts]

                enriched_alts = await _enrich_all_candidates(enriched_alts, req.destination)
                for a in enriched_alts:
                    a["is_alternate"] = True

                alt_sched, alt_unsched = schedule_candidates(
                    enriched_alts, slots,
                    avoid_crowded=req.avoid_crowded,
                    accessibility_needs=req.accessibility_needs,
                    day_dates=day_dates or None,
                )
                scheduled.extend(alt_sched)
                if alt_unsched:
                    still_unscheduled.extend(alt_unsched)
                logger.info(
                    f"[gen] S5 alt: {len(alt_sched)} placed for '{failed.get('place_name')}'"
                )

            unscheduled = still_unscheduled

        # ─ S6: Weather warnings ─
        weather_warnings = []
        if req.travel_dates:
            try:
                forecast = await get_weather_forecast(req.destination, req.days)
                weather_warnings = [f["warning"] for f in forecast if f.get("warning")]
            except Exception as we:
                logger.warning(f"Weather skipped: {we}")

        # ─ Build response ─
        slot_template = [
            TimeSlot(
                slot_id=s["slot_id"], day=s["day"],
                slot_name=s["slot_name"],
                start_time=s["start_time"], end_time=s["end_time"],
                available_mins=s["available_mins"],
                remaining_mins=s["remaining_mins"],
                meal_gap_after=s.get("meal_gap_after")
            )
            for s in slots
        ]

        stops_out: list[ScheduledStop] = []
        for s in sorted(scheduled, key=lambda x: (x["day"], x["start_time"])):
            try:
                stops_out.append(ScheduledStop(**s))
            except Exception as ve:
                logger.warning(f"Stop skipped (validation): {ve}")

        unscheduled_out = None
        if unscheduled:
            unscheduled_out = []
            for u in unscheduled:
                try:
                    unscheduled_out.append(
                        PlaceCandidate(**u) if isinstance(u, dict) else u
                    )
                except Exception:
                    pass

        unverified = sum(1 for s in stops_out if s.opening_hours_unverified)
        meta = ItineraryMeta(
            destination=req.destination, days=req.days,
            travel_type=req.travel_type.value, budget=req.budget.value,
            mood=req.mood.value, total_places=len(stops_out),
            unscheduled_count=len(unscheduled),
            hours_unverified_count=unverified,
        )

        logger.info(
            f"[gen] Done: {len(stops_out)} stops | "
            f"{len(unscheduled)} unscheduled | {unverified} hrs-unverified"
        )
        return ItineraryResponse(
            success=True, meta=meta,
            slot_template=slot_template,
            itinerary=stops_out,
            unscheduled=unscheduled_out,
            weather_warnings=weather_warnings or None,
        )

    except Exception as e:
        logger.error(f"[gen] {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


# ─────────────────────────────────────────────────────────────
# POST /api/itinerary/enrich
# ─────────────────────────────────────────────────────────────
@router.post("/enrich", response_model=PlaceEnrichResponse)
async def enrich_place(req: PlaceEnrichRequest):
    try:
        data = await enrich_place_with_perplexity(req.place_name, req.city)
        return PlaceEnrichResponse(place_name=req.place_name, city=req.city, **data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


@router.post("/save")
async def save_user_itinerary(user_id: str, itinerary: dict):
    try:
        doc_id = await save_itinerary(user_id, itinerary)
        return {"success": True, "itinerary_id": doc_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


@router.get("/user/{user_id}")
async def get_user_saved_itineraries(user_id: str):
    try:
        itineraries = await get_user_itineraries(user_id)
        return {"success": True, "itineraries": itineraries, "count": len(itineraries)}
    except Exception as e:
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
        await save_place_rating(rating.user_id, rating.place_id, rating.rating, rating.review or "")
        return {"success": True, "message": "Rating saved"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")
