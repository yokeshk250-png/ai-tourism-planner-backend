import logging
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
from services.places_service import geocode_city
from services.scheduler_service import build_day_slots, schedule_candidates
from services.weather_service import get_weather_forecast
from services.firebase_service import (
    save_itinerary, get_itinerary,
    save_place_rating, get_user_itineraries
)

logger = logging.getLogger(__name__)
router = APIRouter()


def _compute_day_dates(travel_dates: str, days: int) -> dict:
    """
    Parse 'YYYY-MM-DD' travel start date and return
    {1: date, 2: date+1day, ...} so the scheduler can check
    weekday-sensitive opening hours (e.g. 'Closed on Tuesday').
    Returns empty dict on parse failure.
    """
    try:
        start = datetime.strptime(travel_dates, "%Y-%m-%d").date()
        return {i: start + timedelta(days=i - 1) for i in range(1, days + 1)}
    except ValueError as e:
        logger.warning(f"[gen] Could not parse travel_dates '{travel_dates}': {e}")
        return {}


async def _geocode_candidates(candidates: list, destination: str) -> list:
    """
    Resolve lat/lon for candidates that are missing coordinates.
    Failures are silently skipped; the scheduler handles missing coords
    with a default travel-time estimate.
    """
    enriched = []
    for raw in candidates:
        c = dict(raw) if isinstance(raw, dict) else raw.model_dump()
        if not c.get("lat") or not c.get("lon"):
            try:
                coords = await geocode_city(f"{c.get('place_name', '')}, {destination}")
                if coords:
                    c["lat"] = coords["lat"]
                    c["lon"] = coords["lon"]
            except Exception as ge:
                logger.debug(f"Geocode skip '{c.get('place_name')}': {ge}")
        enriched.append(c)
    return enriched


# ─────────────────────────────────────────
# POST /api/itinerary/generate
# ─────────────────────────────────────────
@router.post("/generate", response_model=ItineraryResponse)
async def generate_itinerary(req: TripRequest):
    try:
        logger.info(
            f"[gen] {req.destination} | {req.days}d | {req.travel_type.value} | "
            f"{req.mood.value} | dates={req.travel_dates}"
        )

        # ─ Stage 1: Groq → top must-visit place candidates (NO meals) ─
        raw_candidates = await generate_place_candidates_llm(req)
        logger.info(f"[gen] S1: {len(raw_candidates)} candidates from Groq")

        # ─ Stage 2: Geocode candidates (resolve lat/lon) ─
        candidates = await _geocode_candidates(raw_candidates, req.destination)
        logger.info(f"[gen] S2: geocoded {len(candidates)} candidates")

        # ─ Stage 3: Build slot template + day dates ─
        slots     = build_day_slots(req.days)
        day_dates = _compute_day_dates(req.travel_dates, req.days) if req.travel_dates else {}
        if day_dates:
            logger.info(f"[gen] S3: day dates computed: {day_dates}")
        else:
            logger.info("[gen] S3: no travel_dates — weekday hour checks skipped")

        # ─ Stage 4: Schedule (opening-hours enforced) ─
        scheduled, unscheduled = schedule_candidates(
            candidates, slots,
            avoid_crowded=req.avoid_crowded,
            accessibility_needs=req.accessibility_needs,
            day_dates=day_dates or None,
        )
        logger.info(f"[gen] S4: {len(scheduled)} placed | {len(unscheduled)} unscheduled")

        # ─ Stage 5: Conflict resolution — ask Groq for alternates ─
        if unscheduled:
            logger.info(f"[gen] S5: resolving {len(unscheduled)} conflicts via Groq")
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

                enriched_alts = await _geocode_candidates(alts, req.destination)
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

        # ─ Stage 6: Weather warnings (non-critical) ─
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
                slot_id=s["slot_id"],
                day=s["day"],
                slot_name=s["slot_name"],
                start_time=s["start_time"],
                end_time=s["end_time"],
                available_mins=s["available_mins"],
                remaining_mins=s["remaining_mins"],
                meal_gap_after=s.get("meal_gap_after")
            )
            for s in slots
        ]

        stops_out = []
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

        # Count unverified-hours stops for meta
        unverified_count = sum(1 for s in stops_out if s.opening_hours_unverified)

        meta = ItineraryMeta(
            destination=req.destination,
            days=req.days,
            travel_type=req.travel_type.value,
            budget=req.budget.value,
            mood=req.mood.value,
            total_places=len(stops_out),
            unscheduled_count=len(unscheduled),
            hours_unverified_count=unverified_count,
        )

        logger.info(
            f"[gen] Done: {len(stops_out)} stops, "
            f"{len(unscheduled)} unscheduled, "
            f"{unverified_count} hours-unverified"
        )
        return ItineraryResponse(
            success=True,
            meta=meta,
            slot_template=slot_template,
            itinerary=stops_out,
            unscheduled=unscheduled_out,
            weather_warnings=weather_warnings or None,
        )

    except Exception as e:
        logger.error(f"[gen] {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


# ─────────────────────────────────────────
# POST /api/itinerary/enrich
# ─────────────────────────────────────────
@router.post("/enrich", response_model=PlaceEnrichResponse)
async def enrich_place(req: PlaceEnrichRequest):
    try:
        data = await enrich_place_with_perplexity(req.place_name, req.city)
        return PlaceEnrichResponse(place_name=req.place_name, city=req.city, **data)
    except Exception as e:
        logger.error(f"[enrich] {type(e).__name__}: {e}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


# ─────────────────────────────────────────
# POST /api/itinerary/save
# ─────────────────────────────────────────
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
