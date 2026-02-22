import asyncio
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
router  = APIRouter()

# Enrich only the top N candidates (by priority) to stay within Groq rate limits.
# Lower-priority candidates that likely won’t be scheduled are skipped.
MAX_ENRICH = 15

# Map enrich’s best_time_to_visit string → scheduler slot name
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


def _merge_enrich_data(candidate: dict, enrich_data: dict) -> dict:
    """
    Merge Groq enrich response into a candidate dict.
    Enrich values override the LLM’s initial guesses because they come
    from a focused, per-place prompt — more accurate for scheduling.

    Mapping:
      opening_hours           → opening_hours   (used by opening-hours check)
      closed_on               → closed_on        (used by opening-hours check)
      avg_visit_duration_hrs  → duration_hrs     (used by fits-in-slot check)
      best_time_to_visit      → best_slot        (slot preference for placement)
      entry_fee_indian        → entry_fee        (INR, shown in UI)
      entry_fee_foreign       → entry_fee_foreign(shown in UI)
      local_tip               → tip
      nearby_food             → nearby_food      (shown in UI below stop card)
    """
    c = dict(candidate)

    if enrich_data.get("opening_hours"):
        c["opening_hours"] = enrich_data["opening_hours"]

    # closed_on may be an empty list — still overrides LLM guess
    if enrich_data.get("closed_on") is not None:
        c["closed_on"] = enrich_data["closed_on"]

    if enrich_data.get("avg_visit_duration_hrs"):
        c["duration_hrs"] = float(enrich_data["avg_visit_duration_hrs"])

    # Map best_time_to_visit → slot name (exact, then substring)
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
    Only enriches the top MAX_ENRICH by priority — rest are kept as-is.

    Concurrency: 5 simultaneous calls (safe within Groq’s 30 req/min free tier).
    Failures are silently swallowed — original candidate is returned unchanged.
    """
    sem = asyncio.Semaphore(5)

    async def _enrich_one(c: dict) -> dict:
        async with sem:
            try:
                data = await enrich_place_with_perplexity(
                    c.get("place_name", ""), destination
                )
                return _merge_enrich_data(c, data) if data else c
            except Exception as e:
                logger.warning(f"[enrich] skip '{c.get('place_name')}': {e}")
                return c

    # Sort by priority so we enrich the most important candidates first
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


async def _geocode_candidates(candidates: list, destination: str) -> list:
    """
    Resolve lat/lon for each candidate missing coordinates (Geoapify).
    Failures are silently skipped — scheduler uses a default travel estimate.
    """
    out = []
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
        out.append(c)
    return out


# ───────────────────────────────────────────────
# POST /api/itinerary/generate
#
# 6-stage pipeline:
#   S1  — Groq: top must-visit candidates (no meals)
#   S2  — Geocode: lat/lon via Geoapify
#   S2.5— Enrich: opening hrs, real duration, fees, slot ← NEW
#   S3  — Build day slots
#   S4  — Schedule (opening-hours enforced)
#   S5  — Conflict resolver (Groq alternates)
#   S6  — Weather warnings
# ───────────────────────────────────────────────
@router.post("/generate", response_model=ItineraryResponse)
async def generate_itinerary(req: TripRequest):
    try:
        logger.info(
            f"[gen] {req.destination} | {req.days}d | {req.travel_type.value} | "
            f"{req.mood.value} | dates={req.travel_dates}"
        )

        # ─ S1: Groq candidates (NO meals) ─
        raw_candidates = await generate_place_candidates_llm(req)
        logger.info(f"[gen] S1: {len(raw_candidates)} candidates")

        # ─ S2: Geocode (lat/lon) ─
        candidates = await _geocode_candidates(raw_candidates, req.destination)
        logger.info(f"[gen] S2: geocoded {len(candidates)} candidates")

        # ─ S2.5: Enrich (opening hours, real duration, fees, best_slot) ─
        candidates = await _enrich_all_candidates(candidates, req.destination)
        logger.info(f"[gen] S2.5: enriched {len(candidates)} candidates")

        # ─ S3: Build slots + day dates ─
        slots     = build_day_slots(req.days)
        day_dates = _compute_day_dates(req.travel_dates, req.days) if req.travel_dates else {}
        logger.info(
            f"[gen] S3: {len(slots)} slots | "
            f"day_dates={'set' if day_dates else 'not set (weekday checks skipped)'}"
        )

        # ─ S4: Schedule (with opening-hours enforcement) ─
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

                # geocode + enrich + tag alternates before scheduling
                enriched_alts = await _geocode_candidates(alts, req.destination)
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


# ───────────────────────────────────────────────
# POST /api/itinerary/enrich
# ───────────────────────────────────────────────
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
