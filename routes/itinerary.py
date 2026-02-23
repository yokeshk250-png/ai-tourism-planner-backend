import asyncio
import logging
import math
import re
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
# Curated coordinates for places that Geoapify commonly
# mis-geocodes or returns no result for.
#
# IMPORTANT — Resolution priority in _geocode_candidates():
#   1st: KNOWN_COORDS (always checked first for listed places)
#   2nd: geocode_place() API (for all non-KNOWN places)
#   3rd: KNOWN_COORDS again (if API returned None or >MAX_PLACE_DISTANCE_KM)
#   4th: lat=None (scheduler uses default 15-min travel estimate)
#
# Key format: re.sub(r'[^a-z0-9 ]', '', name.lower()).strip()[:35]
# i.e. lowercase, strip non-alphanumeric (except space), first 35 chars.
# ─────────────────────────────────────────────────────────────
KNOWN_COORDS: dict[str, tuple[float, float]] = {
    # ── Madurai ───────────────────────────────────────────────────────────────
    "goripalayam dargah":                (9.9275,  78.1133),
    "uchi pillaiyar temple":             (9.9330,  78.1191),
    "vilachery pottery village":         (9.8973,  78.1476),
    "samanar hills":                     (9.9189,  78.0556),
    "aayiram kaal mandapam":             (9.9197,  78.1185),
    "nayak mahal thirumalai":            (9.9148,  78.1239),
    "madurai railway museum":            (9.9161,  78.1062),
    "madurai corporation ecopark":       (9.9355,  78.1382),
    # ── Kodaikanal — viewpoints / walks ───────────────────────────────────────
    "coakers walk kodaikanal":           (10.2301, 77.4940),
    "moir point":                        (10.2367, 77.4910),
    "moir point kodaikanal":             (10.2367, 77.4910),
    "moir point theni":                  (10.2367, 77.4910),  # LLM city-suffix variant
    "moier point":                       (10.2367, 77.4910),  # LLM typo variant
    "moier point kodaikanal":            (10.2367, 77.4910),
    "echo point kodaikanal":             (10.2367, 77.4910),
    "echo point":                        (10.2367, 77.4910),
    "green valley view":                 (10.2355, 77.4952),
    "green valley view kodaikanal":      (10.2355, 77.4952),
    "mannavanur view point":             (10.1667, 77.4167),
    # ── Kodaikanal — lakes ────────────────────────────────────────────────────
    # Berijam Lake is ~22 km SW of Kodaikanal town; Geoapify returns the town
    # lake coords instead — this entry forces the correct location.
    "berijam lake":                      (10.1700, 77.3990),
    "berijam lake kodaikanal":           (10.1700, 77.3990),
    # ── Kodaikanal — caves / rocks ────────────────────────────────────────────
    "dolmen circle kodaikanal":          (10.2381, 77.4891),
    "pillar rocks":                      (10.2227, 77.4804),
    "pillar rocks kodaikanal":           (10.2227, 77.4804),
    "guna cave":                         (10.2227, 77.4804),
    "guna cave kodaikanal":              (10.2227, 77.4804),
    # ── Kodaikanal — waterfalls ───────────────────────────────────────────────
    "silver cascade falls":              (10.2420, 77.5103),
    "silver cascade falls kodaikanal":   (10.2420, 77.5103),
    "bear shola falls":                  (10.2396, 77.4786),
    "bear shola falls kodaikanal":       (10.2396, 77.4786),
    "vattakanal falls":                  (10.2097, 77.4659),
    "vattakanal falls dindigul":         (10.2097, 77.4659),
    "liril falls":                       (10.2097, 77.4659),
    "fairy falls":                       (10.2237, 77.4668),
    "fairy falls kodaikanal":            (10.2237, 77.4668),
    "pambar falls":                      (10.1950, 77.4120),
    "thalaiyar falls":                   (10.1133, 77.4167),
    # ── Kodaikanal — parks / gardens ─────────────────────────────────────────
    "bryant park kodaikanal":            (10.2306, 77.4924),
    "bryant park entrance garden":       (10.2306, 77.4924),
    "chettiar park kodaikanal":          (10.2505, 77.4991),
    "kodai gardens":                     (10.2429, 77.4981),
    # ── Kodaikanal — religious / cultural ────────────────────────────────────
    "kurinji andavar temple":            (10.2532, 77.5020),
    "kurinji andavar temple kodaikanal": (10.2532, 77.5020),
    "la salette church":                 (10.2365, 77.4894),
    "la salette church kodaikanal":      (10.2365, 77.4894),
    "shembaganur museum":                (10.2340, 77.4940),
    "shembaganur museum of natural hist": (10.2340, 77.4940),
    # ── Kodaikanal — sports / recreation ─────────────────────────────────────
    "kodaikanal golf club":              (10.2145, 77.4708),
    "kodaikanal golf club kodaikanal":   (10.2145, 77.4708),
    "kodaikanal cricket club":           (10.2340, 77.4889),
    "kodaikanal cricket club kodaikana": (10.2340, 77.4889),
    # ── Kodaikanal — forests / trails ────────────────────────────────────────
    "pine forest kodaikanal":            (10.2134, 77.4586),
    # ── Kodaikanal — other ────────────────────────────────────────────────────
    "vattakanal viewpoint":              (10.2097, 77.4659),
    "vattakanal viewpoint dindigul":     (10.2097, 77.4659),
    # ── Generic fallbacks for other hill stations can be added here ──────────
}

# Set of KNOWN_COORDS keys for O(1) priority-check lookup
_KNOWN_KEYS: frozenset = frozenset(KNOWN_COORDS.keys())


def _coords_key(name: str) -> str:
    """Normalise place_name to a KNOWN_COORDS lookup key."""
    return re.sub(r'[^a-z0-9 ]', '', name.lower()).strip()[:35]


def _lookup_known(key: str) -> tuple[float, float] | None:
    """Return curated coords for key (exact) or first prefix/substring match."""
    fb = KNOWN_COORDS.get(key)
    if fb:
        return fb
    for k, v in KNOWN_COORDS.items():
        if k in key or key.startswith(k[:20]):
            return v
    return None


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


def _canonical_name(name: str) -> str:
    """Normalise place name for duplicate detection."""
    base = name.split(",")[0].strip().lower()
    return re.sub(r'[^a-z0-9]', '', base)[:30]


def _dedup_candidates(candidates: list) -> list:
    """
    Remove duplicate place candidates (same canonical name).
    When duplicates exist, keep the one with the highest priority.
    """
    seen: dict[str, dict] = {}
    for c in candidates:
        key = _canonical_name(c.get("place_name", ""))
        if key not in seen:
            seen[key] = c
        else:
            existing_pri = int(seen[key].get("priority", 3))
            this_pri     = int(c.get("priority", 3))
            if this_pri > existing_pri:
                seen[key] = c
    deduped = list(seen.values())
    removed = len(candidates) - len(deduped)
    if removed:
        logger.info(f"[dedup] removed {removed} duplicate candidate(s)")
    return deduped


def _remove_already_scheduled(unscheduled: list, scheduled: list) -> list:
    """
    Remove from unscheduled any place already present in scheduled.
    """
    sched_names = {_canonical_name(s.get("place_name", "")) for s in scheduled}
    filtered    = [u for u in unscheduled if _canonical_name(u.get("place_name", "")) not in sched_names]
    removed     = len(unscheduled) - len(filtered)
    if removed:
        logger.info(f"[dedup] removed {removed} already-scheduled item(s) from unscheduled list")
    return filtered


def _merge_enrich_data(candidate: dict, enrich_data: dict) -> dict:
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
    Resolve lat/lon for every candidate.

    Resolution order:
      0. KNOWN_COORDS priority check — if the place_name key is in the curated
         table we use it immediately WITHOUT calling the API. This prevents
         Geoapify from silently returning plausible-but-wrong coords for known
         mis-geocoded places (e.g. Berijam Lake → Kodaikanal Lake area).
      1. geocode_place() API call (only for non-KNOWN places).
      2. KNOWN_COORDS fallback (if API returned None or > MAX_PLACE_DISTANCE_KM).
      3. lat=None — kept in list, scheduler uses default 15-min travel estimate.

    Hard drop: places whose final resolved coords exceed MAX_PLACE_DISTANCE_KM
    from the city centre and are NOT in KNOWN_COORDS are removed (wrong-city).
    """
    out     = []
    dropped = 0

    for raw in candidates:
        c    = dict(raw) if isinstance(raw, dict) else raw.model_dump()
        name = c.get("place_name", "?")
        key  = _coords_key(name)

        lat, lon = None, None
        source   = "none"

        # ── 0: KNOWN_COORDS priority override ──────────────────────────────
        known = _lookup_known(key)
        if known:
            lat, lon = known
            source   = "known_coords_priority"
            logger.debug(f"[geocode] '{name}' → KNOWN_COORDS (priority) ({lat}, {lon})")

        # ── 1: API geocode (only if not overridden by KNOWN_COORDS) ────────
        if lat is None:
            try:
                coords = await geocode_place(place_name=name, city=destination)
            except Exception as ge:
                logger.debug(f"[geocode] skip '{name}': {ge}")
                coords = None

            if coords:
                api_lat, api_lon = coords["lat"], coords["lon"]
                dist = _haversine_km(city_lat, city_lon, api_lat, api_lon)
                if dist <= MAX_PLACE_DISTANCE_KM:
                    lat, lon = api_lat, api_lon
                    source   = f"api ({dist:.1f}km)"
                else:
                    logger.debug(
                        f"[geocode] '{name}' API coords too far ({dist:.0f}km) — trying KNOWN_COORDS"
                    )

        # ── 2: KNOWN_COORDS fallback (API failed or too far) ───────────────
        if lat is None:
            fb = _lookup_known(key)
            if fb:
                lat, lon = fb
                source   = "known_coords_fallback"
                logger.debug(f"[geocode] '{name}' → KNOWN_COORDS (fallback) ({lat}, {lon})")

        # ── 3: No coords ────────────────────────────────────────────────────
        if lat is None:
            c["lat"] = None
            c["lon"] = None
            logger.debug(f"[geocode] '{name}': no result, coords=None")
            out.append(c)
            continue

        # Hard distance check
        dist_final = _haversine_km(city_lat, city_lon, lat, lon)
        if dist_final > MAX_PLACE_DISTANCE_KM:
            logger.warning(
                f"[geocode] DROPPING '{name}': final coords {dist_final:.0f}km from "
                f"{destination} (src={source})"
            )
            dropped += 1
            continue

        c["lat"] = lat
        c["lon"] = lon
        logger.debug(f"[geocode] '{name}' → ({lat:.5f}, {lon:.5f}) [{source}]")
        out.append(c)

    if dropped:
        logger.warning(
            f"[geocode] {dropped} candidate(s) dropped (>{MAX_PLACE_DISTANCE_KM}km from {destination})"
        )
    return out


# ─────────────────────────────────────────────────────────────
# POST /api/itinerary/generate
#
# 6-stage pipeline:
#   S1  — Groq: top must-visit candidates (no meals)
#   S1.5— Dedup: remove duplicate place names from LLM output
#   S2  — Geocode: KNOWN_COORDS priority → API → KNOWN_COORDS fallback → None
#   S2.5— Enrich: opening hrs, real duration, fees, slot (city-anchored)
#   S3  — Build day slots
#   S4  — Schedule (opening-hours enforced)
#   S4.5— Cross-list dedup: remove already-scheduled from unscheduled
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

        # ─ Resolve city centre coords once ─
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

        # ─ S1.5: Dedup LLM output ─
        raw_candidates = _dedup_candidates(
            [c if isinstance(c, dict) else c.model_dump() for c in raw_candidates]
        )
        logger.info(f"[gen] S1.5: {len(raw_candidates)} after dedup")

        # ─ S2: Geocode — KNOWN_COORDS priority → API → KNOWN_COORDS fallback ─
        if city_lat is not None:
            candidates = await _geocode_candidates(
                raw_candidates, req.destination, city_lat, city_lon
            )
        else:
            candidates = [
                c if isinstance(c, dict) else c.model_dump()
                for c in raw_candidates
            ]
        logger.info(f"[gen] S2: {len(candidates)} candidates after geocode+filter")

        # ─ S2.5: Enrich ─
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

        # ─ S4.5: Remove already-scheduled from unscheduled list ─
        unscheduled = _remove_already_scheduled(unscheduled, scheduled)
        logger.info(f"[gen] S4.5: {len(unscheduled)} unscheduled after cross-list dedup")

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

                if city_lat is not None:
                    enriched_alts = await _geocode_candidates(
                        alts, req.destination, city_lat, city_lon
                    )
                else:
                    enriched_alts = [dict(a) if isinstance(a, dict) else a.model_dump() for a in alts]

                enriched_alts = await _enrich_all_candidates(enriched_alts, req.destination)
                enriched_alts = _dedup_candidates(enriched_alts)
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
                    alt_unsched = _remove_already_scheduled(alt_unsched, scheduled)
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
