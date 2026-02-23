"""
routes/customize.py

Itinerary Customization Endpoints
==================================
All endpoints operate on a saved itinerary document stored at:
  Firestore: users/{user_id}/itineraries/{itinerary_id}

Endpoints
---------
POST /api/customize/{itinerary_id}/swap
    Replace one scheduled stop with a new candidate.
    Re-computes times for the affected slot/day after replacement.

POST /api/customize/{itinerary_id}/remove
    Delete a scheduled stop.
    Re-computes times for the affected slot/day after removal.

POST /api/customize/{itinerary_id}/add
    Insert a new stop into the best-fit slot (Pass 1 strict, then Pass 2
    relaxed) across any day.  Re-computes times for the affected slot.

POST /api/customize/{itinerary_id}/reorder
    Change the order of stops within a single day.
    Re-computes start/end times for all stops in that day after reordering.
    Pinned stops are treated as immovable and kept in their original position.

POST /api/customize/{itinerary_id}/pin
    Toggle the pinned flag on a stop.  Pinned stops cannot be moved by
    reorder or by future auto-replanning.

PATCH /api/customize/{itinerary_id}/settings
    Partially update trip-level preferences (pace, budget, avoid_crowded,
    accessibility_needs, start_time override).  Uses exclude_unset so only
    provided fields are written.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel, Field

from services.firebase_service import (
    get_itinerary_by_id_for_user,
    update_itinerary,
    get_db,
)
from services.scheduler_service import (
    _hm_to_mins,
    _mins_to_time,
    _fits_in_slot_v2,
    _compute_stop_times,
    _actual_travel,
    _slot_has_any_overlap,
    _SLOT_WINDOWS,
    estimate_travel_minutes,
    build_day_slots,
    schedule_candidates,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# ─────────────────────────────────────────────────────────────
# Auth helper  (no Firebase Auth SDK dependency — caller passes
# user_id in header; swap for JWT verify when Auth is wired)
# ─────────────────────────────────────────────────────────────
def _get_user_id(x_user_id: Optional[str] = Header(None, alias="X-User-Id")) -> str:
    """
    Minimal auth: caller passes Firebase UID in X-User-Id header.

    TODO: replace with Firebase ID token verification once Auth is
    integrated in the frontend:
        from firebase_admin import auth
        decoded = auth.verify_id_token(token)
        return decoded['uid']
    """
    if not x_user_id or not x_user_id.strip():
        raise HTTPException(status_code=401, detail="X-User-Id header required")
    return x_user_id.strip()


# ─────────────────────────────────────────────────────────────
# Request schemas
# ─────────────────────────────────────────────────────────────
class StopCandidate(BaseModel):
    """A new place to insert into the itinerary (swap or add)."""
    place_name:        str
    category:          Optional[str]   = None
    priority:          int             = 3
    duration_hrs:      float           = 1.0
    best_slot:         Optional[Literal["morning", "afternoon", "evening", "night"]] = None
    opening_hours:     Optional[str]   = None
    closed_on:         Optional[List[str]] = None
    entry_fee:         Optional[int]   = None
    entry_fee_foreign: Optional[int]   = None
    tip:               Optional[str]   = None
    nearby_food:       Optional[str]   = None
    why_must_visit:    Optional[str]   = None
    lat:               Optional[float] = None
    lon:               Optional[float] = None
    is_alternate:      bool            = False


class SwapRequest(BaseModel):
    day:            int   = Field(..., ge=1)
    slot_id:        str
    old_place_name: str
    new_place:      StopCandidate


class RemoveRequest(BaseModel):
    day:        int = Field(..., ge=1)
    slot_id:    str
    place_name: str


class AddRequest(BaseModel):
    new_place:     StopCandidate
    preferred_day: Optional[int] = Field(None, ge=1)


class ReorderRequest(BaseModel):
    day:                   int        = Field(..., ge=1)
    new_order_place_names: List[str]  # full ordered list for that day


class PinRequest(BaseModel):
    day:        int  = Field(..., ge=1)
    place_name: str
    pinned:     bool = True


class SettingsPatch(BaseModel):
    """
    Partial trip-level overrides.  Only provided fields are written
    (model_dump(exclude_unset=True)).
    """
    pace:                Optional[Literal["relaxed", "normal", "fast"]] = None
    budget:              Optional[Literal["low", "medium", "high"]]     = None
    avoid_crowded:       Optional[bool]  = None
    accessibility_needs: Optional[bool]  = None


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────
def _canonical(name: str) -> str:
    return re.sub(r'[^a-z0-9]', '', name.lower())[:30]


def _recompute_day(itinerary: List[Dict[str, Any]], day: int) -> List[Dict[str, Any]]:
    """
    Re-compute start_time / end_time / travel_mins_from_prev for every
    stop in `day`, in their current list order, respecting pinned stops.

    Algorithm:
      - Walk stops in order.
      - For each stop compute travel from previous stop's lat/lon.
      - start = max(cursor, place_open_clamp) + travel
      - end   = start + duration_mins
      - cursor advances to end.
      - Pinned stops: keep their original start_time as a hard lower bound
        (do not move them earlier or later than original time).

    Returns the full itinerary list with that day's times updated.
    """
    # split out the target day, preserve order of other days
    day_stops  = [s for s in itinerary if s.get("day") == day]
    other_stops = [s for s in itinerary if s.get("day") != day]

    if not day_stops:
        return itinerary

    # Group by slot, recompute per-slot
    from collections import defaultdict
    slot_groups: Dict[str, List[Dict]] = defaultdict(list)
    for s in day_stops:
        slot_groups[s.get("slot_id", "")].append(s)

    slot_starts = {
        f"day{day}_morning":   _hm_to_mins("09:00"),
        f"day{day}_afternoon": _hm_to_mins("13:00"),
        f"day{day}_evening":   _hm_to_mins("16:30"),
        f"day{day}_night":     _hm_to_mins("20:00"),
    }
    slot_ends = {
        f"day{day}_morning":   _hm_to_mins("12:00"),
        f"day{day}_afternoon": _hm_to_mins("16:00"),
        f"day{day}_evening":   _hm_to_mins("19:30"),
        f"day{day}_night":     _hm_to_mins("21:30"),
    }

    updated: List[Dict] = []
    for slot_id, stops in slot_groups.items():
        cursor   = slot_starts.get(slot_id, _hm_to_mins("09:00"))
        prev_lat = None
        prev_lon = None
        is_first = True

        for stop in stops:
            pinned       = stop.get("pinned", False)
            dur_mins     = int(float(stop.get("duration_hrs", 1.0)) * 60)
            s_lat        = stop.get("lat")
            s_lon        = stop.get("lon")

            travel = 0 if is_first else estimate_travel_minutes(prev_lat, prev_lon, s_lat, s_lon)

            if pinned:
                # Honour original start time as lower bound
                try:
                    pinned_start = _hm_to_mins(stop["start_time"])
                except Exception:
                    pinned_start = cursor
                start_mins = max(cursor, pinned_start) + (0 if is_first else travel)
            else:
                start_mins = cursor + travel

            end_mins = start_mins + dur_mins

            updated_stop = dict(stop)
            updated_stop["start_time"]            = _mins_to_time(start_mins)
            updated_stop["end_time"]              = _mins_to_time(end_mins)
            updated_stop["travel_mins_from_prev"] = 0 if is_first else travel

            cursor   = end_mins
            prev_lat = s_lat
            prev_lon = s_lon
            is_first = False
            updated.append(updated_stop)

    return other_stops + updated


def _find_stop(itinerary: List[Dict], day: int, slot_id: str, place_name: str) -> Optional[int]:
    """Return index of the matching stop in itinerary list, or None."""
    canon = _canonical(place_name)
    for i, s in enumerate(itinerary):
        if (
            s.get("day") == day
            and s.get("slot_id") == slot_id
            and _canonical(s.get("place_name", "")) == canon
        ):
            return i
    return None


def _stop_from_candidate(
    cand: StopCandidate,
    day: int,
    slot_id: str,
    slot_name: str,
    start_time: str,
    end_time: str,
    travel_mins: int,
    opening_hours_unverified: bool = False,
) -> Dict[str, Any]:
    """Convert a StopCandidate + scheduling metadata into a stop dict."""
    return {
        "day":                      day,
        "slot_id":                  slot_id,
        "slot_name":                slot_name,
        "start_time":               start_time,
        "end_time":                 end_time,
        "travel_mins_from_prev":    travel_mins,
        "place_name":               cand.place_name,
        "category":                 cand.category,
        "priority":                 cand.priority,
        "duration_hrs":             cand.duration_hrs,
        "opening_hours":            cand.opening_hours,
        "closed_on":                cand.closed_on,
        "entry_fee":                cand.entry_fee,
        "entry_fee_foreign":        cand.entry_fee_foreign,
        "tip":                      cand.tip,
        "nearby_food":              cand.nearby_food,
        "why_must_visit":           cand.why_must_visit,
        "lat":                      cand.lat,
        "lon":                      cand.lon,
        "is_alternate":             cand.is_alternate,
        "opening_hours_unverified": opening_hours_unverified,
        "hours_conflict":           False,
        "pinned":                   False,
    }


def _build_slots_from_itinerary(
    itinerary: List[Dict], days: int
) -> List[Dict]:
    """
    Reconstruct live slot state (remaining_mins, last_lat, last_lon,
    last_end_mins) from the current itinerary so that _fits_in_slot_v2
    and _compute_stop_times work correctly when adding a new stop.
    """
    slots = build_day_slots(days)
    slot_map = {s["slot_id"]: s for s in slots}

    # sort by day + start_time to replay in order
    ordered = sorted(itinerary, key=lambda x: (x.get("day", 0), x.get("start_time", "00:00")))
    for stop in ordered:
        sid = stop.get("slot_id")
        if sid not in slot_map:
            continue
        slot = slot_map[sid]
        end_mins = _hm_to_mins(stop.get("end_time", slot["end_time"]))
        slot_end = _hm_to_mins(slot["end_time"])
        slot["last_end_mins"]  = end_mins
        slot["remaining_mins"] = max(0, slot_end - end_mins)
        if stop.get("lat"):
            slot["last_lat"] = stop["lat"]
        if stop.get("lon"):
            slot["last_lon"] = stop["lon"]
        slot.setdefault("stops", []).append(stop)

    return slots


# ─────────────────────────────────────────────────────────────
# POST /api/customize/{itinerary_id}/swap
# ─────────────────────────────────────────────────────────────
@router.post("/{itinerary_id}/swap")
async def swap_stop(
    itinerary_id: str,
    body: SwapRequest,
    user_id: str = Header(None, alias="X-User-Id"),
):
    """
    Replace `old_place_name` in `slot_id` on `day` with `new_place`.
    Re-computes times for the affected day after substitution.
    """
    uid = _get_user_id(user_id)
    doc = await get_itinerary_by_id_for_user(uid, itinerary_id)
    if not doc:
        raise HTTPException(404, "Itinerary not found")

    itinerary: List[Dict] = list(doc.get("itinerary", []))

    idx = _find_stop(itinerary, body.day, body.slot_id, body.old_place_name)
    if idx is None:
        raise HTTPException(404, f"Stop '{body.old_place_name}' not found in {body.slot_id} on day {body.day}")

    old_stop   = itinerary[idx]
    slot_name  = old_stop.get("slot_name", "morning")
    start_time = old_stop.get("start_time", "09:00")
    end_mins   = _hm_to_mins(start_time) + int(body.new_place.duration_hrs * 60)
    end_time   = _mins_to_time(end_mins)
    travel     = old_stop.get("travel_mins_from_prev", 0)

    new_stop = _stop_from_candidate(
        body.new_place,
        day=body.day, slot_id=body.slot_id, slot_name=slot_name,
        start_time=start_time, end_time=end_time,
        travel_mins=travel,
        opening_hours_unverified=(
            body.new_place.opening_hours is None or body.new_place.opening_hours.strip() == ""
        ),
    )
    itinerary[idx] = new_stop

    # Recompute the whole day to cascade time changes
    days = doc.get("meta", {}).get("days", max(s.get("day", 1) for s in itinerary))
    itinerary = _recompute_day(itinerary, body.day)

    await update_itinerary(uid, itinerary_id, {"itinerary": itinerary})
    logger.info(f"[customize] swap: '{body.old_place_name}' → '{body.new_place.place_name}' in {body.slot_id}")
    return {"success": True, "itinerary": itinerary}


# ─────────────────────────────────────────────────────────────
# POST /api/customize/{itinerary_id}/remove
# ─────────────────────────────────────────────────────────────
@router.post("/{itinerary_id}/remove")
async def remove_stop(
    itinerary_id: str,
    body: RemoveRequest,
    user_id: str = Header(None, alias="X-User-Id"),
):
    """
    Delete a stop from the itinerary and recompute times for that day.
    """
    uid = _get_user_id(user_id)
    doc = await get_itinerary_by_id_for_user(uid, itinerary_id)
    if not doc:
        raise HTTPException(404, "Itinerary not found")

    itinerary: List[Dict] = list(doc.get("itinerary", []))
    canon = _canonical(body.place_name)
    before = len(itinerary)
    itinerary = [
        s for s in itinerary
        if not (
            s.get("day") == body.day
            and s.get("slot_id") == body.slot_id
            and _canonical(s.get("place_name", "")) == canon
        )
    ]
    if len(itinerary) == before:
        raise HTTPException(404, f"Stop '{body.place_name}' not found")

    days = doc.get("meta", {}).get("days", max((s.get("day", 1) for s in itinerary), default=1))
    itinerary = _recompute_day(itinerary, body.day)

    await update_itinerary(uid, itinerary_id, {"itinerary": itinerary})
    logger.info(f"[customize] remove: '{body.place_name}' from {body.slot_id} day {body.day}")
    return {"success": True, "itinerary": itinerary}


# ─────────────────────────────────────────────────────────────
# POST /api/customize/{itinerary_id}/add
# ─────────────────────────────────────────────────────────────
@router.post("/{itinerary_id}/add")
async def add_stop(
    itinerary_id: str,
    body: AddRequest,
    user_id: str = Header(None, alias="X-User-Id"),
):
    """
    Insert a new stop into the best available slot.
    Uses Pass 1 (strict OH) then Pass 2 (relaxed, overlap guard) — same
    logic as the main scheduler so all existing rules are respected.
    preferred_day constrains the search to that day only.
    """
    uid = _get_user_id(user_id)
    doc = await get_itinerary_by_id_for_user(uid, itinerary_id)
    if not doc:
        raise HTTPException(404, "Itinerary not found")

    itinerary: List[Dict] = list(doc.get("itinerary", []))
    days = doc.get("meta", {}).get("days", max((s.get("day", 1) for s in itinerary), default=1))

    # Check not already present
    canon_new = _canonical(body.new_place.place_name)
    if any(_canonical(s.get("place_name", "")) == canon_new for s in itinerary):
        raise HTTPException(409, f"'{body.new_place.place_name}' is already in the itinerary")

    # Build live slot state from current itinerary
    slots = _build_slots_from_itinerary(itinerary, days)
    if body.preferred_day:
        slots = [s for s in slots if s["day"] == body.preferred_day]
        if not slots:
            raise HTTPException(400, f"No slots found for day {body.preferred_day}")

    # Run the scheduler for just this one candidate
    cand_dict = body.new_place.model_dump()
    cand_dict.setdefault("best_slot", body.new_place.best_slot or "morning")

    placed_list, unplaced = schedule_candidates([cand_dict], slots)

    if unplaced:
        raise HTTPException(
            409,
            f"No slot available for '{body.new_place.place_name}' "
            f"(duration={body.new_place.duration_hrs}h, opening_hours={body.new_place.opening_hours})"
        )

    placed_stop = placed_list[0]
    itinerary.append(placed_stop)
    # Sort by day + start_time for clean ordering
    itinerary.sort(key=lambda x: (x.get("day", 0), x.get("start_time", "00:00")))

    affected_day = placed_stop["day"]
    itinerary = _recompute_day(itinerary, affected_day)

    await update_itinerary(uid, itinerary_id, {"itinerary": itinerary})
    logger.info(
        f"[customize] add: '{body.new_place.place_name}' → {placed_stop['slot_id']} @ {placed_stop['start_time']}"
    )
    return {"success": True, "placed_stop": placed_stop, "itinerary": itinerary}


# ─────────────────────────────────────────────────────────────
# POST /api/customize/{itinerary_id}/reorder
# ─────────────────────────────────────────────────────────────
@router.post("/{itinerary_id}/reorder")
async def reorder_stops(
    itinerary_id: str,
    body: ReorderRequest,
    user_id: str = Header(None, alias="X-User-Id"),
):
    """
    Reorder all stops on a given day.
    `new_order_place_names` must be a list of all place_names for that day
    in the desired order.  Pinned stops stay in their original position
    (the new order is applied around them).
    """
    uid = _get_user_id(user_id)
    doc = await get_itinerary_by_id_for_user(uid, itinerary_id)
    if not doc:
        raise HTTPException(404, "Itinerary not found")

    itinerary: List[Dict] = list(doc.get("itinerary", []))
    day_stops   = [s for s in itinerary if s.get("day") == body.day]
    other_stops = [s for s in itinerary if s.get("day") != body.day]

    if not day_stops:
        raise HTTPException(404, f"No stops found on day {body.day}")

    # Validate: new order must contain exactly the same set of place names
    existing_canons = {_canonical(s["place_name"]) for s in day_stops}
    new_canons      = [_canonical(n) for n in body.new_order_place_names]
    if set(new_canons) != existing_canons:
        missing  = existing_canons - set(new_canons)
        extra    = set(new_canons) - existing_canons
        detail   = ""
        if missing:  detail += f" Missing: {list(missing)}."
        if extra:    detail += f" Unknown: {list(extra)}."
        raise HTTPException(422, f"new_order_place_names must include all stops for day {body.day}.{detail}")

    # Build a lookup from canon name → stop dict
    stop_by_canon = {_canonical(s["place_name"]): s for s in day_stops}

    # Separate pinned stops (keep slot positions)
    pinned_by_slot: Dict[str, List[Dict]] = {}
    free: List[Dict] = []
    for s in day_stops:
        if s.get("pinned"):
            slot = s.get("slot_id", "")
            pinned_by_slot.setdefault(slot, []).append(s)
        else:
            free.append(s)

    # Rebuild day order following new_order_place_names (only for free stops)
    reordered: List[Dict] = []
    for name in body.new_order_place_names:
        stop = stop_by_canon.get(_canonical(name))
        if stop and not stop.get("pinned"):
            reordered.append(stop)

    # Re-insert pinned stops at their original position by slot order
    for slot_id, pinned_stops in sorted(pinned_by_slot.items()):
        for ps in pinned_stops:
            # insert pinned stop at the correct slot boundary position
            target_slot_order = ["morning", "afternoon", "evening", "night"]
            slot_name = ps.get("slot_name", "morning")
            insert_after = -1
            for i, s in enumerate(reordered):
                if s.get("slot_name") in target_slot_order[
                    :target_slot_order.index(slot_name) + 1
                ]:
                    insert_after = i
            reordered.insert(insert_after + 1, ps)

    itinerary = other_stops + reordered
    itinerary = _recompute_day(itinerary, body.day)

    await update_itinerary(uid, itinerary_id, {"itinerary": itinerary})
    logger.info(f"[customize] reorder: day {body.day} → {body.new_order_place_names}")
    return {"success": True, "itinerary": itinerary}


# ─────────────────────────────────────────────────────────────
# POST /api/customize/{itinerary_id}/pin
# ─────────────────────────────────────────────────────────────
@router.post("/{itinerary_id}/pin")
async def pin_stop(
    itinerary_id: str,
    body: PinRequest,
    user_id: str = Header(None, alias="X-User-Id"),
):
    """
    Toggle the pinned flag on a stop.
    Pinned stops cannot be moved by reorder or future replanning.
    """
    uid = _get_user_id(user_id)
    doc = await get_itinerary_by_id_for_user(uid, itinerary_id)
    if not doc:
        raise HTTPException(404, "Itinerary not found")

    itinerary: List[Dict] = list(doc.get("itinerary", []))
    canon = _canonical(body.place_name)
    found = False
    for s in itinerary:
        if s.get("day") == body.day and _canonical(s.get("place_name", "")) == canon:
            s["pinned"] = body.pinned
            found = True
            break

    if not found:
        raise HTTPException(404, f"Stop '{body.place_name}' not found on day {body.day}")

    await update_itinerary(uid, itinerary_id, {"itinerary": itinerary})
    action = "pinned" if body.pinned else "unpinned"
    logger.info(f"[customize] {action}: '{body.place_name}' on day {body.day}")
    return {"success": True, "place_name": body.place_name, "pinned": body.pinned}


# ─────────────────────────────────────────────────────────────
# PATCH /api/customize/{itinerary_id}/settings
# ─────────────────────────────────────────────────────────────
@router.patch("/{itinerary_id}/settings")
async def patch_settings(
    itinerary_id: str,
    settings: SettingsPatch,
    user_id: str = Header(None, alias="X-User-Id"),
):
    """
    Partially update trip-level preferences.
    Only fields explicitly set in the request body are written.
    Does NOT re-run the scheduler — this is a metadata-only update.
    (Attach a POST /generate with updated preferences to re-plan.)
    """
    uid = _get_user_id(user_id)
    doc = await get_itinerary_by_id_for_user(uid, itinerary_id)
    if not doc:
        raise HTTPException(404, "Itinerary not found")

    updates = settings.model_dump(exclude_unset=True)  # only provided fields
    if not updates:
        raise HTTPException(400, "No settings provided")

    meta = dict(doc.get("meta", {}))
    overrides = dict(meta.get("overrides", {}))
    overrides.update(updates)
    meta["overrides"] = overrides

    await update_itinerary(uid, itinerary_id, {"meta": meta})
    logger.info(f"[customize] settings patch for {itinerary_id}: {updates}")
    return {"success": True, "applied": updates, "meta": meta}
