import math
import logging
from typing import List, Tuple, Dict, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Slot template — 4 slots per day with meal gap markers
# Meal gaps are NOT slots — they appear between slots in the response
# so the user can manually insert their own meal preferences.
# ─────────────────────────────────────────────────────────────
SLOT_TEMPLATE = [
    {
        "slot_name": "morning",
        "start": "09:00",
        "end": "12:00",
        "available_mins": 180,
        "meal_gap_after": "12:00 – 13:00 (🍽️ Lunch gap — add your own)"
    },
    {
        "slot_name": "afternoon",
        "start": "13:00",
        "end": "16:00",
        "available_mins": 180,
        "meal_gap_after": None
    },
    {
        "slot_name": "evening",
        "start": "16:30",
        "end": "19:30",
        "available_mins": 180,
        "meal_gap_after": "19:30 – 20:00 (🍽️ Dinner gap — add your own)"
    },
    {
        "slot_name": "night",
        "start": "20:00",
        "end": "21:30",
        "available_mins": 90,
        "meal_gap_after": None
    },
]

TRAVEL_SPEED_KMPH  = 20   # Avg city travel speed
TRAVEL_BUFFER_MINS = 10   # Parking / auto / walking buffer
DEFAULT_TRAVEL_MINS = 15  # Used when coords are missing


# ─────────────────────────────────────────────────────────────
def build_day_slots(days: int) -> List[Dict]:
    """
    Returns a flat list of all time slots across all days.
    Each slot carries mutable scheduling state (remaining_mins, stops, last_lat/lon).
    """
    slots = []
    for day in range(1, days + 1):
        for tmpl in SLOT_TEMPLATE:
            slots.append({
                "slot_id":        f"day{day}_{tmpl['slot_name']}",
                "day":            day,
                "slot_name":      tmpl["slot_name"],
                "start_time":     tmpl["start"],
                "end_time":       tmpl["end"],
                "available_mins": tmpl["available_mins"],
                "remaining_mins": tmpl["available_mins"],  # decremented as stops are placed
                "meal_gap_after": tmpl.get("meal_gap_after"),
                "stops":          [],      # placed stops (internal use)
                "last_lat":       None,    # last stop lat (for travel calc)
                "last_lon":       None,    # last stop lon
            })
    return slots


# ─────────────────────────────────────────────────────────────
def _time_to_mins(t: str) -> int:
    """'09:30' → 570"""
    h, m = map(int, t.split(":"))
    return h * 60 + m


def _mins_to_time(mins: int) -> str:
    """570 → '09:30'"""
    h = mins // 60
    m = mins % 60
    return f"{h:02d}:{m:02d}"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Straight-line distance in km between two GPS coordinates."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def estimate_travel_minutes(lat1, lon1, lat2, lon2) -> int:
    """Realistic one-way travel time between two points (minutes)."""
    if None in (lat1, lon1, lat2, lon2):
        return DEFAULT_TRAVEL_MINS
    dist_km = haversine_km(lat1, lon1, lat2, lon2)
    drive_mins = (dist_km / TRAVEL_SPEED_KMPH) * 60
    return max(5, int(drive_mins + TRAVEL_BUFFER_MINS))


def _actual_travel(slot: dict, travel_mins: int) -> int:
    """No travel overhead for the very first stop in a slot."""
    return 0 if not slot["stops"] else travel_mins


def _fits_in_slot(candidate: dict, slot: dict, travel_mins: int) -> bool:
    """Check if (travel + duration) fits within slot's remaining time."""
    needed = _actual_travel(slot, travel_mins) + int(candidate.get("duration_hrs", 1.0) * 60)
    return needed <= slot["remaining_mins"]


def _compute_stop_times(slot: dict, travel_mins: int, duration_hrs: float) -> Tuple[str, str]:
    """Compute actual start and end time of a stop within a slot."""
    base_mins    = _time_to_mins(slot["start_time"])
    used_mins    = slot["available_mins"] - slot["remaining_mins"]
    actual_t     = _actual_travel(slot, travel_mins)
    start_mins   = base_mins + used_mins + actual_t
    end_mins     = start_mins + int(duration_hrs * 60)
    return _mins_to_time(start_mins), _mins_to_time(end_mins)


# ─────────────────────────────────────────────────────────────
def schedule_candidates(
    candidates: list,
    slots: list,
    avoid_crowded: bool = False,
    accessibility_needs: bool = False,
) -> Tuple[list, list]:
    """
    Greedy scheduler: fits place candidates into time slots.

    Algorithm:
    1. Sort candidates by priority desc (highest priority placed first).
    2. For each candidate, try preferred slot first (best_slot hint from Groq).
    3. If preferred slot is full, try remaining slots across all days.
    4. Compute travel time from last placed stop in the slot.
    5. If it fits: place it, update slot state.
    6. If nothing fits anywhere: mark as unscheduled.

    Returns:
        scheduled   — list of stop dicts with day/slot/timing fields
        unscheduled — list of candidate dicts that couldn't be placed
    """
    sorted_cands = sorted(candidates, key=lambda c: -int(c.get("priority", 3)))

    slot_map  = {s["slot_id"]: s for s in slots}
    days      = sorted(set(s["day"] for s in slots))
    day_slots = {
        d: [s["slot_id"] for s in slots if s["day"] == d]
        for d in days
    }

    scheduled   = []
    unscheduled = []

    for cand in sorted_cands:
        placed          = False
        pref_slot       = cand.get("best_slot", "morning").lower()
        cand_lat        = cand.get("lat")
        cand_lon        = cand.get("lon")
        duration_hrs    = float(cand.get("duration_hrs", 1.0))

        for day in days:
            if placed:
                break

            all_slots  = day_slots[day]
            preferred  = [sid for sid in all_slots if pref_slot in sid]
            fallback   = [sid for sid in all_slots if pref_slot not in sid]
            trial_order = preferred + fallback

            for slot_id in trial_order:
                slot        = slot_map[slot_id]
                last_lat    = slot.get("last_lat")
                last_lon    = slot.get("last_lon")
                travel_mins = estimate_travel_minutes(last_lat, last_lon, cand_lat, cand_lon)

                if not _fits_in_slot(cand, slot, travel_mins):
                    continue

                start_t, end_t = _compute_stop_times(slot, travel_mins, duration_hrs)
                actual_t       = _actual_travel(slot, travel_mins)

                stop = {
                    "day":                  day,
                    "slot_id":              slot_id,
                    "slot_name":            slot["slot_name"],
                    "start_time":           start_t,
                    "end_time":             end_t,
                    "place_name":           cand.get("place_name", "Unknown"),
                    "category":             cand.get("category"),
                    "priority":             int(cand.get("priority", 3)),
                    "duration_hrs":         duration_hrs,
                    "travel_mins_from_prev": actual_t,
                    "entry_fee":            cand.get("entry_fee"),
                    "tip":                  cand.get("tip"),
                    "lat":                  cand_lat,
                    "lon":                  cand_lon,
                    "why_must_visit":       cand.get("why_must_visit"),
                    "opening_hours":        cand.get("opening_hours"),
                    "closed_on":            cand.get("closed_on"),
                    "is_alternate":         bool(cand.get("is_alternate", False)),
                }

                # Commit to slot
                slot["remaining_mins"] -= (actual_t + int(duration_hrs * 60))
                slot["stops"].append(stop)
                if cand_lat and cand_lon:
                    slot["last_lat"] = cand_lat
                    slot["last_lon"] = cand_lon

                scheduled.append(stop)
                placed = True
                logger.debug(f"Placed '{cand.get('place_name')}' → {slot_id} @ {start_t}")
                break

            if placed:
                break

        if not placed:
            logger.info(f"Unscheduled: '{cand.get('place_name')}' (dur={duration_hrs}h, pref={pref_slot})")
            unscheduled.append(cand)

    logger.info(f"Scheduler: {len(scheduled)} placed, {len(unscheduled)} unscheduled")
    return scheduled, unscheduled
