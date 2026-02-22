import math
import re
import logging
from datetime import date
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Slot template
# ─────────────────────────────────────────────────────────────────────────────
SLOT_TEMPLATE = [
    {
        "slot_name": "morning",
        "start": "09:00",
        "end": "12:00",
        "available_mins": 180,
        "meal_gap_after": "12:00 \u2013 13:00 (\U0001f37d\ufe0f Lunch gap \u2014 add your own)"
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
        "meal_gap_after": "19:30 \u2013 20:00 (\U0001f37d\ufe0f Dinner gap \u2014 add your own)"
    },
    {
        "slot_name": "night",
        "start": "20:00",
        "end": "21:30",
        "available_mins": 90,
        "meal_gap_after": None
    },
]

TRAVEL_SPEED_KMPH   = 20
TRAVEL_BUFFER_MINS  = 10
DEFAULT_TRAVEL_MINS = 15


# ─────────────────────────────────────────────────────────────────────────────
# Opening hours parser
# Handles: 24/7, "9:00 AM - 6:00 PM", "09:00-18:00",
#          "Tue-Sun 9 AM to 6 PM", OSM "Mo-Fr 08:00-17:00; Sa off"
# ─────────────────────────────────────────────────────────────────────────────

_OSM_ABBR   = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
_OSM_TO_IDX = {d: i for i, d in enumerate(_OSM_ABBR)}

_DAY_NAME_TO_IDX = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}


def _hm_to_mins(hm: str) -> int:
    h, m = hm.strip().split(":")
    return int(h) * 60 + int(m)


def _mins_to_time(mins: int) -> str:
    return f"{mins // 60:02d}:{mins % 60:02d}"


def _ampm_to_mins(s: str) -> Optional[int]:
    s = s.strip().lower().replace(".", "").replace(" ", "")
    m = re.match(r"^(\d{1,2})(?::(\d{2}))?(am|pm)$", s)
    if not m:
        return None
    hh, mm, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    hh = (0 if hh == 12 else hh) if ap == "am" else (hh if hh == 12 else hh + 12)
    return hh * 60 + mm


def _parse_time_str(s: str) -> Optional[int]:
    s = s.strip()
    if re.match(r"^\d{1,2}:\d{2}$", s):
        return _hm_to_mins(s)
    if re.search(r"(am|pm)", s, re.IGNORECASE):
        return _ampm_to_mins(s)
    return None


def _parse_daily_range(opening_hours: str) -> Optional[Tuple[int, int]]:
    s = opening_hours.strip()
    if re.search(r"(24/7|open\s*24|always\s*open)", s, re.IGNORECASE):
        return (0, 24 * 60)
    tokens = re.findall(r"\d{1,2}(?::\d{2})?\s*(?:am|pm)|\d{2}:\d{2}", s, re.IGNORECASE)
    if len(tokens) < 2:
        return None
    t1 = _parse_time_str(tokens[0].strip())
    t2 = _parse_time_str(tokens[-1].strip())
    if t1 is None or t2 is None:
        return None
    if t2 < t1:
        t2 += 24 * 60
    return (t1, t2)


def _expand_osm_days(token: str) -> List[int]:
    token = token.strip()
    if "," in token:
        out: List[int] = []
        for t in token.split(","):
            out.extend(_expand_osm_days(t.strip()))
        return out
    if "-" in token:
        a, b = [x.strip() for x in token.split("-", 1)]
        if a in _OSM_TO_IDX and b in _OSM_TO_IDX:
            ia, ib = _OSM_TO_IDX[a], _OSM_TO_IDX[b]
            return list(range(ia, ib + 1)) if ia <= ib else list(range(ia, 7)) + list(range(0, ib + 1))
        return []
    return [_OSM_TO_IDX[token]] if token in _OSM_TO_IDX else []


def _parse_osm_schedule(field: str) -> Optional[Dict[int, List[Tuple[int, int]]]]:
    field = (field or "").strip()
    if not field or not re.search(r"\b(Mo|Tu|We|Th|Fr|Sa|Su)\b", field):
        return None

    out: Dict[int, List[Tuple[int, int]]] = {i: [] for i in range(7)}
    any_rule = False

    for part in [p.strip() for p in field.split(";") if p.strip()]:
        m = re.match(r"^([A-Za-z,\-]+)\s+(.+)$", part)
        if not m:
            continue
        days_str, times_str = m.group(1).strip(), m.group(2).strip()
        days = _expand_osm_days(days_str)
        if not days:
            continue
        if times_str.lower() == "off":
            any_rule = True
            for d in days:
                out[d] = []
            continue
        intervals: List[Tuple[int, int]] = []
        for t in times_str.split(","):
            t = t.strip()
            tm = re.match(r"^(\d{1,2}:\d{2})-(\d{1,2}:\d{2})$", t)
            if not tm:
                continue
            o, c = _hm_to_mins(tm.group(1)), _hm_to_mins(tm.group(2))
            if c < o:
                c += 24 * 60
            intervals.append((o, c))
        if intervals:
            any_rule = True
            for d in days:
                out[d].extend(intervals)

    return out if any_rule else None


def is_open_for_slot(
    opening_hours: Optional[str],
    closed_on: Optional[List[str]],
    slot_start: str,
    slot_end: str,
    duration_mins: int,
    slot_date: Optional[date] = None,
) -> Tuple[Optional[bool], str]:
    """
    Check whether a place can be visited during a slot.

    Valid when:
        visit_start = max(slot_start, place_open)
        visit_end   = visit_start + duration_mins
        visit_end  <= min(slot_end,  place_close)

    Returns:
        (True,  "")       confirmed open and fits
        (False, reason)   confirmed closed / won't fit  → skip slot
        (None,  reason)   unknown / unparseable          → schedule with warning
    """
    s_mins = _hm_to_mins(slot_start)
    e_mins = _hm_to_mins(slot_end)

    if closed_on and slot_date:
        wd = slot_date.weekday()
        for day_str in closed_on:
            idx = _DAY_NAME_TO_IDX.get(day_str.strip().lower())
            if idx is not None and idx == wd:
                return (False, f"Closed on {day_str}")

    if not opening_hours:
        return (None, "No opening hours data")

    low = opening_hours.lower().replace(" ", "")
    if low in ("24/7", "open24hrs", "open24hours", "open24h", "alwaysopen", "open24"):
        return (True, "")

    daily = _parse_daily_range(opening_hours)
    if daily:
        o, c = daily
        visit_start = max(s_mins, o)
        visit_end   = visit_start + duration_mins
        if visit_end <= min(e_mins, c):
            return (True, "")
        return (
            False,
            f"Hours {_mins_to_time(o)}-{_mins_to_time(c % (24*60))}: "
            f"slot {slot_start}-{slot_end}, need {duration_mins}min"
        )

    osm = _parse_osm_schedule(opening_hours)
    if osm is not None:
        if slot_date is None:
            return (None, "Travel date unknown — weekday hours unverifiable")
        wd = slot_date.weekday()
        intervals = osm.get(wd, [])
        if not intervals:
            return (False, f"Closed on {slot_date.strftime('%A')}")
        for o, c in intervals:
            visit_start = max(s_mins, o)
            visit_end   = visit_start + duration_mins
            if visit_end <= min(e_mins, c):
                return (True, "")
        return (
            False,
            f"No interval on {slot_date.strftime('%A')} fits "
            f"{slot_start}-{slot_end} ({duration_mins}min)"
        )

    return (None, f"Unparseable hours: {opening_hours!r:.60}")


# ─────────────────────────────────────────────────────────────────────────────
# Slot builder
# ─────────────────────────────────────────────────────────────────────────────
def build_day_slots(days: int) -> List[Dict]:
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
                "remaining_mins": tmpl["available_mins"],
                "meal_gap_after": tmpl.get("meal_gap_after"),
                "stops":          [],
                "last_lat":       None,
                "last_lon":       None,
            })
    return slots


# ─────────────────────────────────────────────────────────────────────────────
# Travel estimation
# ─────────────────────────────────────────────────────────────────────────────
def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def estimate_travel_minutes(lat1, lon1, lat2, lon2) -> int:
    if None in (lat1, lon1, lat2, lon2):
        return DEFAULT_TRAVEL_MINS
    return max(5, int((haversine_km(lat1, lon1, lat2, lon2) / TRAVEL_SPEED_KMPH) * 60 + TRAVEL_BUFFER_MINS))


def _actual_travel(slot: dict, travel_mins: int) -> int:
    """First stop in a slot has zero travel overhead."""
    return 0 if not slot["stops"] else travel_mins


def _fits_in_slot(candidate: dict, slot: dict, travel_mins: int) -> bool:
    needed = _actual_travel(slot, travel_mins) + int(candidate.get("duration_hrs", 1.0) * 60)
    return needed <= slot["remaining_mins"]


def _compute_stop_times(slot: dict, travel_mins: int, duration_hrs: float) -> Tuple[str, str]:
    base_mins  = _hm_to_mins(slot["start_time"])
    used_mins  = slot["available_mins"] - slot["remaining_mins"]
    actual_t   = _actual_travel(slot, travel_mins)
    start_mins = base_mins + used_mins + actual_t
    return _mins_to_time(start_mins), _mins_to_time(start_mins + int(duration_hrs * 60))


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers — build + commit a stop
# ─────────────────────────────────────────────────────────────────────────────
def _build_stop(cand: dict, day: int, slot: dict, travel_mins: int,
                duration_hrs: float, open_ok: Optional[bool]) -> dict:
    """Construct the full stop dict from candidate + slot placement info."""
    actual_t        = _actual_travel(slot, travel_mins)
    start_t, end_t  = _compute_stop_times(slot, travel_mins, duration_hrs)
    return {
        "day":                    day,
        "slot_id":                slot["slot_id"],
        "slot_name":              slot["slot_name"],
        "start_time":             start_t,
        "end_time":               end_t,
        "travel_mins_from_prev":  actual_t,
        "place_name":             cand.get("place_name", "Unknown"),
        "category":               cand.get("category"),
        "priority":               int(cand.get("priority", 3)),
        "why_must_visit":         cand.get("why_must_visit"),
        "is_alternate":           bool(cand.get("is_alternate", False)),
        "opening_hours":          cand.get("opening_hours"),
        "closed_on":              cand.get("closed_on") or None,
        "duration_hrs":           duration_hrs,
        "entry_fee":              cand.get("entry_fee"),
        "entry_fee_foreign":      cand.get("entry_fee_foreign"),
        "tip":                    cand.get("tip"),
        "nearby_food":            cand.get("nearby_food"),
        "lat":                    cand.get("lat"),
        "lon":                    cand.get("lon"),
        "opening_hours_unverified": open_ok is None,
    }


def _commit_stop(slot: dict, stop: dict) -> None:
    """Append a stop to a slot and update its mutable state."""
    actual_t      = stop["travel_mins_from_prev"]
    duration_mins = int(float(stop["duration_hrs"]) * 60)
    slot["remaining_mins"] -= (actual_t + duration_mins)
    slot["stops"].append(stop)
    if stop.get("lat") and stop.get("lon"):
        slot["last_lat"] = stop["lat"]
        slot["last_lon"] = stop["lon"]


def _uncommit_last_stop(slot: dict) -> Optional[dict]:
    """
    Remove the most-recently-placed stop from a slot and restore slot state.
    Returns the removed stop, or None if slot is empty.

    Only the LAST stop is removed — avoids cascade recalculation for earlier stops.
    Restoration is exact because travel_mins_from_prev is stored on the stop itself.
    """
    if not slot["stops"]:
        return None

    stop          = slot["stops"].pop()
    actual_t      = stop["travel_mins_from_prev"]
    duration_mins = int(float(stop["duration_hrs"]) * 60)
    slot["remaining_mins"] += (actual_t + duration_mins)

    # Restore last_lat/lon to the new last stop (or None if slot is now empty)
    if slot["stops"]:
        prev = slot["stops"][-1]
        slot["last_lat"] = prev.get("lat")
        slot["last_lon"] = prev.get("lon")
    else:
        slot["last_lat"] = None
        slot["last_lon"] = None

    return stop


# ─────────────────────────────────────────────────────────────────────────────
# Bump rule helpers
# ─────────────────────────────────────────────────────────────────────────────
def _can_fit_after_bump(
    cand: dict,
    slot: dict,
    slot_date: Optional[date],
    duration_mins: int,
) -> bool:
    """
    Simulate: if the last stop were removed from `slot`, would `cand` fit?
    Does NOT modify the slot.
    """
    if not slot["stops"]:
        return False

    last_stop   = slot["stops"][-1]
    reclaim     = last_stop["travel_mins_from_prev"] + int(float(last_stop["duration_hrs"]) * 60)
    sim_remain  = slot["remaining_mins"] + reclaim

    # Simulated last position after removing the last stop
    if len(slot["stops"]) >= 2:
        sim_lat = slot["stops"][-2].get("lat")
        sim_lon = slot["stops"][-2].get("lon")
    else:
        sim_lat = sim_lon = None

    travel_to_cand = estimate_travel_minutes(sim_lat, sim_lon, cand.get("lat"), cand.get("lon"))
    # After bump, slot may be empty → no travel overhead
    actual_travel  = 0 if len(slot["stops"]) == 1 else travel_to_cand
    needed         = actual_travel + duration_mins

    if needed > sim_remain:
        return False

    # Check opening hours for cand in this slot
    open_ok, _ = is_open_for_slot(
        cand.get("opening_hours"), cand.get("closed_on") or [],
        slot["start_time"], slot["end_time"],
        duration_mins, slot_date
    )
    return open_ok is not False   # True or None (lenient) both OK


def _try_replace_bumped(
    bumped_stop: dict,
    all_slots: List[dict],
    skip_slot_id: str,
    day_dates: Optional[Dict[int, date]],
    scheduled: list,
) -> bool:
    """
    Try to re-place a bumped stop in any other available slot.
    Modifies `all_slots` and `scheduled` in-place.
    Returns True if successfully re-placed.
    """
    b_dur  = float(bumped_stop.get("duration_hrs", 1.0))
    b_dmin = int(b_dur * 60)

    for slot in all_slots:
        if slot["slot_id"] == skip_slot_id:
            continue

        re_day      = slot["day"]
        slot_date   = day_dates.get(re_day) if day_dates else None
        travel_mins = estimate_travel_minutes(
            slot.get("last_lat"), slot.get("last_lon"),
            bumped_stop.get("lat"), bumped_stop.get("lon")
        )

        open_ok, _ = is_open_for_slot(
            bumped_stop.get("opening_hours"), bumped_stop.get("closed_on") or [],
            slot["start_time"], slot["end_time"],
            b_dmin, slot_date
        )
        if open_ok is False:
            continue

        actual_t = _actual_travel(slot, travel_mins)
        if (actual_t + b_dmin) > slot["remaining_mins"]:
            continue

        # Re-place bumped stop in this new slot
        re_stop = _build_stop(bumped_stop, re_day, slot, travel_mins, b_dur, open_ok)
        _commit_stop(slot, re_stop)
        scheduled.append(re_stop)
        logger.info(
            f"[bump] \u21aa re-placed '{bumped_stop.get('place_name')}' "
            f"\u2192 {slot['slot_id']} (was bumped out)"
        )
        return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Main greedy scheduler  +  bump pass
# ─────────────────────────────────────────────────────────────────────────────
def schedule_candidates(
    candidates: list,
    slots: list,
    avoid_crowded: bool = False,
    accessibility_needs: bool = False,
    day_dates: Optional[Dict[int, date]] = None,
) -> Tuple[list, list]:
    """
    Two-pass greedy scheduler.

    PASS 1 — Standard greedy (priority-ordered):
        Try the candidate's preferred slot first, then every other slot
        across all days.  Check opening hours + duration fit.

    PASS 2 — Bump rule (only when Pass 1 fails):
        Collect every slot whose LAST stop has a strictly lower priority
        than the incoming candidate.  Sort by that stop's priority ASC
        (bump the least-important stop first).
        For each candidate slot:
          a. Simulate removing its last stop.
          b. Check whether the incoming candidate fits after the removal.
          c. If yes:
               - _uncommit_last_stop  →  frees slot capacity
               - Remove bumped stop from `scheduled`
               - Place incoming candidate in the freed slot
               - Try to re-place the bumped stop anywhere else (_try_replace_bumped)
               - If it can't be re-placed → add to unscheduled for Groq alternate
               - Stop (placed = True)

    A candidate reaches `unscheduled` only if both passes fail entirely.
    """
    sorted_cands = sorted(candidates, key=lambda c: -int(c.get("priority", 3)))

    slot_map   = {s["slot_id"]: s for s in slots}
    days       = sorted(set(s["day"] for s in slots))
    day_slots  = {d: [s["slot_id"] for s in slots if s["day"] == d] for d in days}

    scheduled:   list = []
    unscheduled: list = []

    for cand in sorted_cands:
        placed        = False
        pref_slot     = cand.get("best_slot", "morning").lower()
        cand_lat      = cand.get("lat")
        cand_lon      = cand.get("lon")
        duration_hrs  = float(cand.get("duration_hrs", 1.0))
        duration_mins = int(duration_hrs * 60)
        oh_str        = cand.get("opening_hours")
        closed_on     = cand.get("closed_on") or []

        # ══════════════════════════════════════════
        # PASS 1 — Standard greedy
        # ══════════════════════════════════════════
        for day in days:
            if placed:
                break
            preferred = [sid for sid in day_slots[day] if pref_slot in sid]
            fallback  = [sid for sid in day_slots[day] if pref_slot not in sid]

            for slot_id in (preferred + fallback):
                slot      = slot_map[slot_id]
                slot_date = day_dates.get(day) if day_dates else None
                t_mins    = estimate_travel_minutes(
                    slot.get("last_lat"), slot.get("last_lon"), cand_lat, cand_lon
                )

                open_ok, open_reason = is_open_for_slot(
                    oh_str, closed_on,
                    slot["start_time"], slot["end_time"],
                    duration_mins, slot_date
                )
                if open_ok is False:
                    logger.debug(f"[sched P1] skip '{cand.get('place_name')}' in {slot_id}: {open_reason}")
                    continue

                if not _fits_in_slot(cand, slot, t_mins):
                    continue

                stop = _build_stop(cand, day, slot, t_mins, duration_hrs, open_ok)
                _commit_stop(slot, stop)
                scheduled.append(stop)
                placed = True
                logger.debug(
                    f"[sched P1] {'✅' if open_ok else '⚠️'} "
                    f"'{cand.get('place_name')}' → {slot_id} @ {stop['start_time']}"
                )
                break

            if placed:
                break

        if placed:
            continue

        # ══════════════════════════════════════════
        # PASS 2 — Bump rule
        # Only attempt if candidate has priority > 1
        # ══════════════════════════════════════════
        cand_priority = int(cand.get("priority", 3))
        if cand_priority <= 1:
            logger.info(f"[sched P2] skip bump for low-priority '{cand.get('place_name')}'")
            unscheduled.append(cand)
            continue

        # Gather slots where last stop has lower priority than this candidate
        bump_targets: List[Tuple[int, str, int]] = []   # (last_stop_priority, slot_id, day)
        for day in days:
            for slot_id in day_slots[day]:
                slot = slot_map[slot_id]
                if not slot["stops"]:
                    continue
                last_p = int(slot["stops"][-1].get("priority", 3))
                if last_p < cand_priority:
                    bump_targets.append((last_p, slot_id, day))

        # Sort: try to bump from the slot with the lowest-priority last stop first
        bump_targets.sort(key=lambda x: x[0])

        for last_p, slot_id, day in bump_targets:
            slot      = slot_map[slot_id]
            slot_date = day_dates.get(day) if day_dates else None

            if not _can_fit_after_bump(cand, slot, slot_date, duration_mins):
                continue

            # ── Perform bump ──
            bumped_stop = _uncommit_last_stop(slot)   # frees slot capacity
            if bumped_stop is None:
                continue

            # Remove bumped stop from the scheduled list
            for i, s in enumerate(scheduled):
                if s is bumped_stop:
                    scheduled.pop(i)
                    break

            # Place the incoming candidate in the freed slot
            t_mins    = estimate_travel_minutes(
                slot.get("last_lat"), slot.get("last_lon"), cand_lat, cand_lon
            )
            open_ok, _ = is_open_for_slot(
                oh_str, closed_on,
                slot["start_time"], slot["end_time"],
                duration_mins, slot_date
            )
            stop = _build_stop(cand, day, slot, t_mins, duration_hrs, open_ok)
            _commit_stop(slot, stop)
            scheduled.append(stop)
            placed = True

            logger.info(
                f"[sched P2] \U0001f504 bumped '{bumped_stop.get('place_name')}' "
                f"(p={last_p}) out of {slot_id} "
                f"to fit '{cand.get('place_name')}' (p={cand_priority})"
            )

            # ── Try to re-place the bumped stop elsewhere ──
            re_placed = _try_replace_bumped(
                bumped_stop, slots, skip_slot_id=slot_id,
                day_dates=day_dates, scheduled=scheduled
            )
            if not re_placed:
                logger.info(
                    f"[sched P2] '{bumped_stop.get('place_name')}' "
                    f"could not be re-placed → unscheduled"
                )
                unscheduled.append(bumped_stop)

            break   # candidate placed, stop trying other bump targets

        if not placed:
            logger.info(
                f"[sched] unscheduled: '{cand.get('place_name')}' "
                f"(p={cand_priority}, dur={duration_hrs}h) — no slot or bump available"
            )
            unscheduled.append(cand)

    logger.info(f"[sched] done: {len(scheduled)} placed, {len(unscheduled)} unscheduled")
    return scheduled, unscheduled
