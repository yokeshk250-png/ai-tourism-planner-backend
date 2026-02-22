import math
import re
import logging
from datetime import date
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Slot template  —  4 slots per day, meal gaps between slots
# ─────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────
# Opening hours parser
# Handles:
#   •  24/7 / Open 24hrs
#   •  Simple daily: "9:00 AM - 6:00 PM"  or  "09:00-18:00"
#   •  With day prefix: "Tue-Sun 9:00 AM - 6:00 PM"  (Groq often adds this)
#   •  OSM-style:  "Mo-Fr 08:00-17:00; Sa 10:00-14:00"
# ─────────────────────────────────────────────────────────────

# OSM 2-letter day abbreviations
_OSM_ABBR   = ["Mo", "Tu", "We", "Th", "Fr", "Sa", "Su"]
_OSM_TO_IDX = {d: i for i, d in enumerate(_OSM_ABBR)}

# Full / partial day names used in closed_on lists (Mon=0 .. Sun=6)
_DAY_NAME_TO_IDX = {
    "monday": 0,    "mon": 0,
    "tuesday": 1,   "tue": 1,  "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3,  "thu": 3,  "thur": 3, "thurs": 3,
    "friday": 4,    "fri": 4,
    "saturday": 5,  "sat": 5,
    "sunday": 6,    "sun": 6,
}


def _hm_to_mins(hm: str) -> int:
    """'09:30' → 570"""
    h, m = hm.strip().split(":")
    return int(h) * 60 + int(m)


def _mins_to_time(mins: int) -> str:
    """570 → '09:30'"""
    h = mins // 60
    m = mins % 60
    return f"{h:02d}:{m:02d}"


def _ampm_to_mins(s: str) -> Optional[int]:
    """
    '9:00 AM' / '9 AM' / '12:30pm' → minutes since midnight.
    Returns None if unparseable.
    """
    s = s.strip().lower().replace(".", "").replace(" ", "")
    m = re.match(r"^(\d{1,2})(?::(\d{2}))?(am|pm)$", s)
    if not m:
        return None
    hh = int(m.group(1))
    mm = int(m.group(2) or 0)
    ap = m.group(3)
    if ap == "am":
        hh = 0 if hh == 12 else hh          # 12 AM = midnight
    else:
        hh = hh if hh == 12 else hh + 12    # 12 PM = noon, others +12
    return hh * 60 + mm


def _parse_time_str(s: str) -> Optional[int]:
    """Try 24-h 'HH:MM' first, then AM/PM. Returns None on failure."""
    s = s.strip()
    if re.match(r"^\d{1,2}:\d{2}$", s):
        return _hm_to_mins(s)
    if re.search(r"(am|pm)", s, re.IGNORECASE):
        return _ampm_to_mins(s)
    return None


def _parse_daily_range(opening_hours: str) -> Optional[Tuple[int, int]]:
    """
    Extract the first two time values found in the opening_hours string.
    Works for: '9:00 AM - 6:00 PM', '09:00-18:00',
               'Tuesday to Sunday 9:00 AM to 6:00 PM',
               'Everyday 6:00 AM - 9:00 PM'.
    Returns (open_mins, close_mins) or None.
    """
    s = opening_hours.strip()

    # Quick 24/7 check
    if re.search(r"(24/7|open\s*24|always\s*open)", s, re.IGNORECASE):
        return (0, 24 * 60)

    # Find all time-like tokens anywhere in the string
    time_pat = r"\d{1,2}(?::\d{2})?\s*(?:am|pm)|\d{2}:\d{2}"
    tokens = re.findall(time_pat, s, re.IGNORECASE)

    if len(tokens) < 2:
        return None

    t1 = _parse_time_str(tokens[0].strip())
    t2 = _parse_time_str(tokens[-1].strip())

    if t1 is None or t2 is None:
        return None

    # Over-midnight: e.g. 20:00-02:00
    if t2 < t1:
        t2 += 24 * 60

    return (t1, t2)


def _expand_osm_days(token: str) -> List[int]:
    """
    'Mo-Fr' → [0,1,2,3,4] | 'Sa,Su' → [5,6] | 'Mo' → [0]
    """
    token = token.strip()
    if "," in token:
        out: List[int] = []
        for t in token.split(","):
            out.extend(_expand_osm_days(t.strip()))
        return out
    if "-" in token:
        a, b = token.split("-", 1)
        a, b = a.strip(), b.strip()
        if a in _OSM_TO_IDX and b in _OSM_TO_IDX:
            ia, ib = _OSM_TO_IDX[a], _OSM_TO_IDX[b]
            return list(range(ia, ib + 1)) if ia <= ib else list(range(ia, 7)) + list(range(0, ib + 1))
        return []
    return [_OSM_TO_IDX[token]] if token in _OSM_TO_IDX else []


def _parse_osm_schedule(field: str) -> Optional[Dict[int, List[Tuple[int, int]]]]:
    """
    Parse OSM-style opening_hours string.
    Example: 'Mo-Fr 08:00-17:00; Sa 10:00-14:00; Su off'
    Returns weekday_idx -> [(open_mins, close_mins), ...] or None.
    """
    field = (field or "").strip()
    if not field:
        return None

    # Must contain OSM day abbreviations
    if not re.search(r"\b(Mo|Tu|We|Th|Fr|Sa|Su)\b", field):
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

        # "off" keyword = explicitly closed that day
        if times_str.lower() == "off":
            any_rule = True
            for d in days:
                out[d] = []  # empty = closed
            continue

        # Parse comma-separated HH:MM-HH:MM intervals
        intervals: List[Tuple[int, int]] = []
        for t in times_str.split(","):
            t = t.strip()
            tm = re.match(r"^(\d{1,2}:\d{2})-(\d{1,2}:\d{2})$", t)
            if not tm:
                continue
            o = _hm_to_mins(tm.group(1))
            c = _hm_to_mins(tm.group(2))
            if c < o:
                c += 24 * 60   # over-midnight
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
    Decide whether a place can be visited in a given time slot.

    The visit is valid when:
        visit_start = max(slot_start, place_open)
        visit_end   = visit_start + duration_mins
        visit_end  <= min(slot_end,  place_close)

    Returns:
        (True,  "")             — confirmed open and fits
        (False, reason)         — confirmed closed / won’t fit  → skip slot
        (None,  reason)         — unknown / unparseable         → schedule with warning
    """
    s_mins = _hm_to_mins(slot_start)
    e_mins = _hm_to_mins(slot_end)

    # ─ 1. closed_on list check (requires travel date) ─
    if closed_on and slot_date:
        wd = slot_date.weekday()   # Mon=0 … Sun=6
        for day_str in closed_on:
            idx = _DAY_NAME_TO_IDX.get(day_str.strip().lower())
            if idx is not None and idx == wd:
                return (False, f"Closed on {day_str}")

    if not opening_hours:
        return (None, "No opening hours data")

    # ─ 2. 24/7 shortcut ─
    low = opening_hours.lower().replace(" ", "")
    if low in ("24/7", "open24hrs", "open24hours", "open24h", "alwaysopen", "open24"):
        return (True, "")

    # ─ 3. Simple daily range (AM/PM or HH:MM) ─
    daily = _parse_daily_range(opening_hours)
    if daily:
        o, c = daily
        visit_start = max(s_mins, o)
        visit_end   = visit_start + duration_mins
        if visit_end <= min(e_mins, c):
            return (True, "")
        reason = (
            f"Hours {_mins_to_time(o)}-{_mins_to_time(c % (24*60))}: "
            f"slot {slot_start}-{slot_end}, need {duration_mins}min"
        )
        return (False, reason)

    # ─ 4. OSM weekday schedule ─
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
            f"No interval on {slot_date.strftime('%A')} fits {slot_start}-{slot_end} ({duration_mins}min)"
        )

    # ─ 5. Unparseable — lenient (schedule with warning) ─
    return (None, f"Unparseable hours: {opening_hours!r:.60}")


# ─────────────────────────────────────────────────────────────
# Slot builder
# ─────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────
# Travel estimation
# ─────────────────────────────────────────────────────────────
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
    dist_km = haversine_km(lat1, lon1, lat2, lon2)
    return max(5, int((dist_km / TRAVEL_SPEED_KMPH) * 60 + TRAVEL_BUFFER_MINS))


def _actual_travel(slot: dict, travel_mins: int) -> int:
    """First stop in a slot has zero travel overhead."""
    return 0 if not slot["stops"] else travel_mins


def _fits_in_slot(candidate: dict, slot: dict, travel_mins: int) -> bool:
    needed = _actual_travel(slot, travel_mins) + int(candidate.get("duration_hrs", 1.0) * 60)
    return needed <= slot["remaining_mins"]


def _compute_stop_times(slot: dict, travel_mins: int, duration_hrs: float) -> Tuple[str, str]:
    base_mins   = _hm_to_mins(slot["start_time"])
    used_mins   = slot["available_mins"] - slot["remaining_mins"]
    actual_t    = _actual_travel(slot, travel_mins)
    start_mins  = base_mins + used_mins + actual_t
    end_mins    = start_mins + int(duration_hrs * 60)
    return _mins_to_time(start_mins), _mins_to_time(end_mins)


# ─────────────────────────────────────────────────────────────
# Main greedy scheduler  (with opening hours enforcement)
# ─────────────────────────────────────────────────────────────
def schedule_candidates(
    candidates: list,
    slots: list,
    avoid_crowded: bool = False,
    accessibility_needs: bool = False,
    day_dates: Optional[Dict[int, date]] = None,
) -> Tuple[list, list]:
    """
    Greedy scheduler with opening-hours enforcement.

    For each candidate (sorted by priority desc):
      1. Try preferred slot (best_slot hint) then fallback to others.
      2. Check opening hours — skip slot if closed (lenient if unknown).
      3. Check travel+duration fits remaining slot time.
      4. Place the stop, updating slot state.

    Opening-hours outcomes:
      - True  → confirmed open    → schedule normally.
      - False → confirmed closed   → skip this slot, try next.
      - None  → unknown/unparseable → schedule with opening_hours_unverified=True.

    If nothing fits anywhere → add to unscheduled (triggers Groq alternate request).
    """
    sorted_cands = sorted(candidates, key=lambda c: -int(c.get("priority", 3)))

    slot_map  = {s["slot_id"]: s for s in slots}
    days      = sorted(set(s["day"] for s in slots))
    day_slots = {d: [s["slot_id"] for s in slots if s["day"] == d] for d in days}

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

        for day in days:
            if placed:
                break

            all_slots  = day_slots[day]
            preferred  = [sid for sid in all_slots if pref_slot in sid]
            fallback   = [sid for sid in all_slots if pref_slot not in sid]

            for slot_id in (preferred + fallback):
                slot        = slot_map[slot_id]
                slot_date   = day_dates.get(day) if day_dates else None
                last_lat    = slot.get("last_lat")
                last_lon    = slot.get("last_lon")
                travel_mins = estimate_travel_minutes(last_lat, last_lon, cand_lat, cand_lon)

                # ── Opening hours check ──
                open_ok, open_reason = is_open_for_slot(
                    oh_str, closed_on,
                    slot["start_time"], slot["end_time"],
                    duration_mins, slot_date
                )

                if open_ok is False:
                    logger.debug(
                        f"[sched] skip '{cand.get('place_name')}' in {slot_id}: {open_reason}"
                    )
                    continue   # try next slot / day

                # ── Duration + travel fit check ──
                if not _fits_in_slot(cand, slot, travel_mins):
                    continue

                start_t, end_t = _compute_stop_times(slot, travel_mins, duration_hrs)
                actual_t       = _actual_travel(slot, travel_mins)

                stop = {
                    "day":                    day,
                    "slot_id":                slot_id,
                    "slot_name":              slot["slot_name"],
                    "start_time":             start_t,
                    "end_time":               end_t,
                    "place_name":             cand.get("place_name", "Unknown"),
                    "category":               cand.get("category"),
                    "priority":               int(cand.get("priority", 3)),
                    "duration_hrs":           duration_hrs,
                    "travel_mins_from_prev":  actual_t,
                    "entry_fee":              cand.get("entry_fee"),
                    "tip":                    cand.get("tip"),
                    "lat":                    cand_lat,
                    "lon":                    cand_lon,
                    "why_must_visit":         cand.get("why_must_visit"),
                    "opening_hours":          oh_str,
                    "closed_on":              closed_on or None,
                    "is_alternate":           bool(cand.get("is_alternate", False)),
                    # True if we could not verify hours (lenient pass)
                    "opening_hours_unverified": open_ok is None,
                }

                # Commit to slot
                slot["remaining_mins"] -= (actual_t + duration_mins)
                slot["stops"].append(stop)
                if cand_lat and cand_lon:
                    slot["last_lat"] = cand_lat
                    slot["last_lon"] = cand_lon

                scheduled.append(stop)
                placed = True
                verified = "\u2705" if open_ok else "\u26a0\ufe0f"
                logger.debug(
                    f"[sched] {verified} '{cand.get('place_name')}' → {slot_id} @ {start_t}"
                )
                break

            if placed:
                break

        if not placed:
            logger.info(
                f"[sched] unscheduled: '{cand.get('place_name')}' "
                f"(dur={duration_hrs}h, pref={pref_slot})"
            )
            unscheduled.append(cand)

    logger.info(f"[sched] done: {len(scheduled)} placed, {len(unscheduled)} unscheduled")
    return scheduled, unscheduled
