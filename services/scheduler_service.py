import math
import re
import logging
from datetime import date
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Slot template
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
#   • 24/7, "9:00 AM - 6:00 PM", "09:00-18:00"
#   • Multi-session: "5:30 AM - 12:00 PM, 4:00 PM - 8:00 PM"
#   • Day-prefixed: "Tue-Sun 9 AM to 6 PM"
#   • OSM: "Mo-Fr 08:00-17:00; Sa off"
# ─────────────────────────────────────────────────────────────

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

# Regex for a single time token: "9:00 AM", "09:00", "6 PM"
_TIME_PAT = r"\d{1,2}(?::\d{2})?\s*(?:am|pm)|\d{2}:\d{2}"


def _hm_to_mins(hm: str) -> int:
    h, m = hm.strip().split(":")
    return int(h) * 60 + int(m)


def _mins_to_time(mins: int) -> str:
    h = mins // 60
    m = mins % 60
    return f"{h:02d}:{m:02d}"


def _ampm_to_mins(s: str) -> Optional[int]:
    s = s.strip().lower().replace(".", "").replace(" ", "")
    m = re.match(r"^(\d{1,2})(?::(\d{2}))?(am|pm)$", s)
    if not m:
        return None
    hh, mm, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if ap == "am":
        hh = 0 if hh == 12 else hh
    else:
        hh = hh if hh == 12 else hh + 12
    return hh * 60 + mm


def _parse_time_str(s: str) -> Optional[int]:
    s = s.strip()
    if re.match(r"^\d{1,2}:\d{2}$", s):
        return _hm_to_mins(s)
    if re.search(r"(am|pm)", s, re.IGNORECASE):
        return _ampm_to_mins(s)
    return None


def _parse_sessions(opening_hours: str) -> List[Tuple[int, int]]:
    """
    Parse opening_hours into a list of (open_mins, close_mins) sessions.

    Handles both single-session and multi-session (comma-separated) formats:
      • "9:00 AM - 6:00 PM"                  → [(540, 1080)]
      • "5:30 AM - 12:00 PM, 4:00 PM - 8:00 PM" → [(330, 720), (960, 1200)]
      • "09:00-18:00"                         → [(540, 1080)]

    Returns [] if nothing parseable is found.
    """
    s = opening_hours.strip()
    if re.search(r"(24/7|open\s*24|always\s*open)", s, re.IGNORECASE):
        return [(0, 24 * 60)]

    sessions: List[Tuple[int, int]] = []

    # Split on comma or semicolon, then try to extract a pair of time tokens
    # from each chunk.  This handles "X AM - Y PM, A PM - B PM" correctly.
    chunks = re.split(r",|;", s)
    for chunk in chunks:
        chunk = chunk.strip()
        tokens = re.findall(_TIME_PAT, chunk, re.IGNORECASE)
        if len(tokens) < 2:
            continue
        t1 = _parse_time_str(tokens[0].strip())
        t2 = _parse_time_str(tokens[-1].strip())
        if t1 is None or t2 is None:
            continue
        if t2 <= t1:          # overnight wrap (e.g. 10 PM – 2 AM)
            t2 += 24 * 60
        sessions.append((t1, t2))

    return sessions


def _parse_daily_range(opening_hours: str) -> Optional[Tuple[int, int]]:
    """
    Legacy single-range helper kept for validate_candidate_slots / _first_valid_slot.
    Returns the UNION envelope (earliest open, latest close) across all sessions.
    For multi-session hours callers should use _parse_sessions() directly.
    """
    sessions = _parse_sessions(opening_hours)
    if not sessions:
        return None
    return (min(o for o, _ in sessions), max(c for _, c in sessions))


def _expand_osm_days(token: str) -> List[int]:
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
    field = (field or "").strip()
    if not field:
        return None
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
            o = _hm_to_mins(tm.group(1))
            c = _hm_to_mins(tm.group(2))
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
) -> Tuple[Optional[bool], str, Optional[int]]:
    """
    Check whether a place can be visited during a slot.

    Returns a 3-tuple:
        (open_ok, reason, clamped_start_mins)

    open_ok:
        True   — confirmed open and duration fits
        False  — confirmed closed / won't fit  → skip slot
        None   — unknown / unparseable          → schedule with warning

    clamped_start_mins:
        The actual minute-of-day when the visit can BEGIN, accounting for
        when the place opens (max(slot_start, place_open)).  None when
        open_ok is not True.
    """
    s_mins = _hm_to_mins(slot_start)
    e_mins = _hm_to_mins(slot_end)

    # 1. closed_on list
    if closed_on and slot_date:
        wd = slot_date.weekday()
        for day_str in closed_on:
            idx = _DAY_NAME_TO_IDX.get(day_str.strip().lower())
            if idx is not None and idx == wd:
                return (False, f"Closed on {day_str}", None)

    if not opening_hours:
        return (None, "No opening hours data", None)

    low = opening_hours.lower().replace(" ", "")
    if low in ("24/7", "open24hrs", "open24hours", "open24h", "alwaysopen", "open24"):
        return (True, "", s_mins)

    # 2. Multi-session daily schedule
    sessions = _parse_sessions(opening_hours)
    if sessions:
        for o, c in sessions:
            visit_start = max(s_mins, o)
            visit_end   = visit_start + duration_mins
            if visit_end <= min(e_mins, c):
                return (True, "", visit_start)
        # Build a human-readable description of all sessions
        sess_str = ", ".join(
            f"{_mins_to_time(o)}-{_mins_to_time(c % (24*60))}" for o, c in sessions
        )
        return (
            False,
            f"Hours {sess_str}: slot {slot_start}-{slot_end}, need {duration_mins}min",
            None,
        )

    # 3. OSM weekday schedule
    osm = _parse_osm_schedule(opening_hours)
    if osm is not None:
        if slot_date is None:
            return (None, "Travel date unknown — weekday hours unverifiable", None)
        wd = slot_date.weekday()
        intervals = osm.get(wd, [])
        if not intervals:
            return (False, f"Closed on {slot_date.strftime('%A')}", None)
        for o, c in intervals:
            visit_start = max(s_mins, o)
            visit_end   = visit_start + duration_mins
            if visit_end <= min(e_mins, c):
                return (True, "", visit_start)
        return (
            False,
            f"No interval on {slot_date.strftime('%A')} fits "
            f"{slot_start}-{slot_end} ({duration_mins}min)",
            None,
        )

    return (None, f"Unparseable hours: {opening_hours!r:.60}", None)


# ─────────────────────────────────────────────────────────────
# Pre-schedule: validate & auto-correct best_slot vs opening hours
# ─────────────────────────────────────────────────────────────

_SLOT_WINDOWS = {
    "morning":   (_hm_to_mins("09:00"), _hm_to_mins("12:00")),
    "afternoon": (_hm_to_mins("13:00"), _hm_to_mins("16:00")),
    "evening":   (_hm_to_mins("16:30"), _hm_to_mins("19:30")),
    "night":     (_hm_to_mins("20:00"), _hm_to_mins("21:30")),
}


def _first_valid_slot(opening_hours: str, duration_mins: int) -> Optional[str]:
    """
    Return the name of the first slot window where the place is open and
    the duration fits.  Uses multi-session-aware _parse_sessions().
    """
    sessions = _parse_sessions(opening_hours)
    if not sessions:
        return None

    slot_order = ["morning", "afternoon", "evening", "night"]
    for slot_name in slot_order:
        s_mins, e_mins = _SLOT_WINDOWS[slot_name]
        for o, c in sessions:
            visit_start = max(s_mins, o)
            visit_end   = visit_start + duration_mins
            if visit_end <= min(e_mins, c):
                return slot_name
    return None


def validate_candidate_slots(candidates: list) -> list:
    """
    Pre-scheduling pass: verify that each candidate's best_slot aligns
    with its actual opening_hours (multi-session aware).

    Outcomes:
      1. No / unparseable hours → keep best_slot as-is.
      2. best_slot valid → no change.
      3. best_slot wrong → auto-correct to first truly-open slot.
      4. No slot fits at all → set hours_conflict=True.
    """
    for cand in candidates:
        name     = cand.get("place_name", "?")
        oh_str   = cand.get("opening_hours") or ""
        pref     = (cand.get("best_slot") or "morning").lower()
        dur_mins = int(float(cand.get("duration_hrs", 1.0)) * 60)

        if not oh_str.strip():
            continue

        low = oh_str.lower().replace(" ", "")
        if low in ("24/7", "open24hrs", "open24hours", "open24h", "alwaysopen", "open24"):
            continue

        sessions = _parse_sessions(oh_str)
        if not sessions:
            logger.debug(f"[validate] '{name}': unparseable hours — keeping best_slot")
            continue

        s_mins, e_mins = _SLOT_WINDOWS.get(pref, _SLOT_WINDOWS["morning"])
        slot_ok = any(
            (max(s_mins, o) + dur_mins) <= min(e_mins, c)
            for o, c in sessions
        )

        if slot_ok:
            logger.debug(f"[validate] '{name}': best_slot='{pref}' ✔️")
            continue

        corrected = _first_valid_slot(oh_str, dur_mins)
        if corrected and corrected != pref:
            sess_str = ", ".join(f"{_mins_to_time(o)}-{_mins_to_time(c%1440)}" for o, c in sessions)
            logger.info(
                f"[validate] '{name}': best_slot '{pref}' ✖ "
                f"(hours: {sess_str}) → corrected to '{corrected}'"
            )
            cand["best_slot"] = corrected
        elif not corrected:
            logger.warning(f"[validate] '{name}': NO slot fits — flagged hours_conflict")
            cand["hours_conflict"] = True

    return candidates


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
    return max(5, int((haversine_km(lat1, lon1, lat2, lon2) / TRAVEL_SPEED_KMPH) * 60 + TRAVEL_BUFFER_MINS))


def _actual_travel(slot: dict, travel_mins: int) -> int:
    return 0 if not slot["stops"] else travel_mins


def _fits_in_slot(candidate: dict, slot: dict, travel_mins: int) -> bool:
    needed = _actual_travel(slot, travel_mins) + int(candidate.get("duration_hrs", 1.0) * 60)
    return needed <= slot["remaining_mins"]


def _compute_stop_times(
    slot: dict,
    travel_mins: int,
    duration_hrs: float,
    clamped_start: Optional[int] = None,
) -> Tuple[str, str]:
    """
    Compute the actual start/end times for a stop.

    clamped_start: minute-of-day returned by is_open_for_slot() —
      the point where the visit can actually begin (max(slot_start, place_open)).
      When provided this overrides the raw slot-position calculation so the
      displayed start_time is never earlier than when the place opens.
    """
    actual_t = _actual_travel(slot, travel_mins)
    if clamped_start is not None:
        start_mins = clamped_start + actual_t
    else:
        base_mins  = _hm_to_mins(slot["start_time"])
        used_mins  = slot["available_mins"] - slot["remaining_mins"]
        start_mins = base_mins + used_mins + actual_t
    end_mins = start_mins + int(duration_hrs * 60)
    return _mins_to_time(start_mins), _mins_to_time(end_mins)


# ─────────────────────────────────────────────────────────────
# Main greedy scheduler
# ─────────────────────────────────────────────────────────────
def schedule_candidates(
    candidates: list,
    slots: list,
    avoid_crowded: bool = False,
    accessibility_needs: bool = False,
    day_dates: Optional[Dict[int, date]] = None,
) -> Tuple[list, list]:
    """
    Greedy scheduler.

    Step 0: validate_candidate_slots() — pre-correct best_slot vs opening hours.
    Step 1: Sort by priority descending.
    Step 2: For each candidate try preferred slot then fallbacks (day by day).
    Step 3: Multi-session opening hours + fit check (is_open_for_slot).
    Step 4: Clamp displayed start_time to when the place actually opens.
    """
    candidates = validate_candidate_slots(candidates)
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

            preferred = [sid for sid in day_slots[day] if pref_slot in sid]
            fallback  = [sid for sid in day_slots[day] if pref_slot not in sid]

            for slot_id in (preferred + fallback):
                slot      = slot_map[slot_id]
                slot_date = day_dates.get(day) if day_dates else None
                travel_mins = estimate_travel_minutes(
                    slot.get("last_lat"), slot.get("last_lon"), cand_lat, cand_lon
                )

                # ─ Multi-session opening hours check ─
                open_ok, open_reason, clamped_start = is_open_for_slot(
                    oh_str, closed_on,
                    slot["start_time"], slot["end_time"],
                    duration_mins, slot_date
                )
                if open_ok is False:
                    logger.debug(
                        f"[sched] skip '{cand.get('place_name')}' in {slot_id}: {open_reason}"
                    )
                    continue

                # ─ Duration + travel fit check ─
                if not _fits_in_slot(cand, slot, travel_mins):
                    continue

                # ─ Compute times — start clamped to place-open if known ─
                start_t, end_t = _compute_stop_times(
                    slot, travel_mins, duration_hrs, clamped_start
                )
                actual_t = _actual_travel(slot, travel_mins)

                stop = {
                    # ─ scheduling metadata ─
                    "day":                      day,
                    "slot_id":                  slot_id,
                    "slot_name":                slot["slot_name"],
                    "start_time":               start_t,
                    "end_time":                 end_t,
                    "travel_mins_from_prev":     actual_t,
                    # ─ place identity ─
                    "place_name":               cand.get("place_name", "Unknown"),
                    "category":                 cand.get("category"),
                    "priority":                 int(cand.get("priority", 3)),
                    "why_must_visit":            cand.get("why_must_visit"),
                    "is_alternate":              bool(cand.get("is_alternate", False)),
                    # ─ from enrich ─
                    "opening_hours":             oh_str,
                    "closed_on":                 closed_on or None,
                    "duration_hrs":              duration_hrs,
                    "entry_fee":                 cand.get("entry_fee"),
                    "entry_fee_foreign":         cand.get("entry_fee_foreign"),
                    "tip":                       cand.get("tip"),
                    "nearby_food":               cand.get("nearby_food"),
                    # ─ from geocode ─
                    "lat":                       cand_lat,
                    "lon":                       cand_lon,
                    # ─ flags ─
                    "opening_hours_unverified":  open_ok is None,
                    "hours_conflict":            bool(cand.get("hours_conflict", False)),
                }

                slot["remaining_mins"] -= (actual_t + duration_mins)
                slot["stops"].append(stop)
                if cand_lat and cand_lon:
                    slot["last_lat"] = cand_lat
                    slot["last_lon"] = cand_lon

                scheduled.append(stop)
                placed = True
                flag = "\u2705" if open_ok else "\u26a0\ufe0f"
                logger.debug(f"[sched] {flag} '{cand.get('place_name')}' → {slot_id} @ {start_t}")
                break

            if placed:
                break

        if not placed:
            logger.info(f"[sched] unscheduled: '{cand.get('place_name')}' (dur={duration_hrs}h)")
            unscheduled.append(cand)

    logger.info(f"[sched] done: {len(scheduled)} placed, {len(unscheduled)} unscheduled")
    return scheduled, unscheduled
