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

TRAVEL_SPEED_KMPH   = 20
TRAVEL_BUFFER_MINS  = 10
DEFAULT_TRAVEL_MINS = 15

# Sentinel keywords that indicate the LLM embedded reasoning into a place name.
# Any candidate whose place_name contains one of these (case-insensitive) is
# treated as invalid and dropped before scheduling.
_INVALID_NAME_PHRASES = [
    "skipping",
    "not a specific",
    "ambiguous",
    "unknown place",
    "n/a",
]


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
    """
    s = opening_hours.strip()
    if re.search(r"(24/7|open\s*24|always\s*open)", s, re.IGNORECASE):
        return [(0, 24 * 60)]

    sessions: List[Tuple[int, int]] = []
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
        if t2 <= t1:
            t2 += 24 * 60
        sessions.append((t1, t2))

    return sessions


def _parse_daily_range(opening_hours: str) -> Optional[Tuple[int, int]]:
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
    Pre-check: can the place fit *somewhere* within the slot window?

    Returns (open_ok, reason, clamped_start_mins).

    clamped_start_mins = max(slot_start, place_open) — the earliest minute
    in the slot when the place is open. This is passed into _compute_stop_times
    and the final _fits_in_slot_v2 check, which verifies the *actual* computed
    interval (based on last_end_mins cursor) against open sessions.

    NOTE: This function validates feasibility at slot level only. A second
    hard check in _fits_in_slot_v2 rejects placements where the cursor has
    moved past the closing time.
    """
    s_mins = _hm_to_mins(slot_start)
    e_mins = _hm_to_mins(slot_end)

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

    sessions = _parse_sessions(opening_hours)
    if sessions:
        for o, c in sessions:
            visit_start = max(s_mins, o)
            visit_end   = visit_start + duration_mins
            if visit_end <= min(e_mins, c):
                return (True, "", visit_start)
        sess_str = ", ".join(
            f"{_mins_to_time(o)}-{_mins_to_time(c % (24*60))}" for o, c in sessions
        )
        return (
            False,
            f"Hours {sess_str}: slot {slot_start}-{slot_end}, need {duration_mins}min",
            None,
        )

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


def _slot_has_any_overlap(oh_str: Optional[str], slot_start_mins: int, slot_end_mins: int) -> bool:
    """
    Returns True if the slot window [slot_start_mins, slot_end_mins) has ANY
    temporal overlap with the place's open sessions.

    Used as a Pass 2 guard: even in relaxed mode we hard-skip slots that are
    entirely outside the place's operating hours (e.g. night slot 20:00-21:30
    for a place that closes at 18:00). Only slots with at least some overlap
    are allowed through, and the result is marked opening_hours_unverified.

    If oh_str is None or unparseable, returns True (lenient — no data means
    we cannot rule it out).
    """
    if not oh_str:
        return True

    low = oh_str.lower().replace(" ", "")
    if low in ("24/7", "open24hrs", "open24hours", "open24h", "alwaysopen", "open24"):
        return True

    sessions = _parse_sessions(oh_str)
    if not sessions:
        return True  # unparseable — lenient

    # Overlap condition: slot_start < session_close AND slot_end > session_open
    for o, c in sessions:
        if slot_start_mins < c and slot_end_mins > o:
            return True
    return False


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


def _is_bad_place_name(name: str) -> bool:
    """
    Detect LLM-generated garbage place names where the model embedded its own
    reasoning / rejection commentary into the place_name field.

    Examples caught:
      'Subrahmanya Temple, not a specific enough name, skipping'
      'Unknown place, ambiguous'
      'N/A'

    Returns True if the name should be treated as invalid and dropped.
    """
    if not name or not name.strip():
        return True
    name_lower = name.lower()
    for phrase in _INVALID_NAME_PHRASES:
        if phrase in name_lower:
            return True
    # Names longer than 120 chars are almost certainly LLM commentary
    if len(name) > 120:
        return True
    return False


def validate_candidate_slots(candidates: list) -> list:
    valid = []
    for cand in candidates:
        name = cand.get("place_name", "") or ""

        # ─ Drop garbage LLM names before any scheduling logic ────────────────
        if _is_bad_place_name(name):
            logger.warning(
                f"[validate] dropping invalid place_name: {name!r:.80}"
            )
            continue  # skip entirely — do not add to valid list

        oh_str   = cand.get("opening_hours") or ""
        pref     = (cand.get("best_slot") or "morning").lower()
        dur_mins = int(float(cand.get("duration_hrs", 1.0)) * 60)

        if not oh_str.strip():
            valid.append(cand)
            continue

        low = oh_str.lower().replace(" ", "")
        if low in ("24/7", "open24hrs", "open24hours", "open24h", "alwaysopen", "open24"):
            valid.append(cand)
            continue

        sessions = _parse_sessions(oh_str)
        if not sessions:
            logger.debug(f"[validate] '{name}': unparseable hours — keeping best_slot")
            valid.append(cand)
            continue

        s_mins, e_mins = _SLOT_WINDOWS.get(pref, _SLOT_WINDOWS["morning"])
        slot_ok = any(
            (max(s_mins, o) + dur_mins) <= min(e_mins, c)
            for o, c in sessions
        )

        if slot_ok:
            logger.debug(f"[validate] '{name}': best_slot='{pref}' ✔️")
            valid.append(cand)
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

        valid.append(cand)

    return valid


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
                # ── Cursor tracking ──────────────────────────────────────────
                # Tracks the real end-time of the last placed stop in minutes.
                # Initialized to slot start. Updated after every placement.
                "last_end_mins":  _hm_to_mins(tmpl["start"]),
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


def _compute_stop_times(
    slot: dict,
    travel_mins: int,
    duration_hrs: float,
    clamped_start: Optional[int] = None,
) -> Tuple[str, str]:
    """
    Compute the actual start/end times for a stop.

    Uses slot['last_end_mins'] as the real cursor — the end time of
    the previously placed stop (or slot start if no prior stops).

    Formula:
        start = max(last_end_mins, clamped_start) + actual_travel
        end   = start + duration_mins

    This guarantees:
      1. start >= last_end_mins  → no overlap with previous stop
      2. start >= clamped_start  → no visit before place opens
      3. travel is always added on top of both constraints
    """
    actual_t = _actual_travel(slot, travel_mins)
    cursor   = slot["last_end_mins"]   # real end of last stop

    if clamped_start is not None:
        start_mins = max(cursor, clamped_start) + actual_t
    else:
        start_mins = cursor + actual_t

    end_mins = start_mins + int(duration_hrs * 60)
    return _mins_to_time(start_mins), _mins_to_time(end_mins)


def _fits_in_slot_v2(
    slot: dict,
    travel_mins: int,
    duration_hrs: float,
    clamped_start: Optional[int],
    oh_str: Optional[str],
    ignore_oh_check: bool = False,
) -> bool:
    """
    Check whether the candidate physically fits in the slot given the
    current cursor position.

    FIX 1 — Use last_end_mins cursor for fit check (not remaining_mins arithmetic).
    FIX 2 — Revalidate the *actual* computed interval against open sessions.

    Both checks must pass:
      A. computed_end <= slot_end_mins  (stop fits in the slot window)
      B. computed interval lies within at least one open session
         (prevents scheduling past closing time when cursor moved forward)

    Check B is skipped when:
      - opening hours are unparseable (lenient mode), OR
      - ignore_oh_check=True (Pass 2 relaxed scheduling — best-effort
        placement when all strict slots are full; output is flagged as
        opening_hours_unverified so the UI can warn the user)

    Note: Pass 2 also runs _slot_has_any_overlap() BEFORE calling this
    function to ensure the slot is not entirely outside open hours.
    """
    actual_t   = _actual_travel(slot, travel_mins)
    cursor     = slot["last_end_mins"]
    slot_end   = _hm_to_mins(slot["end_time"])

    if clamped_start is not None:
        start = max(cursor, clamped_start) + actual_t
    else:
        start = cursor + actual_t
    end = start + int(duration_hrs * 60)

    # A — must fit within slot window
    if end > slot_end:
        return False

    # B — must still be within an open session at actual computed time
    # Skipped in relaxed mode (ignore_oh_check=True) or unparseable hours.
    if not ignore_oh_check and oh_str:
        sessions = _parse_sessions(oh_str)
        if sessions:  # only hard-check when hours are parseable
            actually_open = any(o <= start and end <= c for o, c in sessions)
            if not actually_open:
                logger.debug(
                    f"[fits] rejected: actual {_mins_to_time(start)}-{_mins_to_time(end)} "
                    f"outside open sessions {[(f'{_mins_to_time(o)}-{_mins_to_time(c%(24*60))}') for o,c in sessions]}"
                )
                return False

    return True


# ─────────────────────────────────────────────────────────────
# Cross-day capacity-aware slot ordering
# ─────────────────────────────────────────────────────────────
def _build_fallback_order(
    slots: list,
    pref_slot: str,
    days: list,
    day_slots: dict,
    slot_map: dict,
    duration_mins: int,
) -> list:
    """
    Build the ordered list of slot_ids to try for a candidate.

    Strategy:
      1. Preferred slot across all days (in day order) — greedy fill preferred slot first.
      2. All other slots, sorted by remaining_mins DESC (roomiest slot first).
         This ensures a candidate with 90-min duration tries day3_evening (180 min free)
         before day1_evening (90 min free), maximising future packing density.

    Within the preferred group, day order is preserved (day 1 first).
    Within the fallback group, slots are sorted by remaining_mins descending
    so the candidate lands in the slot with most headroom, leaving tighter
    slots for later (smaller) candidates.
    """
    preferred_ids: list[str] = []
    fallback_ids:  list[str] = []

    for day in days:
        for sid in day_slots[day]:
            slot = slot_map[sid]
            if pref_slot in slot["slot_name"]:
                preferred_ids.append(sid)
            else:
                fallback_ids.append(sid)

    # Sort fallback by remaining capacity descending — try roomiest first.
    fallback_ids.sort(
        key=lambda sid: slot_map[sid]["remaining_mins"],
        reverse=True,
    )

    return preferred_ids + fallback_ids


def _build_any_slot_order(slots: list, slot_map: dict, duration_mins: int) -> list:
    """
    Build a slot order for Pass 2 (relaxed): ALL slots sorted purely by
    remaining_mins DESC, ignoring best_slot preference entirely.

    Only includes slots that have at least duration_mins of remaining
    capacity (raw check — actual fit is verified by _fits_in_slot_v2).
    """
    eligible = [
        s["slot_id"] for s in slots
        if s["remaining_mins"] >= duration_mins
    ]
    eligible.sort(key=lambda sid: slot_map[sid]["remaining_mins"], reverse=True)
    return eligible


# ─────────────────────────────────────────────────────────────
# Main greedy scheduler — two-pass with conflict relaxation
# ─────────────────────────────────────────────────────────────
def schedule_candidates(
    candidates: list,
    slots: list,
    avoid_crowded: bool = False,
    accessibility_needs: bool = False,
    day_dates: Optional[Dict[int, date]] = None,
) -> Tuple[list, list]:
    """
    Two-pass greedy scheduler.

    PASS 1 — Strict (original behaviour):
      Step 0: validate_candidate_slots() — pre-correct best_slot vs opening hours.
              Also drops candidates with garbage LLM-generated place names.
      Step 1: Sort by priority descending.
      Step 2: For each candidate try preferred slot then fallbacks.
              Fallback order is capacity-aware: roomiest slot tried first.
      Step 3: Slot-level opening hours feasibility check (is_open_for_slot).
      Step 4: Actual-time fit check (_fits_in_slot_v2) — cursor-aware + OH check.
      Step 5: Compute and record times; update slot cursor.

    PASS 2 — Relaxed (only for candidates that failed Pass 1):
      For each remaining unscheduled candidate, retry across ALL slots
      sorted by remaining_mins DESC — ignoring best_slot preference.

      Opening hours check is PARTIALLY RELAXED:
        • _slot_has_any_overlap() is checked FIRST (hard skip if slot window
          has zero overlap with open hours — e.g. night slot for a place
          that closes at 18:00).
        • If overlap exists, _fits_in_slot_v2 is called with
          ignore_oh_check=True — only window fit (Check A) is enforced.
        • closed_on weekday is still a hard skip.
        • Result is marked opening_hours_unverified=True for UI warning.

      This prevents nonsensical placements (Vattakanal Falls at 20:00
      when it closes at 18:00) while still allowing borderline evening
      placements (Golf Club 17:19-18:19 when it closes at 18:00).
    """
    candidates = validate_candidate_slots(candidates)
    sorted_cands = sorted(candidates, key=lambda c: -int(c.get("priority", 3)))

    slot_map  = {s["slot_id"]: s for s in slots}
    days      = sorted(set(s["day"] for s in slots))
    day_slots = {d: [s["slot_id"] for s in slots if s["day"] == d] for d in days}

    scheduled:   list = []

    # ── PASS 1: strict scheduling ──────────────────────────────────────────────
    pass1_unscheduled: list = []

    for cand in sorted_cands:
        placed        = False
        pref_slot     = cand.get("best_slot", "morning").lower()
        cand_lat      = cand.get("lat")
        cand_lon      = cand.get("lon")
        duration_hrs  = float(cand.get("duration_hrs", 1.0))
        duration_mins = int(duration_hrs * 60)
        oh_str        = cand.get("opening_hours")
        closed_on     = cand.get("closed_on") or []

        slot_order = _build_fallback_order(
            slots, pref_slot, days, day_slots, slot_map, duration_mins
        )

        for slot_id in slot_order:
            slot      = slot_map[slot_id]
            day       = slot["day"]
            slot_date = day_dates.get(day) if day_dates else None
            travel_mins = estimate_travel_minutes(
                slot.get("last_lat"), slot.get("last_lon"), cand_lat, cand_lon
            )

            open_ok, open_reason, clamped_start = is_open_for_slot(
                oh_str, closed_on,
                slot["start_time"], slot["end_time"],
                duration_mins, slot_date
            )
            if open_ok is False:
                logger.debug(f"[P1] skip '{cand.get('place_name')}' in {slot_id}: {open_reason}")
                continue

            if not _fits_in_slot_v2(slot, travel_mins, duration_hrs, clamped_start, oh_str):
                logger.debug(
                    f"[P1] no fit '{cand.get('place_name')}' in {slot_id} "
                    f"(cursor={_mins_to_time(slot['last_end_mins'])}, "
                    f"travel={travel_mins}, dur={duration_mins})"
                )
                continue

            start_t, end_t = _compute_stop_times(slot, travel_mins, duration_hrs, clamped_start)
            actual_t = _actual_travel(slot, travel_mins)
            end_mins = _hm_to_mins(end_t)

            stop = {
                "day":                      day,
                "slot_id":                  slot_id,
                "slot_name":                slot["slot_name"],
                "start_time":               start_t,
                "end_time":                 end_t,
                "travel_mins_from_prev":    actual_t,
                "place_name":               cand.get("place_name", "Unknown"),
                "category":                 cand.get("category"),
                "priority":                 int(cand.get("priority", 3)),
                "why_must_visit":           cand.get("why_must_visit"),
                "is_alternate":             bool(cand.get("is_alternate", False)),
                "opening_hours":            oh_str,
                "closed_on":                closed_on or None,
                "duration_hrs":             duration_hrs,
                "entry_fee":                cand.get("entry_fee"),
                "entry_fee_foreign":        cand.get("entry_fee_foreign"),
                "tip":                      cand.get("tip"),
                "nearby_food":              cand.get("nearby_food"),
                "lat":                      cand_lat,
                "lon":                      cand_lon,
                "opening_hours_unverified": open_ok is None,
                "hours_conflict":           bool(cand.get("hours_conflict", False)),
            }

            slot["last_end_mins"] = end_mins
            slot["remaining_mins"] = _hm_to_mins(slot["end_time"]) - end_mins
            slot["stops"].append(stop)
            if cand_lat and cand_lon:
                slot["last_lat"] = cand_lat
                slot["last_lon"] = cand_lon

            scheduled.append(stop)
            placed = True
            flag = "✅" if open_ok else "⚠️"
            logger.debug(f"[P1] {flag} '{cand.get('place_name')}' → {slot_id} @ {start_t}")
            break

        if not placed:
            logger.info(f"[P1] unscheduled: '{cand.get('place_name')}' (dur={duration_hrs}h)")
            pass1_unscheduled.append(cand)

    logger.info(
        f"[P1] done: {len(scheduled)} placed, {len(pass1_unscheduled)} sent to Pass 2"
    )

    # ── PASS 2: relaxed scheduling (ignore best_slot + partial OH relaxation) ───
    #
    # For each candidate that failed Pass 1, retry across ALL slots by
    # remaining capacity (roomiest first).
    #
    # KEY CHANGE vs previous version:
    #   Before allowing a slot, call _slot_has_any_overlap() to check
    #   whether the slot window has ANY overlap with the place's open hours.
    #   If overlap = zero (e.g. night 20:00-21:30 vs close=18:00), hard-skip.
    #   If overlap exists, proceed with ignore_oh_check=True (only window
    #   fit Check A is enforced) and flag as opening_hours_unverified.
    #
    # This prevents absurd placements (waterfall at 20:00 when closed at 18:00)
    # while still allowing borderline evening placements.
    pass2_unscheduled: list = []

    for cand in pass1_unscheduled:
        placed        = False
        cand_lat      = cand.get("lat")
        cand_lon      = cand.get("lon")
        duration_hrs  = float(cand.get("duration_hrs", 1.0))
        duration_mins = int(duration_hrs * 60)
        oh_str        = cand.get("opening_hours")
        closed_on     = cand.get("closed_on") or []

        slot_order = _build_any_slot_order(slots, slot_map, duration_mins)

        for slot_id in slot_order:
            slot      = slot_map[slot_id]
            day       = slot["day"]
            slot_date = day_dates.get(day) if day_dates else None
            travel_mins = estimate_travel_minutes(
                slot.get("last_lat"), slot.get("last_lon"), cand_lat, cand_lon
            )

            # Hard-skip if closed on this weekday
            if closed_on and slot_date:
                wd = slot_date.weekday()
                closed_today = any(
                    _DAY_NAME_TO_IDX.get(d.strip().lower()) == wd
                    for d in closed_on
                )
                if closed_today:
                    logger.debug(f"[P2] skip '{cand.get('place_name')}' in {slot_id}: closed today")
                    continue

            # Hard-skip if slot has ZERO overlap with open hours.
            # This blocks night-slot placements for daytime-only attractions.
            slot_s = _hm_to_mins(slot["start_time"])
            slot_e = _hm_to_mins(slot["end_time"])
            if not _slot_has_any_overlap(oh_str, slot_s, slot_e):
                logger.debug(
                    f"[P2] skip '{cand.get('place_name')}' in {slot_id}: "
                    f"zero overlap with open hours"
                )
                continue

            # Relaxed fit: bypass OH session check, only enforce window fit.
            clamped_start = slot_s
            if not _fits_in_slot_v2(
                slot, travel_mins, duration_hrs, clamped_start, oh_str,
                ignore_oh_check=True
            ):
                logger.debug(
                    f"[P2] no fit '{cand.get('place_name')}' in {slot_id} "
                    f"(cursor={_mins_to_time(slot['last_end_mins'])}, "
                    f"travel={travel_mins}, dur={duration_mins})"
                )
                continue

            start_t, end_t = _compute_stop_times(slot, travel_mins, duration_hrs, clamped_start)
            actual_t = _actual_travel(slot, travel_mins)
            end_mins = _hm_to_mins(end_t)

            stop = {
                "day":                      day,
                "slot_id":                  slot_id,
                "slot_name":                slot["slot_name"],
                "start_time":               start_t,
                "end_time":                 end_t,
                "travel_mins_from_prev":    actual_t,
                "place_name":               cand.get("place_name", "Unknown"),
                "category":                 cand.get("category"),
                "priority":                 int(cand.get("priority", 3)),
                "why_must_visit":           cand.get("why_must_visit"),
                "is_alternate":             bool(cand.get("is_alternate", False)),
                "opening_hours":            oh_str,
                "closed_on":                closed_on or None,
                "duration_hrs":             duration_hrs,
                "entry_fee":                cand.get("entry_fee"),
                "entry_fee_foreign":        cand.get("entry_fee_foreign"),
                "tip":                      cand.get("tip"),
                "nearby_food":              cand.get("nearby_food"),
                "lat":                      cand_lat,
                "lon":                      cand_lon,
                # Always unverified in Pass 2 — OH enforcement was relaxed
                "opening_hours_unverified": True,
                "hours_conflict":           bool(cand.get("hours_conflict", False)),
            }

            slot["last_end_mins"] = end_mins
            slot["remaining_mins"] = _hm_to_mins(slot["end_time"]) - end_mins
            slot["stops"].append(stop)
            if cand_lat and cand_lon:
                slot["last_lat"] = cand_lat
                slot["last_lon"] = cand_lon

            scheduled.append(stop)
            placed = True
            logger.info(
                f"[P2] ⚠️ '{cand.get('place_name')}' → {slot_id} @ {start_t} "
                f"(relaxed — OH bypassed, unverified flag set)"
            )
            break

        if not placed:
            logger.info(f"[P2] truly unscheduled: '{cand.get('place_name')}' (dur={duration_hrs}h)")
            pass2_unscheduled.append(cand)

    unscheduled = pass2_unscheduled
    logger.info(
        f"[sched] final: {len(scheduled)} placed "
        f"(P1={len(scheduled) - len(pass1_unscheduled) + len(pass2_unscheduled)}, "
        f"P2={len(pass1_unscheduled) - len(pass2_unscheduled)}), "
        f"{len(unscheduled)} truly unscheduled"
    )
    return scheduled, unscheduled
