"""
routes/customize_bot.py

Customize Bot Endpoint
======================
POST /api/customize-bot/{itinerary_id}/chat

  Accepts a free-text user message like:
    "On day 2 remove shopping places and add one beach."
    "Swap the temple on day 1 with a fort. Make the pace relaxed."
    "Move Marina Beach from day 1 to day 3."

  Flow:
    1. Load itinerary from Firestore.
    2. Ask Groq (run_customize_bot_llm) to convert the message
       into a structured ActionPlan JSON.
    3. Apply each operation in order:
       remove → swap → add → reorder → pin → settings
    4. Recompute start/end times for every touched day.
    5. Validate: no duplicate place across different days.
    6. Persist updated itinerary to Firestore.
    7. Return ActionPlan + updated itinerary + meta.

Constraints enforced:
  - Pinned stops are never moved/removed unless user explicitly requests it.
  - No place_name can appear on more than one day.
  - Route and travel times are recomputed automatically after any change.
"""

import logging
from typing import Any, Dict, List, Literal, Optional, Union

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel, Field

from services.firebase_service import get_itinerary_by_id_for_user, update_itinerary
from services.scheduler_service import (
    _hm_to_mins,
    _mins_to_time,
    build_day_slots,
    schedule_candidates,
)
from routes.customize import _canonical as _canon, _recompute_day
from services.llm_service import run_customize_bot_llm

logger = logging.getLogger(__name__)
router = APIRouter()


# ────────────────────────────────────────────
# Request / response schemas
# ────────────────────────────────────────────

class BotMessage(BaseModel):
    user_message: str = Field(..., min_length=3, description="Free-text customization request")
    user_id: str       = Field(..., description="Firebase UID")


# —————————— Operation schemas ——————————

class OpRemove(BaseModel):
    op: Literal["remove"]
    day: int
    place_name: str


class OpAdd(BaseModel):
    op: Literal["add"]
    preferred_day: Optional[int] = None
    new_place: Dict[str, Any]


class OpSwap(BaseModel):
    op: Literal["swap"]
    day: int
    old_place_name: str
    new_place: Dict[str, Any]


class OpReorder(BaseModel):
    op: Literal["reorder"]
    day: int
    new_order_place_names: List[str]


class OpPin(BaseModel):
    op: Literal["pin"]
    day: int
    place_name: str
    pinned: bool


class OpSettings(BaseModel):
    op: Literal["settings"]
    pace: Optional[str]  = None
    budget: Optional[str] = None
    avoid_crowded: Optional[bool] = None
    accessibility_needs: Optional[bool] = None


ActionOp = Union[OpRemove, OpAdd, OpSwap, OpReorder, OpPin, OpSettings]


class ActionPlan(BaseModel):
    intent: Literal["modify_itinerary"]
    target_days: Optional[List[int]] = None
    operations: List[Dict[str, Any]]  # raw dicts; typed ops resolved at runtime


class BotResponse(BaseModel):
    success: bool
    action_plan: ActionPlan
    itinerary: List[Dict[str, Any]]
    meta: Dict[str, Any]
    applied_ops: List[str]   # human-readable summary of what changed


# ────────────────────────────────────────────
# Internal operation helpers
# ────────────────────────────────────────────

def _ensure_no_duplicates_across_days(itinerary: List[Dict[str, Any]]) -> None:
    """
    Raise 409 if any place_name (canonical) appears on more than one day.
    """
    seen: Dict[str, int] = {}
    for s in itinerary:
        key = _canon(s.get("place_name", ""))
        day = s.get("day")
        if key in seen and seen[key] != day:
            raise HTTPException(
                409,
                f"Duplicate place across days detected: '{s.get('place_name')}' "
                f"on day {seen[key]} and day {day}. "
                f"Use remove + add to move a place.",
            )
        seen[key] = day


def _apply_remove(itinerary: List[Dict], op: OpRemove, log: List[str]) -> List[Dict]:
    target = _canon(op.place_name)
    before = len(itinerary)
    itinerary = [
        s for s in itinerary
        if not (
            s.get("day") == op.day
            and _canon(s.get("place_name", "")) == target
        )
    ]
    if len(itinerary) < before:
        log.append(f"removed '{op.place_name}' from day {op.day}")
    else:
        logger.info(f"[bot] remove: '{op.place_name}' not found on day {op.day} — skipped")
    return itinerary


def _apply_swap(
    itinerary: List[Dict], op: OpSwap, log: List[str]
) -> List[Dict]:
    target = _canon(op.old_place_name)
    idx = None
    for i, s in enumerate(itinerary):
        if s.get("day") == op.day and _canon(s.get("place_name", "")) == target:
            idx = i
            break
    if idx is None:
        logger.info(f"[bot] swap: '{op.old_place_name}' not found on day {op.day} — skipped")
        return itinerary

    old = itinerary[idx]
    dur = float(op.new_place.get("duration_hrs", old.get("duration_hrs", 1.0)))
    start = old.get("start_time", "09:00")
    start_m = _hm_to_mins(start)
    end_m = start_m + int(dur * 60)

    new_stop = dict(old)          # inherit slot_id, day, pinned, etc.
    new_stop.update(op.new_place) # overlay LLM-provided fields
    new_stop["duration_hrs"] = dur
    new_stop["place_name"]   = op.new_place.get("place_name", old["place_name"])
    new_stop["start_time"]   = start
    new_stop["end_time"]     = _mins_to_time(end_m)
    new_stop["pinned"]       = False  # swapped stop starts unpinned

    itinerary[idx] = new_stop
    log.append(f"swapped '{op.old_place_name}' → '{new_stop['place_name']}' on day {op.day}")
    return itinerary


def _apply_add(
    itinerary: List[Dict], meta: Dict[str, Any], op: OpAdd, log: List[str]
) -> List[Dict]:
    days = meta.get("days") or max((s.get("day", 1) for s in itinerary), default=1)
    slots = build_day_slots(days)
    slot_map = {s["slot_id"]: s for s in slots}

    # Rebuild live slot state from existing itinerary
    ordered = sorted(
        itinerary, key=lambda x: (x.get("day", 0), x.get("start_time", "00:00"))
    )
    for st in ordered:
        sid = st.get("slot_id")
        if sid not in slot_map:
            continue
        slot = slot_map[sid]
        end_m    = _hm_to_mins(st.get("end_time", slot["end_time"]))
        slot_end = _hm_to_mins(slot["end_time"])
        slot["last_end_mins"]  = end_m
        slot["remaining_mins"] = max(0, slot_end - end_m)
        if st.get("lat"):
            slot["last_lat"] = st["lat"]
        if st.get("lon"):
            slot["last_lon"] = st["lon"]

    cand = dict(op.new_place)
    cand.setdefault("best_slot", cand.get("best_slot") or "morning")
    cand.setdefault("duration_hrs", 1.0)
    cand.setdefault("priority", 3)

    target_slots = slots
    if op.preferred_day:
        target_slots = [s for s in slots if s["day"] == op.preferred_day]
        if not target_slots:
            raise HTTPException(400, f"No slots found for day {op.preferred_day}")

    placed, unplaced = schedule_candidates([cand], target_slots)
    if unplaced:
        raise HTTPException(
            409,
            f"No slot available for '{cand.get('place_name')}' "
            f"(duration={cand.get('duration_hrs', 1.0)}h). "
            f"Try a shorter duration or a different day.",
        )
    itinerary.append(placed[0])
    log.append(
        f"added '{cand['place_name']}' to day {placed[0]['day']} "
        f"({placed[0].get('slot_name','?')} slot @ {placed[0].get('start_time','?')})"
    )
    return itinerary


def _apply_reorder(
    itinerary: List[Dict], op: OpReorder, log: List[str]
) -> List[Dict]:
    day_stops = [s for s in itinerary if s.get("day") == op.day]
    other     = [s for s in itinerary if s.get("day") != op.day]
    if not day_stops:
        return itinerary

    lookup = {_canon(s["place_name"]): s for s in day_stops}
    reordered: List[Dict] = []
    for name in op.new_order_place_names:
        key = _canon(name)
        if key in lookup:
            reordered.append(lookup[key])

    # Append any stops that were in day but not mentioned (safety)
    mentioned = {_canon(n) for n in op.new_order_place_names}
    for s in day_stops:
        if _canon(s["place_name"]) not in mentioned:
            reordered.append(s)

    log.append(f"reordered day {op.day} stops")
    return other + reordered


def _apply_pin(
    itinerary: List[Dict], op: OpPin, log: List[str]
) -> List[Dict]:
    target = _canon(op.place_name)
    for s in itinerary:
        if s.get("day") == op.day and _canon(s.get("place_name", "")) == target:
            s["pinned"] = op.pinned
            action = "pinned" if op.pinned else "unpinned"
            log.append(f"{action} '{op.place_name}' on day {op.day}")
            return itinerary
    logger.info(f"[bot] pin: '{op.place_name}' not found on day {op.day} — skipped")
    return itinerary


def _apply_settings(
    meta: Dict[str, Any], op: OpSettings, log: List[str]
) -> Dict[str, Any]:
    overrides = dict(meta.get("overrides", {}))
    changes = []
    if op.pace is not None:
        overrides["pace"] = op.pace
        changes.append(f"pace={op.pace}")
    if op.budget is not None:
        overrides["budget"] = op.budget
        changes.append(f"budget={op.budget}")
    if op.avoid_crowded is not None:
        overrides["avoid_crowded"] = op.avoid_crowded
        changes.append(f"avoid_crowded={op.avoid_crowded}")
    if op.accessibility_needs is not None:
        overrides["accessibility_needs"] = op.accessibility_needs
        changes.append(f"accessibility_needs={op.accessibility_needs}")
    meta["overrides"] = overrides
    if changes:
        log.append("settings updated: " + ", ".join(changes))
    return meta


def _resolve_op(raw: Dict[str, Any]) -> Optional[ActionOp]:
    """
    Convert a raw dict from the LLM JSON into a typed operation model.
    Returns None if op type is unknown (safe skip).
    """
    op_type = raw.get("op", "")
    try:
        if op_type == "remove":  return OpRemove(**raw)
        if op_type == "add":     return OpAdd(**raw)
        if op_type == "swap":    return OpSwap(**raw)
        if op_type == "reorder": return OpReorder(**raw)
        if op_type == "pin":     return OpPin(**raw)
        if op_type == "settings":return OpSettings(**raw)
    except Exception as e:
        logger.warning(f"[bot] could not parse op '{op_type}': {e}  raw={raw}")
    return None


def _recompute_touched_days(
    itinerary: List[Dict], ops: List[ActionOp]
) -> List[Dict]:
    """
    Recompute start/end times for every day that was touched by at least
    one operation.  Respects pinned stops and re-estimates travel minutes.
    """
    touched = sorted(
        {getattr(op, "day", None) or
         getattr(op, "preferred_day", None)
         for op in ops
         if hasattr(op, "day") or hasattr(op, "preferred_day")}
        - {None}
    )
    # Also include the day the added stop actually landed on
    for op in ops:
        if isinstance(op, OpAdd):
            # find the newly added stop
            for s in itinerary:
                if _canon(s.get("place_name", "")) == _canon(
                    op.new_place.get("place_name", "")
                ):
                    touched.append(s.get("day"))
                    break

    for d in sorted(set(touched)):
        itinerary = _recompute_day(itinerary, d)
    return itinerary


# ────────────────────────────────────────────
# POST /api/customize-bot/{itinerary_id}/chat
# ────────────────────────────────────────────

@router.post("/{itinerary_id}/chat", response_model=BotResponse)
async def customize_bot_chat(
    itinerary_id: str,
    payload: BotMessage,
    x_user_id: str = Header(None, alias="X-User-Id"),
):
    """
    Free-text → ActionPlan → apply → updated itinerary.

    Example requests:
      { "user_message": "On day 2 remove the temple and add a beach.", "user_id": "uid123" }
      { "user_message": "Swap the fort on day 1 with a museum. Make pace relaxed.", "user_id": "uid123" }
      { "user_message": "Move Marina Beach from day 1 to day 3.", "user_id": "uid123" }
    """
    user_id = (payload.user_id or x_user_id or "").strip()
    if not user_id:
        raise HTTPException(401, "user_id or X-User-Id header is required")

    # 1) Load itinerary from Firestore
    doc = await get_itinerary_by_id_for_user(user_id, itinerary_id)
    if not doc:
        raise HTTPException(404, "Itinerary not found")

    itinerary: List[Dict] = list(doc.get("itinerary", []))
    meta: Dict[str, Any]  = dict(doc.get("meta", {}))

    # 2) Ask Groq to produce a structured ActionPlan
    try:
        plan_dict = await run_customize_bot_llm(
            user_message=payload.user_message,
            itinerary=itinerary,
            meta=meta,
        )
    except ValueError as e:
        raise HTTPException(500, f"Bot LLM error: {e}")

    # Validate top-level structure
    if plan_dict.get("intent") != "modify_itinerary":
        raise HTTPException(500, "Bot returned unexpected intent; try rephrasing your request.")

    action_plan = ActionPlan(
        intent="modify_itinerary",
        target_days=plan_dict.get("target_days"),
        operations=plan_dict.get("operations", []),
    )

    # 3) Resolve raw dicts → typed op objects
    typed_ops: List[ActionOp] = []
    for raw_op in action_plan.operations:
        op = _resolve_op(raw_op)
        if op is not None:
            typed_ops.append(op)

    if not typed_ops:
        raise HTTPException(422, "Bot produced no valid operations. Try a more specific request.")

    # 4) Apply operations in order
    applied_log: List[str] = []
    for op in typed_ops:
        if isinstance(op, OpRemove):
            itinerary = _apply_remove(itinerary, op, applied_log)
        elif isinstance(op, OpSwap):
            itinerary = _apply_swap(itinerary, op, applied_log)
        elif isinstance(op, OpAdd):
            itinerary = _apply_add(itinerary, meta, op, applied_log)
        elif isinstance(op, OpReorder):
            itinerary = _apply_reorder(itinerary, op, applied_log)
        elif isinstance(op, OpPin):
            itinerary = _apply_pin(itinerary, op, applied_log)
        elif isinstance(op, OpSettings):
            meta = _apply_settings(meta, op, applied_log)

    # 5) Recompute times for touched days (route + travel time)
    itinerary = _recompute_touched_days(itinerary, typed_ops)

    # Sort by day + start_time for clean display
    itinerary.sort(key=lambda x: (x.get("day", 0), x.get("start_time", "00:00")))

    # 6) Hard constraint: no duplicate place across days
    _ensure_no_duplicates_across_days(itinerary)

    # 7) Persist to Firestore
    await update_itinerary(user_id, itinerary_id, {"itinerary": itinerary, "meta": meta})

    logger.info(
        f"[bot] {itinerary_id} by {user_id}: "
        f"{len(typed_ops)} ops applied — {applied_log}"
    )

    return BotResponse(
        success=True,
        action_plan=action_plan,
        itinerary=itinerary,
        meta=meta,
        applied_ops=applied_log,
    )
