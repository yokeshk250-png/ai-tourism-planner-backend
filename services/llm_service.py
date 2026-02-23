import json
import os
import logging
from dotenv import load_dotenv
from models.schemas import TripRequest

load_dotenv()

logger = logging.getLogger(__name__)

GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
_client = None


def get_client():
    global _client
    if _client is not None:
        return _client
    try:
        from groq import AsyncGroq
    except ImportError:
        raise ImportError("groq not installed. Run: pip install groq")
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not set.\n"
            "Add GROQ_API_KEY=gsk_... to .env\n"
            "Free key: https://console.groq.com/keys"
        )
    _client = AsyncGroq(api_key=api_key)
    logger.info(f"Groq ready ✅  model={GROQ_MODEL}  key=...{api_key[-6:]}")
    return _client


async def _call_groq(messages: list, max_tokens: int = 4096, temperature: float = 0.2) -> str:
    """
    Low-level Groq JSON-mode call.
    Raises clear ValueError on 401/429 errors.
    Strips markdown fences from response.
    """
    client = get_client()
    try:
        resp = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            response_format={"type": "json_object"},
            temperature=temperature,
            max_tokens=max_tokens,
        )
    except Exception as e:
        err = str(e)
        if "401" in err or "invalid_api_key" in err.lower():
            raise ValueError("Groq API key invalid (401). Update GROQ_API_KEY in .env.")
        if "429" in err or "rate_limit" in err.lower():
            raise ValueError("Groq rate limit hit (429). Wait 1 min and retry.")
        raise ValueError(f"Groq error: {err}")

    raw = resp.choices[0].message.content or ""
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()
    return raw


def _parse_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Groq returned invalid JSON: {e}\nRaw: {raw[:300]}")


# ─────────────────────────────────────────────────────────────
# Stage 1 — Generate PLACE CANDIDATES
# ─────────────────────────────────────────────────────────────
async def generate_place_candidates_llm(req: TripRequest) -> list:
    """
    Ask Groq for top/must-visit tourist attractions ONLY.
    NO meals, NO restaurants, NO hotels.

    Prompting rules that prevent generic/wrong-city names:
      • Every place must include a locality/area qualifier.
      • No generic names like "Vishnu Temple" or "Shiva Temple".
      • Wikipedia-verifiable, named places only.
      • Strictly within or near the destination city.
    """
    num = max(req.days * 7, 18)  # Ample candidates for scheduler to pick from

    system_prompt = (
        "You are an expert Indian travel planner with deep local knowledge. "
        "Suggest ONLY real, named tourist attractions that physically exist in or very near the given city. "
        "Every place must be uniquely identifiable — include the area/locality in the name if needed. "
        "NEVER use generic names like 'Vishnu Temple', 'Shiva Temple', 'Subramanya Temple' without "
        "the specific temple name used locally. "
        "NEVER suggest restaurants, cafes, hotels, or any food/meal stops. "
        "ONLY suggest places that are physically within 50 km of the city centre. "
        "Return ONLY valid JSON."
    )

    user_prompt = f"""Suggest the top {num} must-visit tourist attractions specifically in {req.destination}, India.

IMPORTANT RULES FOR PLACE NAMES:
1. Use the FULL local name (e.g. "Koodal Azhagar Temple, West Masi Street" not just "Vishnu Temple").
2. Every place must be PHYSICALLY INSIDE or within 50 km of {req.destination} city centre.
3. Prioritise places from this known list first, then add others:
   - Meenakshi Amman Temple
   - Thirumalai Nayak Palace
   - Vandiyur Mariamman Teppakulam
   - Koodal Azhagar Temple
   - Gandhi Memorial Museum
   - Goripalayam Dargah (Madurai)
   - Thirupparankundram Murugan Temple
   - Alagar Kovil (Azhagar Kovil)
   - Pazhamudhircholai Murugan Temple
   - Samanar Hills (Keelakuyilkudi)
   - Yanaimalai
   - Vilachery Pottery Village, Madurai
   - Uchi Pillaiyar Temple, Madurai
   - St. Mary's Cathedral, Madurai
   - Madurai Railway Museum
   - Aayiram Kaal Mandapam, Madurai
4. Strictly NO generic temple names without locality qualifier.
5. ATTRACTIONS ONLY — zero restaurants, zero cafes, zero meal stops.

Traveller profile:
- Budget: {req.budget.value}
- Travel type: {req.travel_type.value}
- Mood/Theme: {req.mood.value}
- Interests: {', '.join(req.interests)}
- Avoid crowded: {req.avoid_crowded}
- Accessibility needs: {req.accessibility_needs}

Additional rules:
- priority: 5=must-visit, 4=highly recommended, 3=good, 2=if time permits, 1=optional.
- Realistic visit duration in hours (0.5 to 4.0).
- best_slot: morning (09-12) | afternoon (13-16) | evening (16:30-19:30) | night (20-21:30).
- Use actual opening hours and INR entry fees (0 if free).
- One practical tip per place (max 15 words).

Return ONLY this JSON:
{{
  "candidates": [
    {{
      "place_name": "Meenakshi Amman Temple",
      "category": "temple",
      "priority": 5,
      "duration_hrs": 2.5,
      "best_slot": "morning",
      "why_must_visit": "One of India's largest and most ornate Dravidian temples",
      "opening_hours": "5:00 AM - 12:30 PM, 4:00 PM - 10:00 PM",
      "closed_on": [],
      "entry_fee": 0,
      "tip": "Arrive before 9 AM to beat crowds and see morning rituals"
    }}
  ]
}}"""

    raw = await _call_groq(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        max_tokens=4096
    )
    data = _parse_json(raw)
    candidates = data.get("candidates", [])
    logger.info(f"Groq: {len(candidates)} candidates for {req.destination}")
    return candidates


# ─────────────────────────────────────────────────────────────
# Slot window reference (mirrors scheduler_service.py SLOT_TEMPLATE)
# ─────────────────────────────────────────────────────────────
_SLOT_WINDOWS = {
    "morning":   ("09:00", "12:00"),
    "afternoon": ("13:00", "16:00"),
    "evening":   ("16:30", "19:30"),
    "night":     ("20:00", "21:30"),
}

_SLOT_OH_HINT = {
    "morning":   "open from at least 09:00",
    "afternoon": "open from at least 13:00",
    "evening":   "open until at least 18:30 (ideally 19:30+)",
    "night":     "open until at least 21:00 (e.g. temples, forts, promenades, food streets)",
}


# ─────────────────────────────────────────────────────────────
# Stage 5 — Conflict resolver: ask Groq for alternate places
# ─────────────────────────────────────────────────────────────
async def suggest_alternates_llm(
    destination: str,
    failed_place: dict,
    scheduled_places: list,
    slot_name: str,
    free_slots: list[str] | None = None,
) -> list:
    """
    When a candidate cannot be scheduled, ask Groq for shorter/alternative places
    that fit one of the currently FREE slots.

    `free_slots` — ordered list of slot names that still have capacity
    (e.g. ["evening", "night", "morning"]). When provided, the LLM is told
    to target these slots specifically and to match their opening-hours
    requirements. This prevents the classic failure where S5 alternates
    all come back with '9am-5pm' OH while only evening/night slots are free.
    """
    scheduled_names = [s.get("place_name", "") for s in scheduled_places]
    max_dur = max(0.5, round(failed_place.get("duration_hrs", 1.5) - 0.5, 1))

    # ── Build slot-awareness block ────────────────────────────────────────────
    if free_slots:
        target_slot = free_slots[0]   # best available slot to anchor the example
        slot_lines = []
        for fs in free_slots:
            win = _SLOT_WINDOWS.get(fs, ("?", "?"))
            hint = _SLOT_OH_HINT.get(fs, "open during this window")
            slot_lines.append(f"  • {fs} ({win[0]}–{win[1]}): must be {hint}")
        slot_block = (
            f"AVAILABLE SLOTS WITH CAPACITY (suggest places that fit ONE of these):\n"
            + "\n".join(slot_lines)
            + "\n\n"
            "CRITICAL — Opening hours MUST match the target slot:\n"
            + "\n".join(
                f"  • For {fs} slot → place must be open during {_SLOT_WINDOWS.get(fs,('?','?'))[0]}–"
                f"{_SLOT_WINDOWS.get(fs,('?','?'))[1]}"
                for fs in free_slots
            )
        )
    else:
        target_slot = slot_name
        slot_block = (
            f"Target slot: {slot_name} ({_SLOT_WINDOWS.get(slot_name, ('?','?'))[0]}–"
            f"{_SLOT_WINDOWS.get(slot_name, ('?','?'))[1]})\n"
            f"Place must be {_SLOT_OH_HINT.get(slot_name, 'open during this window')}."
        )

    system_prompt = (
        "You are an expert Indian travel planner with deep local knowledge. "
        "Suggest real, specifically-named TOURIST ATTRACTIONS only — no restaurants, no meals. "
        "Every suggestion must physically exist in or within 50 km of the destination city. "
        "NEVER use generic names like 'Vishnu Temple' without the specific local temple name. "
        "Pay close attention to opening hours — they MUST match the requested time slot. "
        "Return ONLY valid JSON."
    )

    user_prompt = f"""Building a trip itinerary for {destination}, India.

The following place CANNOT be scheduled (timing/capacity conflict):
- Place: {failed_place.get('place_name')}
- Category: {failed_place.get('category', 'general')}
- Its opening hours: {failed_place.get('opening_hours', 'unknown')}

Already scheduled (DO NOT suggest these):
{', '.join(scheduled_names) if scheduled_names else 'None'}

{slot_block}

Suggest 3 alternative tourist attractions that:
1. Same or similar category as "{failed_place.get('category', 'general')}" (or any category if none fits).
2. Visit duration ≤ {max_dur} hours.
3. MUST fit in one of the available slots listed above — opening hours are CRITICAL.
4. NOT already in the scheduled list above.
5. MUST be real, specifically-named places in or within 50 km of {destination}.
6. ATTRACTIONS ONLY — no restaurants, no hotels.

Return ONLY this JSON:
{{
  "alternates": [
    {{
      "place_name": "Example Place, {destination}",
      "category": "viewpoint",
      "priority": 3,
      "duration_hrs": 0.5,
      "best_slot": "{target_slot}",
      "why_must_visit": "Short reason",
      "opening_hours": "open hours matching the target slot",
      "closed_on": [],
      "entry_fee": 0,
      "tip": "Short practical tip"
    }}
  ]
}}"""

    try:
        raw = await _call_groq(
            [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            max_tokens=1024, temperature=0.3
        )
        data = _parse_json(raw)
        alts = data.get("alternates", [])
        logger.info(
            f"Groq: {len(alts)} alternates for '{failed_place.get('place_name')}' "
            f"(free_slots={free_slots})"
        )
        return alts
    except Exception as e:
        logger.warning(f"suggest_alternates_llm failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────
# Enrich a single place
# ─────────────────────────────────────────────────────────────
async def enrich_place_with_perplexity(place_name: str, city: str) -> dict:
    """
    Enrich a single place using Groq (function name kept for backward compat).
    city is always passed so Groq grounds its answer to the right location.
    """
    prompt = f"""Provide factual information about the tourist attraction "{place_name}" located in {city}, India.
If this place does not physically exist in {city}, return empty strings and zeros.
Return ONLY this JSON:
{{
  "opening_hours": "9:00 AM - 6:00 PM",
  "closed_on": ["Tuesday"],
  "entry_fee_indian": 20,
  "entry_fee_foreign": 300,
  "best_time_to_visit": "morning",
  "avg_visit_duration_hrs": 1.5,
  "local_tip": "short practical tip here",
  "nearby_food": "name of a nearby restaurant or food street"
}}"""
    try:
        raw = await _call_groq([{"role": "user", "content": prompt}], max_tokens=512, temperature=0.1)
        return _parse_json(raw)
    except Exception as e:
        logger.warning(f"enrich_place failed for '{place_name}': {e}")
        return {}
