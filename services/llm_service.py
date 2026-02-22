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
# Stage 1 — Generate top/must-visit PLACE CANDIDATES (no meals)
# ─────────────────────────────────────────────────────────────
async def generate_place_candidates_llm(req: TripRequest) -> list:
    """
    Ask Groq for top/must-visit tourist attractions ONLY.
    NO meals, NO restaurants, NO hotels.
    Returns more candidates than needed so the scheduler has options.
    """
    num = max(req.days * 7, 14)  # Ample candidates for scheduler to pick from

    system_prompt = (
        "You are an expert Indian travel planner. "
        "Suggest ONLY tourist attractions — temples, beaches, museums, forts, parks, viewpoints, heritage sites. "
        "NEVER suggest restaurants, cafes, dhabas, hotels, or any food/meal stops. "
        "Return ONLY valid JSON."
    )

    user_prompt = f"""Suggest the top {num} must-visit tourist attractions in {req.destination}, India.

Traveller profile:
- Budget: {req.budget.value}\n- Travel type: {req.travel_type.value}
- Mood/Theme: {req.mood.value}\n- Interests: {', '.join(req.interests)}
- Avoid crowded: {req.avoid_crowded}\n- Accessibility needs: {req.accessibility_needs}

Rules:
1. ATTRACTIONS ONLY — zero restaurants, zero cafes, zero meal stops.
2. Mix 70% famous/iconic + 30% lesser-known gems.
3. priority: 5=must-visit, 4=highly recommended, 3=good, 2=if time permits, 1=optional.
4. Realistic visit duration in hours (0.5 to 4.0).
5. best_slot: morning (09-12) | afternoon (13-16) | evening (16:30-19:30) | night (20-21:30).
6. Use actual opening hours and INR entry fees (0 if free).
7. One practical tip per place (max 15 words).

Return ONLY this JSON:
{{
  "candidates": [
    {{
      "place_name": "Marina Beach",
      "category": "beach",
      "priority": 5,
      "duration_hrs": 1.5,
      "best_slot": "morning",
      "why_must_visit": "Longest natural urban beach in India",
      "opening_hours": "Open 24hrs",
      "closed_on": [],
      "entry_fee": 0,
      "tip": "Visit at sunrise for golden light and fewer crowds"
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
# Stage 3 — Conflict resolver: ask Groq for alternate places
# ─────────────────────────────────────────────────────────────
async def suggest_alternates_llm(
    destination: str,
    failed_place: dict,
    scheduled_places: list,
    slot_name: str
) -> list:
    """
    When a candidate cannot be scheduled, ask Groq for shorter/alternative places.
    Returns list of alternate candidate dicts.
    """
    scheduled_names = [s.get("place_name", "") for s in scheduled_places]
    max_dur = max(0.5, round(failed_place.get("duration_hrs", 1.5) - 0.5, 1))

    system_prompt = (
        "You are an expert Indian travel planner. "
        "Suggest alternative TOURIST ATTRACTIONS only — no restaurants or meals. "
        "Return ONLY valid JSON."
    )

    user_prompt = f"""Building a trip itinerary for {destination}, India.

The following place CANNOT be scheduled (timing conflict):
- Place: {failed_place.get('place_name')}
- Category: {failed_place.get('category', 'general')}
- Failed slot: {slot_name}

Already scheduled (DO NOT suggest these):
{', '.join(scheduled_names) if scheduled_names else 'None'}

Suggest 3 alternative tourist attractions that:
1. Same or similar category as "{failed_place.get('category', 'general')}".
2. Shorter visit duration (≤ {max_dur} hours).
3. Can fit in the {slot_name} time slot.
4. NOT already in the scheduled list.
5. ATTRACTIONS ONLY — no restaurants, no hotels.

Return ONLY this JSON:
{{
  "alternates": [
    {{
      "place_name": "Santhome Cathedral",
      "category": "heritage",
      "priority": 4,
      "duration_hrs": 1.0,
      "best_slot": "{slot_name}",
      "why_must_visit": "Historic Portuguese cathedral on the beach",
      "opening_hours": "6:00 AM - 8:00 PM",
      "closed_on": [],
      "entry_fee": 0,
      "tip": "Visit on weekday mornings to avoid crowds"
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
        logger.info(f"Groq: {len(alts)} alternates for '{failed_place.get('place_name')}'")
        return alts
    except Exception as e:
        logger.warning(f"suggest_alternates_llm failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────
# Enrich a single place (backward compat)
# ─────────────────────────────────────────────────────────────
async def enrich_place_with_perplexity(place_name: str, city: str) -> dict:
    """
    Enrich a single place using Groq (function name kept for backward compat).
    """
    prompt = f"""Provide factual information about "{place_name}" in {city}, India.
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
