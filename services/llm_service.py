import json
import os
import logging
from dotenv import load_dotenv
from models.schemas import TripRequest

load_dotenv()

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Groq Cloud — FREE 14,400 req/day, no credit card
# Uses official groq-python SDK (async native)
# Get key: https://console.groq.com/keys
# Docs:    https://console.groq.com/docs/openai
# ─────────────────────────────────────────────────────────────
_client = None

# Available Groq free-tier models:
# llama-3.3-70b-versatile  → Best quality, 32K ctx, 14,400 req/day
# llama3-70b-8192          → Solid quality, 8K ctx
# llama3-8b-8192           → Fastest, lightest
# mixtral-8x7b-32768       → Great for structured JSON, 32K ctx
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def get_client():
    """
    Returns a cached Groq AsyncGroq client.
    Raises a clear error if GROQ_API_KEY is missing.
    """
    global _client
    if _client is not None:
        return _client

    try:
        from groq import AsyncGroq
    except ImportError:
        raise ImportError(
            "groq package not installed.\n"
            "Fix: pip install groq"
        )

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is not set.\n"
            "Fix: Add GROQ_API_KEY=gsk_... to your .env file.\n"
            "Get FREE key (no credit card): https://console.groq.com/keys"
        )

    _client = AsyncGroq(api_key=api_key)
    logger.info(f"Groq client ready ✅  model={GROQ_MODEL}  key=...{api_key[-6:]}")
    return _client


# ─────────────────────────────────────────────────────────────
# Generate full itinerary via Groq (fully async)
# ─────────────────────────────────────────────────────────────
async def generate_itinerary_llm(req: TripRequest) -> list:
    """
    Generate a day-wise itinerary using Groq llama-3.3-70b-versatile.
    Forces JSON output via response_format={"type": "json_object"}.
    Fully async — no thread pool needed.
    """
    client = get_client()

    system_prompt = (
        "You are an expert Indian travel planner with deep knowledge of tourist "
        "destinations, local culture, opening times, and travel logistics across India. "
        "Always return factual information. "
        "Return ONLY valid JSON — no markdown, no explanation."
    )

    user_prompt = f"""
    Create a detailed {req.days}-day travel itinerary for {req.destination}, India.

    Traveller Profile:
    - Budget: {req.budget}  (low=budget stays & street food | medium=mid-range | high=luxury)
    - Travel type: {req.travel_type}
    - Mood/Theme: {req.mood}
    - Interests: {', '.join(req.interests)}
    - Travel date: {req.travel_dates or 'flexible'}
    - Avoid crowded spots: {req.avoid_crowded}
    - Accessibility needs: {req.accessibility_needs}

    Planning Rules:
    1. Schedule places in logical geographic order to minimize travel time.
    2. Realistic timings: morning 06:00-12:00, afternoon 12:00-17:00, evening 17:00-21:00.
    3. Include meal stops: breakfast (08:00), lunch (13:00), dinner (19:30).
    4. Estimate visit duration in hours for each stop.
    5. Use actual Indian attraction opening/closing hours.
    6. Include entry fees in INR (0 if free).
    7. Add a short practical local tip per place (max 15 words).
    8. Mix famous landmarks with lesser-known gems.

    Return ONLY this JSON structure:
    {{
      "itinerary": [
        {{
          "day": 1,
          "time": "06:00",
          "place_name": "Marina Beach",
          "category": "beach",
          "duration_hrs": 1.5,
          "entry_fee": 0,
          "tip": "Visit at sunrise to avoid weekend crowds"
        }}
      ]
    }}
    """

    try:
        response = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            response_format={"type": "json_object"},  # ← Forces clean JSON output
            temperature=0.2,
            max_tokens=4096
        )
    except Exception as e:
        err = str(e)
        if "401" in err or "invalid_api_key" in err.lower():
            raise ValueError(
                "Groq API key is invalid (401).\n"
                "Fix: Update GROQ_API_KEY in .env.\n"
                "Get key: https://console.groq.com/keys"
            )
        if "429" in err or "rate_limit" in err.lower():
            raise ValueError(
                "Groq rate limit hit (429).\n"
                "Free tier: 30 req/min, 14,400 req/day.\n"
                "Wait a minute and retry."
            )
        raise ValueError(f"Groq API error: {err}")

    raw = response.choices[0].message.content
    logger.debug(f"Groq raw (first 200): {raw[:200]}")

    # Strip markdown fences (safety fallback)
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from Groq: {e}\nRaw: {raw[:300]}")

    result = data.get("itinerary") if isinstance(data, dict) else data
    logger.info(f"Groq generated {len(result)} stops for {req.destination}")
    return result


# ─────────────────────────────────────────────────────────────
# Enrich a single place via Groq
# ─────────────────────────────────────────────────────────────
async def enrich_place_with_perplexity(place_name: str, city: str) -> dict:
    """
    Fetch real-time details for a specific place using Groq.
    Function name kept for backward compatibility.
    """
    client = get_client()

    prompt = f"""
    Provide factual information about "{place_name}" in {city}, India.
    Return ONLY this JSON, no extra text:
    {{
      "opening_hours": "9:00 AM - 6:00 PM",
      "closed_on": ["Tuesday"],
      "entry_fee_indian": 20,
      "entry_fee_foreign": 300,
      "best_time_to_visit": "morning",
      "avg_visit_duration_hrs": 1.5,
      "local_tip": "short practical tip here",
      "nearby_food": "name of a nearby restaurant or food street"
    }}
    """

    try:
        response = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=512
        )
        raw = response.choices[0].message.content
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Place enrichment failed for '{place_name}': {e}")
        return {}
