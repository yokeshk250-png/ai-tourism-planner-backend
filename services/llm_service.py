import json
import os
import logging
from dotenv import load_dotenv
from models.schemas import TripRequest

load_dotenv()

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Uses google-genai (new unified SDK) with gemini-2.0-flash
# pip install google-genai
# Docs: https://googleapis.github.io/python-genai/
# ─────────────────────────────────────────────────────────────
_client = None
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")


def get_client():
    """
    Returns a cached google-genai Client.
    Raises a clear error if GEMINI_API_KEY is missing.
    """
    global _client
    if _client is not None:
        return _client

    try:
        from google import genai
    except ImportError:
        raise ImportError(
            "google-genai package not installed.\n"
            "Fix: pip install google-genai"
        )

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set.\n"
            "Fix: Add GEMINI_API_KEY=AIza... to your .env file.\n"
            "Get FREE key: https://aistudio.google.com/app/apikey"
        )

    _client = genai.Client(api_key=api_key)
    logger.info(f"Gemini client ready ✅ model={GEMINI_MODEL} key=...{api_key[-6:]}")
    return _client


# ─────────────────────────────────────────────────────────────
# Generate itinerary — async via google-genai native async API
# ─────────────────────────────────────────────────────────────
async def generate_itinerary_llm(req: TripRequest) -> list:
    """
    Generate a day-wise itinerary using Gemini 2.0 Flash.
    Uses native async API (client.aio.models.generate_content).
    Forces JSON output via response_mime_type.
    """
    from google.genai import types

    client = get_client()

    prompt = f"""
    You are an expert Indian travel planner. Create a detailed {req.days}-day travel
    itinerary for {req.destination}, India.

    Traveller Profile:
    - Budget: {req.budget} (low=budget | medium=mid-range | high=luxury)
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
    4. Estimate visit duration in hours per stop.
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

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.2,
        max_output_tokens=4096,
    )

    try:
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=config
        )
    except Exception as e:
        err = str(e)
        if "API_KEY_INVALID" in err or "API key not valid" in err:
            raise ValueError(
                "Gemini API key is invalid.\n"
                "Fix: Update GEMINI_API_KEY in .env.\n"
                "Get key: https://aistudio.google.com/app/apikey"
            )
        if "404" in err or "not found" in err.lower():
            raise ValueError(
                f"Model '{GEMINI_MODEL}' not found.\n"
                "Fix: Check GEMINI_MODEL in .env. Valid: gemini-2.0-flash | gemini-1.5-flash-latest"
            )
        raise ValueError(f"Gemini API error: {err}")

    raw = response.text
    logger.debug(f"Gemini raw (first 200): {raw[:200]}")

    # Strip markdown fences if present
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON from Gemini: {e}\nRaw: {raw[:300]}")

    result = data.get("itinerary") if isinstance(data, dict) else data
    logger.info(f"Generated {len(result)} stops for {req.destination}")
    return result


# ─────────────────────────────────────────────────────────────
# Enrich a single place via Gemini Flash
# ─────────────────────────────────────────────────────────────
async def enrich_place_with_perplexity(place_name: str, city: str) -> dict:
    """
    Enrich a single place with real-time details using Gemini Flash.
    Function name kept for backward compatibility with existing routes.
    """
    from google.genai import types

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

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        temperature=0.1,
        max_output_tokens=512,
    )

    try:
        response = await client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=config
        )
        raw = response.text
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        return json.loads(raw)
    except Exception as e:
        logger.warning(f"Place enrichment failed for '{place_name}': {e}")
        return {}
