import json
import os
import logging
from dotenv import load_dotenv
from models.schemas import TripRequest

load_dotenv()

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Lazy Gemini client — initialized only on first call
# Avoids crash at import time if GEMINI_API_KEY not yet loaded
# ─────────────────────────────────────────────────────────────
_model = None

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

def get_model():
    """
    Returns a cached Gemini GenerativeModel instance.
    Raises a clear error if GEMINI_API_KEY is missing.
    """
    global _model
    if _model is not None:
        return _model

    try:
        import google.generativeai as genai
    except ImportError:
        raise ImportError(
            "google-generativeai package not installed.\n"
            "Fix: pip install google-generativeai"
        )

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY is not set.\n"
            "Fix: Add GEMINI_API_KEY=AIza... to your .env file.\n"
            "Get key: https://aistudio.google.com/app/apikey"
        )

    genai.configure(api_key=api_key)
    _model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        generation_config={
            "temperature": 0.2,
            "max_output_tokens": 4096,
            "response_mime_type": "application/json",   # Force JSON output
        }
    )
    logger.info(f"Gemini client initialized ✅ (model: {GEMINI_MODEL}, key: ...{api_key[-6:]})")
    return _model


# ─────────────────────────────────────────────────────────────
# Generate full itinerary via Gemini 1.5 Flash
# ─────────────────────────────────────────────────────────────
async def generate_itinerary_llm(req: TripRequest) -> list:
    """
    Use Gemini 1.5 Flash to generate a structured day-wise itinerary.
    Returns a list of PlaceStop dicts.
    """
    import asyncio
    model = get_model()

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
    4. Estimate visit duration in hours for each stop.
    5. Use actual Indian attraction opening/closing hours.
    6. Include entry fees in INR (0 if free).
    7. Add a short practical local tip per place (max 15 words).
    8. Mix famous landmarks with lesser-known gems.

    Return ONLY this JSON structure, no extra text:
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
        # Gemini SDK is sync — run in thread pool to avoid blocking async server
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content(prompt)
        )
    except Exception as e:
        err = str(e)
        if "API_KEY_INVALID" in err or "API key not valid" in err:
            raise ValueError(
                "Gemini API key is invalid (403).\n"
                "Fix: Update GEMINI_API_KEY in your .env file.\n"
                "Get key: https://aistudio.google.com/app/apikey"
            )
        raise ValueError(f"Gemini API error: {err}")

    raw = response.text
    logger.debug(f"Gemini raw response (first 200): {raw[:200]}")

    # Strip markdown fences if present (safety fallback)
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Gemini returned invalid JSON: {e}\nRaw: {raw[:300]}")

    # Handle {"itinerary": [...]} or direct [...]
    if isinstance(data, dict):
        result = data.get("itinerary") or list(data.values())[0]
    else:
        result = data

    logger.info(f"Gemini generated {len(result)} stops for {req.destination}")
    return result


# ─────────────────────────────────────────────────────────────
# Enrich a single place via Gemini Flash
# ─────────────────────────────────────────────────────────────
async def enrich_place_with_perplexity(place_name: str, city: str) -> dict:
    """
    Use Gemini Flash to fetch real-time details for a specific place.
    Function name kept for backward compatibility with existing routes.
    """
    import asyncio
    model = get_model()

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
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model.generate_content(prompt)
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
