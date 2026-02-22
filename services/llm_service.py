import json
import os
import logging
from dotenv import load_dotenv
from openai import AsyncOpenAI, AuthenticationError, APIStatusError
from models.schemas import TripRequest

load_dotenv()

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# Lazy Perplexity client
# ─────────────────────────────────────────────────────────────
_client = None

def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "PERPLEXITY_API_KEY is not set.\n"
                "Fix: Add PERPLEXITY_API_KEY=pplx-xxxx to your .env file.\n"
                "Get key: https://www.perplexity.ai/settings/api"
            )
        # Reset cached client if key changes (e.g., during dev)
        _client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.perplexity.ai"
        )
        logger.info(f"Perplexity client initialized (key: ...{api_key[-6:]})")
    return _client


PERPLEXITY_MODEL = os.getenv("PERPLEXITY_MODEL", "sonar-pro")


# ─────────────────────────────────────────────────────────────
# Generate itinerary via Perplexity sonar-pro
# ─────────────────────────────────────────────────────────────
async def generate_itinerary_llm(req: TripRequest) -> list:
    client = get_client()

    system_prompt = """
    You are an expert Indian travel planner with deep knowledge of tourist
    destinations, local culture, timings, and travel logistics across India.
    Always return factual, up-to-date information about places.
    Return ONLY valid JSON — no markdown, no explanation, no extra text.
    """

    user_prompt = f"""
    Create a detailed {req.days}-day travel itinerary for {req.destination}, India.

    Traveller Profile:
    - Budget: {req.budget}
    - Travel type: {req.travel_type}
    - Mood/Theme: {req.mood}
    - Interests: {', '.join(req.interests)}
    - Travel date: {req.travel_dates or 'flexible'}
    - Avoid crowded spots: {req.avoid_crowded}
    - Accessibility needs: {req.accessibility_needs}

    Planning Rules:
    1. Schedule places in logical geographic order to minimize travel time.
    2. Realistic timings: morning 06:00-12:00, afternoon 12:00-17:00, evening 17:00-21:00.
    3. Include meal stops at breakfast (08:00), lunch (13:00), and dinner (19:30).
    4. Estimate visit duration in hours for each stop.
    5. Respect actual Indian attraction opening/closing hours.
    6. Include entry fees in INR (0 if free).
    7. Add a short local tip for each place (max 15 words).
    8. Mix famous landmarks with lesser-known gems.

    Return ONLY this exact JSON structure:
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
            model=PERPLEXITY_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=0.2,
            max_tokens=4000
        )
    except AuthenticationError:
        raise ValueError(
            "Perplexity API key is invalid or expired (401 Unauthorized).\n"
            "Fix: Update PERPLEXITY_API_KEY in your .env file.\n"
            "Get a new key: https://www.perplexity.ai/settings/api"
        )
    except APIStatusError as e:
        raise ValueError(f"Perplexity API error {e.status_code}: {e.message}")

    raw = response.choices[0].message.content
    logger.debug(f"Raw LLM response (first 200 chars): {raw[:200]}")

    # Strip markdown code fences if present
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM returned invalid JSON: {e}\nRaw: {raw[:300]}")

    if isinstance(data, dict):
        return list(data.values())[0]
    return data


# ─────────────────────────────────────────────────────────────
# Enrich a single place via Perplexity sonar
# ─────────────────────────────────────────────────────────────
async def enrich_place_with_perplexity(place_name: str, city: str) -> dict:
    client = get_client()

    prompt = f"""
    Provide current factual information about "{place_name}" in {city}, India.
    Return ONLY JSON with these exact fields:
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
            model="sonar",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=500
        )
    except AuthenticationError:
        raise ValueError(
            "Perplexity API key is invalid (401). "
            "Update PERPLEXITY_API_KEY in .env"
        )

    raw = response.choices[0].message.content
    try:
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        return json.loads(raw)
    except Exception:
        return {}
