import json
import os
from dotenv import load_dotenv
from openai import AsyncOpenAI
from models.schemas import TripRequest

load_dotenv()  # Ensure .env is loaded when this module is imported directly

# ─────────────────────────────────────────────────────────────
# Lazy Perplexity client — initialized only when first called
# Avoids OpenAIError at import time if .env not yet loaded
# ─────────────────────────────────────────────────────────────
_client = None

def get_client() -> AsyncOpenAI:
    """
    Returns a cached Perplexity AsyncOpenAI client.
    Raises a clear error if PERPLEXITY_API_KEY is missing.
    """
    global _client
    if _client is None:
        api_key = os.getenv("PERPLEXITY_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "PERPLEXITY_API_KEY is not set. "
                "Add it to your .env file. "
                "Get your key at: https://www.perplexity.ai/settings/api"
            )
        _client = AsyncOpenAI(
            api_key=api_key,
            base_url="https://api.perplexity.ai"
        )
    return _client

# Available Perplexity models:
# sonar-pro           → Best quality + real-time web search grounding
# sonar               → Fast, cost-effective
# sonar-reasoning     → Deep reasoning (like o1)
# sonar-reasoning-pro → Best reasoning quality
PERPLEXITY_MODEL = os.getenv("PERPLEXITY_MODEL", "sonar-pro")


# ─────────────────────────────────────────────────────────────
# Generate full itinerary using Perplexity sonar-pro
# ─────────────────────────────────────────────────────────────
async def generate_itinerary_llm(req: TripRequest) -> list:
    """
    Use Perplexity sonar-pro to generate a structured day-wise itinerary.
    sonar-pro uses real-time web search grounding — ensures up-to-date
    place info, opening hours, and entry fees for Indian attractions.
    Returns a list of PlaceStop dicts.
    """
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
    - Budget: {req.budget} (low=budget stays & local food | medium=mid-range | high=luxury)
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
    6. Include entry fees (in INR, 0 if free).
    7. Add a short local tip for each place (max 15 words).
    8. Prioritize lesser-known gems alongside famous landmarks.

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

    response = await client.chat.completions.create(
        model=PERPLEXITY_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=4000
    )

    raw = response.choices[0].message.content

    # Strip markdown code blocks if Perplexity wraps output
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()

    data = json.loads(raw)

    # Handle {"itinerary": [...]} or direct [...]
    if isinstance(data, dict):
        return list(data.values())[0]
    return data


# ─────────────────────────────────────────────────────────────
# Enrich a single place with real-time info via Perplexity sonar
# ─────────────────────────────────────────────────────────────
async def enrich_place_with_perplexity(place_name: str, city: str) -> dict:
    """
    Use Perplexity sonar (fast model) to fetch real-time details for a place.
    Returns opening hours, entry fees, best visit time, and local tips.
    """
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

    response = await client.chat.completions.create(
        model="sonar",   # Use faster/cheaper sonar for single-place enrichment
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=500
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
