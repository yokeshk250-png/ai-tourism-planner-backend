import json
import os
from openai import AsyncOpenAI
from models.schemas import TripRequest

# ─────────────────────────────────────────────
# Perplexity API client
# Uses OpenAI-compatible SDK with custom base_url
# Docs: https://docs.perplexity.ai/api-reference
# ─────────────────────────────────────────────
client = AsyncOpenAI(
    api_key=os.getenv("PERPLEXITY_API_KEY"),
    base_url="https://api.perplexity.ai"
)

# Available Perplexity models:
# sonar-pro         → Best quality, supports web search grounding
# sonar             → Fast, cost-effective
# sonar-reasoning   → Deep reasoning (like o1)
# sonar-reasoning-pro → Best reasoning quality
PERPLEXITY_MODEL = os.getenv("PERPLEXITY_MODEL", "sonar-pro")

async def generate_itinerary_llm(req: TripRequest) -> list:
    """
    Use Perplexity sonar-pro to generate a structured day-wise itinerary.
    sonar-pro uses real-time web search grounding — ensures up-to-date
    place info, opening hours, and entry fees for Indian attractions.
    Returns a list of PlaceStop dicts.
    """
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
    - Interests: {', '.join(req.interests)}
    - Travel date: {req.travel_dates or 'flexible'}

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
        temperature=0.2,   # Low temp for factual, consistent output
        max_tokens=4000
    )

    raw = response.choices[0].message.content

    # Strip markdown code blocks if present
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()

    data = json.loads(raw)

    # Handle {"itinerary": [...]} or direct [...]
    if isinstance(data, dict):
        return list(data.values())[0]
    return data


async def enrich_place_with_perplexity(place_name: str, city: str) -> dict:
    """
    Use Perplexity sonar to fetch real-time details for a specific place.
    Great for getting up-to-date opening hours, entry fees, and local tips.
    """
    prompt = f"""
    Provide current factual information about "{place_name}" in {city}, India.
    Return ONLY JSON with these fields:
    {{
      "opening_hours": "9:00 AM - 6:00 PM",
      "closed_on": ["Tuesday"],
      "entry_fee_indian": 20,
      "entry_fee_foreign": 300,
      "best_time_to_visit": "morning",
      "avg_visit_duration_hrs": 1.5,
      "local_tip": "short tip here",
      "nearby_food": "restaurant name nearby"
    }}
    """

    response = await client.chat.completions.create(
        model="sonar",  # Use faster sonar for enrichment
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=500
    )

    raw = response.choices[0].message.content
    try:
        if "```" in raw:
            raw = raw.split("```json")[-1].split("```")[0].strip() if "```json" in raw \
                  else raw.split("```")[1].split("```")[0].strip()
        return json.loads(raw)
    except Exception:
        return {}
