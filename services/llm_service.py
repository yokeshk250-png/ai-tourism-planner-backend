import json
import os
from openai import AsyncOpenAI
from models.schemas import TripRequest

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

async def generate_itinerary_llm(req: TripRequest) -> list:
    """
    Use GPT-4o to generate a structured day-wise itinerary.
    Returns a list of PlaceStop dicts.
    """
    prompt = f"""
    You are an expert travel planner for Indian tourism.

    Create a {req.days}-day travel itinerary for {req.destination}.
    - Budget level: {req.budget}
    - Travel type: {req.travel_type}
    - Interests: {', '.join(req.interests)}
    - Travel date: {req.travel_dates or 'flexible'}

    Rules:
    1. Schedule places in logical geographic order to minimize travel time.
    2. Include realistic visit times (morning: 6-12, afternoon: 12-17, evening: 17-21).
    3. Estimate duration in hours for each stop.
    4. Include local food stops at meal times.
    5. Add a short practical tip for each place.
    6. Respect typical Indian attraction opening hours.

    Return ONLY a valid JSON array with this exact structure:
    [
      {{
        "day": 1,
        "time": "06:00",
        "place_name": "Marina Beach",
        "category": "beach",
        "duration_hrs": 1.5,
        "entry_fee": 0,
        "tip": "Visit at sunrise to avoid crowds"
      }}
    ]
    """

    response = await client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.7
    )

    content = response.choices[0].message.content
    data = json.loads(content)

    # Handle both {"itinerary": [...]} and direct [...] responses
    if isinstance(data, dict):
        return list(data.values())[0]
    return data
