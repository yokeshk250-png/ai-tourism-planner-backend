from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum

class BudgetLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

class TravelType(str, Enum):
    solo = "solo"
    couple = "couple"
    family = "family"
    group = "group"

class TripRequest(BaseModel):
    destination: str = Field(..., example="Chennai")
    days: int = Field(..., ge=1, le=14, example=2)
    budget: BudgetLevel = Field(default=BudgetLevel.medium)
    travel_type: TravelType = Field(default=TravelType.solo)
    interests: List[str] = Field(
        default=["temples", "beach", "history"],
        example=["temples", "beach", "food"]
    )
    travel_dates: Optional[str] = Field(None, example="2026-03-10")

class PlaceStop(BaseModel):
    day: int
    time: str
    place_name: str
    place_id: Optional[str] = None
    category: Optional[str] = None
    duration_hrs: float
    entry_fee: Optional[int] = None
    tip: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    opening_hours: Optional[str] = None
    rating: Optional[float] = None
    photo_url: Optional[str] = None
    warning: Optional[str] = None

class ItineraryResponse(BaseModel):
    success: bool
    itinerary: List[PlaceStop]
