from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from enum import Enum
from datetime import datetime


class BudgetLevel(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"

class TravelType(str, Enum):
    solo = "solo"
    couple = "couple"
    family = "family"
    group = "group"

class TripMood(str, Enum):
    relaxed = "relaxed"
    adventure = "adventure"
    spiritual = "spiritual"
    romantic = "romantic"
    cultural = "cultural"
    foodie = "foodie"

class PlaceCategory(str, Enum):
    beach = "beach"
    temple = "temple"
    museum = "museum"
    park = "park"
    heritage = "heritage"
    food = "food"
    shopping = "shopping"
    adventure = "adventure"
    viewpoint = "viewpoint"
    other = "other"


class TripRequest(BaseModel):
    destination: str = Field(..., example="Chennai")
    days: int = Field(..., ge=1, le=14, example=2)
    budget: BudgetLevel = Field(default=BudgetLevel.medium)
    travel_type: TravelType = Field(default=TravelType.solo)
    mood: TripMood = Field(default=TripMood.cultural)
    interests: List[str] = Field(
        default=["temples", "beach", "history"],
        example=["temples", "beach", "food"]
    )
    travel_dates: Optional[str] = Field(None, example="2026-03-10")
    avoid_crowded: bool = Field(default=False)
    accessibility_needs: bool = Field(default=False)


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
    closed_on: Optional[List[str]] = None
    best_time_to_visit: Optional[str] = None
    rating: Optional[float] = None
    photo_url: Optional[str] = None
    nearby_food: Optional[str] = None
    warning: Optional[str] = None
    source: Optional[str] = None


class ItineraryMeta(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    destination: str
    days: int
    travel_type: str
    budget: str
    mood: str
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    model_used: str = "groq/llama-3.3-70b-versatile"

class ItineraryResponse(BaseModel):
    success: bool
    meta: Optional[ItineraryMeta] = None
    itinerary: List[PlaceStop]
    weather_warnings: Optional[List[str]] = None


class PlaceEnrichRequest(BaseModel):
    place_name: str = Field(..., example="Kapaleeshwarar Temple")
    city: str = Field(..., example="Chennai")

class PlaceEnrichResponse(BaseModel):
    place_name: str
    city: str
    opening_hours: Optional[str] = None
    closed_on: Optional[List[str]] = None
    entry_fee_indian: Optional[int] = None
    entry_fee_foreign: Optional[int] = None
    best_time_to_visit: Optional[str] = None
    avg_visit_duration_hrs: Optional[float] = None
    local_tip: Optional[str] = None
    nearby_food: Optional[str] = None


class UserRating(BaseModel):
    user_id: str
    place_id: str
    place_name: str
    rating: int = Field(..., ge=1, le=5)
    review: Optional[str] = None


class PlaceResult(BaseModel):
    source: str
    place_id: Optional[str] = None
    place_name: str
    lat: Optional[float] = None
    lon: Optional[float] = None
    address: Optional[str] = None
    rating: Optional[float] = None
    opening_hours: Optional[str] = None
    tags: Optional[List[str]] = None
    photo_url: Optional[str] = None

class PlaceSearchResponse(BaseModel):
    success: bool
    destination: str
    category: str
    count: int
    places: List[PlaceResult]
