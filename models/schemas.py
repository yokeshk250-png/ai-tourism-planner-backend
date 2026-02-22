from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional
from enum import Enum
from datetime import datetime


# ─────────────────────────────────────────
class BudgetLevel(str, Enum):
    low    = "low"
    medium = "medium"
    high   = "high"

class TravelType(str, Enum):
    solo   = "solo"
    couple = "couple"
    family = "family"
    group  = "group"

class TripMood(str, Enum):
    relaxed   = "relaxed"
    adventure = "adventure"
    spiritual = "spiritual"
    romantic  = "romantic"
    cultural  = "cultural"
    foodie    = "foodie"


# ─────────────────────────────────────────
class TripRequest(BaseModel):
    destination:         str         = Field(..., example="Chennai")
    days:                int         = Field(..., ge=1, le=14, example=2)
    budget:              BudgetLevel = Field(default=BudgetLevel.medium)
    travel_type:         TravelType  = Field(default=TravelType.solo)
    mood:                TripMood    = Field(default=TripMood.cultural)
    interests:           List[str]   = Field(default=["temples", "beach", "history"])
    travel_dates:        Optional[str] = Field(None, example="2026-03-10")
    avoid_crowded:       bool        = Field(default=False)
    accessibility_needs: bool        = Field(default=False)


# ─────────────────────────────────────────
# Candidate  (raw from Groq, before scheduling)
# ─────────────────────────────────────────
class PlaceCandidate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    place_name:        str
    category:          Optional[str]       = None
    priority:          int                 = 3
    duration_hrs:      float               = 1.0
    best_slot:         Optional[str]       = None
    why_must_visit:    Optional[str]       = None
    opening_hours:     Optional[str]       = None
    closed_on:         Optional[List[str]] = None
    entry_fee:         Optional[int]       = None
    entry_fee_foreign: Optional[int]       = None
    tip:               Optional[str]       = None
    nearby_food:       Optional[str]       = None
    lat:               Optional[float]     = None
    lon:               Optional[float]     = None
    is_alternate:      bool                = False


# ─────────────────────────────────────────
# Scheduled stop  (placed by scheduler)
# ─────────────────────────────────────────
class ScheduledStop(BaseModel):
    """
    A place placed into a time slot by the scheduler.
    Fields are populated from enrichment data (opening hrs, fees, duration)
    then used to enforce scheduling rules.

    opening_hours_unverified=True means opening hours exist but couldn’t be
    parsed — the stop was scheduled in lenient mode, verify manually.
    """
    model_config = ConfigDict(extra="ignore")

    day:                      int
    slot_id:                  str
    slot_name:                str
    start_time:               str
    end_time:                 str
    place_name:               str
    category:                 Optional[str]       = None
    priority:                 int                 = 3
    duration_hrs:             float
    travel_mins_from_prev:    int                 = 0
    # ─ from enrich ─
    opening_hours:            Optional[str]       = None
    closed_on:                Optional[List[str]] = None
    entry_fee:                Optional[int]       = None     # INR Indian fee
    entry_fee_foreign:        Optional[int]       = None     # INR / USD Foreign fee
    nearby_food:              Optional[str]       = None     # nearest food spot
    tip:                      Optional[str]       = None
    # ─ from geocode ─
    lat:                      Optional[float]     = None
    lon:                      Optional[float]     = None
    # ─ from LLM candidate ─
    why_must_visit:           Optional[str]       = None
    is_alternate:             bool                = False
    # ─ scheduler-set flags ─
    opening_hours_unverified: bool                = False


class TimeSlot(BaseModel):
    slot_id:        str
    day:            int
    slot_name:      str
    start_time:     str
    end_time:       str
    available_mins: int
    remaining_mins: int
    meal_gap_after: Optional[str] = None


class ItineraryMeta(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    destination:            str
    days:                   int
    travel_type:            str
    budget:                 str
    mood:                   str
    generated_at:           str   = Field(default_factory=lambda: datetime.utcnow().isoformat())
    model_used:             str   = "groq/llama-3.3-70b-versatile"
    total_places:           int   = 0
    unscheduled_count:      int   = 0
    hours_unverified_count: int   = 0


class ItineraryResponse(BaseModel):
    success:          bool
    meta:             Optional[ItineraryMeta]        = None
    slot_template:    Optional[List[TimeSlot]]       = None
    itinerary:        List[ScheduledStop]
    unscheduled:      Optional[List[PlaceCandidate]] = None
    weather_warnings: Optional[List[str]]            = None


# ─────────────────────────────────────────
# Places / enrich schemas
# ─────────────────────────────────────────
class PlaceStop(BaseModel):
    model_config = ConfigDict(extra="ignore")
    day:          int
    time:         str
    place_name:   str
    place_id:     Optional[str]   = None
    category:     Optional[str]   = None
    duration_hrs: float
    entry_fee:    Optional[int]   = None
    tip:          Optional[str]   = None
    lat:          Optional[float] = None
    lon:          Optional[float] = None


class PlaceEnrichRequest(BaseModel):
    place_name: str = Field(..., example="Kapaleeshwarar Temple")
    city:       str = Field(..., example="Chennai")


class PlaceEnrichResponse(BaseModel):
    place_name:             str
    city:                   str
    opening_hours:          Optional[str]       = None
    closed_on:              Optional[List[str]] = None
    entry_fee_indian:       Optional[int]       = None
    entry_fee_foreign:      Optional[int]       = None
    best_time_to_visit:     Optional[str]       = None
    avg_visit_duration_hrs: Optional[float]     = None
    local_tip:              Optional[str]       = None
    nearby_food:            Optional[str]       = None


class UserRating(BaseModel):
    user_id:    str
    place_id:   str
    place_name: str
    rating:     int = Field(..., ge=1, le=5)
    review:     Optional[str] = None


class PlaceResult(BaseModel):
    source:        str
    place_id:      Optional[str]       = None
    place_name:    str
    lat:           Optional[float]     = None
    lon:           Optional[float]     = None
    address:       Optional[str]       = None
    rating:        Optional[float]     = None
    opening_hours: Optional[str]       = None
    tags:          Optional[List[str]] = None
    photo_url:     Optional[str]       = None


class PlaceSearchResponse(BaseModel):
    success:     bool
    destination: str
    category:    str
    count:       int
    places:      List[PlaceResult]
