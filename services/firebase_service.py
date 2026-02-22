import os
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime
from typing import Optional, List

# ─────────────────────────────────────────
# Initialize Firebase once
# ─────────────────────────────────────────
if not firebase_admin._apps:
    cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "serviceAccountKey.json")
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)

db = firestore.client()


# ─────────────────────────────────────────
# Itinerary CRUD
# ─────────────────────────────────────────
async def save_itinerary(user_id: str, itinerary: dict) -> str:
    """
    Save a generated itinerary under users/{user_id}/itineraries.
    Returns the new document ID.
    """
    doc_ref = db.collection("users").document(user_id) \
                .collection("itineraries").document()
    doc_ref.set({
        **itinerary,
        "user_id": user_id,
        "created_at": datetime.utcnow().isoformat()
    })
    return doc_ref.id


async def get_itinerary(itinerary_id: str) -> Optional[dict]:
    """
    Fetch a specific itinerary document by ID (searches across all users).
    """
    docs = db.collection_group("itineraries").stream()
    for doc in docs:
        if doc.id == itinerary_id:
            return {"id": doc.id, **doc.to_dict()}
    return None


async def get_user_itineraries(user_id: str) -> List[dict]:
    """
    Fetch all saved itineraries for a specific user.
    Returns newest first.
    """
    docs = db.collection("users").document(user_id) \
              .collection("itineraries") \
              .order_by("created_at", direction=firestore.Query.DESCENDING) \
              .stream()
    return [{"id": doc.id, **doc.to_dict()} for doc in docs]


# ─────────────────────────────────────────
# User Ratings
# ─────────────────────────────────────────
async def save_place_rating(
    user_id: str,
    place_id: str,
    rating: int,
    review: str = ""
) -> None:
    """
    Save or overwrite a user's rating for a visited place.
    Document ID format: {user_id}_{place_id}
    """
    db.collection("ratings").document(f"{user_id}_{place_id}").set({
        "user_id": user_id,
        "place_id": place_id,
        "rating": rating,
        "review": review,
        "rated_at": datetime.utcnow().isoformat()
    })


async def get_place_ratings(place_id: str) -> List[dict]:
    """
    Fetch all user ratings for a specific place.
    Used to compute average rating for the recommendation engine.
    """
    docs = db.collection("ratings") \
              .where("place_id", "==", place_id) \
              .stream()
    ratings = [doc.to_dict() for doc in docs]
    return ratings


async def get_avg_rating(place_id: str) -> Optional[float]:
    """
    Compute the average user rating for a place from Firestore.
    """
    ratings = await get_place_ratings(place_id)
    if not ratings:
        return None
    return round(sum(r["rating"] for r in ratings) / len(ratings), 1)


# ─────────────────────────────────────────
# User Preferences
# ─────────────────────────────────────────
async def save_user_preferences(user_id: str, preferences: dict) -> None:
    """
    Save or update user's travel preferences profile in Firestore.
    """
    db.collection("users").document(user_id).set(
        {"preferences": preferences, "updated_at": datetime.utcnow().isoformat()},
        merge=True
    )


async def get_user_preferences(user_id: str) -> Optional[dict]:
    """
    Fetch a user's saved travel preferences.
    """
    doc = db.collection("users").document(user_id).get()
    if doc.exists:
        return doc.to_dict().get("preferences", {})
    return None
