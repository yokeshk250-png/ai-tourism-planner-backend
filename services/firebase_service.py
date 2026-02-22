import os
import firebase_admin
from firebase_admin import credentials, firestore
from datetime import datetime

# Initialize Firebase only once
if not firebase_admin._apps:
    cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "serviceAccountKey.json")
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)

db = firestore.client()

async def save_itinerary(user_id: str, itinerary: dict) -> str:
    """
    Save a generated itinerary to Firestore under users/{user_id}/itineraries
    """
    doc_ref = db.collection("users").document(user_id) \
                .collection("itineraries").document()
    doc_ref.set({
        **itinerary,
        "created_at": datetime.utcnow(),
        "user_id": user_id
    })
    return doc_ref.id

async def get_itinerary(itinerary_id: str) -> dict:
    """
    Fetch a specific itinerary by ID from Firestore.
    """
    # Search across all users (admin query)
    docs = db.collection_group("itineraries") \
              .where("__name__", "==", itinerary_id).stream()
    for doc in docs:
        return doc.to_dict()
    return None

async def save_place_rating(user_id: str, place_id: str, rating: int, review: str = ""):
    """
    Save user rating for a visited place.
    """
    db.collection("ratings").document(f"{user_id}_{place_id}").set({
        "user_id": user_id,
        "place_id": place_id,
        "rating": rating,
        "review": review,
        "rated_at": datetime.utcnow()
    })
