import os
import logging
from datetime import datetime
from typing import Optional, List
from dotenv import load_dotenv

load_dotenv()  # ← Ensure .env is loaded before Firebase init

logger = logging.getLogger(__name__)

_db = None


def get_db():
    global _db
    if _db is not None:
        return _db
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        if not firebase_admin._apps:
            cred_path = os.getenv("FIREBASE_CREDENTIALS_PATH", "serviceAccountKey.json")
            if not os.path.exists(cred_path):
                raise FileNotFoundError(
                    f"Firebase credentials not found at '{cred_path}'. "
                    "Download serviceAccountKey.json from Firebase Console "
                    "and place it in the project root."
                )
            cred = credentials.Certificate(cred_path)
            firebase_admin.initialize_app(cred)
            logger.info("Firebase initialized ✅")

        _db = firestore.client()
        return _db
    except Exception as e:
        logger.warning(f"Firebase unavailable: {e}")
        return None


async def save_itinerary(user_id: str, itinerary: dict) -> str:
    db = get_db()
    if not db:
        raise RuntimeError("Firebase not configured. Add serviceAccountKey.json to project root.")
    doc_ref = db.collection("users").document(user_id) \
                .collection("itineraries").document()
    doc_ref.set({**itinerary, "user_id": user_id, "created_at": datetime.utcnow().isoformat()})
    return doc_ref.id


async def get_itinerary(itinerary_id: str) -> Optional[dict]:
    db = get_db()
    if not db:
        return None
    docs = db.collection_group("itineraries").stream()
    for doc in docs:
        if doc.id == itinerary_id:
            return {"id": doc.id, **doc.to_dict()}
    return None


async def get_user_itineraries(user_id: str) -> List[dict]:
    from firebase_admin import firestore as fs
    db = get_db()
    if not db:
        return []
    docs = db.collection("users").document(user_id) \
              .collection("itineraries") \
              .order_by("created_at", direction=fs.Query.DESCENDING) \
              .stream()
    return [{"id": doc.id, **doc.to_dict()} for doc in docs]


async def save_place_rating(user_id: str, place_id: str, rating: int, review: str = "") -> None:
    db = get_db()
    if not db:
        raise RuntimeError("Firebase not configured.")
    db.collection("ratings").document(f"{user_id}_{place_id}").set({
        "user_id": user_id, "place_id": place_id,
        "rating": rating, "review": review,
        "rated_at": datetime.utcnow().isoformat()
    })


async def get_place_ratings(place_id: str) -> List[dict]:
    db = get_db()
    if not db:
        return []
    docs = db.collection("ratings").where("place_id", "==", place_id).stream()
    return [doc.to_dict() for doc in docs]


async def get_avg_rating(place_id: str) -> Optional[float]:
    ratings = await get_place_ratings(place_id)
    if not ratings:
        return None
    return round(sum(r["rating"] for r in ratings) / len(ratings), 1)


async def save_user_preferences(user_id: str, preferences: dict) -> None:
    db = get_db()
    if not db:
        return
    db.collection("users").document(user_id).set(
        {"preferences": preferences, "updated_at": datetime.utcnow().isoformat()},
        merge=True
    )


async def get_user_preferences(user_id: str) -> Optional[dict]:
    db = get_db()
    if not db:
        return None
    doc = db.collection("users").document(user_id).get()
    if doc.exists:
        return doc.to_dict().get("preferences", {})
    return None
