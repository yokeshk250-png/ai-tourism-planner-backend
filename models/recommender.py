import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
from typing import List, Dict

# Interest categories supported
ALL_CATEGORIES = [
    "temples", "beach", "history", "museum", "nature",
    "food", "shopping", "adventure", "heritage", "pilgrimage"
]

mlb = MultiLabelBinarizer(classes=ALL_CATEGORIES)
mlb.fit([[c] for c in ALL_CATEGORIES])

def encode_interests(interests: List[str]) -> np.ndarray:
    """One-hot encode a list of user interests."""
    return mlb.transform([interests])

def encode_place_tags(tags: List[str]) -> np.ndarray:
    """One-hot encode a list of place tags."""
    return mlb.transform([tags])

def recommend_places(
    user_interests: List[str],
    places: List[Dict],
    top_k: int = 10
) -> List[Dict]:
    """
    Rank places based on cosine similarity with user interests.
    Each place dict must have a 'tags' key with a list of strings.
    """
    if not places:
        return []

    user_vec = encode_interests(user_interests)
    place_vecs = np.vstack([
        encode_place_tags(p.get("tags", [])) for p in places
    ])

    scores = cosine_similarity(user_vec, place_vecs)[0]

    ranked = sorted(
        zip(scores, places),
        key=lambda x: x[0],
        reverse=True
    )

    return [place for score, place in ranked[:top_k]]
