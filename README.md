# 🗺️ AI Tourism Planner — Backend

A **FastAPI** backend powering an AI-driven tourism itinerary planner.
Generates personalized day-wise travel plans using **Perplexity sonar-pro** (with real-time web search grounding), enriched with live POI data from Foursquare and Geoapify, and stores user data in Firebase Firestore.

---

## ⚡ Why Perplexity API?

| Feature | Perplexity sonar-pro | GPT-4o |
|---|---|---|
| Real-time web search | ✅ Grounded answers | ❌ Static training data |
| India-specific accuracy | ✅ Up-to-date info | ⚠️ May be outdated |
| Opening hours accuracy | ✅ Fetched live | ⚠️ May hallucinate |
| Cost | 💚 Cheaper | 💛 Costlier |
| Hallucination risk | 🔵 Lower | 🔴 Higher |

---

## 🏗️ Architecture

```
ai-tourism-planner-backend/
├── main.py                    # FastAPI app entry point
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variable template
├── routes/
│   ├── itinerary.py           # POST /api/itinerary/generate
│   ├── places.py              # GET  /api/places/search
│   └── weather.py             # GET  /api/weather/forecast
├── models/
│   ├── schemas.py             # Pydantic request/response models
│   └── recommender.py         # ML-based place recommendation (cosine similarity)
└── services/
    ├── llm_service.py         # ✨ Perplexity sonar-pro itinerary generation
    ├── places_service.py      # Foursquare → Geoapify → OpenTripMap fallback
    ├── weather_service.py     # OpenWeatherMap integration
    └── firebase_service.py    # Firestore CRUD operations
```

---

## 🚀 Setup

### 1. Clone the repo
```bash
git clone https://github.com/yokeshk250-png/ai-tourism-planner-backend.git
cd ai-tourism-planner-backend
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables
```bash
cp .env.example .env
# Fill in your API keys in .env
```

### 5. Add Firebase credentials
Download `serviceAccountKey.json` from Firebase Console and place it in the root directory.

### 6. Run the server
```bash
uvicorn main:app --reload
```

API will be live at: `http://localhost:8000`
Interactive Docs: `http://localhost:8000/docs`

---

## 🔑 Environment Variables

```env
# Perplexity API (Primary LLM)
# Get key: https://www.perplexity.ai/settings/api
PERPLEXITY_API_KEY=your_perplexity_api_key_here
PERPLEXITY_MODEL=sonar-pro   # Options: sonar-pro | sonar | sonar-reasoning

# Places APIs
FOURSQUARE_API_KEY=your_foursquare_api_key_here
GEOAPIFY_API_KEY=your_geoapify_api_key_here
OPENTRIPMAP_API_KEY=your_opentripmap_api_key_here

# Weather
OPENWEATHER_API_KEY=your_openweather_api_key_here

# Firebase
FIREBASE_CREDENTIALS_PATH=serviceAccountKey.json
FIREBASE_PROJECT_ID=your_firebase_project_id
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Health check |
| `GET`  | `/health` | API status |
| `POST` | `/api/itinerary/generate` | Generate AI itinerary via Perplexity |
| `POST` | `/api/itinerary/save` | Save itinerary to Firebase |
| `GET`  | `/api/itinerary/{id}` | Fetch saved itinerary |
| `GET`  | `/api/places/search` | Search tourist places |
| `GET`  | `/api/places/{place_id}` | Get place details |
| `GET`  | `/api/weather/forecast` | Get weather forecast |

---

## 🧪 Sample Request

```bash
curl -X POST http://localhost:8000/api/itinerary/generate \
  -H "Content-Type: application/json" \
  -d '{
    "destination": "Chennai",
    "days": 2,
    "budget": "medium",
    "travel_type": "family",
    "interests": ["temples", "beach", "history"],
    "travel_dates": "2026-03-10"
  }'
```

### Sample Response
```json
{
  "success": true,
  "itinerary": [
    {
      "day": 1,
      "time": "06:00",
      "place_name": "Marina Beach",
      "category": "beach",
      "duration_hrs": 1.5,
      "entry_fee": 0,
      "tip": "Visit at sunrise to avoid weekend crowds",
      "lat": 13.0499,
      "lon": 80.2824
    },
    {
      "day": 1,
      "time": "09:00",
      "place_name": "Kapaleeshwarar Temple",
      "category": "temple",
      "duration_hrs": 1.5,
      "entry_fee": 0,
      "tip": "Remove footwear at entrance; avoid Fridays (crowded)"
    }
  ]
}
```

---

## 🤖 Perplexity Models Available

| Model | Best For | Speed |
|---|---|---|
| `sonar-pro` | Best quality + web search grounding | Medium |
| `sonar` | Fast place enrichment queries | Fast |
| `sonar-reasoning` | Complex multi-day itinerary logic | Slow |
| `sonar-reasoning-pro` | Highest quality reasoning | Slowest |

Set via `PERPLEXITY_MODEL` in `.env`

---

## 🔗 APIs Used

| Service | Purpose | Free Tier |
|---------|---------|----------|
| **Perplexity sonar-pro** | AI itinerary generation (web-grounded) | Pay-per-use (~$1/1M tokens) |
| **Foursquare Places** | POI data, reviews, opening hours | 1K req/day |
| **Geoapify** | Geocoding, fallback POI search | 3K req/day |
| **OpenTripMap** | Tourist-specific place discovery | 1K req/day |
| **OpenWeatherMap** | Weather forecasting | 1K req/day |
| **Firebase Firestore** | User data & itinerary storage | Spark free tier |

---

## 🛠️ Built With
- **FastAPI** — High-performance async Python web framework
- **Pydantic v2** — Data validation and schemas
- **OpenAI SDK** — Used as Perplexity-compatible client (`base_url` overridden)
- **Firebase Admin SDK** — Firestore database integration
- **scikit-learn** — Cosine similarity ML recommendation engine
- **httpx** — Async HTTP client for external API calls

---

## 📄 License
MIT License — feel free to use and modify.
