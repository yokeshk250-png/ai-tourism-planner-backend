# 🗺️ AI Tourism Planner — Backend

A **FastAPI** backend powering an AI-driven tourism itinerary planner.
Generates personalized day-wise travel plans using LLMs, enriched with real-time POI data from Foursquare and Geoapify, and stores user data in Firebase Firestore.

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
│   └── recommender.py         # ML-based place recommendation
└── services/
    ├── llm_service.py         # GPT-4o itinerary generation
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

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/api/itinerary/generate` | Generate AI itinerary |
| `POST` | `/api/itinerary/save` | Save itinerary to Firebase |
| `GET` | `/api/itinerary/{id}` | Fetch saved itinerary |
| `GET` | `/api/places/search` | Search tourist places |
| `GET` | `/api/places/{place_id}` | Get place details |
| `GET` | `/api/weather/forecast` | Get weather forecast |

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
    "interests": ["temples", "beach", "history"]
  }'
```

---

## 🔑 APIs Used

| Service | Purpose | Free Tier |
|---------|---------|----------|
| OpenAI GPT-4o | Itinerary generation | Pay-per-use |
| Foursquare Places | POI data, reviews, hours | 1K req/day |
| Geoapify | Geocoding, fallback POI | 3K req/day |
| OpenTripMap | Tourist-specific places | 1K req/day |
| OpenWeatherMap | Weather forecasting | 1K req/day |
| Firebase Firestore | User data & itinerary storage | Spark free tier |

---

## 🛠️ Built With
- **FastAPI** — High-performance Python web framework
- **Pydantic v2** — Data validation and schemas
- **OpenAI SDK** — LLM itinerary generation
- **Firebase Admin SDK** — Firestore database
- **scikit-learn** — ML-based place recommendation
- **httpx** — Async HTTP client for external APIs

---

## 📄 License
MIT License — feel free to use and modify.
