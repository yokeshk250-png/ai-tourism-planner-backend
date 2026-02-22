from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import itinerary, places, weather

app = FastAPI(
    title="AI Tourism Planner API",
    description="Backend API for AI-powered tourism itinerary planning",
    version="1.0.0"
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routes
app.include_router(itinerary.router, prefix="/api/itinerary", tags=["Itinerary"])
app.include_router(places.router, prefix="/api/places", tags=["Places"])
app.include_router(weather.router, prefix="/api/weather", tags=["Weather"])

@app.get("/")
async def root():
    return {"message": "AI Tourism Planner API is running 🚀"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
