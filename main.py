# ─────────────────────────────────────────
# MUST be first — loads .env before any import reads os.getenv()
# ─────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()  # ← loads .env BEFORE any other import

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from routes import itinerary, places, weather
import logging
import time

# ─────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────
app = FastAPI(
    title="AI Tourism Planner API",
    description="""Backend API for AI-powered tourism itinerary planning.
    Uses **Perplexity sonar-pro** for real-time grounded itinerary generation,
    **Foursquare/Geoapify** for POI data, and **Firebase** for user storage.""",
    version="2.0.0",
    contact={
        "name": "AI Tourism Planner",
        "url": "https://github.com/yokeshk250-png/ai-tourism-planner-backend"
    }
)

# ─────────────────────────────────────────
# CORS — allow React frontend
# ─────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "https://ai-tourism-planner.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────
# Request logging middleware
# ─────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 2)
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({duration}ms)")
    return response

# ─────────────────────────────────────────
# Global exception handler
# ─────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error", "detail": str(exc)}
    )

# ─────────────────────────────────────────
# Register Routes
# ─────────────────────────────────────────
app.include_router(itinerary.router, prefix="/api/itinerary", tags=["Itinerary"])
app.include_router(places.router,    prefix="/api/places",    tags=["Places"])
app.include_router(weather.router,   prefix="/api/weather",   tags=["Weather"])

# ─────────────────────────────────────────
# Root & Health endpoints
# ─────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "AI Tourism Planner API 🗺️",
        "version": "2.0.0",
        "llm": "Perplexity sonar-pro",
        "docs": "/docs"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "version": "2.0.0"}

@app.on_event("startup")
async def on_startup():
    logger.info("AI Tourism Planner API started 🚀")
    logger.info("LLM: Perplexity sonar-pro | POI: Foursquare + Geoapify + OpenTripMap")
