# ─────────────────────────────────────────
# MUST be first — loads .env before any import reads os.getenv()
# ─────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from routes import itinerary, places, weather
import logging
import time
import os

# ─────────────────────────────────────────
# Logging
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
# CORS — allow all origins in development
# (Covers: file://, localhost:3000, localhost:5173, and deployed frontend)
# ─────────────────────────────────────────
APP_ENV = os.getenv("APP_ENV", "development")

if APP_ENV == "development":
    # Allow ALL origins in dev — covers file://, localhost, 127.0.0.1
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,   # Must be False when allow_origins=["*"]
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    # Production — restrict to known frontend URLs
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
# Register API Routes
# ─────────────────────────────────────────
app.include_router(itinerary.router, prefix="/api/itinerary", tags=["Itinerary"])
app.include_router(places.router,    prefix="/api/places",    tags=["Places"])
app.include_router(weather.router,   prefix="/api/weather",   tags=["Weather"])

# ─────────────────────────────────────────
# Serve test_frontend.html at /test
# Access at: http://localhost:8000/test
# ─────────────────────────────────────────
@app.get("/test", tags=["Test"], include_in_schema=False)
async def serve_test_frontend():
    html_path = os.path.join(os.path.dirname(__file__), "test_frontend.html")
    return FileResponse(html_path, media_type="text/html")

# ─────────────────────────────────────────
# Root & Health
# ─────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "AI Tourism Planner API 🗺️",
        "version": "2.0.0",
        "llm": "Perplexity sonar-pro",
        "docs": "/docs",
        "test_ui": "/test"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "version": "2.0.0", "env": APP_ENV}

@app.on_event("startup")
async def on_startup():
    logger.info(f"AI Tourism Planner API v2.0.0 started 🚀 [{APP_ENV}]")
    logger.info("LLM: Perplexity sonar-pro | POI: Foursquare + Geoapify + OpenTripMap")
    logger.info("Test UI available at: http://localhost:8000/test")
