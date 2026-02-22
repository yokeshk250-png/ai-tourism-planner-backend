# ─────────────────────────────────────────
# MUST be first — loads .env before any import
# ─────────────────────────────────────────
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from routes import itinerary, places, weather
import logging
import time
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

APP_ENV = os.getenv("APP_ENV", "development")

app = FastAPI(
    title="AI Tourism Planner API",
    description="""Backend API for AI-powered tourism itinerary planning.
    Uses **Groq llama-3.3-70b-versatile** for fast itinerary generation (free tier),
    **Foursquare/Geoapify** for POI data, and **Firebase** for user storage.""",
    version="2.2.0",
    contact={
        "name": "AI Tourism Planner",
        "url": "https://github.com/yokeshk250-png/ai-tourism-planner-backend"
    }
)

if APP_ENV == "development":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
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

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 2)
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({duration}ms)")
    return response

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error", "detail": str(exc)}
    )

app.include_router(itinerary.router, prefix="/api/itinerary", tags=["Itinerary"])
app.include_router(places.router,    prefix="/api/places",    tags=["Places"])
app.include_router(weather.router,   prefix="/api/weather",   tags=["Weather"])

@app.get("/test", tags=["Test"], include_in_schema=False)
async def serve_test_frontend():
    html_path = os.path.join(os.path.dirname(__file__), "test_frontend.html")
    return FileResponse(html_path, media_type="text/html")

@app.get("/", tags=["Health"])
async def root():
    return {
        "message": "AI Tourism Planner API 🗺️",
        "version": "2.2.0",
        "llm": "Groq llama-3.3-70b-versatile (free)",
        "docs": "/docs",
        "test_ui": "/test"
    }

@app.get("/health", tags=["Health"])
async def health_check():
    return {"status": "healthy", "version": "2.2.0", "llm": "groq/llama-3.3-70b-versatile", "env": APP_ENV}

@app.on_event("startup")
async def on_startup():
    logger.info(f"AI Tourism Planner API v2.2.0 🚀 [{APP_ENV}]")
    logger.info("LLM: Groq llama-3.3-70b-versatile (FREE 14,400 req/day)")
    logger.info("Test UI: http://localhost:8000/test")
