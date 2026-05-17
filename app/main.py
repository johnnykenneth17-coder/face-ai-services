"""
app/main.py — FEECENT Face-Authentication Service
Production FastAPI entry point
"""
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .config import Config
from .routes import face_routes

# ── Logging ─────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan (replaces deprecated on_event) ──────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FEECENT Face-Auth Service starting …")
    # Warm-up: import heavy models so first request isn't slow
    try:
        from .models.face_recognizer import FaceRecognizer  # noqa: F401
        logger.info("Face models pre-loaded ✓")
    except Exception as exc:
        logger.warning(f"Model pre-load skipped: {exc}")
    yield
    logger.info("Face-Auth Service shutting down …")


# ── App factory ──────────────────────────────────────────────
app = FastAPI(
    title="FEECENT Face Authentication API",
    version="2.0.0",
    docs_url="/docs" if Config.DEBUG else None,
    redoc_url=None,
    lifespan=lifespan,
)

# ── CORS (allow your Vercel + Capacitor origins) ─────────────
ALLOWED_ORIGINS = [
    "https://bank-backend-blush.vercel.app",
    "http://localhost:3000",
    "http://localhost:8080",
    "capacitor://localhost",
    "ionic://localhost",
    "http://localhost",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID"],
)


# ── Global exception handler ─────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error"},
    )


# ── Routes ───────────────────────────────────────────────────
app.include_router(face_routes.router, prefix="/auth/face", tags=["Face Auth"])


@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "service": "feecent-face-auth", "version": "2.0.0"}


@app.get("/", tags=["Health"])
async def root():
    return {"message": "FEECENT Face Authentication Service", "status": "running"}