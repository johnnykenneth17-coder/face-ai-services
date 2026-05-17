"""
app/config.py — Centralised configuration
All values are read from environment variables with sensible defaults.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── Security ────────────────────────────────────────────
    SECRET_KEY: str = os.getenv("AI_SECRET_KEY", "")
    API_KEY: str = os.getenv("AI_API_KEY", "")

    if not SECRET_KEY:
        raise RuntimeError(
            "AI_SECRET_KEY environment variable is not set. "
            "Add it to your Render environment variables."
        )

    # ── Face recognition ────────────────────────────────────
    # 0.55 – 0.60 is a good production threshold for ArcFace cosine similarity
    FACE_SIMILARITY_THRESHOLD: float = float(
        os.getenv("FACE_SIMILARITY_THRESHOLD", "0.55")
    )
    INSIGHTFACE_MODEL: str = os.getenv("INSIGHTFACE_MODEL", "buffalo_l")

    # ── Anti-spoofing ───────────────────────────────────────
    ANTI_SPOOF_MODEL_PATH: str = os.getenv(
        "ANTI_SPOOF_MODEL_PATH", "models/anti_spoofing.onnx"
    )
    SPOOF_THRESHOLD: float = float(os.getenv("SPOOF_THRESHOLD", "0.5"))
    MIN_FACE_QUALITY: float = float(os.getenv("MIN_FACE_QUALITY", "0.3"))

    # ── Redis (optional) ────────────────────────────────────
    REDIS_URL: str = os.getenv("REDIS_URL", "")

    # ── Server ──────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
    PORT: int = int(os.getenv("PORT", "8001"))