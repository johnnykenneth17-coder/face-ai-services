"""
app/routes/face_routes.py — REST endpoints for face authentication
"""
import logging
import time
import uuid
from typing import List, Optional

from fastapi import APIRouter, Header, HTTPException, Request
from pydantic import BaseModel, Field

from ..config import Config
from ..services.face_service import FaceAuthenticationService

logger = logging.getLogger(__name__)
router = APIRouter()

# Lazily initialised singleton — avoids slow model load at import time
_service: Optional[FaceAuthenticationService] = None


def get_service() -> FaceAuthenticationService:
    global _service
    if _service is None:
        _service = FaceAuthenticationService()
    return _service


# ── Request / Response models ────────────────────────────────

class RegisterRequest(BaseModel):
    images: List[str] = Field(..., min_items=3, max_items=10,
                               description="Base64-encoded face images (3-10 angles)")
    user_id: str


class VerifyRequest(BaseModel):
    image: str = Field(..., description="Base64-encoded face image")
    stored_embedding: str = Field(..., description="Encrypted embedding from registration")
    user_id: Optional[str] = None


class LivenessRequest(BaseModel):
    frames: List[str] = Field(..., min_items=5,
                               description="List of base64 frames captured over ~3 seconds")


class StartSessionRequest(BaseModel):
    identifier: str  # email or phone


class FaceLoginRequest(BaseModel):
    session_id: str
    frames: List[str]
    final_image: str
    stored_embedding: str
    user_id: str


# ── Simple in-memory session store (swap for Redis in prod) ──
_sessions: dict = {}


def _verify_api_key(x_api_key: Optional[str] = None) -> None:
    """Optional API-key guard — set AI_API_KEY env var to enable."""
    if Config.API_KEY and x_api_key != Config.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ── Endpoints ────────────────────────────────────────────────

@router.post("/register")
async def register_face(req: RegisterRequest,
                        x_api_key: Optional[str] = Header(None)):
    """
    Register a user's face from multiple angles (3–10 images).
    Returns encrypted embedding to store in your main DB.
    """
    _verify_api_key(x_api_key)
    try:
        result = await get_service().register_face(req.images, req.user_id)
        return result
    except Exception as exc:
        logger.error(f"register_face error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/verify")
async def verify_face(req: VerifyRequest,
                      x_api_key: Optional[str] = Header(None)):
    """
    Verify a live face against a stored encrypted embedding.
    """
    _verify_api_key(x_api_key)
    try:
        result = await get_service().verify_face(
            req.image, req.stored_embedding, req.user_id
        )
        return result
    except Exception as exc:
        logger.error(f"verify_face error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/liveness")
async def check_liveness(req: LivenessRequest,
                         x_api_key: Optional[str] = Header(None)):
    """
    Multi-frame liveness check (blink + head-movement detection).
    """
    _verify_api_key(x_api_key)
    try:
        result = await get_service().verify_liveness(req.frames)
        return result
    except Exception as exc:
        logger.error(f"liveness error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/start-session")
async def start_session(req: StartSessionRequest,
                        x_api_key: Optional[str] = Header(None)):
    """
    Create a short-lived session token before face login.
    The caller must present this token in /login within 5 minutes.
    """
    _verify_api_key(x_api_key)
    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "identifier": req.identifier,
        "created_at": time.time(),
        "expires_at": time.time() + 300,  # 5 min
    }
    return {"success": True, "session_id": session_id}


@router.post("/login")
async def face_login(req: FaceLoginRequest,
                     x_api_key: Optional[str] = Header(None)):
    """
    Full face-login pipeline:
    1. Validate session
    2. Liveness check on frames
    3. Face match against stored embedding
    """
    _verify_api_key(x_api_key)

    # Validate session
    session = _sessions.pop(req.session_id, None)
    if not session or session["expires_at"] < time.time():
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    svc = get_service()

    # Liveness
    liveness = await svc.verify_liveness(req.frames)
    if not liveness.get("is_live"):
        return {
            "success": False,
            "error": "Liveness check failed — please blink and move your head slightly",
            "liveness": liveness,
        }

    # Face match
    match = await svc.verify_face(req.final_image, req.stored_embedding, req.user_id)
    return {**match, "liveness": liveness}