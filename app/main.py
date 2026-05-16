from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationToken
from typing import List, Optional
import uvicorn
import logging
from datetime import datetime

from .config import Config
from .services.face_service import FaceAuthenticationService

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(title="Face Authentication Service", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Initialize service
face_service = FaceAuthenticationService()

# API Key verification
async def verify_api_key(token: HTTPAuthorizationToken = Depends(security)):
    if token.credentials != Config.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return token.credentials

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/v1/face/register")
async def register_face(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """
    Register a user's face from multiple images
    """
    try:
        body = await request.json()
        images = body.get('images', [])
        user_id = body.get('user_id')
        
        if not images or not user_id:
            raise HTTPException(status_code=400, detail="Missing images or user_id")
        
        if len(images) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 images")
        
        result = await face_service.register_face(images, user_id)
        
        if not result['success']:
            raise HTTPException(status_code=400, detail=result['error'])
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/face/verify")
async def verify_face(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """
    Verify a face against stored embedding
    """
    try:
        body = await request.json()
        image = body.get('image')
        stored_embedding = body.get('stored_embedding')
        user_id = body.get('user_id')
        
        if not image or not stored_embedding:
            raise HTTPException(status_code=400, detail="Missing image or stored_embedding")
        
        result = await face_service.verify_face(image, stored_embedding, user_id)
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Verification error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/v1/face/liveness")
async def verify_liveness(
    request: Request,
    api_key: str = Depends(verify_api_key)
):
    """
    Verify liveness using multiple frames
    """
    try:
        body = await request.json()
        frames = body.get('frames', [])
        
        if len(frames) < 5:
            raise HTTPException(status_code=400, detail="Need at least 5 frames")
        
        result = await face_service.verify_liveness(frames)
        
        return result
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Liveness error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)