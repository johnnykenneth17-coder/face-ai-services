# app/config.py
import os
import sys
from dotenv import load_dotenv
import logging

# Configure logging at the config level
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
try:
    load_dotenv()
    logger.info(".env file loaded successfully.")
except Exception as e:
    logger.warning(f"Could not load .env file: {e}. Relying on system environment variables.")

class Config:
    """Configuration class for the AI service."""
    
    # Security
    SECRET_KEY = os.environ.get("AI_SECRET_KEY")
    API_KEY = os.environ.get("AI_API_KEY")
    
    # Face Recognition Settings
    FACE_SIMILARITY_THRESHOLD = float(os.environ.get("FACE_SIMILARITY_THRESHOLD", "0.65"))
    LIVENESS_THRESHOLD = float(os.environ.get("LIVENESS_THRESHOLD", "0.7"))
    
    # InsightFace Settings
    INSIGHTFACE_MODEL = os.environ.get("INSIGHTFACE_MODEL", "buffalo_l")
    INSIGHTFACE_ROOT = os.environ.get("INSIGHTFACE_ROOT", "~/.insightface")
    
    # Image Quality Thresholds (derived from research and best practices [citation:4])
    MIN_FACE_SIZE = 100
    MIN_BRIGHTNESS = 80
    MAX_BRIGHTNESS = 220
    MIN_SHARPNESS = 50 # Laplacian variance threshold
    
    # Model Paths (for anti-spoofing model)
    ANTI_SPOOFING_MODEL_PATH = os.environ.get("ANTI_SPOOFING_MODEL_PATH", "models/anti_spoofing.onnx")
    
    # Redis for Rate Limiting
    REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379")
    
    # Logging
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    
    # Validate critical configuration on startup
    if not SECRET_KEY or not API_KEY:
        logger.critical("AI_SECRET_KEY and AI_API_KEY must be set in environment variables.")
        # In a production app, you might want to raise an error and exit.
        # For this example, we'll just log a critical error.
        
    logger.info("Configuration loaded successfully.")