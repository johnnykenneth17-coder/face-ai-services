# app/__init__.py
"""
Face Authentication AI Service
Provides face recognition, anti-spoofing, and liveness detection.
"""

from .config import Config
from .services.face_service import FaceAuthenticationService
from .services.encryption import EncryptionService, SessionManager, SecureStorage
from .utils.image_utils import ImageUtils

__version__ = "1.0.0"
__all__ = [
    'Config',
    'FaceAuthenticationService',
    'EncryptionService',
    'SessionManager',
    'SecureStorage',
    'ImageUtils'
]