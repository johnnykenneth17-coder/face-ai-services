# app/services/encryption.py
"""
Encryption service for face embeddings and sensitive data.
Uses AES-256-GCM for authenticated encryption and HMAC for integrity verification.
"""

import os
import base64
import hashlib
import hmac
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
from cryptography.hazmat.primitives import hashes, padding
from cryptography.hazmat.backends import default_backend
from typing import Tuple, Optional, Dict, Any
import json
import logging

from ..config import Config

logger = logging.getLogger(__name__)


class EncryptionService:
    """
    Handles encryption and decryption of face embeddings and other sensitive data.
    Uses industry-standard AES-256-GCM for authenticated encryption.
    """
    
    def __init__(self):
        """Initialize the encryption service with the secret key."""
        self.secret_key = Config.SECRET_KEY
        if not self.secret_key:
            raise ValueError("SECRET_KEY not configured. Please set AI_SECRET_KEY environment variable.")
        
        # Derive a 32-byte key from the secret using PBKDF2
        self.encryption_key = self._derive_key(self.secret_key.encode('utf-8'), 
                                                salt=b'face_auth_salt_2024')
        self.backend = default_backend()
    
    def _derive_key(self, password: bytes, salt: bytes, 
                    iterations: int = 100000, key_length: int = 32) -> bytes:
        """
        Derive a strong encryption key from a password using PBKDF2.
        
        Args:
            password: Input password/secret
            salt: Cryptographic salt
            iterations: Number of PBKDF2 iterations
            key_length: Desired key length in bytes
            
        Returns:
            Derived key bytes
        """
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=key_length,
            salt=salt,
            iterations=iterations,
            backend=self.backend
        )
        return kdf.derive(password)
    
    def encrypt_embedding(self, embedding: bytes) -> str:
        """
        Encrypt a face embedding using AES-256-GCM.
        
        Args:
            embedding: Raw embedding bytes to encrypt
            
        Returns:
            Base64 encoded encrypted data (includes IV and authentication tag)
        """
        try:
            # Generate a random IV (12 bytes recommended for GCM)
            iv = os.urandom(12)
            
            # Create AES-256-GCM cipher
            cipher = Cipher(
                algorithms.AES(self.encryption_key),
                modes.GCM(iv),
                backend=self.backend
            )
            encryptor = cipher.encryptor()
            
            # Encrypt the embedding
            encrypted = encryptor.update(embedding) + encryptor.finalize()
            
            # Get the authentication tag
            tag = encryptor.tag
            
            # Combine IV + encrypted_data + tag for storage
            combined = iv + encrypted + tag
            
            # Return as base64 for safe storage
            return base64.b64encode(combined).decode('utf-8')
        except Exception as e:
            logger.error(f"Embedding encryption failed: {e}")
            raise ValueError(f"Failed to encrypt embedding: {str(e)}")
    
    def decrypt_embedding(self, encrypted_data: str) -> bytes:
        """
        Decrypt a face embedding using AES-256-GCM.
        
        Args:
            encrypted_data: Base64 encoded encrypted data
            
        Returns:
            Decrypted embedding bytes
        """
        try:
            # Decode from base64
            combined = base64.b64decode(encrypted_data)
            
            # Extract components (IV is 12 bytes, tag is 16 bytes)
            iv = combined[:12]
            tag = combined[-16:]
            ciphertext = combined[12:-16]
            
            # Create AES-256-GCM cipher for decryption
            cipher = Cipher(
                algorithms.AES(self.encryption_key),
                modes.GCM(iv, tag),
                backend=self.backend
            )
            decryptor = cipher.decryptor()
            
            # Decrypt the data
            decrypted = decryptor.update(ciphertext) + decryptor.finalize()
            
            return decrypted
        except Exception as e:
            logger.error(f"Embedding decryption failed: {e}")
            raise ValueError(f"Failed to decrypt embedding: {str(e)}")
    
    def encrypt_json(self, data: Dict[str, Any]) -> str:
        """
        Encrypt a JSON object.
        
        Args:
            data: Dictionary to encrypt
            
        Returns:
            Base64 encoded encrypted JSON
        """
        try:
            json_string = json.dumps(data)
            return self.encrypt_embedding(json_string.encode('utf-8'))
        except Exception as e:
            logger.error(f"JSON encryption failed: {e}")
            raise ValueError(f"Failed to encrypt JSON: {str(e)}")
    
    def decrypt_json(self, encrypted_data: str) -> Dict[str, Any]:
        """
        Decrypt and parse JSON data.
        
        Args:
            encrypted_data: Base64 encoded encrypted JSON
            
        Returns:
            Decrypted dictionary
        """
        try:
            decrypted_bytes = self.decrypt_embedding(encrypted_data)
            return json.loads(decrypted_bytes.decode('utf-8'))
        except Exception as e:
            logger.error(f"JSON decryption failed: {e}")
            raise ValueError(f"Failed to decrypt JSON: {str(e)}")
    
    def generate_secure_token(self, length: int = 32) -> str:
        """
        Generate a cryptographically secure random token.
        
        Args:
            length: Token length in bytes
            
        Returns:
            Hex-encoded random token
        """
        return os.urandom(length).hex()
    
    def hash_data(self, data: bytes, algorithm: str = 'sha256') -> str:
        """
        Generate a cryptographic hash of data.
        
        Args:
            data: Input data bytes
            algorithm: Hash algorithm ('sha256', 'sha512')
            
        Returns:
            Hex-encoded hash
        """
        if algorithm == 'sha256':
            return hashlib.sha256(data).hexdigest()
        elif algorithm == 'sha512':
            return hashlib.sha512(data).hexdigest()
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
    
    def verify_hmac(self, data: bytes, signature: str, key: bytes = None) -> bool:
        """
        Verify HMAC signature for data integrity.
        
        Args:
            data: Original data bytes
            signature: Expected HMAC signature (hex)
            key: HMAC key (uses derived key if not provided)
            
        Returns:
            True if signature is valid
        """
        if key is None:
            key = self.encryption_key
        
        expected = hmac.new(key, data, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected, signature)
    
    def sign_data(self, data: bytes, key: bytes = None) -> str:
        """
        Generate HMAC signature for data.
        
        Args:
            data: Data to sign
            key: HMAC key (uses derived key if not provided)
            
        Returns:
            HMAC signature as hex string
        """
        if key is None:
            key = self.encryption_key
        
        return hmac.new(key, data, hashlib.sha256).hexdigest()
    
    def rotate_key(self, old_encrypted_data: str, new_key: bytes) -> str:
        """
        Re-encrypt data with a new key (for key rotation).
        
        Args:
            old_encrypted_data: Data encrypted with old key
            new_key: New encryption key
            
        Returns:
            Data re-encrypted with new key
        """
        try:
            # Decrypt with old key (temporarily store old key)
            old_key = self.encryption_key
            self.encryption_key = self._derive_key(
                Config.SECRET_KEY.encode('utf-8') + b'_old',
                salt=b'face_auth_salt_2024'
            )
            decrypted = self.decrypt_embedding(old_encrypted_data)
            
            # Re-encrypt with new key
            self.encryption_key = new_key
            new_encrypted = self.encrypt_embedding(decrypted)
            
            # Restore old key
            self.encryption_key = old_key
            
            return new_encrypted
        except Exception as e:
            logger.error(f"Key rotation failed: {e}")
            raise ValueError(f"Failed to rotate encryption key: {str(e)}")


class SessionManager:
    """
    Manages secure session tokens for face authentication.
    """
    
    def __init__(self):
        self.encryption = EncryptionService()
        self.session_ttl = 300  # 5 minutes default
    
    def create_session(self, user_id: str, metadata: Dict[str, Any] = None) -> str:
        """
        Create a new secure session token.
        
        Args:
            user_id: User identifier
            metadata: Additional session metadata
            
        Returns:
            Session token string
        """
        import time
        
        session_data = {
            'user_id': user_id,
            'created_at': time.time(),
            'expires_at': time.time() + self.session_ttl,
            'metadata': metadata or {}
        }
        
        return self.encryption.encrypt_json(session_data)
    
    def verify_session(self, token: str) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Verify and decrypt a session token.
        
        Args:
            token: Session token string
            
        Returns:
            Tuple of (is_valid, session_data)
        """
        import time
        
        try:
            session_data = self.encryption.decrypt_json(token)
            
            # Check expiration
            if session_data.get('expires_at', 0) < time.time():
                logger.warning(f"Session expired for user {session_data.get('user_id')}")
                return False, None
            
            return True, session_data
        except Exception as e:
            logger.error(f"Session verification failed: {e}")
            return False, None
    
    def revoke_session(self, token: str) -> bool:
        """
        Revoke a session token (by marking it invalid).
        In production, you'd store revoked tokens in Redis.
        
        Args:
            token: Session token to revoke
            
        Returns:
            True if revoked successfully
        """
        # In production, add token to a revocation list in Redis
        # For now, we'll just log it
        logger.info(f"Session token revoked (would be added to blacklist)")
        return True


class SecureStorage:
    """
    Handles secure storage of sensitive data with encryption at rest.
    """
    
    def __init__(self, redis_client=None):
        self.encryption = EncryptionService()
        self.redis = redis_client
    
    def store_encrypted(self, key: str, value: Dict[str, Any], ttl: int = None) -> bool:
        """
        Store encrypted data.
        
        Args:
            key: Storage key
            value: Data to store (will be encrypted)
            ttl: Time-to-live in seconds
            
        Returns:
            True if stored successfully
        """
        try:
            encrypted = self.encryption.encrypt_json(value)
            
            if self.redis:
                if ttl:
                    self.redis.setex(key, ttl, encrypted)
                else:
                    self.redis.set(key, encrypted)
            else:
                # Fallback to memory (not for production)
                self._memory_storage[key] = (encrypted, ttl)
            
            return True
        except Exception as e:
            logger.error(f"Failed to store encrypted data for key {key}: {e}")
            return False
    
    def retrieve_encrypted(self, key: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve and decrypt stored data.
        
        Args:
            key: Storage key
            
        Returns:
            Decrypted data or None if not found
        """
        try:
            if self.redis:
                encrypted = self.redis.get(key)
                if not encrypted:
                    return None
                encrypted = encrypted.decode('utf-8')
            else:
                item = self._memory_storage.get(key)
                if not item:
                    return None
                encrypted, ttl = item
                
                # Check TTL if applicable
                if ttl:
                    # Simplified TTL check - in production, store timestamp
                    pass
            
            return self.encryption.decrypt_json(encrypted)
        except Exception as e:
            logger.error(f"Failed to retrieve encrypted data for key {key}: {e}")
            return None
    
    def delete_encrypted(self, key: str) -> bool:
        """
        Delete encrypted data.
        
        Args:
            key: Storage key
            
        Returns:
            True if deleted successfully
        """
        try:
            if self.redis:
                self.redis.delete(key)
            else:
                self._memory_storage.pop(key, None)
            return True
        except Exception as e:
            logger.error(f"Failed to delete encrypted data for key {key}: {e}")
            return False
    
    # In-memory fallback (for development only)
    _memory_storage = {}


# Initialize singleton instances
_encryption_service = None
_session_manager = None
_secure_storage = None


def get_encryption_service() -> EncryptionService:
    """Get the singleton encryption service instance."""
    global _encryption_service
    if _encryption_service is None:
        _encryption_service = EncryptionService()
    return _encryption_service


def get_session_manager() -> SessionManager:
    """Get the singleton session manager instance."""
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def get_secure_storage(redis_client=None) -> SecureStorage:
    """Get the singleton secure storage instance."""
    global _secure_storage
    if _secure_storage is None:
        _secure_storage = SecureStorage(redis_client)
    return _secure_storage