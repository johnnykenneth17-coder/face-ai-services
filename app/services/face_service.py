import cv2
import numpy as np
import base64
from typing import Dict, Any, Tuple, Optional, List
import logging
from datetime import datetime
import hashlib
import hmac

from ..models.face_recognizer import FaceRecognizer
from ..models.anti_spoofing import AntiSpoofingDetector
from ..models.preprocessor import FacePreprocessor
from ..config import Config

logger = logging.getLogger(__name__)

class FaceAuthenticationService:
    """Main service for face authentication"""
    
    def __init__(self):
        self.recognizer = FaceRecognizer()
        self.spoof_detector = AntiSpoofingDetector()
        self.preprocessor = FacePreprocessor()
        self.similarity_threshold = Config.FACE_SIMILARITY_THRESHOLD
    
    async def register_face(self, images_data: List[str], user_id: str) -> Dict[str, Any]:
        """
        Register a user's face from multiple angles
        
        Args:
            images_data: List of base64 encoded images from different angles
            user_id: User ID to register
            
        Returns:
            Registration result with encrypted embeddings
        """
        embeddings = []
        quality_scores = []
        
        for idx, img_data in enumerate(images_data):
            # Decode image
            image = self._decode_image(img_data)
            if image is None:
                continue
            
            # Get best face
            face_info = self.recognizer.get_best_face(image)
            if not face_info or face_info['embedding'] is None:
                continue
            
            # Check anti-spoofing
            bbox = face_info['bbox']
            cropped_face = self.preprocessor.crop_and_resize_face(image, bbox)
            if cropped_face is not None:
                spoof_score, is_real = self.spoof_detector.detect_spoof(cropped_face)
                if not is_real:
                    continue
            
            # Quality check
            if cropped_face is not None:
                quality = self.spoof_detector.analyze_face_quality(cropped_face)
                if quality['quality_score'] < 0.3:
                    continue
                quality_scores.append(quality['quality_score'])
            
            embeddings.append(np.array(face_info['embedding']))
        
        if len(embeddings) < 2:
            return {
                'success': False,
                'error': 'Need at least 2 valid face captures from different angles',
                'captures_received': len(images_data),
                'valid_captures': len(embeddings)
            }
        
        # Compute average embedding
        avg_embedding = self.recognizer.compute_average_embedding(embeddings)
        
        # Encrypt embedding for storage
        encrypted_embedding = self._encrypt_embedding(avg_embedding.tobytes())
        
        # Calculate average quality
        avg_quality = np.mean(quality_scores) if quality_scores else 0.5
        
        return {
            'success': True,
            'embedding': encrypted_embedding,
            'average_quality': avg_quality,
            'face_count': len(embeddings)
        }
    
    async def verify_face(self, image_data: str, stored_embedding: bytes, 
                          user_id: str = None) -> Dict[str, Any]:
        """
        Verify a face against stored embedding with multi-frame validation
        
        Args:
            image_data: Base64 encoded image
            stored_embedding: Encrypted stored embedding
            user_id: User ID for logging
            
        Returns:
            Verification result
        """
        # Decode image
        image = self._decode_image(image_data)
        if image is None:
            return {'success': False, 'error': 'Invalid image data'}
        
        # Preprocess image
        processed = self.preprocessor.preprocess_for_recognition(image)
        if processed is None:
            return {'success': False, 'error': 'No face detected in image'}
        
        # Get face embedding
        embedding = self.recognizer.get_face_embedding(processed)
        if embedding is None:
            return {'success': False, 'error': 'Could not extract face features'}
        
        # Anti-spoofing check
        spoof_score, is_real = self.spoof_detector.detect_spoof(processed)
        if not is_real:
            await self._log_verification(user_id, False, spoof_score, 'spoof_detected')
            return {
                'success': False,
                'error': 'Liveness detection failed',
                'spoof_score': spoof_score
            }
        
        # Quality check
        quality = self.spoof_detector.analyze_face_quality(processed)
        if quality['quality_score'] < 0.3:
            return {
                'success': False,
                'error': 'Face quality too low',
                'quality_score': quality['quality_score']
            }
        
        # Decrypt stored embedding
        try:
            stored_vector = self._decrypt_embedding(stored_embedding)
            stored_embedding_np = np.frombuffer(stored_vector, dtype=np.float32)
        except Exception as e:
            logger.error(f"Embedding decryption failed: {e}")
            return {'success': False, 'error': 'Invalid stored face data'}
        
        # Compare embeddings
        similarity = self.recognizer.compare_faces(embedding, stored_embedding_np)
        
        is_match = similarity >= self.similarity_threshold
        
        # Log verification
        await self._log_verification(user_id, is_match, similarity, 
                                     'match' if is_match else 'no_match')
        
        return {
            'success': True,
            'matched': is_match,
            'similarity_score': similarity,
            'threshold': self.similarity_threshold,
            'quality_score': quality['quality_score']
        }
    
    async def verify_liveness(self, frames: List[str]) -> Dict[str, Any]:
        """
        Verify liveness using multiple frames (blink detection, head movement)
        
        Args:
            frames: List of base64 encoded frames
            
        Returns:
            Liveness verification result
        """
        if len(frames) < 5:
            return {'success': False, 'error': 'Need at least 5 frames for liveness detection'}
        
        blink_detected = False
        head_movement_detected = False
        face_quality_scores = []
        
        previous_face_center = None
        eye_states = []
        
        for idx, frame_data in enumerate(frames):
            image = self._decode_image(frame_data)
            if image is None:
                continue
            
            # Detect face
            face_info = self.recognizer.get_best_face(image)
            if not face_info:
                continue
            
            # Check face quality
            bbox = face_info['bbox']
            cropped = self.preprocessor.crop_and_resize_face(image, bbox)
            if cropped is not None:
                quality = self.spoof_detector.analyze_face_quality(cropped)
                face_quality_scores.append(quality['quality_score'])
            
            # Track head movement
            face_center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
            if previous_face_center:
                movement = np.sqrt((face_center[0] - previous_face_center[0])**2 +
                                   (face_center[1] - previous_face_center[1])**2)
                if movement > 20:  # Significant movement
                    head_movement_detected = True
            previous_face_center = face_center
            
            # Detect blink using landmarks
            if face_info.get('landmarks'):
                eye_closed = self._detect_blink(face_info['landmarks'])
                eye_states.append(eye_closed)
        
        # Blink detection - look for closed eyes followed by open eyes
        if len(eye_states) > 3:
            for i in range(1, len(eye_states)):
                if eye_states[i] and not eye_states[i-1]:
                    blink_detected = True
                    break
        
        avg_quality = np.mean(face_quality_scores) if face_quality_scores else 0.5
        
        is_live = blink_detected and head_movement_detected and avg_quality > 0.4
        
        return {
            'success': True,
            'is_live': is_live,
            'blink_detected': blink_detected,
            'head_movement_detected': head_movement_detected,
            'average_quality': avg_quality,
            'frames_processed': len(face_quality_scores)
        }
    
    def _detect_blink(self, landmarks: list) -> bool:
        """Detect if eyes are closed using landmark points"""
        if len(landmarks) < 68:
            return False
        
        # Eye landmark indices (assuming 106-point model)
        # Left eye: 60-67, Right eye: 68-75 (approximate)
        try:
            # Calculate eye aspect ratio
            left_eye = landmarks[60:68]
            right_eye = landmarks[68:76]
            
            left_ear = self._eye_aspect_ratio(left_eye)
            right_ear = self._eye_aspect_ratio(right_eye)
            
            ear = (left_ear + right_ear) / 2
            
            # Threshold for closed eyes (typical value ~0.2)
            return ear < 0.2
        except:
            return False
    
    def _eye_aspect_ratio(self, eye_points: list) -> float:
        """Calculate Eye Aspect Ratio (EAR)"""
        if len(eye_points) < 6:
            return 1.0
        
        # Vertical distances
        p2_p6 = np.linalg.norm(np.array(eye_points[1]) - np.array(eye_points[5]))
        p3_p5 = np.linalg.norm(np.array(eye_points[2]) - np.array(eye_points[4]))
        
        # Horizontal distance
        p1_p4 = np.linalg.norm(np.array(eye_points[0]) - np.array(eye_points[3]))
        
        if p1_p4 == 0:
            return 1.0
        
        ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)
        return ear
    
    def _decode_image(self, image_data: str) -> Optional[np.ndarray]:
        """Decode base64 image to OpenCV format"""
        try:
            # Remove data URL prefix if present
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            # Decode base64
            img_bytes = base64.b64decode(image_data)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            return image
        except Exception as e:
            logger.error(f"Image decode error: {e}")
            return None
    
    def _encrypt_embedding(self, embedding_bytes: bytes) -> str:
        """Encrypt face embedding for storage"""
        # Using HMAC-SHA256 for encryption (simplified)
        # In production, use proper encryption (AES-GCM)
        key = Config.SECRET_KEY.encode()
        h = hmac.new(key, embedding_bytes, hashlib.sha256)
        signature = h.digest()
        
        # Combine embedding with signature
        combined = embedding_bytes + signature
        
        return base64.b64encode(combined).decode()
    
    def _decrypt_embedding(self, encrypted: str) -> bytes:
        """Decrypt and verify face embedding"""
        try:
            combined = base64.b64decode(encrypted)
            
            # Extract embedding and signature
            embedding_bytes = combined[:-32]
            stored_signature = combined[-32:]
            
            # Verify signature
            key = Config.SECRET_KEY.encode()
            h = hmac.new(key, embedding_bytes, hashlib.sha256)
            expected_signature = h.digest()
            
            if not hmac.compare_digest(stored_signature, expected_signature):
                raise ValueError("Invalid signature")
            
            return embedding_bytes
        except Exception as e:
            logger.error(f"Embedding decryption error: {e}")
            raise
    
    async def _log_verification(self, user_id: str, success: bool, 
                                  score: float, reason: str):
        """Log verification attempt for audit"""
        # This would store to database
        logger.info(f"Face verification: user={user_id}, success={success}, "
                    f"score={score:.4f}, reason={reason}")