import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from typing import Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)

class FaceRecognizer:
    """Production-grade face recognition using InsightFace + ArcFace"""
    
    def __init__(self, model_name: str = "buffalo_l"):
        self.model_name = model_name
        self.app = None
        self.face_model = None
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize InsightFace models with optimal settings"""
        try:
            # Face detection and recognition model
            self.app = FaceAnalysis(
                name=self.model_name,
                root='~/.insightface',
                allowed_modules=['detection', 'recognition', 'landmark_2d_106']
            )
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            
            logger.info(f"InsightFace model {self.model_name} loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load InsightFace model: {e}")
            raise
    
    def detect_faces(self, image: np.ndarray) -> List[dict]:
        """
        Detect all faces in image with landmarks and embeddings
        
        Args:
            image: BGR image array (OpenCV format)
            
        Returns:
            List of face dictionaries with bbox, landmarks, embedding
        """
        if image is None or image.size == 0:
            return []
        
        # Convert to RGB if needed (InsightFace expects RGB)
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        faces = self.app.get(rgb_image)
        
        results = []
        for face in faces:
            # Get bounding box
            bbox = face.bbox.astype(int).tolist()
            
            # Get landmarks (106 points)
            landmarks = face.landmark_2d_106.tolist() if hasattr(face, 'landmark_2d_106') else []
            
            # Get face embedding (512-dim vector)
            embedding = face.normed_embedding.tolist() if hasattr(face, 'normed_embedding') else None
            
            # Get detection confidence
            confidence = float(face.det_score) if hasattr(face, 'det_score') else 0.0
            
            results.append({
                'bbox': bbox,
                'landmarks': landmarks,
                'embedding': embedding,
                'confidence': confidence
            })
        
        return results
    
    def get_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from image (for single face)
        
        Args:
            image: BGR image array
            
        Returns:
            Face embedding vector or None if no face
        """
        faces = self.detect_faces(image)
        
        if not faces:
            return None
        
        # Return embedding of the largest face (by area)
        largest_face = max(faces, key=lambda f: 
                          (f['bbox'][2] - f['bbox'][0]) * (f['bbox'][3] - f['bbox'][1]))
        
        return np.array(largest_face['embedding']) if largest_face['embedding'] else None
    
    def get_best_face(self, image: np.ndarray) -> Optional[dict]:
        """
        Get best quality face from image
        
        Args:
            image: BGR image array
            
        Returns:
            Best face dict with bbox, landmarks, embedding
        """
        faces = self.detect_faces(image)
        
        if not faces:
            return None
        
        # Score faces by confidence and size
        for face in faces:
            area = (face['bbox'][2] - face['bbox'][0]) * (face['bbox'][3] - face['bbox'][1])
            face['score'] = face['confidence'] * min(1.0, area / (200 * 200))
        
        return max(faces, key=lambda f: f['score'])
    
    def compare_faces(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compare two face embeddings using cosine similarity
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            Cosine similarity score (0-1)
        """
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        embedding1_norm = embedding1 / norm1
        embedding2_norm = embedding2 / norm2
        
        # Cosine similarity
        similarity = np.dot(embedding1_norm, embedding2_norm)
        
        # Clamp to [0, 1]
        return max(0.0, min(1.0, similarity))
    
    def compute_average_embedding(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        Compute average of multiple embeddings for better accuracy
        
        Args:
            embeddings: List of embedding vectors
            
        Returns:
            Average embedding vector
        """
        if not embeddings:
            return None
        
        valid_embeddings = [e for e in embeddings if e is not None]
        
        if not valid_embeddings:
            return None
        
        # Stack and average
        stacked = np.vstack(valid_embeddings)
        avg_embedding = np.mean(stacked, axis=0)
        
        # Re-normalize
        norm = np.linalg.norm(avg_embedding)
        if norm > 0:
            avg_embedding = avg_embedding / norm
        
        return avg_embedding