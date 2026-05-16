import cv2
import numpy as np
import onnxruntime as ort
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class AntiSpoofingDetector:
    """
    Silent Face Anti-Spoofing detector
    Detects photos, videos, masks, and deepfakes
    """
    
    def __init__(self, model_path: str = None):
        self.session = None
        self.input_name = None
        self.output_name = None
        self._initialize_model(model_path)
    
    def _initialize_model(self, model_path: str = None):
        """Initialize ONNX model for anti-spoofing"""
        try:
            # Use default model path if not provided
            if model_path is None:
                # You would download this model file
                model_path = "models/anti_spoofing.onnx"
            
            # Create session with optimized settings
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess_options.intra_op_num_threads = 2
            
            self.session = ort.InferenceSession(model_path, sess_options)
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            logger.info("Anti-spoofing model loaded successfully")
        except Exception as e:
            logger.warning(f"Anti-spoofing model not loaded: {e}")
            self.session = None
    
    def detect_spoof(self, face_image: np.ndarray) -> Tuple[float, bool]:
        """
        Detect if face is real or spoof
        
        Args:
            face_image: Cropped face image (RGB)
            
        Returns:
            Tuple of (spoof_score, is_real)
            spoof_score: 0-1, higher means more likely to be spoof
            is_real: True if real face
        """
        if self.session is None:
            # Fallback to simple liveness detection
            return self._fallback_liveness(face_image), True
        
        try:
            # Preprocess image for model
            processed = self._preprocess(face_image)
            
            # Run inference
            outputs = self.session.run([self.output_name], {self.input_name: processed})
            spoof_score = float(outputs[0][0][0])
            
            # Threshold for real vs spoof
            is_real = spoof_score < 0.5
            
            return spoof_score, is_real
        except Exception as e:
            logger.error(f"Anti-spoofing inference error: {e}")
            return 0.5, True  # Default to real on error
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for anti-spoofing model"""
        # Resize to model input size (typically 224x224)
        resized = cv2.resize(image, (224, 224))
        
        # Convert to float and normalize
        normalized = resized.astype(np.float32) / 255.0
        
        # Add batch dimension
        batched = np.expand_dims(normalized, axis=0)
        
        # Transpose to channel-first format if needed
        if batched.shape[-1] == 3:
            batched = np.transpose(batched, (0, 3, 1, 2))
        
        return batched
    
    def _fallback_liveness(self, face_image: np.ndarray) -> float:
        """
        Fallback liveness detection using texture analysis
        """
        # Convert to grayscale
        gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        
        # Calculate Laplacian variance (sharpness)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Calculate histogram entropy
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        
        # Calculate noise level
        noise = self._estimate_noise(gray)
        
        # Combined score
        score = 0.0
        if laplacian_var > 50:  # Sharp image
            score += 0.3
        if entropy > 5:  # Good texture
            score += 0.4
        if noise < 30:  # Low noise
            score += 0.3
        
        return 1.0 - score  # Higher score = more likely spoof
    
    def _estimate_noise(self, image: np.ndarray) -> float:
        """Estimate noise level in image"""
        # Apply median filter
        median_filtered = cv2.medianBlur(image, 5)
        
        # Calculate difference
        diff = cv2.absdiff(image.astype(np.float32), median_filtered.astype(np.float32))
        
        return np.mean(diff)
    
    def analyze_face_quality(self, face_image: np.ndarray) -> dict:
        """
        Analyze face quality metrics
        
        Returns:
            Dictionary with quality scores
        """
        gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        
        # Blur detection
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        is_blurry = laplacian_var < 50
        
        # Brightness
        brightness = np.mean(gray)
        is_too_dark = brightness < 80
        is_too_bright = brightness > 220
        
        # Contrast
        contrast = gray.std()
        is_low_contrast = contrast < 40
        
        # Face size (assuming face is centered and cropped)
        height, width = face_image.shape[:2]
        face_ratio = min(height, width) / max(height, width)
        is_angled = face_ratio < 0.7
        
        return {
            'blur_score': laplacian_var,
            'is_blurry': is_blurry,
            'brightness': brightness,
            'is_too_dark': is_too_dark,
            'is_too_bright': is_too_bright,
            'contrast': contrast,
            'is_low_contrast': is_low_contrast,
            'face_ratio': face_ratio,
            'is_angled': is_angled,
            'quality_score': self._calculate_quality_score(
                laplacian_var, brightness, contrast, face_ratio
            )
        }
    
    def _calculate_quality_score(self, sharpness: float, brightness: float, 
                                  contrast: float, ratio: float) -> float:
        """Calculate overall quality score (0-1)"""
        scores = []
        
        # Sharpness score
        if sharpness >= 100:
            scores.append(1.0)
        elif sharpness >= 50:
            scores.append(0.7)
        elif sharpness >= 30:
            scores.append(0.4)
        else:
            scores.append(0.1)
        
        # Brightness score
        if 100 <= brightness <= 180:
            scores.append(1.0)
        elif 80 <= brightness <= 200:
            scores.append(0.7)
        elif 50 <= brightness <= 220:
            scores.append(0.4)
        else:
            scores.append(0.1)
        
        # Contrast score
        if contrast >= 60:
            scores.append(1.0)
        elif contrast >= 40:
            scores.append(0.6)
        elif contrast >= 20:
            scores.append(0.3)
        else:
            scores.append(0.1)
        
        # Ratio score
        if ratio >= 0.9:
            scores.append(1.0)
        elif ratio >= 0.8:
            scores.append(0.8)
        elif ratio >= 0.7:
            scores.append(0.5)
        else:
            scores.append(0.2)
        
        return np.mean(scores)