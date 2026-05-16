# app/utils/image_utils.py
"""
Image processing utilities for face authentication system.
Handles image validation, format conversion, compression, and quality checks.
"""

import cv2
import numpy as np
import base64
import io
from PIL import Image
from typing import Tuple, Optional, List, Union
import logging
import hashlib

logger = logging.getLogger(__name__)


class ImageUtils:
    """Utility class for all image processing operations."""
    
    @staticmethod
    def decode_base64_to_cv2(base64_string: str) -> Optional[np.ndarray]:
        """
        Convert a base64 string to an OpenCV image (BGR format).
        
        Args:
            base64_string: Base64 encoded image string (may include data URL prefix)
            
        Returns:
            OpenCV image array or None if decoding fails
        """
        try:
            # Remove data URL prefix if present (e.g., "data:image/jpeg;base64,")
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]
            
            # Decode base64 to bytes
            img_bytes = base64.b64decode(base64_string)
            
            # Convert bytes to numpy array
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            
            # Decode to OpenCV image (BGR format)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if image is None:
                logger.error("Failed to decode image: cv2.imdecode returned None")
                return None
                
            return image
        except Exception as e:
            logger.error(f"Base64 decoding error: {e}")
            return None
    
    @staticmethod
    def encode_cv2_to_base64(image: np.ndarray, format: str = 'jpeg', 
                              quality: int = 85) -> Optional[str]:
        """
        Convert an OpenCV image to a base64 string.
        
        Args:
            image: OpenCV image array (BGR format)
            format: Output format ('jpeg', 'png')
            quality: JPEG quality (1-100)
            
        Returns:
            Base64 encoded string or None if encoding fails
        """
        try:
            # Determine encoding parameters
            params = []
            if format.lower() == 'jpeg':
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            else:
                encode_param = [int(cv2.IMWRITE_PNG_COMPRESSION), 9]
            
            # Encode image to memory buffer
            success, encoded = cv2.imencode(f'.{format}', image, encode_param)
            
            if not success:
                logger.error("Failed to encode image")
                return None
                
            # Convert to base64
            base64_bytes = base64.b64encode(encoded.tobytes())
            base64_string = base64_bytes.decode('utf-8')
            
            return base64_string
        except Exception as e:
            logger.error(f"Base64 encoding error: {e}")
            return None
    
    @staticmethod
    def resize_image(image: np.ndarray, max_width: int = 640, 
                     max_height: int = 640) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio.
        
        Args:
            image: Input image
            max_width: Maximum width
            max_height: Maximum height
            
        Returns:
            Resized image
        """
        height, width = image.shape[:2]
        
        # Calculate scaling factor
        scale = min(max_width / width, max_height / height)
        
        if scale < 1.0:
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized = cv2.resize(image, (new_width, new_height), 
                                 interpolation=cv2.INTER_AREA)
            return resized
        
        return image
    
    @staticmethod
    def convert_color_space(image: np.ndarray, target: str = 'rgb') -> np.ndarray:
        """
        Convert image color space.
        
        Args:
            image: Input image
            target: Target color space ('rgb', 'bgr', 'gray')
            
        Returns:
            Converted image
        """
        if target.lower() == 'rgb':
            if len(image.shape) == 3 and image.shape[2] == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return image
        elif target.lower() == 'bgr':
            if len(image.shape) == 3 and image.shape[2] == 3:
                return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            return image
        elif target.lower() == 'gray':
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            logger.warning(f"Unknown color space target: {target}")
            return image
    
    @staticmethod
    def normalize_image(image: np.ndarray) -> np.ndarray:
        """
        Normalize image pixel values to [0, 1] range.
        
        Args:
            image: Input image (uint8)
            
        Returns:
            Normalized image (float32)
        """
        if image.dtype == np.uint8:
            return image.astype(np.float32) / 255.0
        return image
    
    @staticmethod
    def denormalize_image(image: np.ndarray) -> np.ndarray:
        """
        Convert normalized image back to uint8 [0, 255].
        
        Args:
            image: Normalized image (float32)
            
        Returns:
            uint8 image
        """
        if image.dtype == np.float32:
            return (image * 255).astype(np.uint8)
        return image
    
    @staticmethod
    def compute_image_hash(image: np.ndarray) -> str:
        """
        Compute a perceptual hash of the image for deduplication.
        
        Args:
            image: Input image
            
        Returns:
            SHA-256 hash string
        """
        # Convert to JPEG bytes for consistent hashing
        success, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if success:
            return hashlib.sha256(encoded.tobytes()).hexdigest()
        return hashlib.sha256(image.tobytes()).hexdigest()
    
    @staticmethod
    def validate_image_format(image_data: str, max_size_mb: int = 5) -> Tuple[bool, str]:
        """
        Validate image format and size before processing.
        
        Args:
            image_data: Base64 image string
            max_size_mb: Maximum allowed size in megabytes
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check size
            if ',' in image_data:
                image_data = image_data.split(',')[1]
            
            import math
            image_size_bytes = len(base64.b64decode(image_data))
            image_size_mb = image_size_bytes / (1024 * 1024)
            
            if image_size_mb > max_size_mb:
                return False, f"Image size ({image_size_mb:.2f}MB) exceeds maximum ({max_size_mb}MB)"
            
            # Try to decode to verify it's a valid image
            img_bytes = base64.b64decode(image_data)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if image is None:
                return False, "Invalid image format or corrupted image"
            
            return True, "Valid image"
        except Exception as e:
            return False, f"Image validation failed: {str(e)}"
    
    @staticmethod
    def extract_frames_from_video(video_path: str, frame_count: int = 15) -> List[np.ndarray]:
        """
        Extract frames from a video file.
        
        Args:
            video_path: Path to video file
            frame_count: Number of frames to extract
            
        Returns:
            List of extracted frames
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Cannot open video: {video_path}")
            return frames
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames == 0:
            return frames
            
        # Calculate step size
        step = max(1, total_frames // frame_count)
        
        for i in range(0, total_frames, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret and len(frames) < frame_count:
                frames.append(frame)
            if len(frames) >= frame_count:
                break
        
        cap.release()
        return frames
    
    @staticmethod
    def enhance_brightness(image: np.ndarray, factor: float = 1.2) -> np.ndarray:
        """
        Enhance image brightness.
        
        Args:
            image: Input image
            factor: Brightness multiplier (>1 brightens, <1 darkens)
            
        Returns:
            Brightness-enhanced image
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = cv2.multiply(hsv[:, :, 2], factor)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    @staticmethod
    def auto_white_balance(image: np.ndarray) -> np.ndarray:
        """
        Apply automatic white balance correction.
        
        Args:
            image: Input image
            
        Returns:
            White-balanced image
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge back
        lab = cv2.merge([l, a, b])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    @staticmethod
    def calculate_face_density(image: np.ndarray, face_bbox: list) -> float:
        """
        Calculate face density (ratio of face area to image area).
        
        Args:
            image: Full image
            face_bbox: Face bounding box [x1, y1, x2, y2]
            
        Returns:
            Face density ratio (0-1)
        """
        img_height, img_width = image.shape[:2]
        img_area = img_width * img_height
        
        x1, y1, x2, y2 = face_bbox
        face_area = (x2 - x1) * (y2 - y1)
        
        if img_area == 0:
            return 0.0
            
        return face_area / img_area
    
    @staticmethod
    def is_face_centered(face_bbox: list, image_shape: tuple, 
                         tolerance: float = 0.2) -> bool:
        """
        Check if face is centered in the image.
        
        Args:
            face_bbox: Face bounding box [x1, y1, x2, y2]
            image_shape: Image dimensions (height, width)
            tolerance: Allowed deviation from center (0-1)
            
        Returns:
            True if face is centered within tolerance
        """
        height, width = image_shape[:2]
        center_x = width / 2
        center_y = height / 2
        
        face_center_x = (face_bbox[0] + face_bbox[2]) / 2
        face_center_y = (face_bbox[1] + face_bbox[3]) / 2
        
        x_deviation = abs(face_center_x - center_x) / width
        y_deviation = abs(face_center_y - center_y) / height
        
        return x_deviation <= tolerance and y_deviation <= tolerance