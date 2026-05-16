import cv2
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class FacePreprocessor:
    """
    Face image preprocessing for optimal recognition
    """
    
    def __init__(self):
        self.target_size = (112, 112)  # InsightFace expects 112x112
    
    def enhance_low_light(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance low-light images using CLAHE and gamma correction
        """
        if image is None:
            return image
        
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_enhanced = clahe.apply(l)
        
        # Merge back
        lab_enhanced = cv2.merge([l_enhanced, a, b])
        enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        # Apply gamma correction if still dark
        brightness = np.mean(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY))
        if brightness < 80:
            gamma = 1.2
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 
                              for i in range(256)]).astype("uint8")
            enhanced = cv2.LUT(enhanced, table)
        
        return enhanced
    
    def correct_illumination(self, image: np.ndarray) -> np.ndarray:
        """
        Correct uneven illumination
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to get illumination pattern
        blur = cv2.GaussianBlur(gray, (0, 0), 30)
        
        # Subtract illumination pattern
        corrected = cv2.subtract(gray, blur)
        
        # Normalize to full range
        corrected = cv2.normalize(corrected, None, 0, 255, cv2.NORM_MINMAX)
        
        # Convert back to BGR
        result = cv2.cvtColor(corrected, cv2.COLOR_GRAY2BGR)
        
        return result
    
    def align_face(self, image: np.ndarray, landmarks: list) -> np.ndarray:
        """
        Align face using eye landmarks for better recognition
        """
        if not landmarks or len(landmarks) < 2:
            return image
        
        # Get left and right eye positions (first two landmarks are eyes)
        left_eye = np.array(landmarks[0][:2])
        right_eye = np.array(landmarks[1][:2])
        
        # Calculate angle between eyes
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Desired eye angle is 0 degrees (horizontal)
        rotation_angle = -angle
        
        # Get center between eyes
        center = ((left_eye[0] + right_eye[0]) / 2, 
                  (left_eye[1] + right_eye[1]) / 2)
        
        # Create rotation matrix
        M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
        
        # Apply rotation
        rotated = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]),
                                  flags=cv2.INTER_CUBIC)
        
        return rotated
    
    def crop_and_resize_face(self, image: np.ndarray, bbox: list, 
                              margin: float = 0.2) -> np.ndarray:
        """
        Crop face from image with margin and resize to target size
        """
        x1, y1, x2, y2 = bbox
        
        # Add margin
        width = x2 - x1
        height = y2 - y1
        margin_x = int(width * margin)
        margin_y = int(height * margin)
        
        x1 = max(0, x1 - margin_x)
        y1 = max(0, y1 - margin_y)
        x2 = min(image.shape[1], x2 + margin_x)
        y2 = min(image.shape[0], y2 + margin_y)
        
        # Crop face
        face = image[y1:y2, x1:x2]
        
        if face.size == 0:
            return None
        
        # Resize to target size
        face_resized = cv2.resize(face, self.target_size, interpolation=cv2.INTER_CUBIC)
        
        return face_resized
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """
        Apply denoising to reduce noise
        """
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    
    def sharpen(self, image: np.ndarray) -> np.ndarray:
        """
        Apply sharpening filter
        """
        kernel = np.array([[-1, -1, -1],
                           [-1,  9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(image, -1, kernel)
        return sharpened
    
    def preprocess_for_recognition(self, image: np.ndarray, bbox: list = None,
                                    landmarks: list = None) -> Optional[np.ndarray]:
        """
        Complete preprocessing pipeline for face recognition
        """
        if image is None:
            return None
        
        # Step 1: Enhance low light
        enhanced = self.enhance_low_light(image)
        
        # Step 2: Denoise
        denoised = self.denoise(enhanced)
        
        # Step 3: Correct illumination
        corrected = self.correct_illumination(denoised)
        
        # Step 4: Align if landmarks available
        if landmarks:
            aligned = self.align_face(corrected, landmarks)
        else:
            aligned = corrected
        
        # Step 5: Crop face if bbox available
        if bbox:
            cropped = self.crop_and_resize_face(aligned, bbox)
        else:
            cropped = cv2.resize(aligned, self.target_size)
        
        if cropped is None:
            return None
        
        # Step 6: Sharpen
        sharpened = self.sharpen(cropped)
        
        return sharpened