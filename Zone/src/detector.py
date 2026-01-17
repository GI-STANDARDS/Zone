"""
Face Detection Module
Supports MTCNN and RetinaFace for face detection and alignment
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Union
import torch
from PIL import Image

try:
    from facenet_pytorch import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False

try:
    from retinaface import RetinaFace
    RETINAFACE_AVAILABLE = True
except ImportError:
    RETINAFACE_AVAILABLE = False

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.config import (
    FACE_DETECTION_MODEL,
    MTCNN_CONFIG,
    DEVICE,
    MAX_FACES_PER_IMAGE
)
from src.logger import get_logger

logger = get_logger(__name__)


class FaceDetector:
    """
    Face detection and alignment using MTCNN or RetinaFace
    """
    
    def __init__(self, model_type: str = FACE_DETECTION_MODEL):
        """
        Initialize face detector
        
        Args:
            model_type: Type of face detection model ("mtcnn" or "retinaface")
        """
        self.model_type = model_type.lower()
        self.device = DEVICE
        self.detector = None
        self._initialize_detector()
        
    def _initialize_detector(self):
        """Initialize the specified face detection model"""
        if self.model_type == "mtcnn":
            if not MTCNN_AVAILABLE:
                raise ImportError("MTCNN not available. Install facenet-pytorch")
            
            self.detector = MTCNN(
                image_size=MTCNN_CONFIG["image_size"],
                margin=MTCNN_CONFIG["margin"],
                min_face_size=MTCNN_CONFIG["min_face_size"],
                thresholds=MTCNN_CONFIG["thresholds"],
                factor=MTCNN_CONFIG["factor"],
                post_process=MTCNN_CONFIG["post_process"],
                select_largest=MTCNN_CONFIG["select_largest"],
                selection_method=MTCNN_CONFIG["selection_method"],
                device=self.device
            )
            logger.info(f"MTCNN detector initialized on {self.device}")
            
        elif self.model_type == "retinaface":
            if not RETINAFACE_AVAILABLE:
                raise ImportError("RetinaFace not available. Install retinaface")
            
            self.detector = RetinaFace(
                backbone="resnet50",
                device=self.device
            )
            logger.info(f"RetinaFace detector initialized on {self.device}")
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def detect_faces(
        self, 
        image: Union[np.ndarray, Image.Image, str],
        return_landmarks: bool = False
    ) -> List[dict]:
        """
        Detect faces in an image
        
        Args:
            image: Input image (numpy array, PIL Image, or file path)
            return_landmarks: Whether to return facial landmarks
            
        Returns:
            List of detected faces with bounding boxes and metadata
        """
        # Convert input to PIL Image
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        faces = []
        
        if self.model_type == "mtcnn":
            faces = self._detect_mtcnn(image, return_landmarks)
        elif self.model_type == "retinaface":
            faces = self._detect_retinaface(image, return_landmarks)
        
        # Limit number of faces
        if len(faces) > MAX_FACES_PER_IMAGE:
            faces = faces[:MAX_FACES_PER_IMAGE]
            logger.warning(f"Limited to {MAX_FACES_PER_IMAGE} faces")
        
        logger.info(f"Detected {len(faces)} faces using {self.model_type}")
        return faces
    
    def _detect_mtcnn(self, image: Image.Image, return_landmarks: bool) -> List[dict]:
        """Detect faces using MTCNN"""
        try:
            # Detect faces and get bounding boxes
            boxes, probs, landmarks = self.detector.detect(image, landmarks=True)
            
            faces = []
            if boxes is not None:
                for i, (box, prob) in enumerate(zip(boxes, probs)):
                    if prob > 0.9:  # Confidence threshold
                        face_data = {
                            'bbox': box.tolist(),
                            'confidence': float(prob),
                            'landmarks': landmarks[i].tolist() if landmarks is not None and return_landmarks else None
                        }
                        faces.append(face_data)
            
            return faces
            
        except Exception as e:
            logger.error(f"MTCNN detection failed: {e}")
            return []
    
    def _detect_retinaface(self, image: Image.Image, return_landmarks: bool) -> List[dict]:
        """Detect faces using RetinaFace"""
        try:
            # Convert PIL to numpy array
            img_array = np.array(image)
            
            # Detect faces
            faces = self.detector.predict(img_array)
            
            result = []
            for face_key, face_data in faces.items():
                if isinstance(face_data, dict):
                    bbox = face_data.get('facial_area', [])
                    confidence = face_data.get('score', 0)
                    
                    if confidence > 0.9:  # Confidence threshold
                        face_info = {
                            'bbox': bbox,
                            'confidence': float(confidence),
                            'landmarks': face_data.get('landmarks', None) if return_landmarks else None
                        }
                        result.append(face_info)
            
            return result
            
        except Exception as e:
            logger.error(f"RetinaFace detection failed: {e}")
            return []
    
    def align_face(self, image: Union[np.ndarray, Image.Image], bbox: List[float]) -> Image.Image:
        """
        Align face based on detected bounding box
        
        Args:
            image: Input image
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Aligned face image
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if self.model_type == "mtcnn":
            # Use MTCNN's built-in alignment
            try:
                aligned_face = self.detector(image, return_prob=False)
                if aligned_face is not None:
                    return aligned_face
            except Exception as e:
                logger.warning(f"MTCNN alignment failed: {e}")
        
        # Fallback: simple crop and resize
        x1, y1, x2, y2 = map(int, bbox)
        face = image.crop((x1, y1, x2, y2))
        
        # Resize to standard size
        face = face.resize((160, 160), Image.Resampling.LANCZOS)
        return face
    
    def extract_faces(
        self, 
        image: Union[np.ndarray, Image.Image, str],
        align: bool = True
    ) -> List[Image.Image]:
        """
        Extract face images from input image
        
        Args:
            image: Input image
            align: Whether to align faces
            
        Returns:
            List of face images
        """
        faces_data = self.detect_faces(image)
        face_images = []
        
        for face_data in faces_data:
            bbox = face_data['bbox']
            
            if align:
                face_img = self.align_face(image, bbox)
            else:
                # Simple crop
                if isinstance(image, str):
                    image = Image.open(image)
                elif isinstance(image, np.ndarray):
                    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                
                x1, y1, x2, y2 = map(int, bbox)
                face_img = image.crop((x1, y1, x2, y2))
                face_img = face_img.resize((160, 160), Image.Resampling.LANCZOS)
            
            face_images.append(face_img)
        
        return face_images
    
    def visualize_detections(
        self, 
        image: Union[np.ndarray, Image.Image],
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize face detections on image
        
        Args:
            image: Input image
            save_path: Path to save visualization (optional)
            
        Returns:
            Image with bounding boxes drawn
        """
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image.copy()
        
        faces_data = self.detect_faces(image)
        
        for i, face_data in enumerate(faces_data):
            bbox = face_data['bbox']
            confidence = face_data['confidence']
            
            x1, y1, x2, y2 = map(int, bbox)
            
            # Draw bounding box
            cv2.rectangle(img_array, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw confidence score
            label = f"Face {i+1}: {confidence:.2f}"
            cv2.putText(
                img_array, label, (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
            )
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
            logger.info(f"Visualization saved to {save_path}")
        
        return img_array


def create_detector(model_type: str = FACE_DETECTION_MODEL) -> FaceDetector:
    """
    Factory function to create face detector
    
    Args:
        model_type: Type of face detection model
        
    Returns:
        FaceDetector instance
    """
    return FaceDetector(model_type)


if __name__ == "__main__":
    # Test the detector
    detector = create_detector()
    
    # Test with a sample image (if available)
    test_image_path = "data/images/test_face.jpg"
    try:
        faces = detector.detect_faces(test_image_path)
        print(f"Detected {len(faces)} faces")
        
        for i, face in enumerate(faces):
            print(f"Face {i+1}: bbox={face['bbox']}, confidence={face['confidence']}")
            
    except FileNotFoundError:
        print(f"Test image not found at {test_image_path}")
        print("Please place a test image at the specified path to test the detector")
