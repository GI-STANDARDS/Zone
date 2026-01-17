"""
Face Recognition Pipeline
Integrates detection, encoding, and database for complete face recognition
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from PIL import Image
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src.detector import FaceDetector, create_detector
from src.encoder import FaceEncoder, create_encoder
from src.database import FaceDatabase, create_database
from config.config import (
    SIMILARITY_THRESHOLD,
    SIMILARITY_METRIC,
    MAX_FACES_PER_IMAGE
)
from src.logger import get_logger

logger = get_logger(__name__)


class FaceRecognizer:
    """
    Complete face recognition pipeline
    """
    
    def __init__(
        self,
        detector_model: str = "mtcnn",
        encoder_model: str = "facenet",
        db_type: str = "sqlite",
        threshold: float = SIMILARITY_THRESHOLD
    ):
        """
        Initialize face recognizer
        
        Args:
            detector_model: Face detection model
            encoder_model: Face embedding model
            db_type: Database type
            threshold: Similarity threshold
        """
        self.detector = create_detector(detector_model)
        self.encoder = create_encoder(encoder_model)
        self.database = create_database(db_type)
        self.threshold = threshold
        
        logger.info(f"FaceRecognizer initialized with {detector_model}+{encoder_model}+{db_type}")
    
    def register_face(
        self,
        name: str,
        image: Union[np.ndarray, Image.Image, str],
        save_image: bool = True
    ) -> Dict[str, any]:
        """
        Register a new face in the database
        
        Args:
            name: Person's name
            image: Face image
            save_image: Whether to save the image
            
        Returns:
            Registration result
        """
        try:
            # Detect faces
            faces_data = self.detector.detect_faces(image)
            
            if not faces_data:
                return {
                    "success": False,
                    "message": "No face detected in the image",
                    "faces_detected": 0
                }
            
            if len(faces_data) > 1:
                return {
                    "success": False,
                    "message": f"Multiple faces detected ({len(faces_data)}). Please provide an image with a single face.",
                    "faces_detected": len(faces_data)
                }
            
            # Extract and align face
            face_images = self.detector.extract_faces(image, align=True)
            if not face_images:
                return {
                    "success": False,
                    "message": "Failed to extract face",
                    "faces_detected": 1
                }
            
            # Generate embedding
            face_image = face_images[0]
            embedding = self.encoder.generate_embedding(face_image)
            
            # Save image if requested
            image_path = None
            if save_image:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = f"data/images/{name}_{timestamp}.jpg"
                face_image.save(image_path)
            
            # Add to database
            face_id = self.database.add_face(name, embedding, image_path)
            
            return {
                "success": True,
                "message": f"Face registered successfully for {name}",
                "face_id": face_id,
                "confidence": faces_data[0]['confidence'],
                "image_path": image_path
            }
            
        except Exception as e:
            logger.error(f"Face registration failed: {e}")
            return {
                "success": False,
                "message": f"Registration failed: {str(e)}",
                "faces_detected": 0
            }
    
    def recognize_faces(
        self,
        image: Union[np.ndarray, Image.Image, str],
        top_k: int = 5
    ) -> Dict[str, any]:
        """
        Recognize faces in an image
        
        Args:
            image: Input image
            top_k: Number of top matches to return
            
        Returns:
            Recognition results
        """
        try:
            start_time = time.time()
            
            # Detect faces
            faces_data = self.detector.detect_faces(image)
            
            if not faces_data:
                return {
                    "success": True,
                    "faces_found": 0,
                    "message": "No faces detected in the image",
                    "processing_time": time.time() - start_time,
                    "results": []
                }
            
            # Extract faces
            face_images = self.detector.extract_faces(image, align=True)
            
            if len(face_images) != len(faces_data):
                return {
                    "success": False,
                    "message": "Face extraction failed",
                    "faces_found": len(faces_data),
                    "processing_time": time.time() - start_time,
                    "results": []
                }
            
            # Generate embeddings and find matches
            results = []
            for i, (face_data, face_image) in enumerate(zip(faces_data, face_images)):
                embedding = self.encoder.generate_embedding(face_image)
                
                # Find similar faces
                similar_faces = self.database.find_similar_faces(
                    embedding,
                    threshold=self.threshold,
                    top_k=top_k
                )
                
                # Prepare result
                result = {
                    "face_index": i,
                    "bbox": face_data['bbox'],
                    "detection_confidence": face_data['confidence'],
                    "matches": similar_faces,
                    "best_match": similar_faces[0] if similar_faces else None,
                    "recognized": len(similar_faces) > 0
                }
                
                results.append(result)
            
            return {
                "success": True,
                "faces_found": len(results),
                "processing_time": time.time() - start_time,
                "results": results
            }
            
        except Exception as e:
            logger.error(f"Face recognition failed: {e}")
            return {
                "success": False,
                "message": f"Recognition failed: {str(e)}",
                "faces_found": 0,
                "processing_time": time.time() - start_time if 'start_time' in locals() else 0,
                "results": []
            }
    
    def recognize_single_face(
        self,
        image: Union[np.ndarray, Image.Image, str]
    ) -> Dict[str, any]:
        """
        Recognize a single face in an image
        
        Args:
            image: Input image
            
        Returns:
            Recognition result
        """
        result = self.recognize_faces(image, top_k=1)
        
        if not result["success"] or result["faces_found"] == 0:
            return result
        
        if result["faces_found"] > 1:
            return {
                "success": False,
                "message": f"Multiple faces detected ({result['faces_found']}). Please provide an image with a single face.",
                "faces_found": result["faces_found"]
            }
        
        # Return single face result
        face_result = result["results"][0]
        return {
            "success": True,
            "face_found": True,
            "bbox": face_result["bbox"],
            "detection_confidence": face_result["detection_confidence"],
            "recognized": face_result["recognized"],
            "match": face_result["best_match"],
            "processing_time": result["processing_time"]
        }
    
    def update_threshold(self, new_threshold: float):
        """Update similarity threshold"""
        if 0.0 <= new_threshold <= 1.0:
            self.threshold = new_threshold
            logger.info(f"Similarity threshold updated to {new_threshold}")
        else:
            raise ValueError("Threshold must be between 0.0 and 1.0")
    
    def get_database_stats(self) -> Dict[str, any]:
        """Get database statistics"""
        return self.database.get_stats()
    
    def visualize_recognition(
        self,
        image: Union[np.ndarray, Image.Image, str],
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualize recognition results on image
        
        Args:
            image: Input image
            save_path: Path to save visualization
            
        Returns:
            Image with recognition results drawn
        """
        # Get recognition results
        results = self.recognize_faces(image)
        
        if not results["success"] or results["faces_found"] == 0:
            # Return original image if no faces found
            if isinstance(image, str):
                img_array = cv2.imread(image)
            elif isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                img_array = image.copy()
            
            return img_array
        
        # Convert to numpy array if needed
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        elif isinstance(image, str):
            img_array = cv2.imread(image)
        else:
            img_array = image.copy()
        
        # Draw results
        for face_result in results["results"]:
            bbox = face_result["bbox"]
            x1, y1, x2, y2 = map(int, bbox)
            
            # Choose color based on recognition
            color = (0, 255, 0) if face_result["recognized"] else (0, 0, 255)
            
            # Draw bounding box
            cv2.rectangle(img_array, (x1, y1), (x2, y2), color, 2)
            
            # Add label
            if face_result["recognized"] and face_result["best_match"]:
                name = face_result["best_match"]["name"]
                confidence = face_result["best_match"]["similarity"]
                label = f"{name}: {confidence:.2f}"
            else:
                label = "Unknown"
            
            cv2.putText(
                img_array, label, (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
            )
        
        # Save if path provided
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))
            logger.info(f"Recognition visualization saved to {save_path}")
        
        return img_array
    
    def close(self):
        """Close database connection"""
        self.database.close()


def create_recognizer(
    detector_model: str = "mtcnn",
    encoder_model: str = "facenet",
    db_type: str = "sqlite",
    threshold: float = SIMILARITY_THRESHOLD
) -> FaceRecognizer:
    """
    Factory function to create face recognizer
    
    Args:
        detector_model: Face detection model
        encoder_model: Face embedding model
        db_type: Database type
        threshold: Similarity threshold
        
    Returns:
        FaceRecognizer instance
    """
    return FaceRecognizer(detector_model, encoder_model, db_type, threshold)


if __name__ == "__main__":
    # Test the recognizer
    recognizer = create_recognizer()
    
    print("Face Recognition System Test")
    print(f"Database stats: {recognizer.get_database_stats()}")
    
    # Test with sample image
    test_image_path = "data/images/test_face.jpg"
    try:
        result = recognizer.recognize_faces(test_image_path)
        print(f"Recognition result: {result}")
    except FileNotFoundError:
        print(f"Test image not found at {test_image_path}")
        print("Please place a test image at the specified path")
    
    recognizer.close()
