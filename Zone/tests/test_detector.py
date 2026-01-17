"""
Unit tests for Face Detection Module
"""
import pytest
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from detector import FaceDetector, create_detector


class TestFaceDetector:
    """Test cases for FaceDetector class"""
    
    @pytest.fixture
    def detector(self):
        """Create a detector instance for testing"""
        return FaceDetector(model_type="mtcnn")
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing"""
        # Create a simple test image with a face-like pattern
        img_array = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        return Image.fromarray(img_array)
    
    def test_detector_initialization(self):
        """Test detector initialization"""
        detector = FaceDetector(model_type="mtcnn")
        assert detector.model_type == "mtcnn"
        assert detector.detector is not None
    
    def test_invalid_model_type(self):
        """Test invalid model type raises error"""
        with pytest.raises(ValueError):
            FaceDetector(model_type="invalid_model")
    
    def test_detect_faces_with_pil_image(self, detector, sample_image):
        """Test face detection with PIL image"""
        faces = detector.detect_faces(sample_image)
        assert isinstance(faces, list)
        # Note: This might return empty list for random image
    
    def test_detect_faces_with_numpy_array(self, detector, sample_image):
        """Test face detection with numpy array"""
        img_array = np.array(sample_image)
        faces = detector.detect_faces(img_array)
        assert isinstance(faces, list)
    
    def test_extract_faces(self, detector, sample_image):
        """Test face extraction"""
        face_images = detector.extract_faces(sample_image)
        assert isinstance(face_images, list)
        # Each face should be a PIL Image
        for face_img in face_images:
            assert isinstance(face_img, Image.Image)
    
    def test_align_face(self, detector, sample_image):
        """Test face alignment"""
        # Create a dummy bounding box
        bbox = [10, 10, 150, 150]
        aligned_face = detector.align_face(sample_image, bbox)
        assert isinstance(aligned_face, Image.Image)
        assert aligned_face.size == (160, 160)
    
    def test_visualize_detections(self, detector, sample_image):
        """Test detection visualization"""
        vis_img = detector.visualize_detections(sample_image)
        assert isinstance(vis_img, np.ndarray)
        assert vis_img.shape == (160, 160, 3)
    
    def test_create_detector_factory(self):
        """Test factory function"""
        detector = create_detector(model_type="mtcnn")
        assert isinstance(detector, FaceDetector)
        assert detector.model_type == "mtcnn"


if __name__ == "__main__":
    pytest.main([__file__])
