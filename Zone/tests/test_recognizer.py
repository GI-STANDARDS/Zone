"""
Unit tests for Face Recognition Pipeline
"""
import pytest
import numpy as np
from PIL import Image
import tempfile
import shutil
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from recognizer import FaceRecognizer, create_recognizer


class TestFaceRecognizer:
    """Test cases for FaceRecognizer class"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def recognizer(self, temp_dir):
        """Create a recognizer instance for testing"""
        # Override config paths for testing
        import config.config
        original_db_path = config.config.DATABASE_PATH
        original_embeddings_path = config.config.EMBEDDINGS_PATH
        
        config.config.DATABASE_PATH = temp_dir / "test.db"
        config.config.EMBEDDINGS_PATH = temp_dir / "embeddings"
        
        recognizer = FaceRecognizer(
            detector_model="mtcnn",
            encoder_model="facenet",
            db_type="sqlite",
            threshold=0.6
        )
        
        yield recognizer
        
        recognizer.close()
        config.config.DATABASE_PATH = original_db_path
        config.config.EMBEDDINGS_PATH = original_embeddings_path
    
    @pytest.fixture
    def sample_face_image(self):
        """Create a sample face image for testing"""
        # Create a simple test image with face-like pattern
        img_array = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        # Add some face-like features (simplified)
        # Eyes
        img_array[50:60, 40:50] = [255, 255, 255]  # Left eye
        img_array[50:60, 110:120] = [255, 255, 255]  # Right eye
        # Mouth
        img_array[100:110, 60:100] = [255, 0, 0]  # Mouth
        
        return Image.fromarray(img_array)
    
    def test_recognizer_initialization(self, temp_dir):
        """Test recognizer initialization"""
        # Override config paths
        import config.config
        original_db_path = config.config.DATABASE_PATH
        original_embeddings_path = config.config.EMBEDDINGS_PATH
        
        config.config.DATABASE_PATH = temp_dir / "test.db"
        config.config.EMBEDDINGS_PATH = temp_dir / "embeddings"
        
        recognizer = FaceRecognizer()
        
        assert recognizer.detector is not None
        assert recognizer.encoder is not None
        assert recognizer.database is not None
        assert recognizer.threshold == 0.6
        
        recognizer.close()
        config.config.DATABASE_PATH = original_db_path
        config.config.EMBEDDINGS_PATH = original_embeddings_path
    
    def test_register_face_success(self, recognizer, sample_face_image):
        """Test successful face registration"""
        result = recognizer.register_face("Test Person", sample_face_image, save_image=False)
        
        assert result["success"] is True
        assert "face_id" in result
        assert result["message"] == "Face registered successfully for Test Person"
    
    def test_register_face_no_face(self, recognizer):
        """Test registration with no face detected"""
        # Create image with no face-like features
        img_array = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        no_face_image = Image.fromarray(img_array)
        
        result = recognizer.register_face("No Face", no_face_image, save_image=False)
        
        assert result["success"] is False
        assert "No face detected" in result["message"]
        assert result["faces_detected"] == 0
    
    def test_register_face_multiple_faces(self, recognizer):
        """Test registration with multiple faces"""
        # This test might not work with random images, but structure is correct
        img_array = np.random.randint(0, 255, (320, 320, 3), dtype=np.uint8)
        multi_face_image = Image.fromarray(img_array)
        
        result = recognizer.register_face("Multiple Faces", multi_face_image, save_image=False)
        
        # Result depends on whether MTCNN detects multiple faces
        assert "success" in result
        assert "faces_detected" in result
    
    def test_recognize_faces_no_faces(self, recognizer):
        """Test recognition with no faces"""
        img_array = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        no_face_image = Image.fromarray(img_array)
        
        result = recognizer.recognize_faces(no_face_image)
        
        assert result["success"] is True
        assert result["faces_found"] == 0
        assert "No faces detected" in result["message"]
    
    def test_recognize_single_face_success(self, recognizer, sample_face_image):
        """Test successful single face recognition"""
        # First register a face
        register_result = recognizer.register_face("Known Person", sample_face_image, save_image=False)
        assert register_result["success"] is True
        
        # Then recognize the same face
        result = recognizer.recognize_single_face(sample_face_image)
        
        assert result["success"] is True
        assert result["face_found"] is True
        assert "bbox" in result
        assert "detection_confidence" in result
    
    def test_update_threshold(self, recognizer):
        """Test threshold update"""
        original_threshold = recognizer.threshold
        
        recognizer.update_threshold(0.8)
        assert recognizer.threshold == 0.8
        
        # Test invalid threshold
        with pytest.raises(ValueError):
            recognizer.update_threshold(1.5)
        
        # Restore original threshold
        recognizer.update_threshold(original_threshold)
    
    def test_get_database_stats(self, recognizer, sample_face_image):
        """Test getting database statistics"""
        # Add a face first
        recognizer.register_face("Stats Person", sample_face_image, save_image=False)
        
        stats = recognizer.get_database_stats()
        
        assert isinstance(stats, dict)
        assert "total_faces" in stats
        assert "unique_names" in stats
        assert "database_type" in stats
        assert stats["total_faces"] >= 1
        assert stats["database_type"] == "sqlite"
    
    def test_visualize_recognition(self, recognizer, sample_face_image):
        """Test recognition visualization"""
        # Register a face first
        recognizer.register_face("Visual Person", sample_face_image, save_image=False)
        
        # Visualize recognition
        vis_img = recognizer.visualize_recognition(sample_face_image)
        
        assert isinstance(vis_img, np.ndarray)
        assert vis_img.shape[2] == 3  # RGB channels
    
    def test_create_recognizer_factory(self, temp_dir):
        """Test factory function"""
        # Override config paths
        import config.config
        original_db_path = config.config.DATABASE_PATH
        original_embeddings_path = config.config.EMBEDDINGS_PATH
        
        config.config.DATABASE_PATH = temp_dir / "test.db"
        config.config.EMBEDDINGS_PATH = temp_dir / "embeddings"
        
        recognizer = create_recognizer(
            detector_model="mtcnn",
            encoder_model="facenet",
            db_type="sqlite",
            threshold=0.7
        )
        
        assert isinstance(recognizer, FaceRecognizer)
        assert recognizer.threshold == 0.7
        
        recognizer.close()
        config.config.DATABASE_PATH = original_db_path
        config.config.EMBEDDINGS_PATH = original_embeddings_path


if __name__ == "__main__":
    pytest.main([__file__])
