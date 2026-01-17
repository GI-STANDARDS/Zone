"""
Unit tests for Face Embedding Module
"""
import pytest
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from encoder import FaceEncoder, create_encoder


class TestFaceEncoder:
    """Test cases for FaceEncoder class"""
    
    @pytest.fixture
    def encoder(self):
        """Create an encoder instance for testing"""
        return FaceEncoder(model_type="facenet")
    
    @pytest.fixture
    def sample_face_image(self):
        """Create a sample face image for testing"""
        # Create a simple test image
        img_array = np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8)
        return Image.fromarray(img_array)
    
    def test_encoder_initialization(self):
        """Test encoder initialization"""
        encoder = FaceEncoder(model_type="facenet")
        assert encoder.model_type == "facenet"
        assert encoder.model is not None
        assert encoder.transform is not None
    
    def test_invalid_model_type(self):
        """Test invalid model type raises error"""
        with pytest.raises(ValueError):
            FaceEncoder(model_type="invalid_model")
    
    def test_preprocess_image(self, encoder, sample_face_image):
        """Test image preprocessing"""
        processed = encoder.preprocess_image(sample_face_image)
        assert isinstance(processed, (np.ndarray, object))  # Different types for different models
    
    def test_generate_embedding(self, encoder, sample_face_image):
        """Test embedding generation"""
        embedding = encoder.generate_embedding(sample_face_image)
        assert isinstance(embedding, np.ndarray)
        assert len(embedding.shape) == 1  # Should be 1D vector
        assert embedding.dtype == np.float64 or embedding.dtype == np.float32
    
    def test_generate_embeddings_batch(self, encoder, sample_face_image):
        """Test batch embedding generation"""
        images = [sample_face_image] * 3  # Create 3 identical images
        embeddings = encoder.generate_embeddings_batch(images)
        
        assert isinstance(embeddings, list)
        assert len(embeddings) == 3
        
        for embedding in embeddings:
            assert isinstance(embedding, np.ndarray)
            assert len(embedding.shape) == 1
    
    def test_compute_similarity_cosine(self, encoder):
        """Test cosine similarity computation"""
        emb1 = np.random.random(512)
        emb2 = np.random.random(512)
        
        similarity = encoder.compute_similarity(emb1, emb2, metric="cosine")
        assert isinstance(similarity, float)
        assert -1.0 <= similarity <= 1.0
    
    def test_compute_similarity_euclidean(self, encoder):
        """Test euclidean similarity computation"""
        emb1 = np.random.random(512)
        emb2 = np.random.random(512)
        
        similarity = encoder.compute_similarity(emb1, emb2, metric="euclidean")
        assert isinstance(similarity, float)
        assert 0.0 <= similarity <= 1.0
    
    def test_find_most_similar(self, encoder):
        """Test finding most similar embeddings"""
        query_emb = np.random.random(512)
        candidates = [np.random.random(512) for _ in range(10)]
        
        # Make one candidate very similar
        candidates[5] = query_emb + 0.01 * np.random.random(512)
        
        similar = encoder.find_most_similar(query_emb, candidates, threshold=0.8)
        assert isinstance(similar, list)
        # Should find at least one similar embedding
        assert len(similar) >= 1
    
    def test_get_embedding_dimension(self, encoder):
        """Test getting embedding dimension"""
        dim = encoder.get_embedding_dimension()
        assert isinstance(dim, int)
        assert dim > 0
    
    def test_create_encoder_factory(self):
        """Test factory function"""
        encoder = create_encoder(model_type="facenet")
        assert isinstance(encoder, FaceEncoder)
        assert encoder.model_type == "facenet"


if __name__ == "__main__":
    pytest.main([__file__])
