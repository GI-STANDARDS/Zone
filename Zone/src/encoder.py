"""
Face Embedding Module
Supports FaceNet, ArcFace, and InsightFace for generating face embeddings
"""
import cv2
import numpy as np
from typing import List, Union, Optional
import torch
from PIL import Image
import torchvision.transforms as transforms

try:
    from facenet_pytorch import InceptionResnetV1
    FACENET_AVAILABLE = True
except ImportError:
    FACENET_AVAILABLE = False

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.config import (
    FACE_EMBEDDING_MODEL,
    FACENET_CONFIG,
    DEVICE,
    BATCH_SIZE
)
from src.logger import get_logger

logger = get_logger(__name__)


class FaceEncoder:
    """
    Face embedding generation using FaceNet, ArcFace, or InsightFace
    """
    
    def __init__(self, model_type: str = FACE_EMBEDDING_MODEL):
        """
        Initialize face encoder
        
        Args:
            model_type: Type of face embedding model ("facenet", "arcface", "insightface")
        """
        self.model_type = model_type.lower()
        self.device = DEVICE
        self.model = None
        self.transform = None
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the specified face embedding model"""
        if self.model_type == "facenet":
            if not FACENET_AVAILABLE:
                raise ImportError("FaceNet not available. Install facenet-pytorch")
            
            self.model = InceptionResnetV1(
                pretrained=None,  # Don't auto-load pretrained weights
                device=self.device
            ).eval()
            
            # Define preprocessing transform
            self.transform = transforms.Compose([
                transforms.Resize(FACENET_CONFIG["image_size"]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5]
                )
            ])
            
            logger.info(f"FaceNet encoder initialized on {self.device}")
            
        elif self.model_type == "arcface":
            if not INSIGHTFACE_AVAILABLE:
                raise ImportError("ArcFace not available. Install insightface")
            
            # Initialize InsightFace with ArcFace model
            self.model = FaceAnalysis(name='arcface_r100_v1', providers=['CPUExecutionProvider'])
            self.model.prepare(ctx_id=0 if self.device == 'cpu' else 0)
            
            logger.info(f"ArcFace encoder initialized on {self.device}")
            
        elif self.model_type == "insightface":
            if not INSIGHTFACE_AVAILABLE:
                raise ImportError("InsightFace not available. Install insightface")
            
            # Initialize InsightFace with general model
            self.model = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
            self.model.prepare(ctx_id=0 if self.device == 'cpu' else 0)
            
            logger.info(f"InsightFace encoder initialized on {self.device}")
            
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def preprocess_image(self, image: Union[np.ndarray, Image.Image]) -> torch.Tensor:
        """
        Preprocess image for embedding generation
        
        Args:
            image: Input image
            
        Returns:
            Preprocessed tensor
        """
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        if self.model_type == "facenet":
            return self.transform(image).unsqueeze(0)
        else:
            # For ArcFace and InsightFace, return numpy array
            return np.array(image)
    
    def generate_embedding(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Generate face embedding for a single image
        
        Args:
            image: Input face image
            
        Returns:
            Face embedding vector
        """
        if self.model_type == "facenet":
            return self._generate_facenet_embedding(image)
        else:
            return self._generate_insightface_embedding(image)
    
    def _generate_facenet_embedding(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Generate embedding using FaceNet"""
        try:
            # Preprocess image
            input_tensor = self.preprocess_image(image).to(self.device)
            
            # Generate embedding
            with torch.no_grad():
                embedding = self.model(input_tensor)
            
            # Convert to numpy and normalize
            if hasattr(embedding, 'cpu'):
                embedding = embedding.cpu().numpy().flatten()
            else:
                embedding = np.array(embedding).flatten()
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"FaceNet embedding generation failed: {e}")
            raise
    
    def _generate_insightface_embedding(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Generate embedding using ArcFace or InsightFace"""
        try:
            # Preprocess image
            img_array = self.preprocess_image(image)
            
            # Generate embedding
            faces = self.model.get(img_array)
            
            if len(faces) == 0:
                raise ValueError("No face detected in the image")
            
            # Use the first detected face
            embedding = faces[0].embedding
            
            # Normalize embedding
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
            
        except Exception as e:
            logger.error(f"{self.model_type} embedding generation failed: {e}")
            raise
    
    def generate_embeddings_batch(self, images: List[Union[np.ndarray, Image.Image]]) -> List[np.ndarray]:
        """
        Generate embeddings for multiple images
        
        Args:
            images: List of input face images
            
        Returns:
            List of face embedding vectors
        """
        embeddings = []
        
        # Process in batches
        for i in range(0, len(images), BATCH_SIZE):
            batch_images = images[i:i + BATCH_SIZE]
            
            if self.model_type == "facenet":
                batch_embeddings = self._generate_facenet_batch(batch_images)
                embeddings.extend(batch_embeddings)
            else:
                # Process individually for ArcFace/InsightFace
                for img in batch_images:
                    embedding = self._generate_insightface_embedding(img)
                    embeddings.append(embedding)
        
        return embeddings
    
    def _generate_facenet_batch(self, images: List[Union[np.ndarray, Image.Image]]) -> List[np.ndarray]:
        """Generate FaceNet embeddings for a batch of images"""
        try:
            # Preprocess all images
            batch_tensors = []
            for img in images:
                tensor = self.preprocess_image(img)
                batch_tensors.append(tensor)
            
            # Stack tensors
            batch_tensor = torch.cat(batch_tensors, dim=0).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                embeddings = self.model(batch_tensor)
            
            # Convert to numpy and normalize
            embeddings = embeddings.cpu().numpy()
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            return [emb.flatten() for emb in embeddings]
            
        except Exception as e:
            logger.error(f"FaceNet batch embedding generation failed: {e}")
            raise
    
    def compute_similarity(
        self, 
        embedding1: np.ndarray, 
        embedding2: np.ndarray,
        metric: str = "cosine"
    ) -> float:
        """
        Compute similarity between two embeddings
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            metric: Similarity metric ("cosine" or "euclidean")
            
        Returns:
            Similarity score
        """
        if metric == "cosine":
            # Cosine similarity
            similarity = np.dot(embedding1, embedding2)
        elif metric == "euclidean":
            # Euclidean distance (converted to similarity)
            distance = np.linalg.norm(embedding1 - embedding2)
            similarity = 1 / (1 + distance)
        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")
        
        return float(similarity)
    
    def find_most_similar(
        self, 
        query_embedding: np.ndarray,
        candidate_embeddings: List[np.ndarray],
        threshold: float = 0.6,
        metric: str = "cosine"
    ) -> List[tuple]:
        """
        Find most similar embeddings to query
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of candidate embeddings
            threshold: Similarity threshold
            metric: Similarity metric
            
        Returns:
            List of (index, similarity) tuples sorted by similarity
        """
        similarities = []
        
        for i, candidate_emb in enumerate(candidate_embeddings):
            similarity = self.compute_similarity(query_embedding, candidate_emb, metric)
            if similarity >= threshold:
                similarities.append((i, similarity))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of the embedding vector
        
        Returns:
            Embedding dimension
        """
        if self.model_type == "facenet":
            return FACENET_CONFIG["embedding_dim"]
        elif self.model_type in ["arcface", "insightface"]:
            return 512  # Standard for ArcFace/InsightFace models
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def save_model(self, path: str):
        """
        Save the trained model (for fine-tuned models)
        
        Args:
            path: Path to save the model
        """
        if self.model_type == "facenet":
            torch.save(self.model.state_dict(), path)
            logger.info(f"FaceNet model saved to {path}")
        else:
            logger.warning(f"Model saving not implemented for {self.model_type}")
    
    def load_model(self, path: str):
        """
        Load a trained model
        
        Args:
            path: Path to the saved model
        """
        if self.model_type == "facenet":
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            logger.info(f"FaceNet model loaded from {path}")
        else:
            logger.warning(f"Model loading not implemented for {self.model_type}")


def create_encoder(model_type: str = FACE_EMBEDDING_MODEL) -> FaceEncoder:
    """
    Factory function to create face encoder
    
    Args:
        model_type: Type of face embedding model
        
    Returns:
        FaceEncoder instance
    """
    return FaceEncoder(model_type)


if __name__ == "__main__":
    # Test the encoder
    encoder = create_encoder()
    
    # Test with a sample image (if available)
    test_image_path = "data/images/test_face.jpg"
    try:
        image = Image.open(test_image_path)
        embedding = encoder.generate_embedding(image)
        
        print(f"Generated embedding with shape: {embedding.shape}")
        print(f"Embedding dimension: {encoder.get_embedding_dimension()}")
        print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
        
    except FileNotFoundError:
        print(f"Test image not found at {test_image_path}")
        print("Please place a test image at the specified path to test the encoder")
