"""
Configuration settings for Face Recognition System
"""
import os
from pathlib import Path
from typing import Dict, Any

# Base paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Data paths
DATABASE_PATH = DATA_DIR / "database" / "faces.db"
EMBEDDINGS_PATH = DATA_DIR / "embeddings"
IMAGES_PATH = DATA_DIR / "images"

# Model configurations
FACE_DETECTION_MODEL = "mtcnn"  # Options: "mtcnn", "retinaface"
FACE_EMBEDDING_MODEL = "facenet"  # Options: "facenet", "arcface", "insightface"

# MTCNN settings
MTCNN_CONFIG = {
    "image_size": 160,
    "margin": 0,
    "min_face_size": 20,
    "thresholds": [0.6, 0.7, 0.8],
    "factor": 0.709,
    "post_process": True,
    "select_largest": True,
    "selection_method": "probability"
}

# FaceNet settings
FACENET_CONFIG = {
    "image_size": 160,
    "embedding_dim": 512
}

# Recognition settings
SIMILARITY_THRESHOLD = 0.6  # Cosine similarity threshold
SIMILARITY_METRIC = "cosine"  # Options: "cosine", "euclidean"
MAX_FACES_PER_IMAGE = 5

# Database settings
DATABASE_TYPE = "sqlite"  # Options: "sqlite", "faiss"
FAISS_INDEX_TYPE = "flat"  # Options: "flat", "ivf", "hnsw"

# UI settings
GRADIO_CONFIG = {
    "title": "Face Recognition System",
    "description": "Upload and recognize faces using deep learning",
    "theme": "default",
    "server_port": 7860,
    "share": False
}

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = LOGS_DIR / "face_recognition.log"

# Performance settings
BATCH_SIZE = 32
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
NUM_WORKERS = 4

# Security settings
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Ethics and privacy
ETHICS_WARNING = """
⚠️ ETHICAL USE WARNING ⚠️
This system is designed for legitimate face recognition purposes only.
- Do not use for unauthorized surveillance
- Obtain consent before storing personal data
- Follow applicable privacy laws and regulations
- This system should not be used for discriminatory purposes
"""

def get_config() -> Dict[str, Any]:
    """Return all configuration as dictionary"""
    return {
        "base_dir": str(BASE_DIR),
        "data_dir": str(DATA_DIR),
        "models_dir": str(MODELS_DIR),
        "logs_dir": str(LOGS_DIR),
        "database_path": str(DATABASE_PATH),
        "embeddings_path": str(EMBEDDINGS_PATH),
        "images_path": str(IMAGES_PATH),
        "face_detection_model": FACE_DETECTION_MODEL,
        "face_embedding_model": FACE_EMBEDDING_MODEL,
        "mtcnn_config": MTCNN_CONFIG,
        "facenet_config": FACENET_CONFIG,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "similarity_metric": SIMILARITY_METRIC,
        "max_faces_per_image": MAX_FACES_PER_IMAGE,
        "database_type": DATABASE_TYPE,
        "faiss_index_type": FAISS_INDEX_TYPE,
        "gradio_config": GRADIO_CONFIG,
        "log_level": LOG_LEVEL,
        "log_format": LOG_FORMAT,
        "log_file": str(LOG_FILE),
        "batch_size": BATCH_SIZE,
        "device": DEVICE,
        "num_workers": NUM_WORKERS,
        "allowed_extensions": ALLOWED_EXTENSIONS,
        "max_file_size": MAX_FILE_SIZE,
        "ethics_warning": ETHICS_WARNING
    }
