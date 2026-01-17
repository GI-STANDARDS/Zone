"""
Database Module
Supports SQLite and FAISS for storing and retrieving face embeddings
"""
import sqlite3
import pickle
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path
import json
from datetime import datetime

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.config import (
    DATABASE_TYPE,
    DATABASE_PATH,
    EMBEDDINGS_PATH,
    FAISS_INDEX_TYPE,
    get_config
)
from src.logger import get_logger

logger = get_logger(__name__)


class FaceDatabase:
    """
    Database for storing and retrieving face embeddings
    Supports both SQLite and FAISS backends
    """
    
    def __init__(self, db_type: str = DATABASE_TYPE):
        """
        Initialize face database
        
        Args:
            db_type: Type of database ("sqlite" or "faiss")
        """
        self.db_type = db_type.lower()
        self.config = get_config()
        
        # Ensure directories exist
        DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)
        EMBEDDINGS_PATH.mkdir(parents=True, exist_ok=True)
        
        if self.db_type == "sqlite":
            self.conn = None
            self._init_sqlite()
        elif self.db_type == "faiss":
            if not FAISS_AVAILABLE:
                raise ImportError("FAISS not available. Install faiss-cpu or faiss-gpu")
            self.index = None
            self.metadata = {}
            self._init_faiss()
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")
        
        logger.info(f"Face database initialized with {self.db_type} backend")
    
    def _init_sqlite(self):
        """Initialize SQLite database"""
        self.conn = sqlite3.connect(str(DATABASE_PATH))
        self._create_tables()
    
    def _create_tables(self):
        """Create SQLite tables"""
        cursor = self.conn.cursor()
        
        # Create faces table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS faces (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                embedding BLOB NOT NULL,
                image_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        ''')
        
        # Insert initial metadata
        cursor.execute('''
            INSERT OR IGNORE INTO metadata (key, value) 
            VALUES ('embedding_dim', ?), ('total_faces', '0')
        ''', (self.config.get('facenet_config', {}).get('embedding_dim', 512),))
        
        self.conn.commit()
        logger.info("SQLite tables created successfully")
    
    def _init_faiss(self):
        """Initialize FAISS index"""
        embedding_dim = self.config.get('facenet_config', {}).get('embedding_dim', 512)
        
        if FAISS_INDEX_TYPE == "flat":
            self.index = faiss.IndexFlatIP(embedding_dim)  # Inner Product for cosine similarity
        elif FAISS_INDEX_TYPE == "ivf":
            nlist = 100  # Number of clusters
            quantizer = faiss.IndexFlatIP(embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, embedding_dim, nlist)
        elif FAISS_INDEX_TYPE == "hnsw":
            self.index = faiss.IndexHNSWFlat(embedding_dim, 32)  # M=32 connections
        
        # Load existing index if available
        index_path = EMBEDDINGS_PATH / "faiss.index"
        metadata_path = EMBEDDINGS_PATH / "metadata.json"
        
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logger.info("Loaded existing FAISS index")
        else:
            # Train index if needed
            if hasattr(self.index, 'train') and not self.index.is_trained:
                logger.info("Training FAISS index...")
                # Create dummy training data
                dummy_data = np.random.random((1000, embedding_dim)).astype('float32')
                self.index.train(dummy_data)
                logger.info("FAISS index training completed")
    
    def add_face(
        self, 
        name: str, 
        embedding: np.ndarray, 
        image_path: Optional[str] = None
    ) -> int:
        """
        Add a new face to the database
        
        Args:
            name: Person's name
            embedding: Face embedding vector
            image_path: Path to face image (optional)
            
        Returns:
            Face ID
        """
        if self.db_type == "sqlite":
            return self._add_face_sqlite(name, embedding, image_path)
        else:
            return self._add_face_faiss(name, embedding, image_path)
    
    def _add_face_sqlite(
        self, 
        name: str, 
        embedding: np.ndarray, 
        image_path: Optional[str] = None
    ) -> int:
        """Add face to SQLite database"""
        cursor = self.conn.cursor()
        
        # Serialize embedding
        embedding_blob = pickle.dumps(embedding)
        
        # Insert face
        cursor.execute('''
            INSERT INTO faces (name, embedding, image_path)
            VALUES (?, ?, ?)
        ''', (name, embedding_blob, image_path))
        
        face_id = cursor.lastrowid
        self.conn.commit()
        
        # Update metadata
        cursor.execute('''
            UPDATE metadata SET value = (
                SELECT COUNT(*) FROM faces
            ) WHERE key = 'total_faces'
        ''')
        self.conn.commit()
        
        logger.info(f"Added face '{name}' with ID {face_id}")
        return face_id
    
    def _add_face_faiss(
        self, 
        name: str, 
        embedding: np.ndarray, 
        image_path: Optional[str] = None
    ) -> int:
        """Add face to FAISS database"""
        # Normalize embedding for cosine similarity
        embedding = embedding.astype('float32')
        embedding = embedding / np.linalg.norm(embedding)
        
        # Add to index
        face_id = len(self.metadata)
        self.index.add(embedding.reshape(1, -1))
        
        # Store metadata
        self.metadata[face_id] = {
            'name': name,
            'image_path': image_path,
            'created_at': datetime.now().isoformat()
        }
        
        # Save to disk
        self._save_faiss()
        
        logger.info(f"Added face '{name}' with ID {face_id}")
        return face_id
    
    def find_similar_faces(
        self, 
        query_embedding: np.ndarray,
        threshold: float = 0.6,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find similar faces in the database
        
        Args:
            query_embedding: Query embedding vector
            threshold: Similarity threshold
            top_k: Maximum number of results
            
        Returns:
            List of similar faces with metadata
        """
        if self.db_type == "sqlite":
            return self._find_similar_sqlite(query_embedding, threshold, top_k)
        else:
            return self._find_similar_faiss(query_embedding, threshold, top_k)
    
    def _find_similar_sqlite(
        self, 
        query_embedding: np.ndarray,
        threshold: float,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Find similar faces using SQLite"""
        cursor = self.conn.cursor()
        
        # Get all faces
        cursor.execute('SELECT id, name, embedding, image_path FROM faces')
        all_faces = cursor.fetchall()
        
        similar_faces = []
        
        for face_id, name, embedding_blob, image_path in all_faces:
            # Deserialize embedding
            stored_embedding = pickle.loads(embedding_blob)
            
            # Compute cosine similarity
            similarity = np.dot(query_embedding, stored_embedding)
            
            if similarity >= threshold:
                similar_faces.append({
                    'id': face_id,
                    'name': name,
                    'similarity': float(similarity),
                    'image_path': image_path
                })
        
        # Sort by similarity and limit results
        similar_faces.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_faces[:top_k]
    
    def _find_similar_faiss(
        self, 
        query_embedding: np.ndarray,
        threshold: float,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Find similar faces using FAISS"""
        # Normalize query embedding
        query_embedding = query_embedding.astype('float32')
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search
        similarities, indices = self.index.search(
            query_embedding.reshape(1, -1), 
            min(top_k, self.index.ntotal)
        )
        
        similar_faces = []
        
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx >= 0 and similarity >= threshold:
                metadata = self.metadata.get(idx, {})
                similar_faces.append({
                    'id': idx,
                    'name': metadata.get('name', f'Unknown_{idx}'),
                    'similarity': float(similarity),
                    'image_path': metadata.get('image_path')
                })
        
        return similar_faces
    
    def get_face_by_name(self, name: str) -> List[Dict[str, Any]]:
        """
        Get all faces for a given name
        
        Args:
            name: Person's name
            
        Returns:
            List of faces with metadata
        """
        if self.db_type == "sqlite":
            cursor = self.conn.cursor()
            cursor.execute('''
                SELECT id, name, image_path, created_at 
                FROM faces WHERE name = ?
            ''', (name,))
            
            faces = []
            for row in cursor.fetchall():
                faces.append({
                    'id': row[0],
                    'name': row[1],
                    'image_path': row[2],
                    'created_at': row[3]
                })
            
            return faces
        else:
            # FAISS implementation
            faces = []
            for face_id, metadata in self.metadata.items():
                if metadata.get('name') == name:
                    faces.append({
                        'id': face_id,
                        'name': metadata.get('name'),
                        'image_path': metadata.get('image_path'),
                        'created_at': metadata.get('created_at')
                    })
            
            return faces
    
    def delete_face(self, face_id: int) -> bool:
        """
        Delete a face from the database
        
        Args:
            face_id: Face ID to delete
            
        Returns:
            True if successful, False otherwise
        """
        if self.db_type == "sqlite":
            cursor = self.conn.cursor()
            cursor.execute('DELETE FROM faces WHERE id = ?', (face_id,))
            deleted = cursor.rowcount > 0
            self.conn.commit()
            
            if deleted:
                # Update metadata
                cursor.execute('''
                    UPDATE metadata SET value = (
                        SELECT COUNT(*) FROM faces
                    ) WHERE key = 'total_faces'
                ''')
                self.conn.commit()
                logger.info(f"Deleted face with ID {face_id}")
            
            return deleted
        else:
            # FAISS implementation (more complex)
            if face_id in self.metadata:
                del self.metadata[face_id]
                # Note: FAISS doesn't support easy removal, would need to rebuild index
                logger.warning(f"FAISS removal not fully implemented for ID {face_id}")
                return True
            return False
    
    def update_face_name(self, face_id: int, new_name: str) -> bool:
        """
        Update the name of a face
        
        Args:
            face_id: Face ID
            new_name: New name
            
        Returns:
            True if successful, False otherwise
        """
        if self.db_type == "sqlite":
            cursor = self.conn.cursor()
            cursor.execute('''
                UPDATE faces SET name = ?, updated_at = CURRENT_TIMESTAMP 
                WHERE id = ?
            ''', (new_name, face_id))
            
            updated = cursor.rowcount > 0
            self.conn.commit()
            
            if updated:
                logger.info(f"Updated face ID {face_id} name to '{new_name}'")
            
            return updated
        else:
            # FAISS implementation
            if face_id in self.metadata:
                self.metadata[face_id]['name'] = new_name
                self.metadata[face_id]['updated_at'] = datetime.now().isoformat()
                self._save_faiss()
                logger.info(f"Updated face ID {face_id} name to '{new_name}'")
                return True
            return False
    
    def get_all_faces(self) -> List[Dict[str, Any]]:
        """
        Get all faces in the database
        
        Returns:
            List of all faces with metadata
        """
        if self.db_type == "sqlite":
            cursor = self.conn.cursor()
            cursor.execute('SELECT id, name, image_path, created_at FROM faces')
            
            faces = []
            for row in cursor.fetchall():
                faces.append({
                    'id': row[0],
                    'name': row[1],
                    'image_path': row[2],
                    'created_at': row[3]
                })
            
            return faces
        else:
            # FAISS implementation
            faces = []
            for face_id, metadata in self.metadata.items():
                faces.append({
                    'id': face_id,
                    'name': metadata.get('name', f'Unknown_{face_id}'),
                    'image_path': metadata.get('image_path'),
                    'created_at': metadata.get('created_at')
                })
            
            return faces
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics
        
        Returns:
            Dictionary with database statistics
        """
        if self.db_type == "sqlite":
            cursor = self.conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM faces')
            total_faces = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(DISTINCT name) FROM faces')
            unique_names = cursor.fetchone()[0]
            
            return {
                'total_faces': total_faces,
                'unique_names': unique_names,
                'database_type': 'sqlite'
            }
        else:
            total_faces = len(self.metadata)
            unique_names = len(set(meta.get('name') for meta in self.metadata.values()))
            
            return {
                'total_faces': total_faces,
                'unique_names': unique_names,
                'database_type': 'faiss',
                'index_type': FAISS_INDEX_TYPE
            }
    
    def _save_faiss(self):
        """Save FAISS index and metadata to disk"""
        if self.db_type == "faiss":
            index_path = EMBEDDINGS_PATH / "faiss.index"
            metadata_path = EMBEDDINGS_PATH / "metadata.json"
            
            faiss.write_index(self.index, str(index_path))
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2)
    
    def close(self):
        """Close database connection"""
        if self.db_type == "sqlite" and self.conn:
            self.conn.close()
            logger.info("SQLite connection closed")
        elif self.db_type == "faiss":
            self._save_faiss()
            logger.info("FAISS index saved")


def create_database(db_type: str = DATABASE_TYPE) -> FaceDatabase:
    """
    Factory function to create face database
    
    Args:
        db_type: Type of database
        
    Returns:
        FaceDatabase instance
    """
    return FaceDatabase(db_type)


if __name__ == "__main__":
    # Test the database
    db = create_database()
    
    # Test adding a face
    test_embedding = np.random.random(512)
    face_id = db.add_face("Test Person", test_embedding)
    print(f"Added face with ID: {face_id}")
    
    # Test finding similar faces
    similar = db.find_similar_faces(test_embedding, threshold=0.5)
    print(f"Found {len(similar)} similar faces")
    
    # Get stats
    stats = db.get_stats()
    print(f"Database stats: {stats}")
    
    db.close()
