"""
Unit tests for Database Module
"""
import pytest
import numpy as np
import tempfile
import shutil
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / "src"))

from database import FaceDatabase, create_database


class TestFaceDatabase:
    """Test cases for FaceDatabase class"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing"""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sqlite_db(self, temp_dir):
        """Create a SQLite database for testing"""
        # Override config paths for testing
        import config.config
        original_db_path = config.config.DATABASE_PATH
        config.config.DATABASE_PATH = temp_dir / "test.db"
        
        db = FaceDatabase(db_type="sqlite")
        
        yield db
        
        db.close()
        config.config.DATABASE_PATH = original_db_path
    
    def test_sqlite_initialization(self, temp_dir):
        """Test SQLite database initialization"""
        # Override config path
        import config.config
        original_db_path = config.config.DATABASE_PATH
        config.config.DATABASE_PATH = temp_dir / "test.db"
        
        db = FaceDatabase(db_type="sqlite")
        assert db.db_type == "sqlite"
        assert db.conn is not None
        
        db.close()
        config.config.DATABASE_PATH = original_db_path
    
    def test_add_face(self, sqlite_db):
        """Test adding a face to database"""
        embedding = np.random.random(512)
        face_id = sqlite_db.add_face("Test Person", embedding)
        
        assert isinstance(face_id, int)
        assert face_id > 0
    
    def test_find_similar_faces(self, sqlite_db):
        """Test finding similar faces"""
        # Add some test faces
        embedding1 = np.random.random(512)
        embedding2 = np.random.random(512)
        embedding3 = np.random.random(512)
        
        sqlite_db.add_face("Person 1", embedding1)
        sqlite_db.add_face("Person 2", embedding2)
        sqlite_db.add_face("Person 3", embedding3)
        
        # Find similar faces
        similar = sqlite_db.find_similar_faces(embedding1, threshold=0.5)
        assert isinstance(similar, list)
        assert len(similar) >= 1  # Should find at least the original
    
    def test_get_face_by_name(self, sqlite_db):
        """Test getting faces by name"""
        embedding = np.random.random(512)
        sqlite_db.add_face("John Doe", embedding)
        
        faces = sqlite_db.get_face_by_name("John Doe")
        assert isinstance(faces, list)
        assert len(faces) >= 1
        assert faces[0]['name'] == "John Doe"
    
    def test_update_face_name(self, sqlite_db):
        """Test updating face name"""
        embedding = np.random.random(512)
        face_id = sqlite_db.add_face("Original Name", embedding)
        
        updated = sqlite_db.update_face_name(face_id, "New Name")
        assert updated is True
        
        faces = sqlite_db.get_face_by_name("New Name")
        assert len(faces) >= 1
    
    def test_delete_face(self, sqlite_db):
        """Test deleting a face"""
        embedding = np.random.random(512)
        face_id = sqlite_db.add_face("To Delete", embedding)
        
        deleted = sqlite_db.delete_face(face_id)
        assert deleted is True
        
        faces = sqlite_db.get_face_by_name("To Delete")
        assert len(faces) == 0
    
    def test_get_all_faces(self, sqlite_db):
        """Test getting all faces"""
        # Add some test faces
        for i in range(3):
            embedding = np.random.random(512)
            sqlite_db.add_face(f"Person {i}", embedding)
        
        all_faces = sqlite_db.get_all_faces()
        assert isinstance(all_faces, list)
        assert len(all_faces) >= 3
    
    def test_get_stats(self, sqlite_db):
        """Test getting database statistics"""
        # Add some test faces
        for i in range(3):
            embedding = np.random.random(512)
            sqlite_db.add_face(f"Person {i}", embedding)
        
        stats = sqlite_db.get_stats()
        assert isinstance(stats, dict)
        assert 'total_faces' in stats
        assert 'unique_names' in stats
        assert 'database_type' in stats
        assert stats['total_faces'] >= 3
        assert stats['database_type'] == 'sqlite'
    
    def test_create_database_factory(self, temp_dir):
        """Test factory function"""
        # Override config path
        import config.config
        original_db_path = config.config.DATABASE_PATH
        config.config.DATABASE_PATH = temp_dir / "test.db"
        
        db = create_database(db_type="sqlite")
        assert isinstance(db, FaceDatabase)
        assert db.db_type == "sqlite"
        
        db.close()
        config.config.DATABASE_PATH = original_db_path


if __name__ == "__main__":
    pytest.main([__file__])
