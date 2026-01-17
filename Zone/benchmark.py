"""
Performance Benchmarking Script for Face Recognition System
"""
import time
import numpy as np
from PIL import Image
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from recognizer import create_recognizer


def create_test_images(num_images: int = 10, image_size: tuple = (160, 160)):
    """Create test images for benchmarking"""
    images = []
    for i in range(num_images):
        # Create random image with some face-like features
        img_array = np.random.randint(0, 255, (*image_size, 3), dtype=np.uint8)
        
        # Add some structure to make it more realistic
        # Add eye-like regions
        img_array[50:70, 40:60] = [200, 200, 200]  # Left eye
        img_array[50:70, 100:120] = [200, 200, 200]  # Right eye
        
        # Add mouth-like region
        img_array[100:120, 60:100] = [150, 50, 50]
        
        images.append(Image.fromarray(img_array))
    
    return images


def benchmark_registration(recognizer, test_images):
    """Benchmark face registration performance"""
    print("Benchmarking Face Registration...")
    
    times = []
    successful_registrations = 0
    
    for i, image in enumerate(test_images):
        start_time = time.time()
        
        result = recognizer.register_face(f"Person_{i}", image, save_image=False)
        
        end_time = time.time()
        processing_time = end_time - start_time
        times.append(processing_time)
        
        if result["success"]:
            successful_registrations += 1
        
        print(f"  Image {i+1}: {processing_time:.3f}s - {'Success' if result['success'] else 'Failed'}")
    
    avg_time = np.mean(times)
    success_rate = successful_registrations / len(test_images)
    
    print(f"\nRegistration Results:")
    print(f"  Average time: {avg_time:.3f}s")
    print(f"  Success rate: {success_rate:.1%}")
    print(f"  Throughput: {1/avg_time:.1f} faces/second")
    
    return avg_time, success_rate


def benchmark_recognition(recognizer, test_images):
    """Benchmark face recognition performance"""
    print("\nBenchmarking Face Recognition...")
    
    times = []
    faces_found = 0
    total_faces_detected = 0
    
    for i, image in enumerate(test_images):
        start_time = time.time()
        
        result = recognizer.recognize_faces(image)
        
        end_time = time.time()
        processing_time = end_time - start_time
        times.append(processing_time)
        
        if result["success"]:
            faces_found += 1
            total_faces_detected += result["faces_found"]
        
        print(f"  Image {i+1}: {processing_time:.3f}s - {result['faces_found']} faces")
    
    avg_time = np.mean(times)
    detection_rate = faces_found / len(test_images)
    avg_faces_per_image = total_faces_detected / len(test_images)
    
    print(f"\nRecognition Results:")
    print(f"  Average time: {avg_time:.3f}s")
    print(f"  Detection rate: {detection_rate:.1%}")
    print(f"  Average faces per image: {avg_faces_per_image:.1f}")
    print(f"  Throughput: {1/avg_time:.1f} images/second")
    
    return avg_time, detection_rate


def benchmark_similarity_search(recognizer, num_queries: int = 100):
    """Benchmark similarity search performance"""
    print("\nBenchmarking Similarity Search...")
    
    # Generate random embeddings for testing
    embedding_dim = recognizer.encoder.get_embedding_dimension()
    query_embeddings = [
        np.random.random(embedding_dim) for _ in range(num_queries)
    ]
    
    times = []
    
    for i, embedding in enumerate(query_embeddings):
        start_time = time.time()
        
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)
        
        # Search for similar faces
        similar_faces = recognizer.database.find_similar_faces(
            embedding, threshold=0.5, top_k=5
        )
        
        end_time = time.time()
        processing_time = end_time - start_time
        times.append(processing_time)
        
        if i % 20 == 0:
            print(f"  Query {i+1}: {processing_time:.4f}s - {len(similar_faces)} matches")
    
    avg_time = np.mean(times)
    
    print(f"\nSimilarity Search Results:")
    print(f"  Average time: {avg_time:.4f}s")
    print(f"  Throughput: {1/avg_time:.0f} queries/second")
    
    return avg_time


def run_full_benchmark():
    """Run complete performance benchmark"""
    print("=" * 60)
    print("FACE RECOGNITION SYSTEM PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # Create recognizer
    recognizer = create_recognizer(
        detector_model="mtcnn",
        encoder_model="facenet",
        db_type="sqlite",
        threshold=0.6
    )
    
    try:
        # Get initial database stats
        initial_stats = recognizer.get_database_stats()
        print(f"\nInitial Database Stats: {initial_stats}")
        
        # Create test images
        print(f"\nGenerating test images...")
        test_images = create_test_images(20, (160, 160))
        print(f"Created {len(test_images)} test images")
        
        # Benchmark registration
        reg_time, reg_success = benchmark_registration(recognizer, test_images[:10])
        
        # Benchmark recognition
        rec_time, rec_success = benchmark_recognition(recognizer, test_images)
        
        # Benchmark similarity search
        search_time = benchmark_similarity_search(recognizer, 50)
        
        # Final database stats
        final_stats = recognizer.get_database_stats()
        print(f"\nFinal Database Stats: {final_stats}")
        
        # Summary
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        print(f"Registration Performance:")
        print(f"  Average time: {reg_time:.3f}s per face")
        print(f"  Success rate: {reg_success:.1%}")
        print(f"  Throughput: {1/reg_time:.1f} faces/second")
        
        print(f"\nRecognition Performance:")
        print(f"  Average time: {rec_time:.3f}s per image")
        print(f"  Detection rate: {rec_success:.1%}")
        print(f"  Throughput: {1/rec_time:.1f} images/second")
        
        print(f"\nSimilarity Search Performance:")
        print(f"  Average time: {search_time:.4f}s per query")
        print(f"  Throughput: {1/search_time:.0f} queries/second")
        
        print(f"\nDatabase Growth:")
        print(f"  Initial faces: {initial_stats['total_faces']}")
        print(f"  Final faces: {final_stats['total_faces']}")
        print(f"  Added during benchmark: {final_stats['total_faces'] - initial_stats['total_faces']}")
        
        # Performance recommendations
        print(f"\nPERFORMANCE RECOMMENDATIONS:")
        if reg_time > 2.0:
            print("  WARNING: Registration time is high. Consider GPU acceleration.")
        if rec_time > 1.0:
            print("  WARNING: Recognition time is high. Consider optimizing detection.")
        if search_time > 0.01:
            print("  WARNING: Search time is high. Consider FAISS for large databases.")
        
        print("\nBenchmark completed successfully!")
        
    finally:
        recognizer.close()


if __name__ == "__main__":
    run_full_benchmark()
