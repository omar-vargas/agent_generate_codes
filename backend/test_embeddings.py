#!/usr/bin/env python3
"""
Test script for embedding functionality
"""
import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import generate_embedding, compute_similarity, find_similar_codes
from models import get_db, Code

def test_embedding_generation():
    """Test embedding generation"""
    print("Testing embedding generation...")

    test_text = "percepcion de los estudiantes sobre la IA generativa"
    embedding = generate_embedding(test_text)

    if embedding:
        print(f"[OK] Embedding generated successfully: {len(embedding)} dimensions")
        print(f"First 5 values: {embedding[:5]}")
        return embedding
    else:
        print("[FAIL] Failed to generate embedding - this may be due to Azure deployment not being available")
        # Return mock embedding for testing similarity functions
        import numpy as np
        mock_embedding = np.random.rand(1536).tolist()  # Standard embedding size
        print(f"[MOCK] Using mock embedding for testing: {len(mock_embedding)} dimensions")
        return mock_embedding

def test_similarity_computation():
    """Test similarity computation"""
    print("\nTesting similarity computation...")

    # Generate embeddings for two similar texts
    text1 = "percepcion de los estudiantes sobre la IA"
    text2 = "opinion de los estudiantes acerca de la inteligencia artificial"

    embedding1 = generate_embedding(text1)
    embedding2 = generate_embedding(text2)

    if embedding1 and embedding2:
        similarity = compute_similarity(embedding1, embedding2)
        print(".3f")
        return similarity
    else:
        print("[MOCK] Using mock embeddings for similarity test")
        # Create mock embeddings - similar ones should have high similarity
        import numpy as np
        base_embedding = np.random.rand(1536)
        embedding1 = base_embedding.tolist()
        embedding2 = (base_embedding + np.random.rand(1536) * 0.1).tolist()  # Add small noise

        similarity = compute_similarity(embedding1, embedding2)
        print(".3f")
        return similarity

def test_database_operations():
    """Test database operations with embeddings"""
    print("\nTesting database operations...")

    db = get_db()
    try:
        # Check existing codes
        codes = db.query(Code).all()
        print(f"Found {len(codes)} codes in database")

        # Count codes with embeddings
        codes_with_embeddings = [c for c in codes if c.embedding is not None]
        print(f"Codes with embeddings: {len(codes_with_embeddings)}")

        # Test finding similar codes if we have embeddings
        if codes_with_embeddings:
            target_code = codes_with_embeddings[0]
            target_embedding = json.loads(target_code.embedding)
            similar_codes = find_similar_codes(target_embedding, target_code.user_id, limit=5)
            print(f"Found {len(similar_codes)} similar codes for '{target_code.name}'")

    finally:
        db.close()

if __name__ == "__main__":
    print("=== Embedding Functionality Test ===\n")

    # Test embedding generation
    embedding = test_embedding_generation()

    # Test similarity computation
    similarity = test_similarity_computation()

    # Test database operations
    test_database_operations()

    print("\n=== Test Complete ===")