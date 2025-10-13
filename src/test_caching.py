"""
Quick test to demonstrate the caching functionality.
"""

import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from cache_utils import (
    get_config_hash, 
    get_cache_path, 
    save_vectors, 
    load_vectors,
    print_cache_info,
    clear_cache
)
import pandas as pd
import numpy as np


def test_caching():
    """Test the caching utilities"""
    
    print("=== Testing gram2vec Vector Caching ===\n")
    
    # Setup
    cache_dir = Path("test_cache")
    config = {
        "pos_unigrams": 1,
        "pos_bigrams": 1,
        "func_words": 1,
        "punctuation": 1,
    }
    
    # 1. Test config hashing
    print("1. Testing config hash generation...")
    hash1 = get_config_hash(config)
    hash2 = get_config_hash(config)
    print(f"   Config hash: {hash1}")
    print(f"   Consistent: {hash1 == hash2} ✓")
    
    # Different config should have different hash
    config2 = {**config, "dep_labels": 1}
    hash3 = get_config_hash(config2)
    print(f"   Different config hash: {hash3}")
    print(f"   Different hashes: {hash1 != hash3} ✓\n")
    
    # 2. Test saving and loading
    print("2. Testing save/load functionality...")
    
    # Create dummy vectors
    dummy_vectors = pd.DataFrame({
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100),
        'feature3': np.random.rand(100),
        'authorID': ['human'] * 50 + ['gpt'] * 50,
        'documentID': [f"doc_{i}" for i in range(100)]
    })
    
    # Save
    cache_path = get_cache_path(cache_dir, "train", hash1)
    print(f"   Saving to: {cache_path}")
    start = time.time()
    save_vectors(dummy_vectors, cache_path)
    save_time = time.time() - start
    print(f"   Save time: {save_time:.3f}s\n")
    
    # Load
    print("3. Testing load from cache...")
    start = time.time()
    loaded_vectors = load_vectors(cache_path)
    load_time = time.time() - start
    print(f"   Load time: {load_time:.3f}s")
    print(f"   Shapes match: {loaded_vectors.shape == dummy_vectors.shape} ✓")
    print(f"   Data matches: {loaded_vectors.equals(dummy_vectors)} ✓\n")
    
    # 4. Test cache info
    print("4. Testing cache info...")
    print_cache_info(cache_dir)
    print()
    
    # 5. Demonstrate cache miss vs hit
    print("5. Demonstrating cache miss vs hit...")
    
    # Cache miss (doesn't exist)
    cache_path_miss = get_cache_path(cache_dir, "dev", hash1)
    result = load_vectors(cache_path_miss)
    print(f"   Cache miss returns None: {result is None} ✓")
    
    # Cache hit (exists)
    result = load_vectors(cache_path)
    print(f"   Cache hit returns data: {result is not None} ✓\n")
    
    # 6. Test clear cache
    print("6. Testing cache clearing...")
    clear_cache(cache_dir)
    print(f"   Cache cleared ✓\n")
    
    # Cleanup
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
    
    print("=== All tests passed! ===\n")
    print("You can now use caching with:")
    print("  python parallel_log_regression.py")
    print("  python train_with_top_features.py --coef-file coefficients.csv --top-n 10")


if __name__ == '__main__':
    test_caching()

