"""
Utilities for caching gram2vec feature vectors to avoid re-extracting features.
"""

import pandas as pd
import hashlib
import json
from pathlib import Path
from typing import Optional, Dict


def get_config_hash(config: Dict, include_items: Optional[list] = None) -> str:
    """
    Generate a hash from a configuration dictionary.
    
    Args:
        config: Configuration dictionary
        include_items: Optional list of additional items to include in hash
        
    Returns:
        Hexadecimal hash string
    """
    # Sort config to ensure consistent hashing
    config_str = json.dumps(config, sort_keys=True)
    
    if include_items:
        for item in include_items:
            config_str += str(item)
    
    return hashlib.md5(config_str.encode()).hexdigest()[:12]


def get_cache_path(cache_dir: Path, split_name: str, config_hash: str) -> Path:
    """
    Get the cache file path for a specific dataset split.
    
    Args:
        cache_dir: Directory for cache files
        split_name: Name of the split (e.g., 'train', 'dev', 'test')
        config_hash: Hash of the configuration
        
    Returns:
        Path to cache file
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"vectors_{split_name}_{config_hash}.parquet"


def save_vectors(df: pd.DataFrame, cache_path: Path) -> None:
    """
    Save feature vectors to a parquet file.
    
    Args:
        df: DataFrame containing feature vectors
        cache_path: Path to save the cache file
    """
    df.to_parquet(cache_path, index=True)
    print(f"  ✓ Cached vectors to {cache_path.name}")


def load_vectors(cache_path: Path) -> Optional[pd.DataFrame]:
    """
    Load feature vectors from a parquet cache file.
    
    Args:
        cache_path: Path to the cache file
        
    Returns:
        DataFrame with feature vectors, or None if cache doesn't exist
    """
    if cache_path.exists():
        print(f"  ✓ Loading cached vectors from {cache_path.name}")
        return pd.read_parquet(cache_path)
    return None


def cache_exists(cache_dir: Path, split_name: str, config_hash: str) -> bool:
    """
    Check if a cache file exists for the given split and config.
    
    Args:
        cache_dir: Directory for cache files
        split_name: Name of the split
        config_hash: Hash of the configuration
        
    Returns:
        True if cache exists, False otherwise
    """
    cache_path = get_cache_path(cache_dir, split_name, config_hash)
    return cache_path.exists()


def clear_cache(cache_dir: Path, config_hash: Optional[str] = None) -> None:
    """
    Clear cached vectors.
    
    Args:
        cache_dir: Directory containing cache files
        config_hash: If provided, only clear caches for this config hash.
                     If None, clear all caches.
    """
    if not cache_dir.exists():
        print("No cache directory found.")
        return
    
    if config_hash:
        pattern = f"vectors_*_{config_hash}.parquet"
    else:
        pattern = "vectors_*.parquet"
    
    cache_files = list(cache_dir.glob(pattern))
    
    if not cache_files:
        print(f"No cache files found matching pattern: {pattern}")
        return
    
    print(f"Clearing {len(cache_files)} cache file(s)...")
    for cache_file in cache_files:
        cache_file.unlink()
        print(f"  ✓ Deleted {cache_file.name}")
    
    print("Cache cleared successfully.")


def get_or_extract_vectors(
    df: pd.DataFrame,
    split_name: str,
    vectorizer,
    config: Dict,
    cache_dir: Path,
    use_cache: bool = True
) -> pd.DataFrame:
    """
    Get vectors from cache or extract them if cache doesn't exist.
    
    Args:
        df: DataFrame with text, doc_id, and author_id columns
        split_name: Name of the split ('train', 'dev', 'test')
        vectorizer: Gram2VecVectorizer instance
        config: Feature configuration dictionary
        cache_dir: Directory for cache files
        use_cache: Whether to use caching (default: True)
        
    Returns:
        DataFrame with feature vectors
    """
    if not use_cache:
        print(f"\n=== Vectorizing {split_name} data (cache disabled) ===")
        return vectorizer.from_documents(
            df["text"].tolist(),
            author_ids=df["author_id"].tolist(),
            document_ids=df["doc_id"].tolist()
        )
    
    # Generate cache hash based on config
    config_hash = get_config_hash(config)
    cache_path = get_cache_path(cache_dir, split_name, config_hash)
    
    # Try to load from cache
    print(f"\n=== {split_name.capitalize()} data ===")
    vectors = load_vectors(cache_path)
    
    if vectors is not None:
        return vectors
    
    # Cache miss - extract features
    print(f"  Cache miss - extracting features...")
    vectors = vectorizer.from_documents(
        df["text"].tolist(),
        author_ids=df["author_id"].tolist(),
        document_ids=df["doc_id"].tolist()
    )
    
    # Save to cache
    save_vectors(vectors, cache_path)
    
    return vectors


def print_cache_info(cache_dir: Path) -> None:
    """
    Print information about cached files.
    
    Args:
        cache_dir: Directory containing cache files
    """
    if not cache_dir.exists():
        print("No cache directory found.")
        return
    
    cache_files = list(cache_dir.glob("vectors_*.parquet"))
    
    if not cache_files:
        print("No cached vectors found.")
        return
    
    print(f"\n=== Cache Information ===")
    print(f"Cache directory: {cache_dir}")
    print(f"Cached files: {len(cache_files)}")
    
    # Group by config hash
    by_config = {}
    for cache_file in cache_files:
        # Extract config hash from filename: vectors_{split}_{hash}.parquet
        parts = cache_file.stem.split('_')
        if len(parts) >= 3:
            config_hash = parts[-1]
            if config_hash not in by_config:
                by_config[config_hash] = []
            by_config[config_hash].append(cache_file)
    
    for config_hash, files in by_config.items():
        print(f"\n  Config hash: {config_hash}")
        total_size = sum(f.stat().st_size for f in files)
        print(f"    Files: {len(files)}")
        print(f"    Total size: {total_size / 1024 / 1024:.2f} MB")
        for f in files:
            size_mb = f.stat().st_size / 1024 / 1024
            print(f"      - {f.name} ({size_mb:.2f} MB)")

