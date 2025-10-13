# Caching Guide for gram2vec Feature Vectors

This guide explains how to use the caching functionality to speed up training by avoiding re-extraction of features.

## Overview

Feature extraction with gram2vec can be time-consuming, especially on large datasets. The caching system saves extracted feature vectors to disk so subsequent runs can load them instantly.

## How It Works

1. **Cache Key**: A hash is generated from your feature configuration (which features are enabled)
2. **Cache Files**: Feature vectors for each split (train/dev/test) are saved as `.parquet` files
3. **Cache Hit**: If vectors exist for your config, they're loaded from cache
4. **Cache Miss**: If not found, features are extracted and then cached for next time

## Basic Usage

### parallel_log_regression.py

```bash
# First run - extracts features and caches them
python parallel_log_regression.py

# Second run - loads from cache (much faster!)
python parallel_log_regression.py

# Force re-extraction (bypass cache)
python parallel_log_regression.py --no-cache

# Use a different cache directory
python parallel_log_regression.py --cache-dir my_cache
```

### train_with_top_features.py

```bash
# First run with top 10 features - extracts and caches
python train_with_top_features.py --coef-file coefficients.csv --top-n 10

# Second run - loads from cache
python train_with_top_features.py --coef-file coefficients.csv --top-n 10

# Try different feature counts - uses same cache (filters after loading)
python train_with_top_features.py --coef-file coefficients.csv --top-n 20
python train_with_top_features.py --coef-file coefficients.csv --top-n 50
```

## Cache Management

### View Cache Information

```bash
# Check what's cached
python parallel_log_regression.py --cache-info

# Or
python train_with_top_features.py --coef-file coefficients.csv --cache-info
```

Example output:
```
=== Cache Information ===
Cache directory: vector_cache
Cached files: 3

  Config hash: a1b2c3d4e5f6
    Files: 3
    Total size: 45.23 MB
      - vectors_train_a1b2c3d4e5f6.parquet (23.45 MB)
      - vectors_dev_a1b2c3d4e5f6.parquet (11.23 MB)
      - vectors_test_a1b2c3d4e5f6.parquet (10.55 MB)
```

### Clear Cache

```bash
# Clear all cached vectors
python parallel_log_regression.py --clear-cache

# Or
python train_with_top_features.py --coef-file coefficients.csv --clear-cache
```

### Manual Cache Management

```bash
# Cache is stored in the specified directory (default: vector_cache/)
ls vector_cache/

# Remove specific cache files
rm vector_cache/vectors_train_*.parquet

# Remove entire cache directory
rm -rf vector_cache/
```

## When Cache is Invalidated

The cache is automatically invalidated (new extraction needed) when:

1. **Feature configuration changes**: Different features enabled/disabled
2. **Cache doesn't exist**: First run or after clearing cache
3. **Forced re-extraction**: Using `--no-cache` flag

The cache is **NOT** invalidated when:

- Different `--top-n` values (in `train_with_top_features.py`)
- Different output filenames
- Different model hyperparameters

## Best Practices

### 1. Use Cache by Default
```bash
# Normal workflow - let caching work automatically
python parallel_log_regression.py
```

### 2. Cache Different Configs Separately
```bash
# Each config gets its own cache automatically
python parallel_log_regression.py  # Full features
python parallel_log_regression.py --no-cache  # If you modify the config in the script
```

### 3. Clear Cache When Needed
```bash
# Clear cache if:
# - Data changes
# - gram2vec code changes
# - Storage is running low
python parallel_log_regression.py --clear-cache
```

### 4. Check Cache Before Long Runs
```bash
# See what's already cached
python parallel_log_regression.py --cache-info

# This helps you know if extraction will be needed
```

## Example Workflow

```bash
# 1. First training run (extracts and caches features)
python parallel_log_regression.py -o coefficients_v1.csv
# Takes ~5-10 minutes

# 2. Try different model settings (uses cache)
python train_with_top_features.py --coef-file coefficients_v1.csv --top-n 10
# Takes ~1 minute (cached!)

python train_with_top_features.py --coef-file coefficients_v1.csv --top-n 50
# Takes ~1 minute (cached!)

# 3. Check cache status
python parallel_log_regression.py --cache-info

# 4. Clean up when done
python parallel_log_regression.py --clear-cache
```

## Cache File Format

- **Format**: Apache Parquet (compressed, efficient)
- **Naming**: `vectors_{split}_{config_hash}.parquet`
- **Location**: `vector_cache/` (default) or custom via `--cache-dir`
- **Size**: Varies by dataset, typically 10-50 MB per split

## Troubleshooting

### Cache Not Working?

1. Check if cache directory exists and is writable
2. Verify config hasn't changed between runs
3. Try `--cache-info` to see what's cached

### Cache Taking Too Much Space?

```bash
# Check cache size
du -sh vector_cache/

# Clear old caches
python parallel_log_regression.py --clear-cache
```

### Want Fresh Features?

```bash
# Force re-extraction
python parallel_log_regression.py --no-cache
```

## Performance Impact

With caching enabled:

- **First run**: Full feature extraction time + ~5 seconds to save cache
- **Subsequent runs**: ~1-5 seconds to load from cache (depending on dataset size)
- **Speedup**: Typically 10-50x faster for subsequent runs

Example:
- Without cache: 10 minutes per run
- With cache (first): 10 minutes + 5 seconds
- With cache (subsequent): 15 seconds

**Total time saved**: ~9.75 minutes per run after the first one!

