import pandas as pd
import polars as pl
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
import os
import sys
from tqdm import tqdm
import argparse

# Add the src directory to the path to allow imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import vectorizer
from vectorizer import Gram2VecVectorizer
from elfen.extractor import Extractor
from utils.cache_utils import get_or_extract_vectors, print_cache_info, clear_cache, get_config_hash
from utils.log_utils import generate_log_filename, start_logging, stop_logging, list_logs

config = {
    
}

def select_top_features(coef_file: str, n_features: int = 10):
    """
    Load coefficients and return the top N feature names.
    
    Args:
        coef_file: Path to coefficients.csv from a previous training run
        n_features: Number of top features to select
        
    Returns:
        List of feature names to keep
    """
    coef_df = pd.read_csv(coef_file)
    
    # Get absolute coefficient values
    coef_df['abs_coef'] = coef_df['human'].abs()
    
    # Sort by absolute coefficient and get top N
    top_features = coef_df.nlargest(n_features, 'abs_coef')
    
    print(f"\n=== Top {n_features} Features ===")
    for idx, row in top_features.iterrows():
        print(f"  {row['Feature']:40s} | coef: {row['human']:7.4f}")
    
    return top_features['Feature'].tolist()


def filter_feature_columns(df: pd.DataFrame, feature_names: list, keep_meta_cols: bool = True):
    """
    Filter a feature dataframe to only include specified feature columns.
    
    Args:
        df: Feature dataframe from gram2vec
        feature_names: List of feature names to keep
        keep_meta_cols: Whether to keep metadata columns (authorID, documentID)
        
    Returns:
        Filtered dataframe
    """
    cols_to_keep = []
    
    if keep_meta_cols:
        # Keep metadata columns if they exist
        meta_cols = ['authorID', 'documentID']
        for col in meta_cols:
            if col in df.columns:
                cols_to_keep.append(col)
    
    # Add the specific feature columns
    for feat in feature_names:
        if feat in df.columns:
            cols_to_keep.append(feat)
        else:
            print(f"Warning: Feature '{feat}' not found in dataframe columns")
    
    print(f"\nFiltered from {len(df.columns)} to {len(cols_to_keep)} columns")
    
    return df[cols_to_keep]


def extract_elfen_features(df, text_col="text", doc_id_col="doc_id", author_id_col="author_id"):
    """Extract elfen features from a dataframe"""
    print(f"Extracting elfen features from {len(df)} documents...")
    
    # Convert to polars if needed
    if isinstance(df, pd.DataFrame):
        pl_df = pl.from_pandas(df[[text_col, doc_id_col, author_id_col]])
    else:
        pl_df = df.select([text_col, doc_id_col, author_id_col])
    
    # Rename columns to what Extractor expects
    pl_df = pl_df.rename({text_col: "text", doc_id_col: "documentID", author_id_col: "authorID"})
    
    # Initialize extractor and extract features
    extractor = Extractor(data=pl_df)
    extractor.extract_features()
    
    # Convert back to pandas for consistency with gram2vec pipeline
    result_df = extractor.data.to_pandas()
    
    return result_df

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train logistic regression classifier with gram2vec or elfen features')
    parser.add_argument('--features', '-f', choices=['gram2vec', 'elfen'], default='gram2vec',
                        help='Feature extraction method to use (default: gram2vec)')
    parser.add_argument('--output', '-o', default='coefficients.csv',
                        help='Output CSV file for coefficients (default: coefficients.csv)')
    parser.add_argument('--coef-file', '-c', default=None,
                        help='Path to coefficients.csv from a previous training run (enables feature selection)')
    parser.add_argument('--top-n', '-n', type=int, default=10,
                        help='Number of top features to use when --coef-file is specified (default: 10)')
    parser.add_argument('--cache-dir', default='vector_cache',
                        help='Directory for caching feature vectors (default: vector_cache)')
    parser.add_argument('--no-cache', action='store_true',
                        help='Disable caching and re-extract all features')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Clear cached vectors and exit')
    parser.add_argument('--cache-info', action='store_true',
                        help='Show cache information and exit')
    parser.add_argument('--log-dir', default='logs',
                        help='Directory for log files (default: logs)')
    parser.add_argument('--no-log', action='store_true',
                        help='Disable logging to file')
    parser.add_argument('--list-logs', action='store_true',
                        help='List recent log files and exit')
    args = parser.parse_args()
    
    # Handle log listing
    if args.list_logs:
        list_logs(args.log_dir)
        exit(0)
    
    # Handle cache management commands
    cache_dir = Path(args.cache_dir)
    
    if args.cache_info:
        print_cache_info(cache_dir)
        exit(0)
    
    if args.clear_cache:
        clear_cache(cache_dir)
        exit(0)
    
    # Setup logging
    logger = None
    log_file = None
    if not args.no_log:
        output_basename = Path(args.output).stem
        log_kwargs = {
            "features": args.features,
            "output": output_basename
        }
        # Add coef and top_n to log filename if feature selection is enabled
        if args.coef_file:
            coef_basename = Path(args.coef_file).stem
            log_kwargs["coef"] = coef_basename
            log_kwargs["top_n"] = args.top_n
        
        log_file = generate_log_filename(
            "full_training",
            log_dir=args.log_dir,
            **log_kwargs
        )
        logger = start_logging(log_file)
    
    try:
        print(f"\n=== Using {args.features} features ===\n")
        
        # Load and combine parquet files
        dfs = []
        data_dir = Path(__file__).parent.parent / "data"
        corpus_files = list(data_dir.glob("*.parquet"))
        for corpus_file in tqdm(corpus_files, desc="Loading corpus files"):
            curr_df = pd.read_parquet(corpus_file)
            dfs.append(curr_df)
        print("Concatenating dataframes...")
        df = pd.concat(dfs, ignore_index=True)

        # Extract author_id
        def get_author_id(doc_id: str) ->str:
            if "chunk" in doc_id:
                return "human"
            elif "gpt" in doc_id:
                return "gpt"
            else:
                return "unknown"
            
        df["author_id"] = df["doc_id"].apply(get_author_id)

        # Train/Dev/Test split (60-20-20)
        train_df, temp_df = train_test_split(
            df, test_size=0.4, stratify=df["author_id"], random_state=42
        )
        dev_df, test_df = train_test_split(
            temp_df, test_size=0.5, stratify=temp_df["author_id"], random_state=42
        )
        print(f"Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")

        # Extract features based on selected method
        if args.features == 'gram2vec':
            default_config = {
                "pos_unigrams": 1,
                "pos_bigrams": 1,
                "dep_labels": 1,
                "morph_tags": 1,
                "sentences": 1,
                "emojis": 1,
                "func_words": 1,
                "punctuation": 1,
                "letters": 0,
                "transitions": 1,
                "unique_transitions": 1,
                "tokens": 1,
                "num_tokens": 1,
                "types": 1,
                "sentence_count": 1,
                "named_entities": 1,
                "suasive_verbs": 1,
                "stative_verbs": 1,
            }
            
            # Show cache status
            use_cache = not args.no_cache
            if use_cache:
                config_hash = get_config_hash(default_config)
                print(f"Cache enabled - config hash: {config_hash}")
                print(f"Cache directory: {cache_dir}")
            else:
                print("Cache disabled - will re-extract all features")
            
            # Vectorize texts
            # Limit n_process to avoid "too many open files" error
            vectorizer = Gram2VecVectorizer(language="en", normalize=True, n_process=4, enabled_features=default_config)

            # Use cached vectors or extract if needed
            train_vectors = get_or_extract_vectors(
                train_df, "train", vectorizer, default_config, cache_dir, use_cache
            )
            dev_vectors = get_or_extract_vectors(
                dev_df, "dev", vectorizer, default_config, cache_dir, use_cache
            )
            test_vectors = get_or_extract_vectors(
                test_df, "test", vectorizer, default_config, cache_dir, use_cache
            )
        
        elif args.features == 'elfen':
            print("\n=== Extracting elfen features from training data ===")
            train_vectors = extract_elfen_features(train_df)
            print("\n=== Extracting elfen features from dev data ===")
            dev_vectors = extract_elfen_features(dev_df)
            print("\n=== Extracting elfen features from test data ===")
            test_vectors = extract_elfen_features(test_df)

        # Apply feature selection if coef_file is provided
        if args.coef_file:
            print(f"\n=== Feature Selection Enabled ===")
            top_features = select_top_features(args.coef_file, args.top_n)
            
            print(f"\n=== Filtering to top {args.top_n} features ===")
            original_cols = train_vectors.shape[1] - 2  # Exclude authorID and documentID
            train_vectors = filter_feature_columns(train_vectors, top_features)
            dev_vectors = filter_feature_columns(dev_vectors, top_features)
            test_vectors = filter_feature_columns(test_vectors, top_features)
            final_cols = train_vectors.shape[1] - 2
            
            print(f"Training with {final_cols} features (reduced from {original_cols})")
            print(f"Feature names: {[col for col in train_vectors.columns if col not in ['authorID', 'documentID']]}")

        # Drop author_id to identify
        X_train, y_train = train_vectors.drop(columns=["authorID", "documentID"]), train_vectors["authorID"]
        X_dev, y_dev   = dev_vectors.drop(columns=["authorID", "documentID"]),   dev_vectors["authorID"]
        X_test, y_test  = test_vectors.drop(columns=["authorID", "documentID"]),  test_vectors["authorID"]

        # Scale features
        print("Scaling features...")
        scaler = StandardScaler(with_mean=False)  
        X_train_scaled = scaler.fit_transform(X_train)
        X_dev_scaled   = scaler.transform(X_dev)
        X_test_scaled  = scaler.transform(X_test)

        # Train logistic regression
        print("Training logistic regression model...")
        authorship_model = LogisticRegression(
            max_iter=1000,
            solver="saga",  # good for large sparse data
            n_jobs=-1,
            verbose=0  # Show training progress
        )
        authorship_model.fit(X_train_scaled, y_train)

        # Evaluate
        print("Evaluating model on test set...")
        y_pred_test = authorship_model.predict(X_test_scaled)

        # Print metrics
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred_test))

        coefs = np.squeeze(authorship_model.coef_)
        coefficients = pd.DataFrame({
            'Feature': X_train.columns,
            'human': coefs,
            'gpt': -coefs
        })
        # Rank by absolute coefficient value (most important features first)
        coefficients['abs_coef'] = np.abs(coefs)
        coefficients = coefficients.sort_values('abs_coef', ascending=False)
        coefficients = coefficients.drop(columns=['abs_coef'])
        coefficients.to_csv(args.output, index=False)
        print(f"\nCoefficients saved to {args.output}")
    
    finally:
        # Stop logging
        if logger and log_file:
            stop_logging(logger, log_file)

