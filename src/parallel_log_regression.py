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

config = {
    
}

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
    args = parser.parse_args()
    
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
        # Vectorize texts
        # Limit n_process to avoid "too many open files" error
        vectorizer = Gram2VecVectorizer(language="en", normalize=True, n_process=4, enabled_features=default_config)

        print("\n=== Vectorizing training data ===")
        train_vectors = vectorizer.from_documents(train_df["text"].tolist(), author_ids=train_df["author_id"].tolist(), document_ids=train_df["doc_id"].tolist())
        print("\n=== Vectorizing dev data ===")
        dev_vectors = vectorizer.from_documents(dev_df["text"].tolist(), author_ids=dev_df["author_id"].tolist(), document_ids=dev_df["doc_id"].tolist())
        print("\n=== Vectorizing test data ===")
        test_vectors = vectorizer.from_documents(test_df["text"].tolist(), author_ids=test_df["author_id"].tolist(), document_ids=test_df["doc_id"].tolist())
    
    elif args.features == 'elfen':
        print("\n=== Extracting elfen features from training data ===")
        train_vectors = extract_elfen_features(train_df)
        print("\n=== Extracting elfen features from dev data ===")
        dev_vectors = extract_elfen_features(dev_df)
        print("\n=== Extracting elfen features from test data ===")
        test_vectors = extract_elfen_features(test_df)

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

