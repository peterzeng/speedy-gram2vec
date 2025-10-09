import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
import os
import sys
from tqdm import tqdm

# Add the src directory to the path to allow imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import vectorizer
from vectorizer import Gram2VecVectorizer

config = {
    
}
if __name__ == '__main__':
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

    # Vectorize texts
    # Limit n_process to avoid "too many open files" error
    vectorizer = Gram2VecVectorizer(language="en", normalize=True, n_process=4)

    print("\n=== Vectorizing training data ===")
    train_vectors = vectorizer.from_documents(train_df["text"].tolist(), author_ids=train_df["author_id"].tolist(), document_ids=train_df["doc_id"].tolist())
    print("\n=== Vectorizing dev data ===")
    dev_vectors = vectorizer.from_documents(dev_df["text"].tolist(), author_ids=dev_df["author_id"].tolist(), document_ids=dev_df["doc_id"].tolist())
    print("\n=== Vectorizing test data ===")
    test_vectors = vectorizer.from_documents(test_df["text"].tolist(), author_ids=test_df["author_id"].tolist(), document_ids=test_df["doc_id"].tolist())

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
        verbose=1  # Show training progress
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
    coefficients.to_csv("coefficients.csv", index=False)

