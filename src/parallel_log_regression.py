import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
import os
import sys

# Add the src directory to the path to allow imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import vectorizer
from vectorizer import Gram2VecVectorizer

# Load and combine parquet files
dfs = []
data_dir = Path(__file__).parent.parent / "data"
for corpus_file in data_dir.glob("*.parquet"):
    print(f"Loading {corpus_file.name}...")
    curr_df = pd.read_parquet(corpus_file)
    dfs.append(curr_df)
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
vectorizer = Gram2VecVectorizer(language="en", normalize=True)

train_vectors = vectorizer.from_documents(train_df["text"].tolist(), author_ids=train_df["author_id"].tolist(), document_ids=train_df["doc_id"].tolist())
dev_vectors = vectorizer.from_documents(dev_df["text"].tolist(), author_ids=dev_df["author_id"].tolist(), document_ids=dev_df["doc_id"].tolist())
test_vectors = vectorizer.from_documents(test_df["text"].tolist(), author_ids=test_df["author_id"].tolist(), document_ids=test_df["doc_id"].tolist())

# Drop author_id to identify
X_train, y_train = train_vectors.drop(columns=["authorID", "documentID"]), train_vectors["authorID"]
X_dev, y_dev   = dev_vectors.drop(columns=["authorID", "documentID"]),   dev_vectors["authorID"]
X_test, y_test  = test_vectors.drop(columns=["authorID", "documentID"]),  test_vectors["authorID"]

# Scale features
scaler = StandardScaler(with_mean=False)  
X_train_scaled = scaler.fit_transform(X_train)
X_dev_scaled   = scaler.transform(X_dev)
X_test_scaled  = scaler.transform(X_test)

# Train logistic regression
authorship_model = LogisticRegression(
    max_iter=1000,
    solver="saga",  # good for large sparse data
    n_jobs=-1
)
authorship_model.fit(X_train_scaled, y_train)

# Evaluate
y_pred_test = authorship_model.predict(X_test_scaled)

# Print metrics
print(classification_report(y_test, y_pred_test))

coefs = np.squeeze(authorship_model.coef_)
coefficients = pd.DataFrame({
    'Feature': X_train.columns,
    'human': -coefs,
    'gpt': coefs
})
coefficients.style
