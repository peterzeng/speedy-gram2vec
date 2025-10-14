#!/usr/bin/env python3
"""
Pairwise classification accuracy distinguishing each LLM from human text.
Uses gram2vec features + logistic regression (our method).

Training/testing on HAP-E (Human-AI Parallel English Corpus).
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from pathlib import Path
import sys
import os
from tqdm import tqdm
import argparse

# Add the src directory to the path to allow imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vectorizer import Gram2VecVectorizer
from utils.cache_utils import get_or_extract_vectors, get_config_hash

def extract_model_from_doc_id(doc_id: str) -> str:
    """Extract model name from doc_id"""
    if 'chunk' in doc_id:
        if 'chunk_2' in doc_id:
            return 'human_chunk_2'
        else:
            return 'human_chunk_1'
    elif '@' in doc_id:
        model = doc_id.split('@')[1]
        return model
    else:
        return 'unknown'

def categorize_model(model: str) -> str:
    """Categorize model into broader groups"""
    if model == 'human_chunk_2':
        return 'human'
    elif model == 'human_chunk_1':
        return 'human_chunk_1'  # We'll filter these out
    elif 'gpt-4o-mini' in model:
        return 'gpt-4o-mini'
    elif 'gpt-4o' in model:
        return 'gpt-4o'
    elif 'Meta-Llama-3-70B-Instruct' in model:
        return 'llama-3-70b-instruct'
    elif 'Meta-Llama-3-8B-Instruct' in model:
        return 'llama-3-8b-instruct'
    elif 'Meta-Llama-3-70B' in model:
        return 'llama-3-70b-base'
    elif 'Meta-Llama-3-8B' in model:
        return 'llama-3-8b-base'
    else:
        return 'other'

def train_pairwise_classifier(df, human_label='human', llm_label='llm', 
                              vectorizer=None, config=None, cache_dir=None,
                              use_cache=True, test_size=0.2, random_state=42):
    """
    Train a logistic regression classifier to distinguish human text from one LLM.
    Uses gram2vec features (matching our method in parallel_log_regression.py).
    
    Args:
        df: DataFrame with 'text' and 'label' columns
        human_label: Label for human class
        llm_label: Label for LLM class
        vectorizer: Gram2VecVectorizer instance
        config: Feature configuration dict
        cache_dir: Cache directory path
        use_cache: Whether to use caching
        test_size: Proportion of data for testing
        random_state: Random seed
        
    Returns:
        Dictionary with training and test accuracies
    """
    # Split data
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df['label'], random_state=random_state
    )
    
    # Extract features
    # Create temporary doc_id and author_id columns for vectorizer
    train_df_copy = train_df.copy()
    test_df_copy = test_df.copy()
    
    train_df_copy['doc_id'] = [f'train_{i}' for i in range(len(train_df_copy))]
    test_df_copy['doc_id'] = [f'test_{i}' for i in range(len(test_df_copy))]
    
    train_df_copy['author_id'] = train_df_copy['label']
    test_df_copy['author_id'] = test_df_copy['label']
    
    # Get or extract vectors
    train_vectors = get_or_extract_vectors(
        train_df_copy, f"pairwise_train_{llm_label}", vectorizer, config, cache_dir, use_cache
    )
    test_vectors = get_or_extract_vectors(
        test_df_copy, f"pairwise_test_{llm_label}", vectorizer, config, cache_dir, use_cache
    )
    
    # Prepare X, y
    X_train = train_vectors.drop(columns=["authorID", "documentID"])
    y_train = train_vectors["authorID"]
    X_test = test_vectors.drop(columns=["authorID", "documentID"])
    y_test = test_vectors["authorID"]
    
    # Scale features (matching parallel_log_regression.py)
    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression (matching parallel_log_regression.py)
    clf = LogisticRegression(
        max_iter=1000,
        solver="saga",
        n_jobs=-1,
        verbose=0
    )
    clf.fit(X_train_scaled, y_train)
    
    # Get predictions
    y_pred_train = clf.predict(X_train_scaled)
    y_pred_test = clf.predict(X_test_scaled)
    
    # Calculate accuracies
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    return {
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'n_train': len(train_df),
        'n_test': len(test_df)
    }

def main():
    parser = argparse.ArgumentParser(
        description='Pairwise classification of human vs. LLM text using gram2vec + logistic regression'
    )
    parser.add_argument('--cache-dir', default='vector_cache_hape',
                       help='Directory for caching feature vectors')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable caching and re-extract all features')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Proportion of data for testing (default: 0.2)')
    parser.add_argument('--output', '-o', default='pairwise_results.csv',
                       help='Output CSV file for results')
    args = parser.parse_args()
    
    print("=" * 80)
    print("HAP-E Pairwise Classification with Gram2Vec")
    print("=" * 80)
    
    # Load dataset
    print("\nLoading HAP-E dataset from HuggingFace...")
    dataset = load_dataset("browndw/human-ai-parallel-corpus", split="train")
    df = dataset.to_pandas()
    
    # Extract metadata
    df['model_full'] = df['doc_id'].apply(extract_model_from_doc_id)
    df['model_category'] = df['model_full'].apply(categorize_model)
    
    print(f"Total samples: {len(df):,}")
    print(f"Model categories: {df['model_category'].value_counts().to_dict()}")
    
    # Filter to only chunk_2 for human (chunk_1 was used to prompt LLMs)
    human_df = df[df['model_category'] == 'human'].copy()
    print(f"\nHuman samples (chunk_2): {len(human_df):,}")
    
    # Get list of LLMs
    llm_categories = [cat for cat in df['model_category'].unique() 
                     if cat not in ['human', 'human_chunk_1', 'other', 'unknown']]
    llm_categories = sorted(llm_categories)
    
    print(f"\nLLMs to compare: {llm_categories}")
    
    # Setup gram2vec vectorizer
    gram2vec_config = {
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
    
    use_cache = not args.no_cache
    cache_dir = Path(args.cache_dir)
    
    if use_cache:
        config_hash = get_config_hash(gram2vec_config)
        print(f"\nCache enabled - config hash: {config_hash}")
        print(f"Cache directory: {cache_dir}")
    else:
        print("\nCache disabled - will re-extract all features")
    
    vectorizer = Gram2VecVectorizer(
        language="en", 
        normalize=True, 
        n_process=4, 
        enabled_features=gram2vec_config
    )
    
    # Results storage
    results = []
    
    # Train pairwise classifiers
    print(f"\n{'=' * 80}")
    print(f"Training Pairwise Classifiers (Logistic Regression + Gram2Vec)")
    print(f"{'=' * 80}\n")
    
    for llm in tqdm(llm_categories, desc="LLM comparisons"):
        print(f"\n--- Human vs. {llm} ---")
        
        # Get LLM data
        llm_df = df[df['model_category'] == llm].copy()
        print(f"LLM samples: {len(llm_df):,}")
        
        # Combine human and LLM data
        combined_df = pd.concat([
            human_df[['text', 'doc_id']].assign(label='human'),
            llm_df[['text', 'doc_id']].assign(label='llm')
        ], ignore_index=True)
        
        print(f"Combined samples: {len(combined_df):,}")
        print(f"  Human: {(combined_df['label'] == 'human').sum():,}")
        print(f"  LLM:   {(combined_df['label'] == 'llm').sum():,}")
        
        # Train classifier
        result = train_pairwise_classifier(
            combined_df,
            human_label='human',
            llm_label=llm,
            vectorizer=vectorizer,
            config=gram2vec_config,
            cache_dir=cache_dir,
            use_cache=use_cache,
            test_size=args.test_size,
            random_state=42
        )
        
        print(f"  Training accuracy:   {result['train_accuracy']:.4f} ({result['train_accuracy']*100:.2f}%)")
        print(f"  Test accuracy:       {result['test_accuracy']:.4f} ({result['test_accuracy']*100:.2f}%)")
        
        # Store results
        results.append({
            'LLM': llm,
            'Training': f"{result['train_accuracy']*100:.1f}%",
            'Test': f"{result['test_accuracy']*100:.1f}%",
            'n_train': result['n_train'],
            'n_test': result['n_test']
        })
    
    # Create results table
    results_df = pd.DataFrame(results)
    
    print(f"\n{'=' * 80}")
    print("RESULTS TABLE (similar to Table S9)")
    print(f"{'=' * 80}\n")
    print(results_df.to_string(index=False))
    
    # Save results
    results_df.to_csv(args.output, index=False)
    print(f"\nResults saved to: {args.output}")
    
    # Also create a formatted version like the table
    print(f"\n{'=' * 80}")
    print("Formatted Results Table")
    print(f"{'=' * 80}")
    print(f"\n{'LLM':<30s} | {'Training':<10s} | {'Test':<10s}")
    print("-" * 55)
    for _, row in results_df.iterrows():
        print(f"{row['LLM']:<30s} | {row['Training']:<10s} | {row['Test']:<10s}")

if __name__ == '__main__':
    main()

