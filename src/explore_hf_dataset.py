#!/usr/bin/env python3
"""
Explore the Human-AI Parallel English Corpus (HAP-E) from HuggingFace
to understand its structure and prepare for training/testing splits.
"""

import pandas as pd
from datasets import load_dataset
from collections import Counter
import re

def extract_model_from_doc_id(doc_id: str) -> str:
    """Extract model name from doc_id"""
    if 'chunk' in doc_id:
        return 'human'
    elif '@' in doc_id:
        # Format: "acad_0001@gpt-4o-2024-08-06"
        model = doc_id.split('@')[1]
        return model
    else:
        return 'unknown'

def extract_text_type(doc_id: str) -> str:
    """Extract text type from doc_id (acad, blog, fic, news, spok, tvm)"""
    # doc_id format: "acad_0001@model" or "acad_0001@chunk_1"
    parts = doc_id.split('_')
    if parts:
        return parts[0]
    return 'unknown'

def categorize_model(model: str) -> str:
    """Categorize model into broader groups for replicating Table S9"""
    if model == 'human':
        return 'human'
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

def main():
    print("Loading Human-AI Parallel English Corpus from HuggingFace...")
    print("This may take a moment...")
    
    # Load the dataset
    dataset = load_dataset("browndw/human-ai-parallel-corpus", split="train")
    
    # Convert to pandas for easier analysis
    df = dataset.to_pandas()
    
    print(f"\n=== Dataset Overview ===")
    print(f"Total rows: {len(df):,}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nFirst few doc_ids:")
    print(df['doc_id'].head(10).tolist())
    
    # Extract metadata from doc_ids
    df['model_full'] = df['doc_id'].apply(extract_model_from_doc_id)
    df['text_type'] = df['doc_id'].apply(extract_text_type)
    df['model_category'] = df['model_full'].apply(categorize_model)
    
    print(f"\n=== Model Categories ===")
    model_counts = df['model_category'].value_counts()
    for model, count in model_counts.items():
        print(f"{model:30s}: {count:6,} samples")
    
    print(f"\n=== Text Types ===")
    text_type_counts = df['text_type'].value_counts()
    for text_type, count in text_type_counts.items():
        print(f"{text_type:10s}: {count:6,} samples")
    
    print(f"\n=== Cross-tabulation: Model Category x Text Type ===")
    cross_tab = pd.crosstab(df['model_category'], df['text_type'])
    print(cross_tab)
    
    print(f"\n=== Unique Full Model Names ===")
    unique_models = df['model_full'].unique()
    print(f"Total unique models/sources: {len(unique_models)}")
    for model in sorted(unique_models):
        count = (df['model_full'] == model).sum()
        print(f"  {model:50s}: {count:6,} samples")
    
    # Analyze text lengths
    df['text_length'] = df['text'].str.len()
    df['word_count'] = df['text'].str.split().str.len()
    
    print(f"\n=== Text Statistics ===")
    print(f"Character length:")
    print(df.groupby('model_category')['text_length'].describe())
    print(f"\nWord count:")
    print(df.groupby('model_category')['word_count'].describe())
    
    # Check for chunk_1 vs chunk_2 for human data
    human_df = df[df['model_category'] == 'human']
    print(f"\n=== Human Data Analysis ===")
    print(f"Total human samples: {len(human_df):,}")
    
    # Check if there are chunk_1 and chunk_2 in the doc_ids
    chunk_1_count = human_df['doc_id'].str.contains('chunk_1').sum()
    chunk_2_count = human_df['doc_id'].str.contains('chunk_2').sum()
    print(f"  chunk_1: {chunk_1_count:,} samples")
    print(f"  chunk_2: {chunk_2_count:,} samples")
    
    # Recommendations for train/test splits
    print(f"\n=== Recommendations for Replicating Table S9 ===")
    print(f"Table S9 shows pairwise classification (human vs. each LLM)")
    print(f"Training scenarios:")
    print(f"  1. Train on HAP-E (this dataset)")
    print(f"  2. Train on CAP (need to clarify what CAP is)")
    print(f"\nFor HAP-E training:")
    print(f"  - Use chunk_2 as human examples (chunk_1 was used to prompt LLMs)")
    print(f"  - For each LLM, create a binary classifier: chunk_2 vs. LLM output")
    print(f"  - Split data 60/20/20 or similar for train/dev/test")
    print(f"\nAvailable pairwise comparisons:")
    llm_models = [m for m in model_counts.index if m != 'human']
    for llm in llm_models:
        llm_count = (df['model_category'] == llm).sum()
        chunk2_count = chunk_2_count  # Same for all comparisons
        print(f"  Human (chunk_2) vs. {llm:30s}: {chunk2_count:5,} vs {llm_count:5,} samples")
    
    # Save summary statistics
    output_file = "hf_dataset_summary.csv"
    summary_df = df.groupby(['model_category', 'text_type']).agg({
        'doc_id': 'count',
        'word_count': ['mean', 'std', 'min', 'max']
    }).reset_index()
    summary_df.to_csv(output_file, index=False)
    print(f"\nSummary statistics saved to: {output_file}")
    
    # Save full dataframe with metadata for further analysis
    meta_output = "hf_dataset_with_metadata.parquet"
    df_to_save = df[['doc_id', 'text', 'model_full', 'model_category', 'text_type', 'word_count']]
    df_to_save.to_parquet(meta_output, index=False)
    print(f"Full dataset with metadata saved to: {meta_output}")

if __name__ == "__main__":
    main()

