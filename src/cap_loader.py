#!/usr/bin/env python3
"""
Utilities to load and reconstruct documents from the COCA-AI Parallel (CAP)
spaCy-tokenized dataset on Hugging Face:

Dataset: browndw/coca-ai-parallel-corpus-spacy

This dataset is token-level with columns like:
- doc_id, sentence_id, token_id, token, lemma, pos, tag, head_token_id, dep_rel

We reconstruct sentence strings by ordering tokens by token_id per (doc_id, sentence_id),
then concatenate sentences per doc_id to produce document-level texts suitable for
feature extraction (e.g., Gram2Vec).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from utils.cache_utils import get_config_hash


def _extract_model_from_doc_id(doc_id: str) -> str:
    """Extract model name/group from a CAP/HAP-E style doc_id.

    Heuristic matches logic in HAP-E utilities:
    - 'chunk_2' => human
    - 'chunk_1' => human_chunk_1 (used as prompts; we typically exclude)
    - '@<model>' => LLM identifier (e.g., gpt-4o-2024-08-06, Meta-Llama-3-70B-Instruct)
    """
    if "chunk" in doc_id:
        if "chunk_2" in doc_id:
            return "human_chunk_2"
        else:
            return "human_chunk_1"
    elif "@" in doc_id:
        return doc_id.split("@")[1]
    else:
        return "unknown"


def _categorize_model(model: str) -> str:
    """Map detailed model names to coarse categories used for binary labeling."""
    if model == "human_chunk_2":
        return "human"
    elif model == "human_chunk_1":
        return "human_chunk_1"
    elif "gpt-4o-mini" in model:
        return "gpt-4o-mini"
    elif "gpt-4o" in model:
        return "gpt-4o"
    elif "Meta-Llama-3-70B-Instruct" in model:
        return "llama-3-70b-instruct"
    elif "Meta-Llama-3-8B-Instruct" in model:
        return "llama-3-8b-instruct"
    elif "Meta-Llama-3-70B" in model:
        return "llama-3-70b-base"
    elif "Meta-Llama-3-8B" in model:
        return "llama-3-8b-base"
    else:
        return "other"


def _postprocess_spacing(text: str) -> str:
    """Lightly fix spacing around punctuation for readability.

    This is not meant to perfectly undo tokenizer decisions, but to avoid cases like
    "word , next". This step is optional for feature extraction but helps sanity checks.
    """
    # Remove spaces before common punctuation
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    # Fix bracket spacing
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    # Collapse multiple spaces
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def load_cap_documents(
    limit_per_class: Optional[int] = None,
    include_models: Optional[List[str]] = None,
    streaming: bool = True,
    cache_dir: str | Path = "cap_doc_cache",
    use_cache: bool = True,
) -> pd.DataFrame:
    """Reconstruct document-level texts from CAP token rows.

    Args:
        limit_per_class: If provided, return at most this many docs for each class
            ("human" and aggregated "llm"). Skips "human_chunk_1" and "other".
        include_models: Optional list of LLM categories to include (e.g., ["gpt-4o"]).
            If None, include all LLM categories.
        streaming: Use HF streaming to avoid loading the full token table.

    Returns:
        pandas.DataFrame with columns: ["doc_id", "text", "author_id"], where
        author_id is "human" or "llm" (binary), derived from doc_id.
    """
    # Check doc-level cache first
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = {
        "dataset": "browndw/coca-ai-parallel-corpus-spacy",
        "limit_per_class": int(limit_per_class) if limit_per_class is not None else None,
        "include_models": sorted(include_models) if include_models is not None else None,
        "streaming": bool(streaming),
    }
    cache_hash = get_config_hash(cache_key)
    cache_path = cache_dir / f"cap_docs_{cache_hash}.parquet"

    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)

    ds = load_dataset(
        "browndw/coca-ai-parallel-corpus-spacy", split="train", streaming=streaming
    )

    # We assume rows are ordered by doc_id, sentence_id, token_id (as produced by the pipeline).
    # We'll accumulate one document at a time to keep memory bounded.
    current_doc_id: Optional[str] = None
    current_sentence_id: Optional[int] = None
    sentence_tokens: List[str] = []
    sentences: List[str] = []

    human_count = 0
    llm_count = 0
    max_human = limit_per_class if limit_per_class is not None else float("inf")
    max_llm = limit_per_class if limit_per_class is not None else float("inf")

    records: List[Dict[str, str]] = []

    def maybe_finalize_doc(doc_id: Optional[str]) -> None:
        nonlocal human_count, llm_count, sentences, sentence_tokens, current_sentence_id
        if doc_id is None:
            return
        # Flush last open sentence
        if sentence_tokens:
            sent_text = _postprocess_spacing(" ".join(sentence_tokens))
            sentences.append(sent_text)
            sentence_tokens = []
            current_sentence_id = None

        if not sentences:
            return

        full_text = _postprocess_spacing(" ".join(sentences))
        model = _extract_model_from_doc_id(doc_id)
        category = _categorize_model(model)

        if category == "human":
            if human_count < max_human:
                records.append({"doc_id": doc_id, "text": full_text, "author_id": "human"})
                human_count += 1
        elif category in {"human_chunk_1", "other"}:
            # Skip
            pass
        else:
            # Any LLM category counts as llm for binary labeling; allow filtering by include_models
            if include_models is None or category in include_models:
                if llm_count < max_llm:
                    records.append({"doc_id": doc_id, "text": full_text, "author_id": "llm"})
                    llm_count += 1

        # Reset per-doc buffers
        sentences = []

    # Iterate token rows with progress (unknown total)
    token_iter = tqdm(ds, desc="CAP tokens", unit="tok")
    for row in token_iter:
        doc_id = row["doc_id"]
        sent_id = int(row["sentence_id"])  # HF may provide as int already
        token = row["token"]

        # If we have gathered enough of both classes, we can early stop once doc boundary passes
        if human_count >= max_human and llm_count >= max_llm:
            # keep consuming until doc boundary change to finalize last doc cleanly
            if current_doc_id is not None and doc_id != current_doc_id:
                break

        # New document boundary
        if current_doc_id is None:
            current_doc_id = doc_id
            current_sentence_id = sent_id
            sentence_tokens = [token]
            continue

        if doc_id != current_doc_id:
            # finalize the document we've been accumulating
            maybe_finalize_doc(current_doc_id)
            current_doc_id = doc_id
            current_sentence_id = sent_id
            sentence_tokens = [token]
            continue

        # Same document, check sentence boundary
        if current_sentence_id is None:
            current_sentence_id = sent_id

        if sent_id != current_sentence_id:
            # finalize sentence
            sent_text = _postprocess_spacing(" ".join(sentence_tokens))
            sentences.append(sent_text)
            # start new sentence
            current_sentence_id = sent_id
            sentence_tokens = [token]
        else:
            sentence_tokens.append(token)

    # Finalize last open doc
    maybe_finalize_doc(current_doc_id)

    df = pd.DataFrame.from_records(records)
    if use_cache:
        df.to_parquet(cache_path, index=False)
    return df


if __name__ == "__main__":
    # Quick manual check (small limits) when running standalone
    sample_df = load_cap_documents(limit_per_class=5)
    print(sample_df.head())
    print({k: v for k, v in sample_df["author_id"].value_counts().to_dict().items()})


