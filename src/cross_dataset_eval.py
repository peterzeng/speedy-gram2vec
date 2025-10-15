#!/usr/bin/env python3
"""
Cross-dataset training/evaluation using Gram2Vec features:
- Train on HAP-E, evaluate on CAP (COCA-AI Parallel spaCy tokens reconstructed)
- Train on CAP, evaluate on HAP-E

HAP-E: browndw/human-ai-parallel-corpus
CAP:   browndw/coca-ai-parallel-corpus-spacy

This script builds a binary classifier (human vs llm) and reports metrics.
"""

import argparse
import atexit
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Local imports
from vectorizer import Gram2VecVectorizer
from utils.cache_utils import get_or_extract_vectors, get_config_hash
from cap_loader import load_cap_documents
from utils.log_utils import generate_log_filename, start_logging, stop_logging, list_logs


def _extract_model_from_doc_id(doc_id: str) -> str:
    if 'chunk' in doc_id:
        if 'chunk_2' in doc_id:
            return 'human_chunk_2'
        else:
            return 'human_chunk_1'
    elif '@' in doc_id:
        return doc_id.split('@')[1]
    else:
        return 'unknown'


def _categorize_model(model: str) -> str:
    if model == 'human_chunk_2':
        return 'human'
    elif model == 'human_chunk_1':
        return 'human_chunk_1'
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


def load_hape_documents(limit_per_class: Optional[int] = None, include_models: Optional[List[str]] = None) -> pd.DataFrame:
    """Load HAP-E data and return doc-level DataFrame with binary labels.

    Returns columns: doc_id, text, author_id in {"human", "llm"}
    Skips human_chunk_1 and other/unknown categories.
    """
    dataset = load_dataset("browndw/human-ai-parallel-corpus", split="train")
    df = dataset.to_pandas()

    df['model_full'] = df['doc_id'].apply(_extract_model_from_doc_id)
    df['model_category'] = df['model_full'].apply(_categorize_model)

    human_df = df[df['model_category'] == 'human'][['doc_id', 'text']].copy()
    if include_models is None:
        llm_mask = ~df['model_category'].isin(['human', 'human_chunk_1', 'other', 'unknown'])
    else:
        llm_mask = df['model_category'].isin(include_models)
    llm_df = df[llm_mask][['doc_id', 'text']].copy()

    if limit_per_class is not None:
        human_df = human_df.sample(n=min(limit_per_class, len(human_df)), random_state=42)
        llm_df = llm_df.sample(n=min(limit_per_class, len(llm_df)), random_state=42)

    human_df['author_id'] = 'human'
    llm_df['author_id'] = 'llm'
    combined = pd.concat([human_df, llm_df], ignore_index=True)
    return combined


def load_cap_local_documents(
    cap_dir: Path,
    limit_per_class: Optional[int] = None,
    include_models: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Load CAP from locally rebuilt parquet files into a binary human/llm dataframe.

    Expects files written by rebuild_cap_hape_format.py:
      - chunk2_human.parquet
      - gpt-4o.parquet
      - gpt-4o-mini.parquet
      - llama-3-70b-base.parquet
      - llama-3-8b-base.parquet
      - llama-3-70b-instruct.parquet
      - llama-3-8b-instruct.parquet
    """
    mapping: Dict[str, str] = {
        "human": "chunk2_human.parquet",
        "gpt-4o": "gpt-4o.parquet",
        "gpt-4o-mini": "gpt-4o-mini.parquet",
        "llama-3-70b-base": "llama-3-70b-base.parquet",
        "llama-3-8b-base": "llama-3-8b-base.parquet",
        "llama-3-70b-instruct": "llama-3-70b-instruct.parquet",
        "llama-3-8b-instruct": "llama-3-8b-instruct.parquet",
    }

    # Human
    human_path = cap_dir / mapping["human"]
    human_df = pd.read_parquet(human_path)
    if limit_per_class is not None and len(human_df) > limit_per_class:
        human_df = human_df.sample(n=limit_per_class, random_state=42)
    human_df = human_df[["doc_id", "text"]].copy()
    human_df["author_id"] = "human"

    # LLM side: include specified models or all
    llm_categories = [k for k in mapping.keys() if k != "human"]
    if include_models is not None:
        llm_categories = [c for c in llm_categories if c in include_models]

    llm_frames = []
    per_llm_limit = limit_per_class
    for cat in llm_categories:
        path = cap_dir / mapping[cat]
        df_curr = pd.read_parquet(path)
        if per_llm_limit is not None and len(df_curr) > per_llm_limit:
            df_curr = df_curr.sample(n=per_llm_limit, random_state=42)
        df_curr = df_curr[["doc_id", "text"]].copy()
        df_curr["author_id"] = "llm"
        llm_frames.append(df_curr)

    llm_df = pd.concat(llm_frames, ignore_index=True) if llm_frames else pd.DataFrame(columns=["doc_id", "text", "author_id"])

    return pd.concat([human_df, llm_df], ignore_index=True)


def annotate_llm_category(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'model_category' column by parsing doc_id (human stays 'human')."""
    def categorize_from_doc(doc_id: str, author_id: str) -> str:
        if author_id == 'human':
            return 'human'
        model = _extract_model_from_doc_id(doc_id)
        return _categorize_model(model)
    df = df.copy()
    df['model_category'] = [categorize_from_doc(d, a) for d, a in zip(df['doc_id'], df['author_id'])]
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-dataset Gram2Vec training/evaluation (HAP-E <-> CAP)")
    parser.add_argument("--train-dataset", choices=["hape", "cap"], required=True)
    parser.add_argument("--test-dataset", choices=["hape", "cap"], required=True)
    parser.add_argument("--limit-per-class", type=int, default=None, help="Max docs per class for each dataset")
    parser.add_argument("--include-models", nargs="*", default=None, help="Optional list of LLM categories to include")
    parser.add_argument("--cache-dir", default="vector_cache_cross", help="Cache directory for Gram2Vec vectors")
    parser.add_argument("--cap-doc-cache", default="cap_doc_cache", help="Cache directory for CAP reconstructed docs (stream source)")
    parser.add_argument("--cap-source", choices=["local", "stream"], default="local", help="Use local rebuilt CAP parquets or HF streaming")
    parser.add_argument("--cap-dir", default=str(Path(__file__).parent.parent / "data"), help="Directory containing rebuilt CAP parquet files")
    parser.add_argument("--no-cache", action="store_true", help="Disable vector cache")
    parser.add_argument("--per-llm", action="store_true", help="Evaluate per LLM category (human vs that LLM) across datasets")
    parser.add_argument("--log-dir", default="logs", help="Directory for log files")
    parser.add_argument("--no-log", action="store_true", help="Disable logging to file")
    parser.add_argument("--list-logs", action="store_true", help="List recent log files and exit")
    parser.add_argument("--intra-split", type=float, default=None, help="If train==test, use this test fraction for an intra-dataset split (e.g., 0.2 for 80/20)")
    parser.add_argument("--intra-seed", type=int, default=42, help="Random seed for intra-dataset split")
    args = parser.parse_args()

    # Optional: list logs and exit
    if args.list_logs:
        list_logs(args.log_dir)
        return

    # Setup tee logging (capture stdout)
    logger = None
    log_file = None
    if not args.no_log:
        log_kwargs = {
            "train": args.train_dataset,
            "test": args.test_dataset,
            "per_llm": args.per_llm,
            "cap_source": args.cap_source,
        }
        if args.include_models:
            log_kwargs["include"] = "-".join(args.include_models)
        if args.limit_per_class is not None:
            log_kwargs["limit"] = args.limit_per_class
        log_file = generate_log_filename("cross_eval", log_dir=args.log_dir, **log_kwargs)
        logger = start_logging(log_file)

        def _stop_logger():
            # Ensure logs are finalized even on early returns
            stop_logging(logger, log_file)

        atexit.register(_stop_logger)

    print("=" * 80)
    print("Cross-Dataset Evaluation (Gram2Vec + Logistic Regression)")
    print("=" * 80)

    # Load datasets (supports intra-dataset split when train==test)
    if args.train_dataset == args.test_dataset:
        # Single dataset load, then stratified split
        dataset_name = args.train_dataset
        if args.intra_split is None:
            args.intra_split = 0.2  # default to 80/20 if not specified
            print(f"\nNo --intra-split provided for same-dataset run; defaulting to 0.2 (80/20).")

        if dataset_name == "hape":
            print("\nLoading full dataset from HAP-E for intra-dataset split...")
            full_df = load_hape_documents(limit_per_class=args.limit_per_class, include_models=args.include_models)
        else:
            if args.cap_source == "local":
                print("\nLoading full dataset from CAP (local parquets) for intra-dataset split...")
                full_df = load_cap_local_documents(Path(args.cap_dir), limit_per_class=args.limit_per_class, include_models=args.include_models)
            else:
                print("\nLoading full dataset from CAP (streaming reconstruct) for intra-dataset split...")
                full_df = load_cap_documents(limit_per_class=args.limit_per_class, include_models=args.include_models, streaming=True, cache_dir=args.cap_doc_cache, use_cache=not args.no_cache)

        train_df, test_df = train_test_split(
            full_df,
            test_size=args.intra_split,
            stratify=full_df["author_id"],
            random_state=args.intra_seed,
        )
        print(f"Performed stratified intra-dataset split with test_size={args.intra_split}, seed={args.intra_seed}")
        print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")
        print(f"Train label counts: {train_df['author_id'].value_counts().to_dict()}")
        print(f"Test  label counts: {test_df['author_id'].value_counts().to_dict()}")
    else:
        # Cross-dataset load as before
        if args.train_dataset == "hape":
            print("\nLoading training data from HAP-E...")
            train_df = load_hape_documents(limit_per_class=args.limit_per_class, include_models=args.include_models)
        else:
            if args.cap_source == "local":
                print("\nLoading training data from CAP (local parquets)...")
                train_df = load_cap_local_documents(Path(args.cap_dir), limit_per_class=args.limit_per_class, include_models=args.include_models)
            else:
                print("\nLoading training data from CAP (streaming reconstruct)...")
                train_df = load_cap_documents(limit_per_class=args.limit_per_class, include_models=args.include_models, streaming=True, cache_dir=args.cap_doc_cache, use_cache=not args.no_cache)

        if args.test_dataset == "hape":
            print("Loading test data from HAP-E...")
            test_df = load_hape_documents(limit_per_class=args.limit_per_class, include_models=args.include_models)
        else:
            if args.cap_source == "local":
                print("Loading test data from CAP (local parquets)...")
                test_df = load_cap_local_documents(Path(args.cap_dir), limit_per_class=args.limit_per_class, include_models=args.include_models)
            else:
                print("Loading test data from CAP (streaming reconstruct)...")
                test_df = load_cap_documents(limit_per_class=args.limit_per_class, include_models=args.include_models, streaming=True, cache_dir=args.cap_doc_cache, use_cache=not args.no_cache)

        print(f"Train size: {len(train_df)} | Test size: {len(test_df)}")
        print(f"Train label counts: {train_df['author_id'].value_counts().to_dict()}")
        print(f"Test  label counts: {test_df['author_id'].value_counts().to_dict()}")

    # Feature config (matches other scripts)
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
        print(f"\nCache enabled - config hash: {get_config_hash(gram2vec_config)}")
        print(f"Cache directory: {cache_dir}")
    else:
        print("\nCache disabled - will re-extract features")

    # Vectorizer
    vectorizer = Gram2VecVectorizer(language="en", normalize=True, n_process=4, enabled_features=gram2vec_config)

    def run_once(run_tag: str, tr_df: pd.DataFrame, te_df: pd.DataFrame):
        tr_vectors = get_or_extract_vectors(tr_df, f"train_{run_tag}", vectorizer, gram2vec_config, cache_dir, use_cache)
        te_vectors = get_or_extract_vectors(te_df, f"test_{run_tag}", vectorizer, gram2vec_config, cache_dir, use_cache)
        X_tr = tr_vectors.drop(columns=["authorID", "documentID"])  # Features
        y_tr = tr_vectors["authorID"]
        X_te = te_vectors.drop(columns=["authorID", "documentID"])    # Features
        y_te = te_vectors["authorID"]
        scaler = StandardScaler(with_mean=False)
        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)
        clf = LogisticRegression(max_iter=1000, solver="saga", n_jobs=-1, verbose=0)
        clf.fit(X_tr_s, y_tr)
        y_pred = clf.predict(X_te_s)
        acc = accuracy_score(y_te, y_pred)
        print(f"\n[{run_tag}] Accuracy: {acc:.4f}")
        print(classification_report(y_te, y_pred, digits=4))

    if not args.per_llm:
        if args.train_dataset == args.test_dataset:
            # Include split percentage and seed in tag for cache uniqueness
            train_pct = int(round((1.0 - float(args.intra_split)) * 100)) if args.intra_split is not None else 100
            run_tag = f"{args.train_dataset}_intra_{train_pct}_seed{args.intra_seed}"
        else:
            run_tag = f"{args.train_dataset}_to_{args.test_dataset}"
        run_once(run_tag, train_df, test_df)
        if logger and log_file:
            try:
                atexit.unregister(_stop_logger)
            except Exception:
                pass
            stop_logging(logger, log_file)
        return

    # Per-LLM mode
    if args.train_dataset == args.test_dataset:
        # Intra-dataset per-LLM: build (human_chunk_2 + LLM) then 80/20 split per category
        if args.intra_split is None:
            args.intra_split = 0.2
        # Use the full_df constructed earlier in same-dataset branch
        # If not present (defensive), reconstruct from train+test
        try:
            full_df  # type: ignore # noqa: F401
        except NameError:
            full_df = pd.concat([train_df, test_df], ignore_index=True)
        df_ann = annotate_llm_category(full_df)
        llm_categories = sorted([c for c in df_ann['model_category'].unique() if c not in ['human', 'human_chunk_1', 'other', 'unknown']])
        if args.include_models:
            llm_categories = [c for c in llm_categories if c in args.include_models]
        print(f"\nPer-LLM intra-dataset evaluation over: {llm_categories}")
        train_pct = int(round((1.0 - float(args.intra_split)) * 100))
        for cat in llm_categories:
            pair_df = pd.concat([
                df_ann[df_ann['author_id'] == 'human'][['doc_id', 'text', 'author_id']],
                df_ann[df_ann['model_category'] == cat][['doc_id', 'text', 'author_id']].assign(author_id='llm'),
            ], ignore_index=True)
            tr_subset, te_subset = train_test_split(
                pair_df,
                test_size=args.intra_split,
                stratify=pair_df['author_id'],
                random_state=args.intra_seed,
            )
            print(f"\nCategory: {cat} | Train: {len(tr_subset)} | Test: {len(te_subset)}")
            run_tag = f"{args.train_dataset}_intra_{cat}_{train_pct}_seed{args.intra_seed}"
            run_once(run_tag, tr_subset, te_subset)
    else:
        # Cross-dataset per-LLM: filter both datasets to human + each LLM category
        train_df_ann = annotate_llm_category(train_df)
        test_df_ann = annotate_llm_category(test_df)
        llm_categories = sorted([c for c in train_df_ann['model_category'].unique() if c not in ['human', 'human_chunk_1', 'other', 'unknown']])
        if args.include_models:
            llm_categories = [c for c in llm_categories if c in args.include_models]
        print(f"\nPer-LLM evaluation over: {llm_categories}")
        for cat in llm_categories:
            tr_subset = pd.concat([
                train_df_ann[train_df_ann['author_id'] == 'human'][['doc_id', 'text', 'author_id']],
                train_df_ann[train_df_ann['model_category'] == cat][['doc_id', 'text', 'author_id']].assign(author_id='llm')
            ], ignore_index=True)
            te_subset = pd.concat([
                test_df_ann[test_df_ann['author_id'] == 'human'][['doc_id', 'text', 'author_id']],
                test_df_ann[test_df_ann['model_category'] == cat][['doc_id', 'text', 'author_id']].assign(author_id='llm')
            ], ignore_index=True)
            print(f"\nCategory: {cat} | Train: {len(tr_subset)} | Test: {len(te_subset)}")
            run_tag = f"{args.train_dataset}_to_{args.test_dataset}_{cat}"
            run_once(run_tag, tr_subset, te_subset)

    if logger and log_file:
        try:
            atexit.unregister(_stop_logger)
        except Exception:
            pass
        stop_logging(logger, log_file)


if __name__ == "__main__":
    main()


