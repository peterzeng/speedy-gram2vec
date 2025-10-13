import os
import sys
import time
import argparse
import pandas as pd
from typing import List, Set

import numpy as np
import pandas as pd


def add_paths(repo_root:str) -> None:
    sys.path.insert(0, os.path.join(repo_root, "src"))
    sys.path.insert(0, os.path.join(repo_root, "speedy-gram2vec", "src"))


def build_default_texts() -> List[str]:
    return [
        "Hello, world! I like to kick things and punch the wall.",
        "This dark chocolate is the best chocolate I have ever had!🥳",
        "What is the capital of France??!",
        "Maria lives in Mexico City on January 1st, 2023.",
        "She is a doctor. The sky was blue. They are happy.",
        "I suggest you try the new restaurant. We recommend the cake.",
        "My cat is smaller than my dog...",
        "I know the answer. She believes in magic.",
        "Apple Inc. announced a new product yesterday.",
        "They see this person and that person.",
    ]


def load_texts(args) -> List[str]:
    if args.csv_path:
        df = pd.read_csv(args.csv_path)
        if args.text_col not in df.columns:
            raise SystemExit(f"Column '{args.text_col}' not in {args.csv_path}")
        texts = df[args.text_col].astype(str).tolist()
    elif args.text_file:
        with open(args.text_file, "r", encoding="utf-8") as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        texts = build_default_texts()
    if args.n_docs:
        base = texts
        while len(texts) < args.n_docs:
            texts.extend(base)
        texts = texts[: args.n_docs]
    return texts


def select_overlap_columns(df_std:pd.DataFrame, df_fast:pd.DataFrame, prefixes:Set[str]) -> List[str]:
    cols_std = {c for c in df_std.columns if any(c.startswith(p) for p in prefixes)}
    cols_fast = {c for c in df_fast.columns if any(c.startswith(p) for p in prefixes)}
    overlap = sorted(cols_std.intersection(cols_fast))
    return overlap


def compare_vectors(df_std:pd.DataFrame, df_fast:pd.DataFrame, cols:List[str], atol:float) -> pd.DataFrame:
    a = df_std.reindex(columns=cols).fillna(0.0).to_numpy(dtype=float)
    b = df_fast.reindex(columns=cols).fillna(0.0).to_numpy(dtype=float)
    diff = np.abs(a - b)
    max_diff = float(diff.max()) if diff.size else 0.0
    mean_diff = float(diff.mean()) if diff.size else 0.0
    mismatches = int((diff > atol).sum())
    print(f"Overlap columns: {len(cols)} | max_abs_diff={max_diff:.6g} | mean_abs_diff={mean_diff:.6g} | num_values>|{atol}|={mismatches}")

    if cols:
        # Show top-k differing features
        k = min(10, len(cols))
        max_per_col = diff.max(axis=0)
        top_idx = np.argsort(-max_per_col)[:k]
        print("Top differing features:")
        for i in top_idx:
            print(f"  {cols[i]}: max_abs_diff={max_per_col[i]:.6g}")

    # Return detailed per-feature diffs
    return pd.DataFrame({
        "feature": cols,
        "max_abs_diff": list(diff.max(axis=0) if diff.size else [])
    })


def benchmark_std(texts:List[str], config:dict) -> float:
    from gram2vec import vectorizer as g2v
    t0 = time.perf_counter()
    _ = g2v.from_documents(texts, config=config, include_content_embedding=False)
    t1 = time.perf_counter()
    return t1 - t0


def benchmark_fast(texts:List[str]) -> float:
    from vectorizer import Gram2VecVectorizer
    vec = Gram2VecVectorizer("en", normalize=True)
    t0 = time.perf_counter()
    _ = vec.vectorize_documents(texts)
    t1 = time.perf_counter()
    return t1 - t0


def run_once(texts:List[str], atol:float, show_shapes:bool, std_all:bool) -> None:
    # Configure original gram2vec
    config = None if std_all else {
        "pos_unigrams": 1,
        "pos_bigrams": 0,  # differs (BOS/EOS) vs speedy implementation
        "func_words": 0,
        "punctuation": 0,
        "letters": 0,
        "emojis": 0,
        "dep_labels": 1,
        "morph_tags": 1,
        "sentences": 1,
        "num_tokens": 0,
    }

    # Compute vectors
    from gram2vec import vectorizer as g2v
    from vectorizer import Gram2VecVectorizer

    df_std = g2v.from_documents(texts, config=config, include_content_embedding=False)
    fast = Gram2VecVectorizer("en", normalize=True)
    df_fast = fast.vectorize_documents(texts)

    if show_shapes:
        print(f"std shape={df_std.shape}, fast shape={df_fast.shape}")

    # Report full feature sets and differences
    std_cols = sorted(df_std.columns.tolist())
    fast_cols = sorted(df_fast.columns.tolist())
    std_set = set(std_cols)
    fast_set = set(fast_cols)

    missing_in_fast = sorted(std_set - fast_set)
    missing_in_std = sorted(fast_set - std_set)

    print("\n--- Feature sets ---")
    print(f"gram2vec features: {len(std_cols)}")
    print(f"speedy features:  {len(fast_cols)}")
    print(f"Missing in speedy (present in gram2vec only): {len(missing_in_fast)}")
    print("  e.g.", missing_in_fast[:20])
    print(f"Missing in gram2vec (present in speedy only): {len(missing_in_std)}")
    print("  e.g.", missing_in_std[:20])

    # Align overlapping comparable features
    prefixes = {"pos_unigrams:", "dep_labels:", "morph_tags:", "sentences:", "func_words:", "punctuation:", "letters:", "emojis:", "pos_bigrams:", "tokens:", "named_entities:", "suasive_verbs:", "stative_verbs:"}
    cols = select_overlap_columns(df_std, df_fast, prefixes)
    _ = compare_vectors(df_std, df_fast, cols, atol=atol)


def main():
    parser = argparse.ArgumentParser(description="Compare gram2vec vs speedy-gram2vec outputs and speed")
    parser.add_argument("--text-file", type=str, default=None, help="Optional path to a newline-delimited text file")
    parser.add_argument("--csv-path", type=str, default=None, help="Optional path to a CSV file")
    parser.add_argument("--text-col", type=str, default=None, help="Column name in CSV containing text")
    parser.add_argument("--n-docs", type=int, default=20, help="Number of documents to process (duplicates default set if needed)")
    parser.add_argument("--spacy-model", type=str, default="en_core_web_lg", help="spaCy model to use")
    parser.add_argument("--atol", type=float, default=1e-6, help="Absolute tolerance for value comparison")
    parser.add_argument("--repeat", type=int, default=1, help="Number of repeats for speed benchmarking")
    parser.add_argument("--std-all", action="store_true", help="Enable all features in original gram2vec (use default config)")
    parser.add_argument("--show-shapes", action="store_true", help="Print dataframe shapes")
    args = parser.parse_args()

    # Environment setup for original gram2vec
    os.environ.setdefault("LANGUAGE", "en")
    os.environ["SPACY_MODEL"] = args.spacy_model

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    add_paths(repo_root)

    texts = load_texts(args)

    print("=== Equality check on overlapping features ===")
    run_once(texts, atol=args.atol, show_shapes=args.show_shapes, std_all=args.std_all)

    print("\n=== Speed benchmark ===")
    # Warmup both
    std_config = None if args.std_all else {
        "pos_unigrams": 1,
        "pos_bigrams": 0,
        "func_words": 0,
        "punctuation": 0,
        "letters": 0,
        "emojis": 0,
        "dep_labels": 1,
        "morph_tags": 1,
        "sentences": 1,
        "num_tokens": 0,
    }
    _ = benchmark_std(texts, std_config)
    _ = benchmark_fast(texts)

    std_times = []
    fast_times = []
    for _ in range(args.repeat):
        std_times.append(benchmark_std(texts, std_config))
        fast_times.append(benchmark_fast(texts))

    n = len(texts)
    std_mean = float(np.mean(std_times))
    fast_mean = float(np.mean(fast_times))
    print(f"Original gram2vec: {std_mean:.3f}s  ({n/std_mean:.2f} docs/s)")
    print(f"Speedy gram2vec:   {fast_mean:.3f}s  ({n/fast_mean:.2f} docs/s)")


if __name__ == "__main__":
    main()


