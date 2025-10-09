import os
import time
import argparse
import pandas as pd

# Ensure original gram2vec is importable
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))

from gram2vec import vectorizer as g2v


def load_reddit(path:str, text_col:str) -> list:
    df = pd.read_csv(path)
    if text_col not in df.columns:
        raise SystemExit(f"Column '{text_col}' not found in {path}")
    return df[text_col].astype(str).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/home/pezeng1/dev/rsp/data/reddit/test.csv")
    ap.add_argument("--text-col", default="document1")
    ap.add_argument("--limit", type=int, default=10000)
    ap.add_argument("--spacy-model", default="en_core_web_lg")
    args = ap.parse_args()

    # Match repo defaults
    os.environ.setdefault("LANGUAGE", "en")
    os.environ.setdefault("SPACY_MODEL", args.spacy_model)

    texts = load_reddit(args.path, args.text_col)
    if args.limit:
        texts = texts[: args.limit]
    print(f"Loaded {len(texts)} texts")

    t0 = time.time()
    df = g2v.from_documents(texts, config=None, include_content_embedding=False)
    t1 = time.time()
    dur = t1 - t0
    print(f"Processed {len(texts)} docs in {dur:.2f}s -> {len(texts)/dur:.2f} docs/s, {len(df.columns)} features")


if __name__ == "__main__":
    main()


