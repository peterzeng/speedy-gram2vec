import os
import time
import argparse
import pandas as pd

from vectorizer import Gram2VecVectorizer

def load_reddit(path:str, text_col:str) -> list:
    df = pd.read_csv(path)
    if text_col not in df.columns:
        raise SystemExit(f"Column '{text_col}' not found in {path}")
    texts = df[text_col].astype(str).tolist()
    return texts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", default="/home/pezeng1/dev/rsp/data/reddit/test.csv")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--limit", type=int, default=5000)
    ap.add_argument("--n-process", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=1000)
    ap.add_argument("--disable-ner", action="store_true")
    args = ap.parse_args()

    texts = load_reddit(args.path, args.text_col)
    if args.limit:
        texts = texts[: args.limit]
    print(f"Loaded {len(texts)} texts")

    enabled = None
    if args.disable_ner:
        enabled = None
        # vectorizer will exclude NER automatically when named_entities is 0
        enabled = {
            "pos_unigrams": 1,
            "pos_bigrams": 1,
            "dep_labels": 1,
            "morph_tags": 1,
            "sentences": 1,
            "emojis": 1,
            "func_words": 1,
            "punctuation": 1,
            "letters": 1,
            "tokens": 1,
            "num_tokens": 1,
            "named_entities": 0,
            "suasive_verbs": 0,
            "stative_verbs": 0,
        }

    vec = Gram2VecVectorizer(
        language="en",
        normalize=True,
        enabled_features=enabled,
        spacy_model="en_core_web_lg",
        n_process=args.n_process,
        batch_size=args.batch_size,
    )

    t0 = time.time()
    df = vec.vectorize_documents(texts)
    t1 = time.time()
    dur = t1 - t0
    print(f"Processed {len(texts)} docs in {dur:.2f}s -> {len(texts)/dur:.2f} docs/s, {len(df.columns)} features")


if __name__ == "__main__":
    main()


