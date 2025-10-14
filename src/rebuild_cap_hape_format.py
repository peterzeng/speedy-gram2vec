#!/usr/bin/env python3
"""
Rebuild CAP (COCA-AI Parallel spaCy tokens) into HAP-E-like parquet files.

Source dataset (token-level): `browndw/coca-ai-parallel-corpus-spacy`
Target output (doc-level): 7 parquet files in the `data` directory:
  - chunk2_human.parquet
  - gpt-4o.parquet
  - gpt-4o-mini.parquet
  - llama-3-70b-base.parquet
  - llama-3-8b-base.parquet
  - llama-3-70b-instruct.parquet
  - llama-3-8b-instruct.parquet

Each file stores rows with columns: ['doc_id', 'text'] matching HAP-E doc format.

We stream tokens, reconstruct sentences by (doc_id, sentence_id) order and
concatenate into document texts. We write documents in small buffered batches per
category to parquet using pyarrow for efficient appends.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq


def _extract_model_from_doc_id(doc_id: str) -> str:
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
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def rebuild_cap_to_parquet(
    output_dir: Path,
    streaming: bool = True,
    batch_size_docs: int = 1000,
    doc_limit_per_category: Optional[int] = None,
    overwrite: bool = True,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Target categories and filenames
    category_to_filename = {
        "human": "chunk2_human.parquet",
        "gpt-4o": "gpt-4o.parquet",
        "gpt-4o-mini": "gpt-4o-mini.parquet",
        "llama-3-70b-base": "llama-3-70b-base.parquet",
        "llama-3-8b-base": "llama-3-8b-base.parquet",
        "llama-3-70b-instruct": "llama-3-70b-instruct.parquet",
        "llama-3-8b-instruct": "llama-3-8b-instruct.parquet",
    }

    # Temporary paths for atomic writes
    category_to_temp = {k: f"{v}.tmp" for k, v in category_to_filename.items()}

    # Prepare buffers and doc counters
    buffers: Dict[str, List[Dict[str, str]]] = {k: [] for k in category_to_filename}
    counters: Dict[str, int] = {k: 0 for k in category_to_filename}

    # Prepare parquet writers lazily
    writers: Dict[str, pq.ParquetWriter] = {}
    arrow_schema = pa.schema([pa.field("doc_id", pa.string()), pa.field("text", pa.string())])

    def flush(category: str, force_close: bool = False) -> None:
        nonlocal writers
        buf = buffers[category]
        if not buf:
            if force_close and category in writers:
                writers[category].close()
                del writers[category]
            return
        table = pa.Table.from_pandas(pd.DataFrame(buf), schema=arrow_schema, preserve_index=False)
        out_tmp_path = output_dir / category_to_temp[category]
        if category not in writers:
            # Ensure fresh temp file when overwriting
            if overwrite and out_tmp_path.exists():
                out_tmp_path.unlink()
            writers[category] = pq.ParquetWriter(out_tmp_path, arrow_schema)
        writers[category].write_table(table)
        buffers[category].clear()
        if force_close and category in writers:
            writers[category].close()
            del writers[category]

    def finalize_document(doc_id: str, sentences: List[str]) -> None:
        # Determine category
        model = _extract_model_from_doc_id(doc_id)
        cat = _categorize_model(model)
        if cat == "human_chunk_1" or cat == "other":
            return
        if cat not in category_to_filename:
            return
        if doc_limit_per_category is not None and counters[cat] >= doc_limit_per_category:
            return
        text = _postprocess_spacing(" ".join(sentences))
        buffers[cat].append({"doc_id": doc_id, "text": text})
        counters[cat] += 1
        if len(buffers[cat]) >= batch_size_docs:
            flush(cat)
        # update doc progress bar if present
        if doc_pbar is not None:
            doc_pbar.update(1)

    # Optional pre-count of documents to set tqdm total
    doc_total: Optional[int] = None
    doc_pbar: Optional[tqdm] = None

    def precount_docs() -> int:
        # Count documents we will actually write (respecting category filters and optional limits)
        ds_count = load_dataset("browndw/coca-ai-parallel-corpus-spacy", split="train", streaming=streaming)
        prev_doc: Optional[str] = None
        totals: Dict[str, int] = {k: 0 for k in category_to_filename}
        total_docs = 0
        for row in tqdm(ds_count, desc="Precount docs", unit="tok"):
            doc_id = row["doc_id"]
            if prev_doc is None or doc_id != prev_doc:
                model = _extract_model_from_doc_id(doc_id)
                cat = _categorize_model(model)
                if cat in totals:
                    if doc_limit_per_category is None or totals[cat] < doc_limit_per_category:
                        totals[cat] += 1
                        total_docs += 1
                        # Early exit if limits reached for all categories
                        if doc_limit_per_category is not None and all(v >= doc_limit_per_category for v in totals.values()):
                            break
                prev_doc = doc_id
        return total_docs

    if doc_limit_per_category is not None:
        # For limited runs, the pre-count is fast (stops early once all caps reached)
        doc_total = precount_docs()
    else:
        # For full runs, skip pre-count by default to avoid a second full pass
        doc_total = None

    # Stream tokens and reconstruct documents (with robust close/rename)
    ds = load_dataset("browndw/coca-ai-parallel-corpus-spacy", split="train", streaming=streaming)
    current_doc_id: Optional[str] = None
    current_sentence_id: Optional[int] = None
    sentence_tokens: List[str] = []
    sentences: List[str] = []

    # Ensure output files are clean if overwriting
    if overwrite:
        for cat, fname in category_to_filename.items():
            out_final = output_dir / fname
            out_tmp = output_dir / category_to_temp[cat]
            if out_final.exists():
                out_final.unlink()
            if out_tmp.exists():
                out_tmp.unlink()

    try:
        pbar = tqdm(ds, desc="Streaming CAP tokens", unit="tok")
        doc_pbar = tqdm(total=doc_total, desc="Reconstructed docs", unit="doc")
        for row in pbar:
            doc_id = row["doc_id"]
            sent_id = int(row["sentence_id"])  # ensure int
            token = row["token"]

            if current_doc_id is None:
                current_doc_id = doc_id
                current_sentence_id = sent_id
                sentence_tokens = [token]
                continue

            if doc_id != current_doc_id:
                # finalize previous doc
                if sentence_tokens:
                    sentences.append(_postprocess_spacing(" ".join(sentence_tokens)))
                    sentence_tokens = []
                finalize_document(current_doc_id, sentences)
                # reset for new doc
                current_doc_id = doc_id
                current_sentence_id = sent_id
                sentences = []
                sentence_tokens = [token]
                # Early stop if all limits reached
                if doc_limit_per_category is not None and all(counters[c] >= doc_limit_per_category for c in counters):
                    break
                continue

            # Same document
            if current_sentence_id is None:
                current_sentence_id = sent_id

            if sent_id != current_sentence_id:
                sentences.append(_postprocess_spacing(" ".join(sentence_tokens)))
                current_sentence_id = sent_id
                sentence_tokens = [token]
            else:
                sentence_tokens.append(token)

        # Finalize last document
        if current_doc_id is not None:
            if sentence_tokens:
                sentences.append(_postprocess_spacing(" ".join(sentence_tokens)))
            finalize_document(current_doc_id, sentences)
    finally:
        # Flush and close all writers
        for cat in list(category_to_filename.keys()):
            flush(cat, force_close=True)

        # Rename temp files to final atomically
        for cat, fname in category_to_filename.items():
            tmp_path = output_dir / category_to_temp[cat]
            final_path = output_dir / fname
            if tmp_path.exists():
                # If a final file somehow exists, replace it
                if final_path.exists():
                    final_path.unlink()
                tmp_path.rename(final_path)

        if doc_pbar is not None:
            doc_pbar.close()
        try:
            pbar.close()
        except Exception:
            pass

    # Print summary
    print("\nDocuments written per category:")
    for cat, n in counters.items():
        print(f"  {cat:24s}: {n}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild CAP dataset into HAP-E-like doc parquet files")
    parser.add_argument("--output-dir", default=str(Path(__file__).parent.parent / "data"))
    parser.add_argument("--batch-size-docs", type=int, default=1000)
    parser.add_argument("--no-stream", action="store_true", help="Disable HF streaming (not recommended)")
    parser.add_argument("--limit-per-category", type=int, default=None, help="Optional doc cap per category for quick test")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    rebuild_cap_to_parquet(
        output_dir=output_dir,
        streaming=not args.no_stream,
        batch_size_docs=args.batch_size_docs,
        doc_limit_per_category=args.limit_per_category,
    )


if __name__ == "__main__":
    main()


