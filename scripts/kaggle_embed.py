"""
QuantScribe — Kaggle H100 Embedding Script

PURPOSE:
    Run this on Kaggle with GPU (H100/T4/P100) to batch-embed all chunks
    and build per-bank FAISS indices. Download the outputs to your local machine.

USAGE:
    1. Upload your chunked JSON files as a Kaggle dataset
    2. Copy this script into a Kaggle notebook
    3. Run with GPU accelerator enabled
    4. Download the .faiss and _metadata.json files from output

INPUTS (upload as Kaggle dataset):
    /kaggle/input/quantscribe-chunks/
    ├── HDFC_BANK_annual_report_FY25_chunks.json
    ├── SBI_annual_report_FY25_chunks.json
    └── ...

OUTPUTS (download from notebook output):
    /kaggle/working/
    ├── HDFC_BANK_annual_report_FY25.faiss
    ├── HDFC_BANK_annual_report_FY25_metadata.json   ← includes "content" field
    ├── SBI_annual_report_FY25.faiss
    ├── SBI_annual_report_FY25_metadata.json
    └── ...

ESTIMATED TIME:
    ~2-3 minutes per 15,000 chunks on H100
    ~15-20 minutes per 15,000 chunks on T4

NOTE ON METADATA FORMAT:
    Each entry in _metadata.json is a ChunkMetadata dict PLUS two extra keys:
        "content":      the raw chunk text (required by the LLM extraction step)
        "content_type": "narrative" | "table_text" | "table_structured"
    Without "content", Gemini receives empty context and hallucinations follow.
"""

import json
import os
import glob
import time

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Update this to match your Kaggle dataset slug ──
# e.g. if your dataset URL is kaggle.com/datasets/vybhav72954/quantscribe-chunks
# then the slug is "quantscribe-chunks"
INPUT_DIR = "/kaggle/input/datasets/vybhavchaturvedi/quantscribe-chunks"
OUTPUT_DIR = "/kaggle/working"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 64

# Verify files are visible
chunk_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_chunks.json")))
print(f"Found {len(chunk_files)} chunk files:")
for f in chunk_files:
    size_mb = os.path.getsize(f) / 1024 / 1024
    print(f"  {os.path.basename(f)} ({size_mb:.1f} MB)")


print(f"Loading: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
print("Model ready.")


for chunk_file in chunk_files:
    file_name = os.path.basename(chunk_file)
    index_name = file_name.replace("_chunks.json", "")

    print(f"\n{'='*60}")
    print(f"Processing: {file_name}")

    with open(chunk_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"  Loaded {len(chunks)} chunks")

    texts = [c["content"] for c in chunks]

    metadata_with_content = []
    for c in chunks:
        entry = c["metadata"].copy()
        entry["content"] = c["content"]
        entry["content_type"] = c["content_type"]
        metadata_with_content.append(entry)

    print(f"  Embedding {len(texts)} chunks...")
    start_time = time.time()
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    embeddings = embeddings.astype(np.float32)
    elapsed = time.time() - start_time
    print(f"  Done in {elapsed:.1f}s ({len(texts)/elapsed:.0f} chunks/sec)")

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    print(f"  FAISS index: {index.ntotal} vectors")

    faiss_path = os.path.join(OUTPUT_DIR, f"{index_name}.faiss")
    meta_path = os.path.join(OUTPUT_DIR, f"{index_name}_metadata.json")

    faiss.write_index(index, faiss_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata_with_content, f, indent=2, ensure_ascii=False)

    print(f"  Saved: {faiss_path}")
    print(f"  Saved: {meta_path}")

print(f"\nDONE. Files are in {OUTPUT_DIR}")

# Verify each index loads and content is present
for chunk_file in chunk_files:
    index_name = os.path.basename(chunk_file).replace("_chunks.json", "")
    meta_path = os.path.join(OUTPUT_DIR, f"{index_name}_metadata.json")
    faiss_path = os.path.join(OUTPUT_DIR, f"{index_name}.faiss")

    idx = faiss.read_index(faiss_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    has_content = all("content" in m and m["content"] for m in meta[:10])
    print(f"{index_name}:")
    print(f"  Vectors: {idx.ntotal}")
    print(f"  Content stored: {has_content}")
    print(f"  Sample: {meta[0]['content'][:100]}...")
    