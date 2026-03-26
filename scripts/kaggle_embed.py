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
    ├── HDFC_BANK_annual_report_FY24_chunks.json
    ├── SBI_annual_report_FY24_chunks.json
    └── ...

OUTPUTS (download from notebook output):
    /kaggle/working/
    ├── HDFC_BANK_annual_report_FY24.faiss
    ├── HDFC_BANK_annual_report_FY24_metadata.json
    ├── SBI_annual_report_FY24.faiss
    ├── SBI_annual_report_FY24_metadata.json
    └── ...

ESTIMATED TIME:
    ~2-3 minutes per 15,000 chunks on H100
    ~15-20 minutes per 15,000 chunks on T4
"""

import json
import os
import glob
import time

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ── Configuration ──
INPUT_DIR = "/kaggle/input/quantscribe-chunks"
OUTPUT_DIR = "/kaggle/working"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
BATCH_SIZE = 64

# ── Load Model ──
print(f"Loading embedding model: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
print(f"Model loaded. Dimension: {EMBEDDING_DIM}")

# ── Process Each Chunk File ──
chunk_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_chunks.json")))
print(f"Found {len(chunk_files)} chunk files to process.")

for chunk_file in chunk_files:
    file_name = os.path.basename(chunk_file)
    index_name = file_name.replace("_chunks.json", "")

    print(f"\n{'='*60}")
    print(f"Processing: {file_name}")
    print(f"Index name: {index_name}")

    # Load chunks
    with open(chunk_file, "r") as f:
        chunks = json.load(f)

    print(f"  Loaded {len(chunks)} chunks")

    # Extract texts and metadata
    texts = [c["content"] for c in chunks]
    metadata = [c["metadata"] for c in chunks]

    # Embed
    print(f"  Embedding {len(texts)} chunks (batch_size={BATCH_SIZE})...")
    start_time = time.time()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,  # L2 normalization for IndexFlatIP
        show_progress_bar=True,
    )
    embeddings = embeddings.astype(np.float32)

    elapsed = time.time() - start_time
    print(f"  Embedding complete in {elapsed:.1f}s ({len(texts)/elapsed:.0f} chunks/sec)")

    # Build FAISS index
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    print(f"  FAISS index built: {index.ntotal} vectors")

    # Save
    faiss_path = os.path.join(OUTPUT_DIR, f"{index_name}.faiss")
    meta_path = os.path.join(OUTPUT_DIR, f"{index_name}_metadata.json")

    faiss.write_index(index, faiss_path)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Saved: {faiss_path}")
    print(f"  Saved: {meta_path}")

    # Quick sanity check — search for first chunk's content
    query = model.encode([texts[0]], normalize_embeddings=True).astype(np.float32)
    scores, indices = index.search(query, 3)
    print(f"  Sanity check — top-3 scores for first chunk: {scores[0].tolist()}")
    print(f"  (Score of ~1.0 for first result confirms correct indexing)")

print(f"\n{'='*60}")
print(f"DONE. Download all .faiss and _metadata.json files from {OUTPUT_DIR}")
print(f"Place them in your local indices/v1.0.0/ directory.")
