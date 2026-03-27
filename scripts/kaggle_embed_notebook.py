"""
QuantScribe — Kaggle Notebook Embedding Script
===============================================

HOW TO USE THIS IN A KAGGLE NOTEBOOK
--------------------------------------
1. Upload your two chunk JSON files as a Kaggle Dataset:
   - Go to kaggle.com → Datasets → New Dataset
   - Upload: HDFC_BANK_annual_report_FY25_chunks.json
             SBI_annual_report_FY25_chunks.json
   - Name the dataset: quantscribe-chunks

2. Create a new Kaggle Notebook:
   - Add the dataset above as input
   - Set accelerator to GPU (T4 is fine; H100 is faster but not required
     for ~10k–15k chunks — T4 handles it in ~10 mins)

3. Paste each cell below into separate notebook cells and run in order.

4. After the final cell, go to the notebook's Output section and download:
   - *.faiss files  →  place in  indices/active/
   - *_metadata.json files  →  place in  indices/active/
   Then upload everything to the shared Google Drive link.

ESTIMATED TIME:
   T4  GPU: ~8–15 min per 15 000 chunks
   H100 GPU: ~2–3 min per 15 000 chunks
   No GPU (CPU): ~45–90 min (not recommended)
"""

# ════════════════════════════════════════════════════════════════════
# CELL 1 — Install dependencies
# (Kaggle already has faiss-gpu and sentence-transformers, but pin versions)
# ════════════════════════════════════════════════════════════════════

# Paste this into Cell 1 and run it:
"""
!pip install -q sentence-transformers==2.7.0 faiss-gpu tqdm
"""

# ════════════════════════════════════════════════════════════════════
# CELL 2 — Imports and configuration
# ════════════════════════════════════════════════════════════════════

import json
import os
import glob
import time

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# ── Configuration ──
# If you named your Kaggle dataset something different, update INPUT_DIR below.
INPUT_DIR = "/kaggle/input/quantscribe-chunks"
OUTPUT_DIR = "/kaggle/working"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384   # Fixed for all-MiniLM-L6-v2
BATCH_SIZE = 128      # Larger batches are faster on GPU; reduce to 64 if OOM

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=== QuantScribe Kaggle Embedding Script ===")
print(f"Input  dir : {INPUT_DIR}")
print(f"Output dir : {OUTPUT_DIR}")
print(f"Batch size : {BATCH_SIZE}")

# ════════════════════════════════════════════════════════════════════
# CELL 3 — Load embedding model
# ════════════════════════════════════════════════════════════════════

print(f"\nLoading model: {MODEL_NAME}")
t0 = time.time()
model = SentenceTransformer(MODEL_NAME)

# Detect device
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
print(f"Model loaded in {time.time() - t0:.1f}s  |  Device: {device}")

if device == "cpu":
    print("⚠️  WARNING: No GPU detected. Embedding will be slow.")
    print("   Make sure you enabled a GPU accelerator in Notebook Settings.")

# ════════════════════════════════════════════════════════════════════
# CELL 4 — Discover chunk files
# ════════════════════════════════════════════════════════════════════

chunk_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*_chunks.json")))

if not chunk_files:
    raise FileNotFoundError(
        f"No *_chunks.json files found in {INPUT_DIR}.\n"
        "Make sure you added the Kaggle dataset and the files are named correctly.\n"
        f"Contents of {INPUT_DIR}: {os.listdir(INPUT_DIR)}"
    )

print(f"\nFound {len(chunk_files)} chunk file(s):")
for f in chunk_files:
    size_mb = os.path.getsize(f) / 1_048_576
    print(f"  {os.path.basename(f)}  ({size_mb:.1f} MB)")

# ════════════════════════════════════════════════════════════════════
# CELL 5 — Embed each file and build FAISS index
# ════════════════════════════════════════════════════════════════════

results_summary = []

for chunk_file in chunk_files:
    file_name  = os.path.basename(chunk_file)
    index_name = file_name.replace("_chunks.json", "")

    print(f"\n{'='*60}")
    print(f"Processing : {file_name}")
    print(f"Index name : {index_name}")

    # ── Load chunks ──
    with open(chunk_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"Chunks loaded : {len(chunks)}")

    if not chunks:
        print("  ⚠️  No chunks found — skipping.")
        continue

    # ── Validate chunk structure ──
    sample = chunks[0]
    if "content" not in sample:
        raise KeyError(
            f"Chunk is missing 'content' key. Keys found: {list(sample.keys())}\n"
            "Make sure you ran run_etl.py with the current codebase."
        )

    texts    = [c["content"]  for c in chunks]
    metadata = [c["metadata"] for c in chunks]

    # ── Embed in batches (with progress bar) ──
    print(f"Embedding {len(texts)} chunks (batch_size={BATCH_SIZE})…")
    t0 = time.time()

    # encode() handles batching internally when batch_size is set
    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,   # L2 norm → enables cosine sim via IndexFlatIP
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    embeddings = embeddings.astype(np.float32)

    elapsed = time.time() - t0
    throughput = len(texts) / elapsed
    print(f"Embedding done in {elapsed:.1f}s  ({throughput:.0f} chunks/sec)")
    print(f"Embedding matrix shape : {embeddings.shape}")

    # ── Sanity check: embedding dimension ──
    if embeddings.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Expected embedding dim {EMBEDDING_DIM}, got {embeddings.shape[1]}.\n"
            "Update EMBEDDING_DIM at the top of this script."
        )

    # ── Build FAISS index ──
    # IndexFlatIP = exact inner product search.
    # With L2-normalised vectors, inner product == cosine similarity.
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    print(f"FAISS index built : {index.ntotal} vectors")

    # ── Sanity check — top-1 for first chunk should score ~1.0 ──
    query_vec = model.encode([texts[0]], normalize_embeddings=True).astype(np.float32)
    scores, idx = index.search(query_vec, 3)
    print(f"Sanity check (top-3 scores for chunk[0]): {scores[0].tolist()}")
    if scores[0][0] < 0.99:
        print("  ⚠️  Top-1 score < 0.99 — normalization may have failed.")

    # ── Save FAISS index and metadata ──
    faiss_path = os.path.join(OUTPUT_DIR, f"{index_name}.faiss")
    meta_path  = os.path.join(OUTPUT_DIR, f"{index_name}_metadata.json")

    faiss.write_index(index, faiss_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    faiss_mb = os.path.getsize(faiss_path) / 1_048_576
    meta_mb  = os.path.getsize(meta_path)  / 1_048_576

    print(f"Saved FAISS index : {faiss_path}  ({faiss_mb:.1f} MB)")
    print(f"Saved metadata    : {meta_path}   ({meta_mb:.1f} MB)")

    results_summary.append({
        "index_name": index_name,
        "chunks": len(chunks),
        "elapsed_s": round(elapsed, 1),
        "faiss_mb": round(faiss_mb, 1),
    })

# ════════════════════════════════════════════════════════════════════
# CELL 6 — Final summary and download instructions
# ════════════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print("ALL DONE ✓")
print(f"{'='*60}\n")

for r in results_summary:
    print(f"  {r['index_name']}")
    print(f"    Chunks  : {r['chunks']}")
    print(f"    Time    : {r['elapsed_s']}s")
    print(f"    FAISS   : {r['faiss_mb']} MB")
    print()

print("FILES TO DOWNLOAD (from the Output tab on the right):")
output_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "*")))
for f in output_files:
    print(f"  {os.path.basename(f)}")

print("""
NEXT STEPS:
  1. Download all .faiss and _metadata.json files from the Output tab.
  2. Place them in your local project:
       indices/active/HDFC_BANK_annual_report_FY25.faiss
       indices/active/HDFC_BANK_annual_report_FY25_metadata.json
       indices/active/SBI_annual_report_FY25.faiss
       indices/active/SBI_annual_report_FY25_metadata.json
  3. Upload ALL files to the shared Google Drive link.
  4. Run the retrieval test:
       python tests/test_retrieval.py
     (from the project root, with your venv active)
""")
