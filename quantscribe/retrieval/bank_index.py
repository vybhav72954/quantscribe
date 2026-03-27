"""
BankIndex: FAISS index wrapper for a single bank-document-year combination.

Each bank gets its own index to prevent cross-entity contamination.
Index naming convention: {bank_name}_{doc_type}_{fiscal_year}
Example: HDFC_BANK_annual_report_FY25
"""

from __future__ import annotations

import json
from pathlib import Path

import faiss
import numpy as np

from quantscribe.schemas.etl import TextChunk
from quantscribe.logging_config import get_logger

logger = get_logger("quantscribe.retrieval.bank_index")


class BankIndex:
    """
    Wraps a single FAISS IndexFlatIP for one bank-document-year combination.

    The metadata_store is a parallel array: metadata_store[i] corresponds to vector[i].

    IMPORTANT: Each metadata_store entry includes the chunk's text content under
    the key "content". This is required for the LLM extraction step — without it,
    the pipeline would have no text to send to Gemini.
    """

    def __init__(self, index_name: str, dimension: int = 384):
        self.index_name = index_name
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata_store: list[dict] = []

    @property
    def size(self) -> int:
        """Number of vectors in this index."""
        return self.index.ntotal

    def add(self, embeddings: np.ndarray, chunks: list[TextChunk]) -> None:
        """
        Add vectors with their metadata AND content.

        Args:
            embeddings: L2-normalized vectors, shape (n, dimension).
            chunks: List of TextChunk objects (same length as embeddings).
                    Both metadata and content are stored so the LLM step
                    has actual text to work with.
        """
        assert len(embeddings) == len(chunks), (
            f"Embedding count ({len(embeddings)}) != chunk count ({len(chunks)})"
        )
        assert embeddings.shape[1] == self.dimension, (
            f"Embedding dimension ({embeddings.shape[1]}) != expected ({self.dimension})"
        )

        self.index.add(embeddings)

        for chunk in chunks:
            # Store metadata fields + content in one dict.
            # content MUST be here — peer_comparison._format_bank_context() reads it.
            entry = chunk.metadata.model_dump()
            entry["content"] = chunk.content
            entry["content_type"] = chunk.content_type
            self.metadata_store.append(entry)

        logger.info(
            "vectors_added",
            index=self.index_name,
            count=len(embeddings),
            total=self.size,
        )

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Search this bank's index.

        Args:
            query_vector: L2-normalized query vector, shape (1, dimension).
            top_k: Number of results to return.

        Returns:
            List of dicts with 'metadata' (ChunkMetadata fields + content)
            and 'score' (cosine similarity).
        """
        if self.size == 0:
            logger.warn("search_empty_index", index=self.index_name)
            return []

        scores, indices = self.index.search(query_vector, min(top_k, self.size))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "metadata": self.metadata_store[idx],
                "score": float(score),
            })
        return results

    def save(self, directory: str | Path) -> None:
        """Persist index and metadata (including content) to disk."""
        dir_path = Path(directory)
        dir_path.mkdir(parents=True, exist_ok=True)

        index_path = dir_path / f"{self.index_name}.faiss"
        meta_path = dir_path / f"{self.index_name}_metadata.json"

        faiss.write_index(self.index, str(index_path))
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata_store, f, indent=2, ensure_ascii=False)

        logger.info(
            "index_saved",
            index=self.index_name,
            vectors=self.size,
            path=str(dir_path),
        )

    def load(self, directory: str | Path) -> None:
        """Load index and metadata (including content) from disk."""
        dir_path = Path(directory)

        index_path = dir_path / f"{self.index_name}.faiss"
        meta_path = dir_path / f"{self.index_name}_metadata.json"

        self.index = faiss.read_index(str(index_path))
        with open(meta_path, "r", encoding="utf-8") as f:
            self.metadata_store = json.load(f)

        logger.info("index_loaded", index=self.index_name, vectors=self.size)
        