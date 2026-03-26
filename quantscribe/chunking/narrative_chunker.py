"""
Narrative text chunker with overlap.

Splits narrative text into chunks on sentence boundaries with configurable
overlap. Every chunk gets a full ChunkMetadata envelope.

Rules:
1. NEVER split mid-sentence.
2. Target chunk_size_words per chunk with overlap_words overlap.
3. Prepend section_header to every chunk.
4. Attach full ChunkMetadata to every chunk.
"""

from __future__ import annotations

import re
from typing import Optional

from quantscribe.schemas.etl import ChunkMetadata, PageType, TextChunk
from quantscribe.config import get_settings
from quantscribe.logging_config import get_logger

logger = get_logger("quantscribe.chunking.narrative")

# Sentence boundary regex — handles abbreviations common in financial text
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def chunk_narrative(
    text: str,
    bank_name: str,
    document_type: str,
    fiscal_year: str,
    page_number: int,
    section_header: Optional[str] = None,
    chunk_size_words: Optional[int] = None,
    overlap_words: Optional[int] = None,
    chunk_index_start: int = 0,
) -> list[TextChunk]:
    """
    Chunk narrative text with overlap on sentence boundaries.

    Args:
        text: The narrative text to chunk.
        bank_name: Standardized bank identifier.
        document_type: Type of source document.
        fiscal_year: Fiscal year in FYxx format.
        page_number: Source page number (1-indexed).
        section_header: Detected section header, if any.
        chunk_size_words: Target words per chunk (default from config).
        overlap_words: Overlap words between chunks (default from config).
        chunk_index_start: Starting chunk index (for pages with multiple content types).

    Returns:
        List of TextChunk objects ready for embedding.
    """
    settings = get_settings()
    chunk_size = chunk_size_words or settings.narrative_chunk_size_words
    overlap = overlap_words or settings.narrative_overlap_words

    # Split on sentence boundaries
    sentences = SENTENCE_SPLIT_RE.split(text)
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        logger.warn("empty_narrative_text", page=page_number, bank=bank_name)
        return []

    chunks: list[TextChunk] = []
    current_sentences: list[str] = []
    current_word_count = 0
    chunk_index = chunk_index_start

    for sentence in sentences:
        word_count = len(sentence.split())
        current_sentences.append(sentence)
        current_word_count += word_count

        if current_word_count >= chunk_size:
            chunk_text = _build_chunk_text(current_sentences, section_header)
            metadata = _build_metadata(
                bank_name, document_type, fiscal_year,
                page_number, section_header, chunk_index, chunk_text,
            )

            chunks.append(TextChunk(
                content=chunk_text,
                metadata=metadata,
                content_type="narrative",
            ))

            # Overlap: keep last N words worth of sentences
            current_sentences = _compute_overlap(current_sentences, overlap)
            current_word_count = sum(len(s.split()) for s in current_sentences)
            chunk_index += 1

    # Flush remaining sentences
    if current_sentences:
        chunk_text = _build_chunk_text(current_sentences, section_header)
        # Only create chunk if it meets minimum length
        if len(chunk_text) >= 10:
            metadata = _build_metadata(
                bank_name, document_type, fiscal_year,
                page_number, section_header, chunk_index, chunk_text,
            )
            chunks.append(TextChunk(
                content=chunk_text,
                metadata=metadata,
                content_type="narrative",
            ))

    logger.info(
        "narrative_chunked",
        page=page_number,
        bank=bank_name,
        num_chunks=len(chunks),
        total_words=len(text.split()),
    )
    return chunks


def _build_chunk_text(sentences: list[str], section_header: Optional[str]) -> str:
    """Join sentences and prepend section header if available."""
    text = " ".join(sentences)
    if section_header:
        text = f"[Section: {section_header}]\n{text}"
    return text


def _build_metadata(
    bank_name: str,
    document_type: str,
    fiscal_year: str,
    page_number: int,
    section_header: Optional[str],
    chunk_index: int,
    chunk_text: str,
) -> ChunkMetadata:
    """Build a ChunkMetadata envelope for a narrative chunk."""
    settings = get_settings()
    return ChunkMetadata(
        chunk_id=ChunkMetadata.generate_chunk_id(
            bank_name, document_type, fiscal_year, page_number, chunk_index,
        ),
        bank_name=bank_name,
        document_type=document_type,
        fiscal_year=fiscal_year,
        page_number=page_number,
        section_header=section_header,
        page_type=PageType.NARRATIVE,
        chunk_index=chunk_index,
        token_count=len(chunk_text.split()),  # Approximate
        parse_version=settings.parse_version,
    )


def _compute_overlap(sentences: list[str], overlap_words: int) -> list[str]:
    """Keep the last N words worth of sentences for overlap."""
    overlap_sentences: list[str] = []
    overlap_count = 0
    for s in reversed(sentences):
        overlap_count += len(s.split())
        overlap_sentences.insert(0, s)
        if overlap_count >= overlap_words:
            break
    return overlap_sentences
