"""
Table chunker.

Keeps tables as atomic units. If a table exceeds the token limit,
splits at row boundaries and repeats column headers in every sub-chunk.

Rules:
1. NEVER split mid-row.
2. If the table fits within max_tokens, keep it as one chunk.
3. If it exceeds max_tokens, split at row boundaries.
4. ALWAYS prepend column headers to every sub-chunk.
"""

from __future__ import annotations

from typing import Optional

from quantscribe.schemas.etl import ChunkMetadata, PageType, TextChunk
from quantscribe.config import get_settings
from quantscribe.logging_config import get_logger

logger = get_logger("quantscribe.chunking.table")


def chunk_table(
    table_data: list[dict],
    bank_name: str,
    document_type: str,
    fiscal_year: str,
    page_number: int,
    section_header: Optional[str] = None,
    max_tokens: Optional[int] = None,
    chunk_index_start: int = 0,
) -> list[TextChunk]:
    """
    Chunk a table into atomic units with header repetition.

    Args:
        table_data: Table as list-of-dicts (each dict = one row, keys = column headers).
        bank_name: Standardized bank identifier.
        document_type: Type of source document.
        fiscal_year: Fiscal year in FYxx format.
        page_number: Source page number (1-indexed).
        section_header: Detected section header, if any.
        max_tokens: Maximum tokens per chunk (default from config).
        chunk_index_start: Starting chunk index.

    Returns:
        List of TextChunk objects ready for embedding.
    """
    settings = get_settings()
    max_tok = max_tokens or settings.table_max_tokens

    if not table_data:
        logger.warn("empty_table", page=page_number, bank=bank_name)
        return []

    headers = list(table_data[0].keys())
    header_line = " | ".join(str(h) for h in headers)
    separator = "-" * len(header_line)

    # Convert entire table to text
    row_lines: list[str] = []
    for row in table_data:
        row_line = " | ".join(str(row.get(h, "")) for h in headers)
        row_lines.append(row_line)

    full_text = "\n".join([header_line, separator] + row_lines)

    # Check if it fits in one chunk
    if len(full_text.split()) <= max_tok:
        chunk_text = _prepend_section(full_text, section_header)
        metadata = _build_table_metadata(
            bank_name, document_type, fiscal_year,
            page_number, section_header, chunk_index_start, chunk_text,
        )
        logger.info("table_single_chunk", page=page_number, bank=bank_name, rows=len(table_data))
        return [TextChunk(
            content=chunk_text,
            metadata=metadata,
            content_type="table_structured",
        )]

    # Split at row boundaries with header repetition
    chunks: list[TextChunk] = []
    current_rows: list[str] = []
    header_tokens = len(header_line.split()) + 1  # header + separator
    current_token_count = header_tokens
    chunk_idx = chunk_index_start

    for row_line in row_lines:
        row_tokens = len(row_line.split())

        if current_token_count + row_tokens > max_tok and current_rows:
            # Flush current chunk
            chunk_text = "\n".join([header_line, separator] + current_rows)
            chunk_text = _prepend_section(chunk_text, section_header)
            metadata = _build_table_metadata(
                bank_name, document_type, fiscal_year,
                page_number, section_header, chunk_idx, chunk_text,
            )
            chunks.append(TextChunk(
                content=chunk_text,
                metadata=metadata,
                content_type="table_structured",
            ))

            current_rows = []
            current_token_count = header_tokens
            chunk_idx += 1

        current_rows.append(row_line)
        current_token_count += row_tokens

    # Flush remaining rows
    if current_rows:
        chunk_text = "\n".join([header_line, separator] + current_rows)
        chunk_text = _prepend_section(chunk_text, section_header)
        metadata = _build_table_metadata(
            bank_name, document_type, fiscal_year,
            page_number, section_header, chunk_idx, chunk_text,
        )
        chunks.append(TextChunk(
            content=chunk_text,
            metadata=metadata,
            content_type="table_structured",
        ))

    logger.info(
        "table_split",
        page=page_number,
        bank=bank_name,
        total_rows=len(table_data),
        num_chunks=len(chunks),
    )
    return chunks


def _prepend_section(text: str, section_header: Optional[str]) -> str:
    """Prepend section header to table text if available."""
    if section_header:
        return f"[Section: {section_header}]\n{text}"
    return text


def _build_table_metadata(
    bank_name: str,
    document_type: str,
    fiscal_year: str,
    page_number: int,
    section_header: Optional[str],
    chunk_index: int,
    chunk_text: str,
) -> ChunkMetadata:
    """Build a ChunkMetadata envelope for a table chunk."""
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
        page_type=PageType.TABULAR,
        chunk_index=chunk_index,
        token_count=len(chunk_text.split()),
        parse_version=settings.parse_version,
    )
