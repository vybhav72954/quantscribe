"""
ETL Pipeline Orchestrator.

Takes a PDF path + bank metadata and produces a list of TextChunk
objects ready for embedding. This is the single entry point for
converting a raw PDF into chunked, metadata-tagged content.

Usage:
    from quantscribe.etl.pipeline import run_etl_pipeline

    chunks = run_etl_pipeline(
        pdf_path="data/pdfs/HDFC_Bank_Annual_Report_FY25.pdf",
        bank_name="HDFC_BANK",
        document_type="annual_report",
        fiscal_year="FY25",
    )
    # chunks is a list of TextChunk objects ready for embedding
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF — used to get total page count

from quantscribe.etl.page_classifier import classify_page
from quantscribe.etl.pdf_parser import extract_narrative, extract_tables
from quantscribe.etl.mixed_page_handler import handle_mixed_page
from quantscribe.etl.section_detector import detect_section_header
from quantscribe.chunking.narrative_chunker import chunk_narrative
from quantscribe.chunking.table_chunker import chunk_table
from quantscribe.schemas.etl import PageType, TextChunk
from quantscribe.logging_config import get_logger

logger = get_logger("quantscribe.etl.pipeline")


def run_etl_pipeline(
    pdf_path: str,
    bank_name: str,
    document_type: str,
    fiscal_year: str,
    page_range: Optional[tuple[int, int]] = None,
) -> list[TextChunk]:
    """
    End-to-end ETL pipeline: PDF → classified pages → extracted content → chunks.

    Args:
        pdf_path:      Path to the PDF file.
        bank_name:     Bank identifier (e.g., "HDFC_BANK", "SBI").
        document_type: One of "annual_report", "earnings_call", "investor_presentation".
        fiscal_year:   Fiscal year in FYxx format (e.g., "FY25").
        page_range:    Optional (start, end) 0-indexed page range.
                       If None, processes all pages.

    Returns:
        List of TextChunk objects with full metadata, ready for embedding.
    """
    start_time = time.time()

    logger.info(
        "pipeline_start",
        pdf=pdf_path,
        bank=bank_name,
        doc_type=document_type,
        fy=fiscal_year,
    )

    # ── Step 1: Determine page range ──
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    doc.close()

    if page_range:
        start, end = page_range
        page_indices = list(range(start, min(end, total_pages)))
    else:
        page_indices = list(range(total_pages))

    # ── Classify pages one at a time (avoids loading all 500+ pages at once) ──
    pages_to_process = []
    for pg_idx in page_indices:
        try:
            pages_to_process.append(classify_page(pg_idx, pdf_path))
        except Exception as e:
            logger.error("classification_failed", page=pg_idx + 1, error=str(e))

    # ── Step 2: Process each page based on classification ──
    all_chunks: list[TextChunk] = []
    last_known_header: Optional[str] = None
    stats = {"narrative": 0, "tabular": 0, "mixed": 0, "graphical": 0, "errors": 0}

    from tqdm import tqdm
    for parsed_page in tqdm(pages_to_process, desc=f"ETL {bank_name}", unit="page"):
        page_idx = parsed_page.page_number - 1  # Convert back to 0-indexed
        page_type = parsed_page.page_type

        try:
            if page_type == PageType.GRAPHICAL:
                stats["graphical"] += 1
                continue

            elif page_type == PageType.NARRATIVE:
                chunks, header = _process_narrative_page(
                    page_idx, pdf_path, bank_name, document_type,
                    fiscal_year, last_known_header,
                )
                if header:
                    last_known_header = header
                all_chunks.extend(chunks)
                stats["narrative"] += 1

            elif page_type == PageType.TABULAR:
                chunks = _process_tabular_page(
                    page_idx, pdf_path, bank_name, document_type,
                    fiscal_year, last_known_header,
                )
                all_chunks.extend(chunks)
                stats["tabular"] += 1

            elif page_type == PageType.MIXED:
                chunks, header = _process_mixed_page(
                    page_idx, pdf_path, bank_name, document_type,
                    fiscal_year, last_known_header,
                )
                if header:
                    last_known_header = header
                all_chunks.extend(chunks)
                stats["mixed"] += 1

        except Exception as e:
            stats["errors"] += 1
            logger.error(
                "page_processing_failed",
                page=parsed_page.page_number,
                page_type=page_type.value,
                error=str(e),
            )
            continue  # Never silently drop — log and continue

    elapsed = time.time() - start_time

    logger.info(
        "pipeline_complete",
        bank=bank_name,
        fy=fiscal_year,
        total_pages=len(pages_to_process),
        total_chunks=len(all_chunks),
        stats=stats,
        elapsed_seconds=round(elapsed, 1),
    )

    return all_chunks


def save_chunks_to_json(chunks: list[TextChunk], output_path: str) -> None:
    """
    Save chunks to a JSON file for upload to Kaggle for embedding.

    The JSON format matches what scripts/kaggle_embed.py expects.
    """
    data = [
        {
            "content": chunk.content,
            "metadata": chunk.metadata.model_dump(),
            "content_type": chunk.content_type,
        }
        for chunk in chunks
    ]

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("chunks_saved", path=output_path, count=len(chunks))


# ── Internal Page Processors ──


def _process_narrative_page(
    page_idx: int,
    pdf_path: str,
    bank_name: str,
    document_type: str,
    fiscal_year: str,
    last_known_header: Optional[str],
) -> tuple[list[TextChunk], Optional[str]]:
    """Extract and chunk a NARRATIVE page. Returns (chunks, detected_header)."""
    result = extract_narrative(page_idx, pdf_path)

    if not result["text"].strip():
        return [], None

    # Detect section header from font metadata
    header = detect_section_header(result["blocks"], page_idx + 1)
    section = header or last_known_header

    chunks = chunk_narrative(
        text=result["text"],
        bank_name=bank_name,
        document_type=document_type,
        fiscal_year=fiscal_year,
        page_number=page_idx + 1,
        section_header=section,
    )

    return chunks, header


def _process_tabular_page(
    page_idx: int,
    pdf_path: str,
    bank_name: str,
    document_type: str,
    fiscal_year: str,
    last_known_header: Optional[str],
) -> list[TextChunk]:
    """Extract and chunk a TABULAR page."""
    tables = extract_tables(page_idx, pdf_path)

    all_chunks: list[TextChunk] = []
    chunk_offset = 0

    for table_data in tables:
        if not table_data:
            continue
        chunks = chunk_table(
            table_data=table_data,
            bank_name=bank_name,
            document_type=document_type,
            fiscal_year=fiscal_year,
            page_number=page_idx + 1,
            section_header=last_known_header,
            chunk_index_start=chunk_offset,
        )
        all_chunks.extend(chunks)
        chunk_offset += len(chunks)

    return all_chunks


def _process_mixed_page(
    page_idx: int,
    pdf_path: str,
    bank_name: str,
    document_type: str,
    fiscal_year: str,
    last_known_header: Optional[str],
) -> tuple[list[TextChunk], Optional[str]]:
    """
    Extract and chunk a MIXED page. Returns (chunks, detected_header).

    If the mixed page handler fails for any reason (malformed PDF objects,
    pdfminer parse errors, etc.), degrades gracefully to pure narrative
    extraction rather than crashing the whole pipeline.
    """
    try:
        result = handle_mixed_page(page_idx, pdf_path)
    except Exception as e:
        logger.error(
            "mixed_page_handler_failed",
            page=page_idx + 1,
            error=str(e)[:300],
            fallback="degrading_to_narrative",
        )
        # Degrade gracefully — treat the whole page as narrative
        return _process_narrative_page(
            page_idx, pdf_path, bank_name, document_type,
            fiscal_year, last_known_header,
        )

    # Detect section header from narrative blocks
    header = detect_section_header(result["narrative_blocks"], page_idx + 1)
    section = header or last_known_header

    all_chunks: list[TextChunk] = []
    chunk_offset = 0

    # Chunk narrative content
    if result["narrative_text"].strip():
        n_chunks = chunk_narrative(
            text=result["narrative_text"],
            bank_name=bank_name,
            document_type=document_type,
            fiscal_year=fiscal_year,
            page_number=page_idx + 1,
            section_header=section,
        )
        all_chunks.extend(n_chunks)
        chunk_offset += len(n_chunks)

    # Chunk table content
    for table_data in result["tables"]:
        if not table_data:
            continue
        t_chunks = chunk_table(
            table_data=table_data,
            bank_name=bank_name,
            document_type=document_type,
            fiscal_year=fiscal_year,
            page_number=page_idx + 1,
            section_header=section,
            chunk_index_start=chunk_offset,
        )
        all_chunks.extend(t_chunks)
        chunk_offset += len(t_chunks)

    return all_chunks, header
