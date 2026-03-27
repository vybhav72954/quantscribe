"""
Mixed page handler.

Handles pages classified as MIXED — containing BOTH narrative text
and tables on the same page. This is the most common and dangerous
page type in Indian annual reports.

Strategy:
1. Get table bounding boxes from pdfplumber.
2. Extract tables within those bounding boxes.
3. Extract text blocks OUTSIDE table regions via PyMuPDF.
4. Return both as separate content streams for the chunker.
"""

from __future__ import annotations

from typing import Optional

import fitz  # PyMuPDF

from quantscribe.etl.pdf_parser import extract_tables, extract_table_bboxes
from quantscribe.etl.text_cleaner import strip_unicode_garbage
from quantscribe.logging_config import get_logger

logger = get_logger("quantscribe.etl.mixed_page_handler")


def handle_mixed_page(page_number: int, pdf_path: str) -> dict:
    """
    Split a mixed page into separate table and narrative content.

    Args:
        page_number: 0-indexed page number.
        pdf_path:    Path to the PDF file.

    Returns:
        Dict with:
        - "narrative_text": Cleaned text from blocks OUTSIDE table regions.
        - "narrative_blocks": Block dicts with font metadata (for section_detector).
        - "tables": List of tables as list-of-row-dicts (from pdfplumber).
        - "warnings": List of any extraction warnings.
    """
    warnings: list[str] = []

    # ── Step 1: Get table bounding boxes ──
    table_bboxes = extract_table_bboxes(page_number, pdf_path)

    # ── Step 2: Extract tables ──
    tables = extract_tables(page_number, pdf_path, use_camelot_fallback=True)

    if not tables and table_bboxes:
        warnings.append("table_bboxes_found_but_extraction_empty")

    # ── Step 3: Extract narrative text OUTSIDE table regions ──
    narrative_text, narrative_blocks = _extract_narrative_outside_tables(
        page_number, pdf_path, table_bboxes,
    )

    if not narrative_text.strip():
        warnings.append("no_narrative_text_outside_tables")

    logger.info(
        "mixed_page_handled",
        page=page_number + 1,
        tables=len(tables),
        table_bboxes=len(table_bboxes),
        narrative_words=len(narrative_text.split()),
        narrative_blocks=len(narrative_blocks),
    )

    return {
        "narrative_text": narrative_text,
        "narrative_blocks": narrative_blocks,
        "tables": tables,
        "warnings": warnings,
    }


def _extract_narrative_outside_tables(
    page_number: int,
    pdf_path: str,
    table_bboxes: list[tuple],
) -> tuple[str, list[dict]]:
    """
    Extract text blocks whose centroids fall OUTSIDE any table bbox.

    Returns:
        (full_text, structured_blocks) where structured_blocks
        have font metadata for section header detection.
    """
    doc = fitz.open(pdf_path)
    mu_page = doc[page_number]
    page_height = mu_page.rect.height
    dict_data = mu_page.get_text("dict")
    doc.close()

    # ── Compute median font size across whole page ──
    all_sizes: list[float] = []
    for block in dict_data["blocks"]:
        if block["type"] != 0:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                all_sizes.append(span["size"])

    from statistics import median
    median_size = median(all_sizes) if all_sizes else 10.0

    # ── Filter blocks outside table regions ──
    narrative_blocks: list[dict] = []
    paragraphs: list[str] = []

    for block in dict_data["blocks"]:
        if block["type"] != 0:
            continue

        # Check if block centroid is inside any table bbox
        bx0, by0, bx1, by1 = block["bbox"]
        cx = (bx0 + bx1) / 2
        cy = (by0 + by1) / 2

        if _point_in_any_bbox(cx, cy, table_bboxes):
            continue  # Skip — this text is part of a table

        # ── Build block text and metadata ──
        block_text_parts: list[str] = []
        block_sizes: list[float] = []
        block_bold = False

        for line in block["lines"]:
            line_parts: list[str] = []
            for span in line["spans"]:
                line_parts.append(span["text"])
                block_sizes.append(span["size"])
                if "Bold" in span["font"] or "Bd" in span["font"]:
                    block_bold = True
            block_text_parts.append(" ".join(line_parts))

        raw_text = "\n".join(block_text_parts)
        cleaned = strip_unicode_garbage(raw_text)

        if not cleaned.strip():
            continue

        # Skip page numbers and footers (very short, at bottom of page)
        if len(cleaned.split()) <= 3 and by0 > page_height * 0.9:
            continue

        block_font_size = max(block_sizes) if block_sizes else median_size

        narrative_blocks.append({
            "text": cleaned,
            "font_size": block_font_size,
            "median_font_size": median_size,
            "y_position": by0,
            "page_height": page_height,
            "is_bold": block_bold,
        })
        paragraphs.append(cleaned)

    full_text = "\n\n".join(paragraphs)
    return full_text, narrative_blocks


def _point_in_any_bbox(
    x: float,
    y: float,
    bboxes: list[tuple],
) -> bool:
    """Check if point (x, y) falls inside any bounding box."""
    for x0, y0, x1, y1 in bboxes:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return True
    return False
