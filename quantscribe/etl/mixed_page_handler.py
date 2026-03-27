"""
Mixed page handler.

Handles pages that contain BOTH narrative text and tables.
Strategy:
1. Use pdfplumber to detect table bounding boxes + extract raw table data.
2. Use PyMuPDF to extract text blocks OUTSIDE table regions.
3. Return both as structured dicts for the pipeline to chunk separately.

Returns a dict with:
    - "narrative_text":   str  — text from outside table regions
    - "narrative_blocks": list — structured block dicts (for section_detector)
    - "tables":           list — list of table-as-list-of-dicts
"""

from __future__ import annotations

from statistics import median

import fitz  # PyMuPDF
import pdfplumber

from quantscribe.etl.text_cleaner import strip_unicode_garbage, clean_table_cell
from quantscribe.logging_config import get_logger

logger = get_logger("quantscribe.etl.mixed_page_handler")


def handle_mixed_page(page_number: int, pdf_path: str) -> dict:
    """
    Split a mixed page into table and narrative sub-regions.

    Args:
        page_number: 0-indexed page number.
        pdf_path:    Absolute path to the PDF file.

    Returns:
        Dict with keys:
        - "narrative_text":   str  — cleaned text outside all table bboxes
        - "narrative_blocks": list — block dicts with font metadata (for section_detector)
        - "tables":           list[list[dict]] — one list-of-dicts per detected table
    """
    # ── Pass 1: pdfplumber — get table bboxes + raw table data ──
    table_bboxes: list[tuple] = []
    raw_tables: list[list[list[str | None]]] = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_number >= len(pdf.pages):
                logger.warn("mixed_page_out_of_range", page=page_number + 1)
                return _empty_result()

            plumber_page = pdf.pages[page_number]
            detected = plumber_page.find_tables()

            for tbl in detected:
                table_bboxes.append(tbl.bbox)
                extracted = tbl.extract()
                if extracted:
                    raw_tables.append(extracted)
    except Exception as e:
        logger.error("mixed_pdfplumber_failed", page=page_number + 1, error=str(e))
        return _empty_result()

    # ── Pass 2: PyMuPDF — extract text blocks with font metadata ──
    narrative_text = ""
    narrative_blocks: list[dict] = []

    try:
        doc = fitz.open(pdf_path)
        mu_page = doc[page_number]
        page_height = mu_page.rect.height
        dict_data = mu_page.get_text("dict")
        doc.close()

        # Collect all font sizes for median (needed by section_detector)
        all_sizes: list[float] = []
        for block in dict_data["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    all_sizes.append(span["size"])

        median_size = median(all_sizes) if all_sizes else 10.0

        outside_paragraphs: list[str] = []

        for block in dict_data["blocks"]:
            if block["type"] != 0:  # Skip image blocks
                continue

            bx0, by0, bx1, by1 = (
                block["bbox"][0], block["bbox"][1],
                block["bbox"][2], block["bbox"][3],
            )
            centroid_x = (bx0 + bx1) / 2
            centroid_y = (by0 + by1) / 2

            # Skip blocks whose centroid falls inside a table region
            inside_table = any(
                tx0 <= centroid_x <= tx1 and ty0 <= centroid_y <= ty1
                for tx0, ty0, tx1, ty1 in table_bboxes
            )
            if inside_table:
                continue

            # Build text + font metadata from spans
            block_text_parts: list[str] = []
            block_sizes: list[float] = []
            block_bold = False

            for line in block["lines"]:
                line_parts: list[str] = []
                for span in line["spans"]:
                    line_parts.append(span["text"])
                    block_sizes.append(span["size"])
                    font_name = span.get("font", "")
                    if "Bold" in font_name or "Bd" in font_name:
                        block_bold = True
                block_text_parts.append(" ".join(line_parts))

            raw_text = "\n".join(block_text_parts)
            cleaned = strip_unicode_garbage(raw_text)

            if not cleaned.strip():
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
            outside_paragraphs.append(cleaned)

        narrative_text = "\n\n".join(outside_paragraphs)

    except Exception as e:
        logger.error("mixed_pymupdf_failed", page=page_number + 1, error=str(e))

    # ── Parse raw tables into list-of-dicts format ──
    tables: list[list[dict]] = []
    for raw_table in raw_tables:
        parsed = _parse_raw_table(raw_table, page_number)
        if parsed:
            tables.append(parsed)

    logger.info(
        "mixed_page_handled",
        page=page_number + 1,
        narrative_words=len(narrative_text.split()),
        tables_found=len(tables),
        narrative_blocks=len(narrative_blocks),
    )

    return {
        "narrative_text": narrative_text,
        "narrative_blocks": narrative_blocks,
        "tables": tables,
    }


# ── Internal Helpers ──


def _parse_raw_table(
    raw_table: list[list[str | None]],
    page_number: int,
) -> list[dict]:
    """
    Convert a raw pdfplumber table to a list of row-dicts.

    Handles None values from merged cells (forward-fill), empty tables,
    and unicode garbage in cell text.
    """
    if not raw_table or len(raw_table) < 2:
        return []

    filled = _forward_fill(raw_table)

    # Build headers from first row
    headers: list[str] = []
    seen: set[str] = set()
    for i, h in enumerate(filled[0]):
        cleaned_h = strip_unicode_garbage(str(h) if h else "").replace("\n", " ").strip()
        if not cleaned_h:
            cleaned_h = f"col_{i}"
        if cleaned_h in seen:
            cleaned_h = f"{cleaned_h}_{i}"
        seen.add(cleaned_h)
        headers.append(cleaned_h)

    rows: list[dict] = []
    for row in filled[1:]:
        if all(not str(cell).strip() for cell in row):
            continue  # Skip fully empty rows
        row_dict: dict[str, str] = {}
        for i, cell in enumerate(row):
            key = headers[i] if i < len(headers) else f"col_{i}"
            row_dict[key] = clean_table_cell(cell)
        rows.append(row_dict)

    if not rows:
        logger.warn("empty_table_after_parsing", page=page_number + 1)

    return rows


def _forward_fill(table: list[list[str | None]]) -> list[list[str]]:
    """Forward-fill None cells from the row above (handles merged cells)."""
    if not table:
        return []

    num_cols = max(len(row) for row in table)
    result: list[list[str]] = []

    for row_idx, row in enumerate(table):
        padded = list(row) + [None] * (num_cols - len(row))
        filled_row: list[str] = []

        for col_idx, cell in enumerate(padded):
            if cell is None and row_idx > 0 and col_idx < len(result[-1]):
                filled_row.append(result[-1][col_idx])
            else:
                filled_row.append(str(cell).strip() if cell else "")

        result.append(filled_row)

    return result


def _empty_result() -> dict:
    """Return a safe empty result dict for failed/out-of-range pages."""
    return {
        "narrative_text": "",
        "narrative_blocks": [],
        "tables": [],
    }
