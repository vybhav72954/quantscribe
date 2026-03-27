"""
PDF text and table extraction.

Routes extraction to the appropriate tool based on page classification:
- NARRATIVE  → PyMuPDF get_text("dict") with font metadata
- TABULAR    → pdfplumber extract_tables() with forward-fill for merged cells
- MIXED      → handled separately by mixed_page_handler
- GRAPHICAL  → skipped

Every extraction returns structured data with font/position metadata
needed by the section_detector and chunkers downstream.
"""

from __future__ import annotations

from statistics import median
from typing import Optional

import fitz  # PyMuPDF
import pdfplumber

from quantscribe.etl.text_cleaner import (
    clean_table_cell,
    strip_unicode_garbage,
)
from quantscribe.logging_config import get_logger

logger = get_logger("quantscribe.etl.pdf_parser")


# ── Narrative Extraction ──


def extract_narrative(page_number: int, pdf_path: str) -> dict:
    """
    Extract narrative text from a page using PyMuPDF.

    Returns structured data including text blocks with font metadata,
    which the section_detector needs for header identification.

    Args:
        page_number: 0-indexed page number.
        pdf_path:    Path to the PDF file.

    Returns:
        Dict with keys:
        - "text": Cleaned full-page text (paragraphs joined).
        - "blocks": List of block dicts with text, font_size,
          median_font_size, y_position, page_height, is_bold.
    """
    doc = fitz.open(pdf_path)
    mu_page = doc[page_number]
    page_height = mu_page.rect.height
    dict_data = mu_page.get_text("dict")
    doc.close()

    # ── Collect all font sizes for median calculation ──
    all_sizes: list[float] = []
    for block in dict_data["blocks"]:
        if block["type"] != 0:  # text blocks only
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                all_sizes.append(span["size"])

    median_size = median(all_sizes) if all_sizes else 10.0

    # ── Build structured blocks ──
    structured_blocks: list[dict] = []
    page_paragraphs: list[str] = []

    for block in dict_data["blocks"]:
        if block["type"] != 0:
            continue

        # Combine all spans in this block into one text
        block_text_parts: list[str] = []
        block_sizes: list[float] = []
        block_bold = False

        for line in block["lines"]:
            line_text_parts: list[str] = []
            for span in line["spans"]:
                line_text_parts.append(span["text"])
                block_sizes.append(span["size"])
                if "Bold" in span["font"] or "Bd" in span["font"]:
                    block_bold = True
            block_text_parts.append(" ".join(line_text_parts))

        raw_text = "\n".join(block_text_parts)
        cleaned = strip_unicode_garbage(raw_text)

        if not cleaned.strip():
            continue

        block_font_size = max(block_sizes) if block_sizes else median_size
        y_position = block["bbox"][1]  # top-y of block

        structured_blocks.append({
            "text": cleaned,
            "font_size": block_font_size,
            "median_font_size": median_size,
            "y_position": y_position,
            "page_height": page_height,
            "is_bold": block_bold,
        })
        page_paragraphs.append(cleaned)

    full_text = "\n\n".join(page_paragraphs)

    logger.info(
        "narrative_extracted",
        page=page_number + 1,
        blocks=len(structured_blocks),
        words=len(full_text.split()),
    )

    return {
        "text": full_text,
        "blocks": structured_blocks,
    }


# ── Table Extraction ──


def extract_tables(
    page_number: int,
    pdf_path: str,
    use_camelot_fallback: bool = True,
) -> list[list[dict]]:
    """
    Extract tables from a page using pdfplumber.

    Each table is returned as a list of row-dicts (keys = column headers).
    Handles merged cells via forward-fill and cleans all cell values.

    Args:
        page_number:          0-indexed page number.
        pdf_path:             Path to the PDF file.
        use_camelot_fallback: Try camelot-py if pdfplumber finds nothing.

    Returns:
        List of tables. Each table is a list of row-dicts.
        Example: [[{"Metric": "NPA", "FY25": "1.2%"}, ...], ...]
        Returns empty list on any extraction failure.
    """
    all_tables: list[list[dict]] = []

    # ── Primary: pdfplumber ──
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_number >= len(pdf.pages):
                return []
            page = pdf.pages[page_number]
            raw_tables = page.extract_tables()

        for table_idx, raw_table in enumerate(raw_tables):
            parsed = _parse_raw_table(raw_table, page_number, table_idx)
            if parsed:
                all_tables.append(parsed)

    except Exception as e:
        logger.warn(
            "pdfplumber_extract_tables_failed",
            page=page_number + 1,
            error=str(e)[:200],
        )
        # Fall through to camelot

    # ── Fallback: camelot-py ──
    if not all_tables and use_camelot_fallback:
        camelot_tables = _try_camelot(page_number, pdf_path)
        if camelot_tables:
            all_tables.extend(camelot_tables)
            logger.info(
                "camelot_fallback_used",
                page=page_number + 1,
                tables_found=len(camelot_tables),
            )

    logger.info(
        "tables_extracted",
        page=page_number + 1,
        table_count=len(all_tables),
        total_rows=sum(len(t) for t in all_tables),
    )

    return all_tables


def extract_table_bboxes(page_number: int, pdf_path: str) -> list[tuple]:
    """
    Get table bounding boxes for a page (used by mixed_page_handler).

    Returns list of (x0, y0, x1, y1) tuples.
    Returns empty list on any failure — callers treat missing bboxes
    as "no tables", degrading gracefully to pure narrative extraction.
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            if page_number >= len(pdf.pages):
                return []
            page = pdf.pages[page_number]
            tables = page.find_tables()
            return [t.bbox for t in tables]
    except Exception as e:
        logger.warn(
            "extract_table_bboxes_failed",
            page=page_number + 1,
            error=str(e)[:200],
        )
        return []


# ── Internal Helpers ──


def _parse_raw_table(
    raw_table: list[list[str | None]],
    page_number: int,
    table_idx: int,
) -> list[dict]:
    """
    Convert a raw pdfplumber table to a list of row-dicts.

    Handles:
    - None values from merged cells (forward-fill from row above)
    - Unicode garbage in cell text
    - Empty/garbage header rows
    """
    if not raw_table or len(raw_table) < 2:
        return []

    # ── Forward-fill None values (merged cells) ──
    filled = _forward_fill(raw_table)

    # ── Build headers from first row ──
    raw_headers = filled[0]
    headers = _clean_headers(raw_headers)

    # ── Build row dicts ──
    rows: list[dict] = []
    for row in filled[1:]:
        # Skip fully empty rows
        if all(not cell.strip() for cell in row):
            continue

        row_dict: dict[str, str] = {}
        for i, cell in enumerate(row):
            key = headers[i] if i < len(headers) else f"col_{i}"
            row_dict[key] = clean_table_cell(cell)
        rows.append(row_dict)

    if not rows:
        logger.warn(
            "empty_table_after_cleaning",
            page=page_number + 1,
            table_idx=table_idx,
        )

    return rows


def _forward_fill(table: list[list[str | None]]) -> list[list[str]]:
    """Forward-fill None cells from the row above (handles merged cells)."""
    if not table:
        return []

    num_cols = max(len(row) for row in table)
    result: list[list[str]] = []

    for row_idx, row in enumerate(table):
        # Pad short rows
        padded = list(row) + [None] * (num_cols - len(row))
        filled_row: list[str] = []

        for col_idx, cell in enumerate(padded):
            if cell is None and row_idx > 0 and col_idx < len(result[-1]):
                filled_row.append(result[-1][col_idx])
            else:
                filled_row.append(str(cell).strip() if cell else "")

        result.append(filled_row)

    return result


def _clean_headers(raw_headers: list[str]) -> list[str]:
    """
    Clean table header row.

    Handles multi-line headers (common in Indian bank PDFs where
    headers span two rows with newlines inside cells).
    """
    headers: list[str] = []
    seen: set[str] = set()

    for i, h in enumerate(raw_headers):
        # Clean the header text
        cleaned = strip_unicode_garbage(h).replace("\n", " ").strip()

        if not cleaned:
            cleaned = f"col_{i}"

        # Deduplicate headers
        if cleaned in seen:
            cleaned = f"{cleaned}_{i}"
        seen.add(cleaned)
        headers.append(cleaned)

    return headers


def _try_camelot(page_number: int, pdf_path: str) -> list[list[dict]]:
    """
    Fallback table extraction using camelot-py.

    Tries lattice mode first (for bordered tables), then stream mode.
    """
    try:
        import camelot
    except ImportError:
        logger.warn("camelot_not_installed", page=page_number + 1)
        return []

    camelot_page = str(page_number + 1)  # camelot uses 1-indexed
    tables: list[list[dict]] = []

    # Try lattice mode first (bordered tables)
    try:
        result = camelot.read_pdf(
            pdf_path,
            pages=camelot_page,
            flavor="lattice",
            suppress_stdout=True,
        )
        for t in result:
            df = t.df
            if len(df) < 2:
                continue
            headers = [str(c).strip() or f"col_{i}" for i, c in enumerate(df.iloc[0])]
            rows = []
            for _, row in df.iloc[1:].iterrows():
                row_dict = {
                    headers[i]: clean_table_cell(str(v))
                    for i, v in enumerate(row)
                    if i < len(headers)
                }
                rows.append(row_dict)
            if rows:
                tables.append(rows)
    except Exception as e:
        logger.warn("camelot_lattice_failed", page=page_number + 1, error=str(e))

    # Try stream mode if lattice found nothing
    if not tables:
        try:
            result = camelot.read_pdf(
                pdf_path,
                pages=camelot_page,
                flavor="stream",
                suppress_stdout=True,
            )
            for t in result:
                df = t.df
                if len(df) < 2:
                    continue
                headers = [str(c).strip() or f"col_{i}" for i, c in enumerate(df.iloc[0])]
                rows = []
                for _, row in df.iloc[1:].iterrows():
                    row_dict = {
                        headers[i]: clean_table_cell(str(v))
                        for i, v in enumerate(row)
                        if i < len(headers)
                    }
                    rows.append(row_dict)
                if rows:
                    tables.append(rows)
        except Exception as e:
            logger.warn("camelot_stream_failed", page=page_number + 1, error=str(e))

    return tables
    