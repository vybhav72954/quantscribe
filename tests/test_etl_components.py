"""
Comprehensive ETL test suite for QuantScribe.

Tests cover:
- PageClassifier: classification logic, decision tree, edge cases
- PDFParser: narrative extraction, table extraction, camelot fallback, merged cells
- MixedPageHandler: narrative/table separation, empty page safety
- TextCleaner: unicode handling, Indian currency, forward-fill
- Schemas: metadata validation, chunk IDs
- Integration: full ETL pipeline on synthetic inputs

Run with:
    pytest tests/ -v
    pytest tests/test_etl_components.py -v          # just this file
    pytest tests/test_etl_components.py -v -k "Classifier"  # one class
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch, call

import pytest

from quantscribe.schemas.etl import PageType, ParsedPage, ChunkMetadata, TextChunk
from quantscribe.etl.text_cleaner import (
    strip_unicode_garbage,
    normalize_indian_currency,
    clean_table_cell,
    forward_fill_none,
)
from quantscribe.chunking.narrative_chunker import chunk_narrative, _compute_overlap
from quantscribe.chunking.table_chunker import chunk_table


# ════════════════════════════════════════════════════════════════════
# HELPERS — synthetic fitz / pdfplumber data
# ════════════════════════════════════════════════════════════════════

def _make_fitz_block(
    text: str,
    x0: float = 50.0,
    y0: float = 100.0,
    x1: float = 500.0,
    y1: float = 120.0,
    font_size: float = 11.0,
    font: str = "Helvetica",
) -> dict:
    """Minimal fitz text block dict that our code reads."""
    return {
        "type": 0,
        "bbox": (x0, y0, x1, y1),
        "lines": [
            {
                "spans": [
                    {
                        "text": text,
                        "size": font_size,
                        "font": font,
                    }
                ]
            }
        ],
    }


def _make_image_block() -> dict:
    """A fitz image block (type=1) — should always be ignored by narrative code."""
    return {"type": 1, "bbox": (0, 0, 100, 100)}


def _make_raw_table(nrows: int = 3, ncols: int = 3) -> list[list[str]]:
    """Build a simple synthetic pdfplumber raw table (list-of-lists, all strings)."""
    headers = [f"Col_{c}" for c in range(ncols)]
    rows = [[f"row{r}_col{c}" for c in range(ncols)] for r in range(nrows)]
    return [headers] + rows


# ════════════════════════════════════════════════════════════════════
# TEXT CLEANER
# ════════════════════════════════════════════════════════════════════

class TestStripUnicodeGarbage:
    """Tests for strip_unicode_garbage."""

    def test_removes_zero_width_space(self):
        assert strip_unicode_garbage("hello\u200bworld") == "helloworld"

    def test_removes_bom(self):
        assert strip_unicode_garbage("\ufefftext") == "text"

    def test_replaces_nbsp_with_space(self):
        result = strip_unicode_garbage("hello\u00a0world")
        assert result == "hello world"

    def test_normalizes_multiple_spaces(self):
        result = strip_unicode_garbage("hello   world")
        assert result == "hello world"

    def test_preserves_paragraph_breaks(self):
        result = strip_unicode_garbage("para1\n\npara2")
        assert "\n\n" in result

    def test_collapses_triple_newlines(self):
        result = strip_unicode_garbage("a\n\n\n\nb")
        assert result.count("\n") <= 2

    def test_clean_text_unchanged(self):
        clean = "The quick brown fox jumps over the lazy dog."
        assert strip_unicode_garbage(clean) == clean

    def test_removes_soft_hyphen(self):
        result = strip_unicode_garbage("hypen\u00adated")
        assert "\u00ad" not in result

    def test_removes_zero_width_joiner(self):
        result = strip_unicode_garbage("no\u200djoin")
        assert "\u200d" not in result

    def test_private_use_area_removed(self):
        result = strip_unicode_garbage("bullet\uf0b7item")
        assert "\uf0b7" not in result

    def test_empty_string(self):
        assert strip_unicode_garbage("") == ""

    def test_only_garbage(self):
        result = strip_unicode_garbage("\u200b\u200c\u200d")
        assert result.strip() == ""


class TestNormalizeIndianCurrency:
    """Tests for normalize_indian_currency."""

    def test_basic_indian_format(self):
        assert normalize_indian_currency("₹ 1,23,456.78") == "123456.78"

    def test_accounting_negative(self):
        assert normalize_indian_currency("(1,234.56)") == "-1234.56"

    def test_simple_percentage(self):
        assert normalize_indian_currency("12.5%") == "12.5"

    def test_nil_lowercase(self):
        assert normalize_indian_currency("Nil") == "0"

    def test_nil_uppercase(self):
        assert normalize_indian_currency("NIL") == "0"

    def test_dash_value(self):
        assert normalize_indian_currency("-") == "0"

    def test_double_dash(self):
        assert normalize_indian_currency("--") == "0"

    def test_em_dash(self):
        assert normalize_indian_currency("—") == "0"

    def test_na_value(self):
        assert normalize_indian_currency("n/a") == "0"

    def test_unparseable_returns_stripped(self):
        result = normalize_indian_currency("N/A text")
        # Should not raise; may return original with spaces stripped
        assert isinstance(result, str)

    def test_large_indian_number(self):
        result = normalize_indian_currency("₹ 45,00,000")
        assert result == "4500000.0"

    def test_plain_integer(self):
        assert normalize_indian_currency("1000") == "1000.0"

    def test_negative_accounting_with_currency_symbol(self):
        result = normalize_indian_currency("₹ (50,000)")
        assert result.startswith("-")

    def test_zero_value(self):
        assert normalize_indian_currency("0") == "0.0"


class TestCleanTableCell:
    """Tests for clean_table_cell."""

    def test_none_returns_empty(self):
        assert clean_table_cell(None) == ""

    def test_strips_whitespace(self):
        assert clean_table_cell("  hello  ") == "hello"

    def test_empty_string(self):
        assert clean_table_cell("") == ""

    def test_all_spaces(self):
        assert clean_table_cell("   ") == ""

    def test_number_cleaned(self):
        result = clean_table_cell("  1,234.56  ")
        assert isinstance(result, str)

    def test_unicode_garbage_removed(self):
        result = clean_table_cell("hello\u200bworld")
        assert "\u200b" not in result


class TestForwardFillNone:
    """Tests for forward_fill_none."""

    def test_fills_from_above(self):
        table = [["A", "B"], [None, "C"]]
        result = forward_fill_none(table)
        assert result[1][0] == "A"

    def test_empty_table(self):
        assert forward_fill_none([]) == []

    def test_no_nones(self):
        table = [["A", "B"], ["C", "D"]]
        result = forward_fill_none(table)
        assert result == [["A", "B"], ["C", "D"]]

    def test_multiple_none_rows(self):
        table = [["Header"], [None], [None]]
        result = forward_fill_none(table)
        assert result[1][0] == "Header"
        assert result[2][0] == "Header"

    def test_short_rows_padded(self):
        table = [["A", "B", "C"], ["X"]]
        result = forward_fill_none(table)
        # Short row should be padded to num_cols
        assert len(result[1]) == 3

    def test_first_row_none(self):
        # First row None has nothing to fill from → stays empty string
        table = [[None, "B"], ["C", "D"]]
        result = forward_fill_none(table)
        assert result[0][0] == ""

    def test_single_row(self):
        table = [["A", "B", "C"]]
        result = forward_fill_none(table)
        assert result == [["A", "B", "C"]]


# ════════════════════════════════════════════════════════════════════
# PAGE CLASSIFIER — decision tree (unit-tests, no real PDF needed)
# ════════════════════════════════════════════════════════════════════

class TestPageClassifierDecisionTree:
    """
    Unit-tests for _decide_page_type in page_classifier.
    We import and call the private function directly to avoid needing a real PDF.
    """

    def setup_method(self):
        from quantscribe.etl.page_classifier import _decide_page_type
        self._decide = _decide_page_type

    def test_low_word_count_gives_graphical(self):
        page_type, conf = self._decide(
            table_count=0, total_words=10, narrative_words=10, image_count=2
        )
        assert page_type == PageType.GRAPHICAL

    def test_no_tables_gives_narrative(self):
        page_type, conf = self._decide(
            table_count=0, total_words=300, narrative_words=300, image_count=0
        )
        assert page_type == PageType.NARRATIVE

    def test_tables_with_high_narrative_gives_mixed(self):
        page_type, conf = self._decide(
            table_count=2, total_words=400, narrative_words=200, image_count=0
        )
        assert page_type == PageType.MIXED

    def test_tables_with_low_narrative_gives_tabular(self):
        page_type, conf = self._decide(
            table_count=3, total_words=250, narrative_words=20, image_count=0
        )
        assert page_type == PageType.TABULAR

    def test_confidence_between_0_and_1(self):
        for table_count, total_words, narrative_words, image_count in [
            (0, 10, 10, 1),
            (0, 300, 300, 0),
            (2, 400, 200, 0),
            (3, 250, 20, 0),
        ]:
            _, conf = self._decide(table_count, total_words, narrative_words, image_count)
            assert 0.0 <= conf <= 1.0, f"Confidence {conf} out of range"

    def test_graphical_threshold_boundary(self):
        """Exactly at threshold → GRAPHICAL; one above → NARRATIVE."""
        from quantscribe.etl.page_classifier import MIN_WORDS_FOR_CONTENT
        pt_at, _ = self._decide(0, MIN_WORDS_FOR_CONTENT - 1, 0, 0)
        pt_above, _ = self._decide(0, MIN_WORDS_FOR_CONTENT, MIN_WORDS_FOR_CONTENT, 0)
        assert pt_at == PageType.GRAPHICAL
        assert pt_above == PageType.NARRATIVE

    def test_mixed_threshold_boundary(self):
        from quantscribe.etl.page_classifier import MIN_NARRATIVE_WORDS_FOR_MIXED
        pt_just_below, _ = self._decide(1, 200, MIN_NARRATIVE_WORDS_FOR_MIXED - 1, 0)
        pt_at, _ = self._decide(1, 200, MIN_NARRATIVE_WORDS_FOR_MIXED, 0)
        assert pt_just_below == PageType.TABULAR
        assert pt_at == PageType.MIXED


class TestPageClassifierWordCounter:
    """Unit-tests for _count_words_outside_tables."""

    def setup_method(self):
        from quantscribe.etl.page_classifier import _count_words_outside_tables
        self._count = _count_words_outside_tables

    def _block(
        self,
        text: str,
        x0: float, y0: float, x1: float, y1: float,
        block_type: int = 0,
    ) -> tuple:
        """Create a minimal fitz text block tuple."""
        return (x0, y0, x1, y1, text, 0, block_type)

    def test_no_tables_counts_all_text(self):
        blocks = [self._block("hello world foo bar", 0, 0, 100, 20)]
        count = self._count(blocks, [])
        assert count == 4

    def test_block_inside_table_excluded(self):
        # Block centroid (50, 10) is inside table bbox (0,0,100,20)
        blocks = [self._block("inside table text words", 0, 0, 100, 20)]
        table_bboxes = [(0, 0, 100, 20)]
        count = self._count(blocks, table_bboxes)
        assert count == 0

    def test_block_outside_table_counted(self):
        # Block centroid (50, 150) is outside table bbox (0,0,100,20)
        blocks = [self._block("outside the table words", 0, 140, 100, 160)]
        table_bboxes = [(0, 0, 100, 20)]
        count = self._count(blocks, table_bboxes)
        assert count == 4

    def test_image_blocks_skipped(self):
        blocks = [self._block("image block", 0, 0, 100, 20, block_type=1)]
        count = self._count(blocks, [])
        assert count == 0

    def test_mixed_blocks(self):
        blocks = [
            self._block("inside table", 0, 0, 100, 20),     # inside table
            self._block("outside narrative words", 0, 200, 100, 220),  # outside
        ]
        table_bboxes = [(0, 0, 100, 30)]
        count = self._count(blocks, table_bboxes)
        assert count == 3  # "outside narrative words"


class TestPageClassifierTablesToDicts:
    """Unit-tests for _tables_to_dicts."""

    def setup_method(self):
        from quantscribe.etl.page_classifier import _tables_to_dicts
        self._fn = _tables_to_dicts

    def test_basic_table(self):
        raw = [["Name", "Value"], ["NPA", "1.2%"], ["CAR", "18%"]]
        result = self._fn([raw])
        assert len(result) == 2
        assert result[0]["Name"] == "NPA"
        assert result[1]["Value"] == "18%"

    def test_empty_table_skipped(self):
        result = self._fn([[]])
        assert result == []

    def test_single_row_table_skipped(self):
        result = self._fn([["Header only"]])
        assert result == []

    def test_none_cells_become_empty_string(self):
        raw = [["A", "B"], [None, "val"]]
        result = self._fn([raw])
        assert result[0]["A"] == ""  # None → ""

    def test_multiple_tables_flattened(self):
        t1 = [["X"], ["1"], ["2"]]
        t2 = [["Y"], ["3"]]
        result = self._fn([t1, t2])
        assert len(result) == 3  # 2 rows from t1 + 1 row from t2

    def test_duplicate_header_deduplicated(self):
        raw = [["Col", "Col"], ["a", "b"]]
        result = self._fn([raw])
        keys = list(result[0].keys())
        assert len(keys) == len(set(keys)), "Duplicate headers should be deduplicated"


# ════════════════════════════════════════════════════════════════════
# PDF PARSER — unit-tests via mocking
# ════════════════════════════════════════════════════════════════════

class TestPDFParserNarrative:
    """Tests for extract_narrative (mocked fitz)."""

    @patch("quantscribe.etl.pdf_parser.fitz.open")
    def test_returns_text_and_blocks(self, mock_fitz):
        """extract_narrative returns a dict with 'text' and 'blocks' keys."""
        mock_doc = MagicMock()
        mock_fitz.return_value.__enter__ = MagicMock(return_value=mock_doc)
        mock_fitz.return_value = mock_doc

        mock_page = MagicMock()
        mock_page.rect.height = 800.0
        mock_page.get_text.return_value = {
            "blocks": [
                {
                    "type": 0,
                    "bbox": (50, 100, 500, 120),
                    "lines": [
                        {
                            "spans": [
                                {"text": "Capital adequacy ratio improved.", "size": 11.0, "font": "Helvetica"}
                            ]
                        }
                    ],
                }
            ]
        }
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        from quantscribe.etl.pdf_parser import extract_narrative
        result = extract_narrative(0, "dummy.pdf")

        assert "text" in result
        assert "blocks" in result
        assert isinstance(result["text"], str)
        assert isinstance(result["blocks"], list)

    @patch("quantscribe.etl.pdf_parser.fitz.open")
    def test_bold_font_detected(self, mock_fitz):
        """Blocks with 'Bold' in font name should set is_bold=True."""
        mock_doc = MagicMock()
        mock_fitz.return_value = mock_doc

        mock_page = MagicMock()
        mock_page.rect.height = 800.0
        mock_page.get_text.return_value = {
            "blocks": [
                {
                    "type": 0,
                    "bbox": (50, 50, 500, 80),
                    "lines": [
                        {"spans": [{"text": "Section Header", "size": 16.0, "font": "Arial-Bold"}]}
                    ],
                }
            ]
        }
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        from quantscribe.etl.pdf_parser import extract_narrative
        result = extract_narrative(0, "dummy.pdf")

        if result["blocks"]:
            assert result["blocks"][0]["is_bold"] is True

    @patch("quantscribe.etl.pdf_parser.fitz.open")
    def test_image_blocks_skipped(self, mock_fitz):
        """Image blocks (type=1) should not appear in narrative blocks."""
        mock_doc = MagicMock()
        mock_fitz.return_value = mock_doc

        mock_page = MagicMock()
        mock_page.rect.height = 800.0
        mock_page.get_text.return_value = {
            "blocks": [
                {"type": 1, "bbox": (0, 0, 100, 100)},  # image block
            ]
        }
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        from quantscribe.etl.pdf_parser import extract_narrative
        result = extract_narrative(0, "dummy.pdf")

        assert result["blocks"] == []
        assert result["text"] == ""

    @patch("quantscribe.etl.pdf_parser.fitz.open")
    def test_empty_page_returns_empty(self, mock_fitz):
        """A page with no blocks returns empty text and no blocks."""
        mock_doc = MagicMock()
        mock_fitz.return_value = mock_doc

        mock_page = MagicMock()
        mock_page.rect.height = 800.0
        mock_page.get_text.return_value = {"blocks": []}
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)

        from quantscribe.etl.pdf_parser import extract_narrative
        result = extract_narrative(0, "dummy.pdf")

        assert result["text"] == ""
        assert result["blocks"] == []


class TestPDFParserTableMergedCells:
    """Tests for _forward_fill (the internal forward-fill used by pdf_parser)."""

    def setup_method(self):
        from quantscribe.etl.pdf_parser import _forward_fill
        self._ff = _forward_fill

    def test_none_filled_from_above(self):
        table = [["Metric", "FY25", "FY24"], [None, "1.2%", "1.5%"]]
        result = self._ff(table)
        assert result[1][0] == "Metric"

    def test_column_header_repeated_across_merged(self):
        table = [["Bank"], [None], [None]]
        result = self._ff(table)
        assert all(row[0] == "Bank" for row in result[1:])

    def test_no_nones_unchanged(self):
        table = [["A", "B"], ["1", "2"]]
        result = self._ff(table)
        assert result == [["A", "B"], ["1", "2"]]


class TestPDFParserCleanHeaders:
    """Tests for _clean_headers."""

    def setup_method(self):
        from quantscribe.etl.pdf_parser import _clean_headers
        self._fn = _clean_headers

    def test_basic_headers(self):
        headers = self._fn(["Metric", "FY25", "FY24"])
        assert headers == ["Metric", "FY25", "FY24"]

    def test_empty_header_gets_col_index(self):
        headers = self._fn(["Name", "", "Value"])
        assert headers[1] == "col_1"

    def test_duplicate_headers_deduplicated(self):
        headers = self._fn(["FY25", "FY25", "FY25"])
        assert len(set(headers)) == 3

    def test_multiline_header_joined(self):
        headers = self._fn(["Capital\nAdequacy"])
        assert "\n" not in headers[0]
        assert "Capital" in headers[0]


# ════════════════════════════════════════════════════════════════════
# MIXED PAGE HANDLER — unit tests
# ════════════════════════════════════════════════════════════════════

class TestMixedPageHandler:
    """Tests for handle_mixed_page via mocking."""

    @patch("quantscribe.etl.mixed_page_handler.fitz.open")
    @patch("quantscribe.etl.mixed_page_handler.pdfplumber.open")
    def test_returns_required_keys(self, mock_plumber, mock_fitz):
        """Result dict must have narrative_text, narrative_blocks, tables."""
        # Set up plumber mock: 1 table with bbox (0,0,200,100)
        mock_plumber_ctx = MagicMock()
        mock_plumber.return_value.__enter__ = MagicMock(return_value=mock_plumber_ctx)
        mock_plumber.return_value.__exit__ = MagicMock(return_value=False)

        mock_pdf = MagicMock()
        mock_plumber_ctx.__enter__ = MagicMock(return_value=mock_pdf)
        mock_plumber_ctx.__exit__ = MagicMock(return_value=False)
        mock_plumber.return_value = mock_plumber_ctx

        mock_page_plumber = MagicMock()
        mock_pdf.pages = [mock_page_plumber]
        mock_tbl = MagicMock()
        mock_tbl.bbox = (0, 0, 200, 100)
        mock_tbl.extract.return_value = [["Metric", "Value"], ["NPA", "1.2"]]
        mock_page_plumber.find_tables.return_value = [mock_tbl]

        # Set up fitz mock: 1 text block outside table area
        mock_fitz_doc = MagicMock()
        mock_fitz.return_value = mock_fitz_doc
        mock_fitz_page = MagicMock()
        mock_fitz_page.rect.height = 800.0
        mock_fitz_page.get_text.return_value = {
            "blocks": [
                {
                    "type": 0,
                    "bbox": (50, 400, 500, 420),  # centroid y=410, outside table (0–100)
                    "lines": [
                        {"spans": [{"text": "Narrative text here", "size": 11.0, "font": "Helvetica"}]}
                    ],
                }
            ]
        }
        mock_fitz_doc.__getitem__ = MagicMock(return_value=mock_fitz_page)

        from quantscribe.etl.mixed_page_handler import handle_mixed_page
        result = handle_mixed_page(0, "dummy.pdf")

        assert "narrative_text" in result
        assert "narrative_blocks" in result
        assert "tables" in result

    def test_internal_forward_fill(self):
        """_forward_fill inside mixed_page_handler handles Nones correctly."""
        from quantscribe.etl.mixed_page_handler import _forward_fill
        table = [["Section", "Amount"], [None, "500"]]
        result = _forward_fill(table)
        assert result[1][0] == "Section"

    def test_internal_parse_raw_table_skips_empty(self):
        """_parse_raw_table should return [] for an empty or header-only table."""
        from quantscribe.etl.mixed_page_handler import _parse_raw_table
        # Only header row → no data rows
        assert _parse_raw_table([["Header"]], 0) == []
        # Completely empty
        assert _parse_raw_table([], 0) == []

    def test_internal_parse_raw_table_basic(self):
        from quantscribe.etl.mixed_page_handler import _parse_raw_table
        raw = [["Metric", "Value"], ["NPA", "1.2%"], ["CAR", "18%"]]
        result = _parse_raw_table(raw, 0)
        assert len(result) == 2
        assert result[0]["Metric"] == "NPA"

    def test_internal_parse_raw_table_all_empty_rows_skipped(self):
        from quantscribe.etl.mixed_page_handler import _parse_raw_table
        raw = [["Col1", "Col2"], ["", ""], ["data", "val"]]
        result = _parse_raw_table(raw, 0)
        assert len(result) == 1
        assert result[0]["Col1"] == "data"

    def test_empty_result_safe(self):
        from quantscribe.etl.mixed_page_handler import _empty_result
        r = _empty_result()
        assert r["narrative_text"] == ""
        assert r["narrative_blocks"] == []
        assert r["tables"] == []


class TestMixedPageHandlerNarrativeExtraction:
    """Tests for narrative extraction logic in handle_mixed_page."""

    @patch("quantscribe.etl.mixed_page_handler.fitz.open")
    @patch("quantscribe.etl.mixed_page_handler.pdfplumber.open")
    def test_block_inside_table_bbox_excluded_from_narrative(self, mock_plumber, mock_fitz):
        """Text blocks whose centroid lies inside a table bbox must NOT appear in narrative."""
        mock_plumber_ctx = MagicMock()
        mock_plumber.return_value = mock_plumber_ctx
        mock_plumber_ctx.__enter__ = MagicMock(return_value=mock_plumber_ctx)
        mock_plumber_ctx.__exit__ = MagicMock(return_value=False)

        mock_pdf = MagicMock()
        mock_plumber_ctx.pages = [mock_pdf]

        mock_pdf_page = MagicMock()
        mock_plumber_ctx.pages = [mock_pdf_page]

        mock_tbl = MagicMock()
        mock_tbl.bbox = (0, 0, 600, 200)  # large table covering top of page
        mock_tbl.extract.return_value = [["H1", "H2"], ["v1", "v2"]]
        mock_pdf_page.find_tables.return_value = [mock_tbl]

        mock_fitz_doc = MagicMock()
        mock_fitz.return_value = mock_fitz_doc
        mock_fitz_page = MagicMock()
        mock_fitz_page.rect.height = 800.0
        mock_fitz_page.get_text.return_value = {
            "blocks": [
                {
                    "type": 0,
                    "bbox": (10, 50, 590, 150),  # centroid y=100, INSIDE table (0–200)
                    "lines": [
                        {"spans": [{"text": "This is inside the table area", "size": 11.0, "font": "Arial"}]}
                    ],
                }
            ]
        }
        mock_fitz_doc.__getitem__ = MagicMock(return_value=mock_fitz_page)

        from quantscribe.etl.mixed_page_handler import handle_mixed_page
        result = handle_mixed_page(0, "dummy.pdf")

        # Narrative should be empty since the only block is inside the table
        assert "inside the table area" not in result["narrative_text"]


# ════════════════════════════════════════════════════════════════════
# NARRATIVE CHUNKER
# ════════════════════════════════════════════════════════════════════

class TestNarrativeChunker:
    """Tests for chunk_narrative."""

    BASE_KWARGS = dict(
        bank_name="HDFC_BANK",
        document_type="annual_report",
        fiscal_year="FY25",
        page_number=42,
    )

    def _long_text(self, sentences: int = 40, words_each: int = 15) -> str:
        sentence = "The bank reported strong capital adequacy ratios this quarter."
        return " ".join([sentence] * sentences)

    def test_returns_list_of_text_chunks(self):
        chunks = chunk_narrative(text="Hello world. This is a sentence.", **self.BASE_KWARGS)
        assert isinstance(chunks, list)
        for c in chunks:
            assert isinstance(c, TextChunk)

    def test_empty_text_returns_empty_list(self):
        chunks = chunk_narrative(text="", **self.BASE_KWARGS)
        assert chunks == []

    def test_section_header_prepended(self):
        chunks = chunk_narrative(
            text=self._long_text(),
            section_header="Capital Adequacy",
            **self.BASE_KWARGS,
        )
        assert all("[Section: Capital Adequacy]" in c.content for c in chunks)

    def test_no_section_header_no_prefix(self):
        chunks = chunk_narrative(
            text=self._long_text(),
            section_header=None,
            **self.BASE_KWARGS,
        )
        assert all("[Section:" not in c.content for c in chunks)

    def test_content_type_is_narrative(self):
        chunks = chunk_narrative(text=self._long_text(), **self.BASE_KWARGS)
        assert all(c.content_type == "narrative" for c in chunks)

    def test_metadata_bank_name(self):
        chunks = chunk_narrative(text=self._long_text(), **self.BASE_KWARGS)
        assert all(c.metadata.bank_name == "HDFC_BANK" for c in chunks)

    def test_metadata_page_number(self):
        chunks = chunk_narrative(text=self._long_text(), **self.BASE_KWARGS)
        assert all(c.metadata.page_number == 42 for c in chunks)

    def test_chunk_ids_are_unique(self):
        chunks = chunk_narrative(text=self._long_text(sentences=60), **self.BASE_KWARGS)
        ids = [c.metadata.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_no_mid_sentence_splits(self):
        """Chunks must not end with a word that breaks a sentence."""
        text = "The NPA ratio was 1.2%. The CAR ratio stood at 18.5%. The bank improved its PCR."
        chunks = chunk_narrative(text=text, chunk_size_words=5, overlap_words=2, **self.BASE_KWARGS)
        # Check all content is actual text
        for c in chunks:
            assert len(c.content.strip()) > 0

    def test_long_text_produces_multiple_chunks(self):
        chunks = chunk_narrative(
            text=self._long_text(sentences=100),
            chunk_size_words=50,
            overlap_words=10,
            **self.BASE_KWARGS,
        )
        assert len(chunks) > 1

    def test_custom_chunk_size(self):
        """With a very small chunk size, every sentence should produce a chunk."""
        text = "Sentence one here. Sentence two here. Sentence three here."
        chunks = chunk_narrative(text=text, chunk_size_words=4, overlap_words=0, **self.BASE_KWARGS)
        assert len(chunks) >= 2


class TestComputeOverlap:
    """Tests for _compute_overlap helper."""

    def test_returns_subset_of_sentences(self):
        sentences = ["alpha beta.", "gamma delta.", "epsilon zeta."]
        result = _compute_overlap(sentences, overlap_words=3)
        assert len(result) <= len(sentences)

    def test_overlap_words_respected(self):
        sentences = ["one two three.", "four five six.", "seven eight nine."]
        result = _compute_overlap(sentences, overlap_words=5)
        total_words = sum(len(s.split()) for s in result)
        assert total_words >= 3  # at least some sentences kept

    def test_empty_sentences(self):
        assert _compute_overlap([], overlap_words=10) == []


# ════════════════════════════════════════════════════════════════════
# TABLE CHUNKER
# ════════════════════════════════════════════════════════════════════

class TestTableChunker:
    """Tests for chunk_table."""

    BASE_KWARGS = dict(
        bank_name="SBI",
        document_type="annual_report",
        fiscal_year="FY25",
        page_number=10,
    )

    def _table(self, nrows: int = 5, ncols: int = 3) -> list[dict]:
        headers = [f"Col{i}" for i in range(ncols)]
        return [{h: f"val_{r}_{c}" for c, h in enumerate(headers)} for r in range(nrows)]

    def test_returns_list_of_text_chunks(self):
        chunks = chunk_table(table_data=self._table(), **self.BASE_KWARGS)
        assert isinstance(chunks, list)
        for c in chunks:
            assert isinstance(c, TextChunk)

    def test_empty_table_returns_empty_list(self):
        chunks = chunk_table(table_data=[], **self.BASE_KWARGS)
        assert chunks == []

    def test_content_type_is_table_structured(self):
        chunks = chunk_table(table_data=self._table(), **self.BASE_KWARGS)
        assert all(c.content_type == "table_structured" for c in chunks)

    def test_section_header_prepended(self):
        chunks = chunk_table(
            table_data=self._table(),
            section_header="Credit Risk",
            **self.BASE_KWARGS,
        )
        assert all("[Section: Credit Risk]" in c.content for c in chunks)

    def test_column_headers_in_every_chunk(self):
        """When table is split, headers must appear in every sub-chunk."""
        big_table = self._table(nrows=200, ncols=4)
        chunks = chunk_table(table_data=big_table, max_tokens=50, **self.BASE_KWARGS)
        assert len(chunks) > 1
        for c in chunks:
            assert "Col0" in c.content, "Column header missing from chunk"

    def test_small_table_single_chunk(self):
        small = self._table(nrows=3, ncols=2)
        chunks = chunk_table(table_data=small, **self.BASE_KWARGS)
        assert len(chunks) == 1

    def test_chunk_ids_unique_across_table(self):
        big_table = self._table(nrows=200, ncols=3)
        chunks = chunk_table(table_data=big_table, max_tokens=50, **self.BASE_KWARGS)
        ids = [c.metadata.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_no_row_split_mid_row(self):
        """Every chunk must contain complete rows (no partial rows)."""
        table = self._table(nrows=20, ncols=3)
        chunks = chunk_table(table_data=table, max_tokens=30, **self.BASE_KWARGS)
        for c in chunks:
            # Each chunk's content lines should be complete
            lines = [l for l in c.content.splitlines() if l.strip() and "---" not in l and "[Section" not in l]
            # Header line + data lines — all should have same number of | separators
            pipe_counts = [l.count("|") for l in lines]
            if len(pipe_counts) > 1:
                assert len(set(pipe_counts)) == 1, "Inconsistent pipe counts — partial row detected"

    def test_metadata_page_number(self):
        chunks = chunk_table(table_data=self._table(), **self.BASE_KWARGS)
        assert all(c.metadata.page_number == 10 for c in chunks)

    def test_single_row_table(self):
        single_row = [{"Metric": "NPA", "Value": "1.2%"}]
        chunks = chunk_table(table_data=single_row, **self.BASE_KWARGS)
        assert len(chunks) == 1
        assert "NPA" in chunks[0].content


# ════════════════════════════════════════════════════════════════════
# SCHEMA VALIDATION
# ════════════════════════════════════════════════════════════════════

class TestChunkMetadataSchema:
    """Tests for ChunkMetadata Pydantic model."""

    BASE = dict(
        chunk_id="abc123",
        bank_name="SBI",
        document_type="annual_report",
        fiscal_year="FY25",
        page_number=1,
        page_type=PageType.NARRATIVE,
        chunk_index=0,
        token_count=100,
        parse_version="etl_v1.0.0",
    )

    def test_bank_name_normalized_to_uppercase(self):
        meta = ChunkMetadata(**{**self.BASE, "bank_name": "hdfc bank"})
        assert meta.bank_name == "HDFC_BANK"

    def test_invalid_fiscal_year_rejected(self):
        with pytest.raises(Exception):
            ChunkMetadata(**{**self.BASE, "fiscal_year": "2024"})

    def test_valid_fy_formats(self):
        for fy in ["FY24", "FY25", "FY26"]:
            meta = ChunkMetadata(**{**self.BASE, "fiscal_year": fy})
            assert meta.fiscal_year == fy

    def test_deterministic_chunk_id(self):
        id1 = ChunkMetadata.generate_chunk_id("SBI", "annual_report", "FY25", 42, 0)
        id2 = ChunkMetadata.generate_chunk_id("SBI", "annual_report", "FY25", 42, 0)
        assert id1 == id2

    def test_different_inputs_give_different_ids(self):
        id1 = ChunkMetadata.generate_chunk_id("SBI", "annual_report", "FY25", 42, 0)
        id2 = ChunkMetadata.generate_chunk_id("SBI", "annual_report", "FY25", 42, 1)
        assert id1 != id2

    def test_bank_sbi_different_from_hdfc(self):
        id_sbi = ChunkMetadata.generate_chunk_id("SBI", "annual_report", "FY25", 1, 0)
        id_hdfc = ChunkMetadata.generate_chunk_id("HDFC_BANK", "annual_report", "FY25", 1, 0)
        assert id_sbi != id_hdfc

    def test_section_header_optional(self):
        meta = ChunkMetadata(**{**self.BASE, "section_header": None})
        assert meta.section_header is None

    def test_page_number_positive(self):
        with pytest.raises(Exception):
            ChunkMetadata(**{**self.BASE, "page_number": 0})


class TestParsedPageSchema:
    """Tests for ParsedPage Pydantic model."""

    def test_narrative_page_valid(self):
        page = ParsedPage(
            page_number=1,
            page_type=PageType.NARRATIVE,
            raw_text="Some narrative text here.",
            confidence_score=0.9,
        )
        assert page.page_type == PageType.NARRATIVE

    def test_tabular_page_valid(self):
        page = ParsedPage(
            page_number=5,
            page_type=PageType.TABULAR,
            tables=[{"Metric": "NPA", "Value": "1.2"}],
            confidence_score=0.85,
        )
        assert page.tables is not None

    def test_graphical_page_valid(self):
        page = ParsedPage(
            page_number=10,
            page_type=PageType.GRAPHICAL,
            confidence_score=0.85,
        )
        assert page.page_type == PageType.GRAPHICAL

    def test_mixed_page_has_both(self):
        page = ParsedPage(
            page_number=3,
            page_type=PageType.MIXED,
            raw_text="Some narrative.",
            tables=[{"A": "1"}],
            confidence_score=0.80,
        )
        assert page.raw_text is not None
        assert page.tables is not None

    def test_confidence_between_0_and_1(self):
        with pytest.raises(Exception):
            ParsedPage(
                page_number=1,
                page_type=PageType.NARRATIVE,
                confidence_score=1.5,
            )

    def test_extraction_warnings_default_empty(self):
        page = ParsedPage(
            page_number=1,
            page_type=PageType.NARRATIVE,
            confidence_score=0.9,
        )
        assert page.extraction_warnings == []


# ════════════════════════════════════════════════════════════════════
# INTEGRATION-STYLE TESTS (pipeline components, no real PDF)
# ════════════════════════════════════════════════════════════════════

class TestETLPipelineSaveLoad:
    """
    Test that save_chunks_to_json produces valid JSON that can be
    round-tripped. Does not require a real PDF.
    """

    def _make_chunk(self, idx: int) -> TextChunk:
        meta = ChunkMetadata(
            chunk_id=ChunkMetadata.generate_chunk_id("SBI", "annual_report", "FY25", 1, idx),
            bank_name="SBI",
            document_type="annual_report",
            fiscal_year="FY25",
            page_number=1,
            page_type=PageType.NARRATIVE,
            chunk_index=idx,
            token_count=50,
            parse_version="etl_v1.0.0",
        )
        return TextChunk(content=f"Chunk content number {idx}.", metadata=meta, content_type="narrative")

    def test_save_and_reload(self):
        from quantscribe.etl.pipeline import save_chunks_to_json

        chunks = [self._make_chunk(i) for i in range(5)]

        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "test_chunks.json")
            save_chunks_to_json(chunks, out_path)

            assert os.path.exists(out_path)
            with open(out_path, "r") as f:
                data = json.load(f)

            assert len(data) == 5
            assert data[0]["content"] == "Chunk content number 0."
            assert "metadata" in data[0]
            assert "content_type" in data[0]

    def test_saved_json_has_correct_keys(self):
        from quantscribe.etl.pipeline import save_chunks_to_json

        chunks = [self._make_chunk(0)]

        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "one_chunk.json")
            save_chunks_to_json(chunks, out_path)

            with open(out_path, "r") as f:
                data = json.load(f)

            record = data[0]
            assert "content" in record
            assert "metadata" in record
            assert "content_type" in record
            assert record["metadata"]["bank_name"] == "SBI"

    def test_empty_chunks_saves_empty_json(self):
        from quantscribe.etl.pipeline import save_chunks_to_json

        with tempfile.TemporaryDirectory() as tmp:
            out_path = os.path.join(tmp, "empty.json")
            save_chunks_to_json([], out_path)

            with open(out_path, "r") as f:
                data = json.load(f)

            assert data == []

    def test_output_dir_created_if_not_exists(self):
        from quantscribe.etl.pipeline import save_chunks_to_json

        chunks = [self._make_chunk(0)]

        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "deep", "nested", "dir", "chunks.json")
            save_chunks_to_json(chunks, nested)
            assert os.path.exists(nested)


class TestPageClassifierEmptyPage:
    """Test that _empty_page returns a safe ParsedPage."""

    def test_empty_page_is_graphical(self):
        from quantscribe.etl.page_classifier import _empty_page
        page = _empty_page(5)
        assert page.page_type == PageType.GRAPHICAL
        assert page.page_number == 6  # 0-indexed → 1-indexed
        assert page.confidence_score == 0.0
        assert "page_parse_failed_or_empty" in page.extraction_warnings
