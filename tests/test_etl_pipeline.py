"""
Tests for ETL pipeline components.

These tests require the HDFC PDF in data/pdfs/.
Skip gracefully if the PDF is not present.

Run with: pytest tests/test_etl_pipeline.py -v
"""

import os
import pytest

from quantscribe.etl.page_classifier import classify_page
from quantscribe.etl.pdf_parser import extract_narrative, extract_tables
from quantscribe.etl.mixed_page_handler import handle_mixed_page
from quantscribe.etl.section_detector import detect_section_header
from quantscribe.etl.text_cleaner import strip_unicode_garbage, normalize_indian_currency
from quantscribe.schemas.etl import PageType, TextChunk

try:
    from quantscribe.etl.pdf_parser import extract_table_bboxes
    HAS_BBOXES = True
except ImportError:
    HAS_BBOXES = False

HDFC_PDF = "data/pdfs/HDFC Bank Report.pdf"
PDF_EXISTS = os.path.exists(HDFC_PDF)

skip_no_pdf = pytest.mark.skipif(not PDF_EXISTS, reason="HDFC PDF not found in data/pdfs/")


# ══════════════════════════════════════════════════
# Page Classifier Tests
# ══════════════════════════════════════════════════


@skip_no_pdf
class TestPageClassifier:
    def test_toc_page_is_narrative(self):
        result = classify_page(4, HDFC_PDF)  # Page 5 = Table of Contents
        assert result.page_type == PageType.NARRATIVE

    def test_financial_schedule_is_tabular(self):
        result = classify_page(319, HDFC_PDF)  # Page 320 = Balance Sheet Schedules
        assert result.page_type == PageType.TABULAR

    def test_cover_page_is_graphical_or_tabular(self):
        result = classify_page(0, HDFC_PDF)  # Page 1 = Cover/AGM info
        assert result.page_type in (PageType.GRAPHICAL, PageType.TABULAR)

    def test_confidence_score_in_range(self):
        result = classify_page(50, HDFC_PDF)
        assert 0.0 <= result.confidence_score <= 1.0

    def test_page_number_is_one_indexed(self):
        result = classify_page(0, HDFC_PDF)  # 0-indexed input
        assert result.page_number == 1  # 1-indexed output

    def test_graphical_page_has_warning(self):
        # Find a low-content page (cover pages are often graphical)
        result = classify_page(0, HDFC_PDF)
        if result.page_type == PageType.GRAPHICAL:
            assert "graphical_page_skipped" in result.extraction_warnings

    def test_narrative_page_has_raw_text(self):
        result = classify_page(4, HDFC_PDF)  # TOC = narrative
        if result.page_type == PageType.NARRATIVE:
            assert result.raw_text is not None
            assert len(result.raw_text) > 0

    def test_tabular_page_has_tables(self):
        result = classify_page(319, HDFC_PDF)  # Financial schedule
        if result.page_type == PageType.TABULAR:
            assert result.tables is not None
            assert len(result.tables) > 0


# ══════════════════════════════════════════════════
# PDF Parser Tests
# ══════════════════════════════════════════════════


@skip_no_pdf
class TestExtractNarrative:
    def test_returns_text_and_blocks(self):
        result = extract_narrative(55, HDFC_PDF)  # Page 56
        assert "text" in result
        assert "blocks" in result
        assert len(result["text"]) > 50

    def test_blocks_have_font_metadata(self):
        result = extract_narrative(55, HDFC_PDF)
        assert len(result["blocks"]) > 0
        block = result["blocks"][0]
        assert "font_size" in block
        assert "median_font_size" in block
        assert "is_bold" in block
        assert "y_position" in block
        assert "page_height" in block

    def test_font_size_is_positive(self):
        result = extract_narrative(55, HDFC_PDF)
        for block in result["blocks"]:
            assert block["font_size"] > 0
            assert block["median_font_size"] > 0

    def test_text_is_cleaned(self):
        result = extract_narrative(55, HDFC_PDF)
        # Should not contain zero-width spaces
        assert "\u200b" not in result["text"]
        assert "\ufeff" not in result["text"]


@skip_no_pdf
class TestExtractTables:
    def test_returns_list_of_tables(self):
        tables = extract_tables(319, HDFC_PDF)  # Page 320
        assert isinstance(tables, list)
        assert len(tables) > 0

    def test_tables_are_list_of_dicts(self):
        tables = extract_tables(319, HDFC_PDF)
        for table in tables:
            assert isinstance(table, list)
            if table:
                assert isinstance(table[0], dict)

    def test_table_has_headers_as_keys(self):
        tables = extract_tables(319, HDFC_PDF)
        if tables and tables[0]:
            keys = list(tables[0][0].keys())
            assert len(keys) > 0
            # Headers should not all be "col_0", "col_1" etc
            non_generic = [k for k in keys if not k.startswith("col_")]
            # At least some meaningful headers
            assert len(non_generic) > 0 or len(keys) > 1

    def test_empty_page_returns_empty(self):
        # Page 0 may not have tables
        tables = extract_tables(0, HDFC_PDF)
        assert isinstance(tables, list)

    @pytest.mark.skipif(not HAS_BBOXES, reason="extract_table_bboxes not available")
    def test_extract_table_bboxes(self):
        bboxes = extract_table_bboxes(319, HDFC_PDF)
        assert isinstance(bboxes, list)
        assert len(bboxes) > 0
        # Each bbox is a tuple of 4 numbers
        for bb in bboxes:
            assert len(bb) == 4
            x0, y0, x1, y1 = bb
            assert x1 > x0  # width positive
            assert y1 > y0  # height positive


# ══════════════════════════════════════════════════
# Mixed Page Handler Tests
# ══════════════════════════════════════════════════


@skip_no_pdf
class TestMixedPageHandler:
    def test_returns_required_keys(self):
        result = handle_mixed_page(349, HDFC_PDF)  # Page 350
        assert "narrative_text" in result
        assert "narrative_blocks" in result
        assert "tables" in result
        assert "warnings" in result

    def test_has_both_narrative_and_tables(self):
        result = handle_mixed_page(349, HDFC_PDF)
        assert len(result["narrative_text"]) > 0
        assert len(result["tables"]) > 0

    def test_narrative_blocks_have_metadata(self):
        result = handle_mixed_page(349, HDFC_PDF)
        if result["narrative_blocks"]:
            block = result["narrative_blocks"][0]
            assert "text" in block
            assert "font_size" in block
            assert "is_bold" in block

    def test_tables_are_list_of_dicts(self):
        result = handle_mixed_page(349, HDFC_PDF)
        for table in result["tables"]:
            assert isinstance(table, list)
            if table:
                assert isinstance(table[0], dict)


# ══════════════════════════════════════════════════
# Section Detector Tests
# ══════════════════════════════════════════════════


class TestSectionDetector:
    def test_detects_known_section(self):
        blocks = [{"text": "Management Discussion and Analysis", "font_size": 14, "median_font_size": 10, "y_position": 50, "page_height": 800, "is_bold": True}]
        result = detect_section_header(blocks, 1)
        assert result == "Management Discussion and Analysis"

    def test_detects_fuzzy_match(self):
        blocks = [{"text": "Management Discussion & Analysis", "font_size": 14, "median_font_size": 10, "y_position": 50, "page_height": 800, "is_bold": True}]
        result = detect_section_header(blocks, 1)
        assert result is not None
        assert "Management" in result

    def test_detects_large_font_header(self):
        blocks = [{"text": "Some Custom Header", "font_size": 16, "median_font_size": 10, "y_position": 50, "page_height": 800, "is_bold": False}]
        result = detect_section_header(blocks, 1)
        assert result == "Some Custom Header"

    def test_skips_long_text(self):
        blocks = [{"text": "A" * 101, "font_size": 16, "median_font_size": 10, "y_position": 50, "page_height": 800, "is_bold": True}]
        result = detect_section_header(blocks, 1)
        assert result is None

    def test_returns_none_for_empty_blocks(self):
        result = detect_section_header([], 1)
        assert result is None

    def test_detects_bold_top_positioned(self):
        blocks = [{"text": "Important Section", "font_size": 10, "median_font_size": 10, "y_position": 30, "page_height": 800, "is_bold": True}]
        result = detect_section_header(blocks, 1)
        assert result == "Important Section"


# ══════════════════════════════════════════════════
# ETL Pipeline Integration Test
# ══════════════════════════════════════════════════


@skip_no_pdf
class TestETLPipelineIntegration:
    def test_pipeline_produces_chunks(self):
        from quantscribe.etl.pipeline import run_etl_pipeline
        chunks = run_etl_pipeline(
            pdf_path=HDFC_PDF,
            bank_name="HDFC_BANK",
            document_type="annual_report",
            fiscal_year="FY25",
            page_range=(49, 52),
        )
        assert len(chunks) > 0

    def test_all_chunks_are_valid_textchunk(self):
        from quantscribe.etl.pipeline import run_etl_pipeline
        chunks = run_etl_pipeline(
            pdf_path=HDFC_PDF,
            bank_name="HDFC_BANK",
            document_type="annual_report",
            fiscal_year="FY25",
            page_range=(49, 52),
        )
        for c in chunks:
            assert isinstance(c, TextChunk)

    def test_chunks_have_correct_bank_name(self):
        from quantscribe.etl.pipeline import run_etl_pipeline
        chunks = run_etl_pipeline(
            pdf_path=HDFC_PDF,
            bank_name="HDFC_BANK",
            document_type="annual_report",
            fiscal_year="FY25",
            page_range=(49, 52),
        )
        for c in chunks:
            assert c.metadata.bank_name == "HDFC_BANK"
            assert c.metadata.fiscal_year == "FY25"
            assert c.metadata.document_type == "annual_report"

    def test_chunks_have_deterministic_ids(self):
        from quantscribe.etl.pipeline import run_etl_pipeline
        chunks1 = run_etl_pipeline(
            pdf_path=HDFC_PDF, bank_name="HDFC_BANK",
            document_type="annual_report", fiscal_year="FY25",
            page_range=(55, 56),
        )
        chunks2 = run_etl_pipeline(
            pdf_path=HDFC_PDF, bank_name="HDFC_BANK",
            document_type="annual_report", fiscal_year="FY25",
            page_range=(55, 56),
        )
        ids1 = [c.metadata.chunk_id for c in chunks1]
        ids2 = [c.metadata.chunk_id for c in chunks2]
        assert ids1 == ids2