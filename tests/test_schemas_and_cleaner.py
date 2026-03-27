"""
Tests for Pydantic schemas and text cleaning utilities.

Run with: pytest tests/ -v
"""

import pytest

from quantscribe.schemas.etl import ChunkMetadata, PageType, TextChunk
from quantscribe.schemas.extraction import CitationTrace, ExtractedMetric, ThematicExtraction
from quantscribe.schemas.evaluation import EvalTestCase
from quantscribe.etl.text_cleaner import (
    strip_unicode_garbage,
    normalize_indian_currency,
    clean_table_cell,
    forward_fill_none,
)


# ── Text Cleaner Tests ──


class TestStripUnicodeGarbage:
    def test_removes_zero_width_space(self):
        assert strip_unicode_garbage("hello\u200bworld") == "helloworld"

    def test_removes_bom(self):
        assert strip_unicode_garbage("\ufefftext") == "text"

    def test_replaces_nbsp_with_space(self):
        assert strip_unicode_garbage("hello\u00a0world") == "hello world"

    def test_normalizes_whitespace(self):
        assert strip_unicode_garbage("hello   world") == "hello world"

    def test_preserves_paragraph_breaks(self):
        result = strip_unicode_garbage("para1\n\npara2")
        assert "\n\n" in result

    def test_clean_text_unchanged(self):
        assert strip_unicode_garbage("clean text") == "clean text"


class TestNormalizeIndianCurrency:
    def test_basic_indian_format(self):
        assert normalize_indian_currency("₹ 1,23,456.78") == "123456.78"

    def test_accounting_negative(self):
        assert normalize_indian_currency("(1,234.56)") == "-1234.56"

    def test_simple_number(self):
        assert normalize_indian_currency("12.5%") == "12.5"

    def test_nil_value(self):
        assert normalize_indian_currency("Nil") == "0"

    def test_dash_value(self):
        assert normalize_indian_currency("-") == "0"

    def test_unparseable_returns_original(self):
        assert normalize_indian_currency("N/A text") == "N/Atext"

    def test_large_indian_number(self):
        assert normalize_indian_currency("₹ 45,00,000") == "4500000.0"


class TestCleanTableCell:
    def test_none_returns_empty(self):
        assert clean_table_cell(None) == ""

    def test_strips_whitespace(self):
        assert clean_table_cell("  hello  ") == "hello"


class TestForwardFillNone:
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


# ── Schema Validation Tests ──


class TestChunkMetadata:
    def test_bank_name_normalization(self):
        meta = ChunkMetadata(
            chunk_id="test123",
            bank_name="hdfc bank",
            document_type="annual_report",
            fiscal_year="FY24",
            page_number=1,
            page_type=PageType.NARRATIVE,
            chunk_index=0,
            token_count=100,
            parse_version="etl_v1.0.0",
        )
        assert meta.bank_name == "HDFC_BANK"

    def test_invalid_fiscal_year_rejected(self):
        with pytest.raises(Exception):
            ChunkMetadata(
                chunk_id="test123",
                bank_name="SBI",
                document_type="annual_report",
                fiscal_year="2024",  # Wrong format
                page_number=1,
                page_type=PageType.NARRATIVE,
                chunk_index=0,
                token_count=100,
                parse_version="etl_v1.0.0",
            )

    def test_deterministic_chunk_id(self):
        id1 = ChunkMetadata.generate_chunk_id("SBI", "annual_report", "FY24", 42, 0)
        id2 = ChunkMetadata.generate_chunk_id("SBI", "annual_report", "FY24", 42, 0)
        assert id1 == id2

    def test_different_inputs_different_ids(self):
        id1 = ChunkMetadata.generate_chunk_id("SBI", "annual_report", "FY24", 42, 0)
        id2 = ChunkMetadata.generate_chunk_id("SBI", "annual_report", "FY24", 42, 1)
        assert id1 != id2


class TestExtractedMetric:
    def test_requires_at_least_one_value(self):
        # Validator no longer raises — it auto-fills qualitative_value="not_disclosed"
        # so Gemini responses with null/null don't crash the entire extraction.
        metric = ExtractedMetric(
            metric_name="test_metric",
            metric_value=None,
            qualitative_value=None,
            confidence="high",
            citation=CitationTrace(
                chunk_id="abc",
                bank_name="SBI",
                document_type="annual_report",
                fiscal_year="FY24",
                page_number=1,
                relevance_score=0.9,
                source_excerpt="some text",
            ),
        )
        assert metric.qualitative_value == "not_disclosed"

    def test_numeric_metric_valid(self):
        metric = ExtractedMetric(
            metric_name="gross_npa_ratio",
            metric_value=1.12,
            metric_unit="%",
            confidence="high",
            citation=CitationTrace(
                chunk_id="abc",
                bank_name="SBI",
                document_type="annual_report",
                fiscal_year="FY24",
                page_number=42,
                relevance_score=0.95,
                source_excerpt="Gross NPA ratio stood at 1.12%",
            ),
        )
        assert metric.metric_value == 1.12


class TestEvalTestCase:
    def test_valid_test_case(self):
        tc = EvalTestCase(
            test_id="HDFC_FY24_credit_risk_001",
            query_theme="credit_risk",
            bank_name="HDFC_BANK",
            fiscal_year="FY24",
            expected_metrics={"gross_npa_ratio": 1.12, "net_npa_ratio": 0.27},
            expected_pages=[142, 143],
            source_document="HDFC_Bank_AR_FY24.pdf",
        )
        assert len(tc.expected_metrics) == 2
        