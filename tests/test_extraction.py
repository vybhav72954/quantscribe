"""
Tests for extraction chain, retrieval, and peer comparison schemas.

These tests validate Pydantic models and FAISS operations.
No Gemini API key required — all tests use mock data.

Run with: pytest tests/test_extraction.py -v
"""

import pytest
import numpy as np

from quantscribe.schemas.extraction import (
    CitationTrace,
    ExtractedMetric,
    ThematicExtraction,
)
from quantscribe.schemas.report import PeerComparisonReport, PeerRankEntry
from quantscribe.schemas.evaluation import EvalTestCase, EvalResult
from quantscribe.schemas.etl import ChunkMetadata, PageType, TextChunk
from quantscribe.retrieval.bank_index import BankIndex
from quantscribe.retrieval.peer_retriever import PeerGroupRetriever
from quantscribe.evaluation.numerical_eval import evaluate_numerical_accuracy


# ══════════════════════════════════════════════════
# Helper: build a valid CitationTrace
# ══════════════════════════════════════════════════

def _make_citation(**overrides) -> CitationTrace:
    defaults = dict(
        chunk_id="abc123",
        bank_name="HDFC_BANK",
        document_type="annual_report",
        fiscal_year="FY25",
        page_number=9,
        relevance_score=0.92,
        source_excerpt="Gross NPA ratio stood at 1.33%",
    )
    defaults.update(overrides)
    return CitationTrace(**defaults)


def _make_metric(**overrides) -> ExtractedMetric:
    defaults = dict(
        metric_name="gross_npa_ratio",
        metric_value=1.33,
        metric_unit="%",
        confidence="high",
        citation=_make_citation(),
    )
    defaults.update(overrides)
    return ExtractedMetric(**defaults)


def _make_extraction(**overrides) -> ThematicExtraction:
    defaults = dict(
        bank_name="HDFC_BANK",
        fiscal_year="FY25",
        theme="credit_risk",
        risk_score=3.5,
        risk_rating="low",
        summary="HDFC Bank maintains strong asset quality.",
        extracted_metrics=[_make_metric()],
        sentiment_score=0.3,
    )
    defaults.update(overrides)
    return ThematicExtraction(**defaults)


# ══════════════════════════════════════════════════
# CitationTrace Tests
# ══════════════════════════════════════════════════


class TestCitationTrace:
    def test_valid_citation(self):
        c = _make_citation()
        assert c.bank_name == "HDFC_BANK"
        assert c.page_number == 9

    def test_relevance_score_in_range(self):
        c = _make_citation(relevance_score=0.5)
        assert 0.0 <= c.relevance_score <= 1.0

    def test_relevance_score_out_of_range_rejected(self):
        with pytest.raises(Exception):
            _make_citation(relevance_score=1.5)

    def test_negative_page_number_rejected(self):
        with pytest.raises(Exception):
            _make_citation(page_number=0)

    def test_excerpt_max_length(self):
        # 500 char limit
        long_excerpt = "x" * 501
        with pytest.raises(Exception):
            _make_citation(source_excerpt=long_excerpt)

    def test_excerpt_at_max_length(self):
        c = _make_citation(source_excerpt="x" * 500)
        assert len(c.source_excerpt) == 500


# ══════════════════════════════════════════════════
# ExtractedMetric Tests
# ══════════════════════════════════════════════════


class TestExtractedMetric:
    def test_valid_numeric_metric(self):
        m = _make_metric(metric_value=1.33, qualitative_value=None)
        assert m.metric_value == 1.33

    def test_valid_qualitative_metric(self):
        m = _make_metric(metric_value=None, qualitative_value="high")
        assert m.qualitative_value == "high"

    def test_both_values_set(self):
        m = _make_metric(metric_value=1.33, qualitative_value="high")
        assert m.metric_value == 1.33
        assert m.qualitative_value == "high"

    def test_neither_value_set_auto_fills_not_disclosed(self):
        # Validator no longer raises — it auto-fills qualitative_value="not_disclosed"
        # so Gemini responses with null/null don't crash the entire extraction.
        m = _make_metric(metric_value=None, qualitative_value=None)
        assert m.qualitative_value == "not_disclosed"
        assert m.metric_value is None

    def test_confidence_must_be_valid_literal(self):
        with pytest.raises(Exception):
            _make_metric(confidence="very_high")

    def test_confidence_high(self):
        m = _make_metric(confidence="high")
        assert m.confidence == "high"

    def test_confidence_medium(self):
        m = _make_metric(confidence="medium")
        assert m.confidence == "medium"

    def test_confidence_low(self):
        m = _make_metric(confidence="low")
        assert m.confidence == "low"


# ══════════════════════════════════════════════════
# ThematicExtraction Tests
# ══════════════════════════════════════════════════


class TestThematicExtraction:
    def test_valid_extraction(self):
        e = _make_extraction()
        assert e.risk_score == 3.5
        assert len(e.extracted_metrics) == 1

    def test_risk_score_zero(self):
        e = _make_extraction(risk_score=0.0)
        assert e.risk_score == 0.0

    def test_risk_score_max(self):
        e = _make_extraction(risk_score=10.0)
        assert e.risk_score == 10.0

    def test_risk_score_above_max_rejected(self):
        with pytest.raises(Exception):
            _make_extraction(risk_score=10.1)

    def test_risk_score_negative_rejected(self):
        with pytest.raises(Exception):
            _make_extraction(risk_score=-0.1)

    def test_empty_metrics_rejected(self):
        with pytest.raises(Exception):
            _make_extraction(extracted_metrics=[])

    def test_risk_rating_valid_values(self):
        for rating in ["very_low", "low", "moderate", "high", "critical"]:
            e = _make_extraction(risk_rating=rating)
            assert e.risk_rating == rating

    def test_risk_rating_invalid_rejected(self):
        with pytest.raises(Exception):
            _make_extraction(risk_rating="extreme")

    def test_sentiment_range(self):
        e = _make_extraction(sentiment_score=-1.0)
        assert e.sentiment_score == -1.0
        e = _make_extraction(sentiment_score=1.0)
        assert e.sentiment_score == 1.0

    def test_sentiment_out_of_range_rejected(self):
        with pytest.raises(Exception):
            _make_extraction(sentiment_score=1.1)

    def test_summary_max_length(self):
        with pytest.raises(Exception):
            _make_extraction(summary="x" * 1001)

    def test_multiple_metrics(self):
        metrics = [
            _make_metric(metric_name="gross_npa_ratio", metric_value=1.33),
            _make_metric(metric_name="net_npa_ratio", metric_value=0.43),
            _make_metric(metric_name="provision_coverage_ratio", metric_value=67.86),
        ]
        e = _make_extraction(extracted_metrics=metrics)
        assert len(e.extracted_metrics) == 3


# ══════════════════════════════════════════════════
# PeerComparisonReport Tests
# ══════════════════════════════════════════════════


class TestPeerComparisonReport:
    def test_valid_report(self):
        report = PeerComparisonReport(
            query_theme="credit_risk",
            peer_group=["HDFC_BANK", "SBI"],
            extractions=[_make_extraction()],
            peer_ranking=[
                PeerRankEntry(bank="HDFC_BANK", risk_score=3.5, rank=1),
                PeerRankEntry(bank="SBI", risk_score=5.2, rank=2),
            ],
            cross_cutting_insights="HDFC has lower credit risk than SBI.",
            generated_at="2025-03-26T00:00:00",
        )
        assert report.query_theme == "credit_risk"
        assert len(report.peer_ranking) == 2

    def test_single_bank_peer_group_rejected(self):
        with pytest.raises(Exception):
            PeerComparisonReport(
                query_theme="credit_risk",
                peer_group=["HDFC_BANK"],  # min_length=2
                extractions=[],
                peer_ranking=[],
                cross_cutting_insights="N/A",
                generated_at="2025-03-26T00:00:00",
            )

    def test_insights_max_length(self):
        with pytest.raises(Exception):
            PeerComparisonReport(
                query_theme="credit_risk",
                peer_group=["HDFC_BANK", "SBI"],
                extractions=[],
                peer_ranking=[],
                cross_cutting_insights="x" * 2001,
                generated_at="2025-03-26T00:00:00",
            )

    def test_ranking_order(self):
        report = PeerComparisonReport(
            query_theme="credit_risk",
            peer_group=["HDFC_BANK", "SBI"],
            extractions=[],
            peer_ranking=[
                PeerRankEntry(bank="HDFC_BANK", risk_score=3.5, rank=1),
                PeerRankEntry(bank="SBI", risk_score=5.2, rank=2),
            ],
            cross_cutting_insights="Test",
            generated_at="2025-03-26T00:00:00",
        )
        assert report.peer_ranking[0].rank < report.peer_ranking[1].rank
        assert report.peer_ranking[0].risk_score < report.peer_ranking[1].risk_score


# ══════════════════════════════════════════════════
# BankIndex Tests (FAISS)
# ══════════════════════════════════════════════════


class TestBankIndex:
    def _make_index(self, dim=4):
        return BankIndex("TEST_BANK_annual_report_FY25", dimension=dim)

    def _make_chunk(self, chunk_id="a1", page=1) -> TextChunk:
        """Build a minimal TextChunk for BankIndex tests.
        BankIndex.add() now takes TextChunk objects (not bare ChunkMetadata)
        so it can store chunk content alongside metadata.
        """
        meta = ChunkMetadata(
            chunk_id=chunk_id,
            bank_name="TEST_BANK",
            document_type="annual_report",
            fiscal_year="FY25",
            page_number=page,
            page_type=PageType.NARRATIVE,
            chunk_index=0,
            token_count=100,
            parse_version="etl_v1.0.0",
        )
        return TextChunk(
            content=f"Sample content for chunk {chunk_id} on page {page}.",
            metadata=meta,
            content_type="narrative",
        )

    def test_add_and_size(self):
        idx = self._make_index()
        vectors = np.array([[1, 0, 0, 0]], dtype=np.float32)
        idx.add(vectors, [self._make_chunk()])
        assert idx.size == 1

    def test_add_multiple(self):
        idx = self._make_index()
        vectors = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        chunks = [self._make_chunk("a1", 1), self._make_chunk("a2", 2)]
        idx.add(vectors, chunks)
        assert idx.size == 2

    def test_search_returns_correct_result(self):
        idx = self._make_index()
        vectors = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        chunks = [self._make_chunk("a1", 1), self._make_chunk("a2", 2)]
        idx.add(vectors, chunks)

        query = np.array([[1, 0, 0, 0]], dtype=np.float32)
        results = idx.search(query, top_k=1)
        assert len(results) == 1
        assert results[0]["metadata"]["page_number"] == 1
        assert results[0]["score"] > 0.9  # Should be ~1.0 for exact match

    def test_search_top_k(self):
        idx = self._make_index()
        vectors = np.array([[1, 0, 0, 0], [0.9, 0.1, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        chunks = [self._make_chunk("a1", 1), self._make_chunk("a2", 2), self._make_chunk("a3", 3)]
        idx.add(vectors, chunks)

        query = np.array([[1, 0, 0, 0]], dtype=np.float32)
        results = idx.search(query, top_k=2)
        assert len(results) == 2
        # First result should be the exact match
        assert results[0]["metadata"]["page_number"] == 1

    def test_search_empty_index(self):
        idx = self._make_index()
        query = np.array([[1, 0, 0, 0]], dtype=np.float32)
        results = idx.search(query)
        assert results == []

    def test_save_and_load(self, tmp_path):
        idx = self._make_index()
        vectors = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        chunks = [self._make_chunk("a1", 1), self._make_chunk("a2", 2)]
        idx.add(vectors, chunks)

        # Save
        idx.save(str(tmp_path))

        # Load into new index
        idx2 = self._make_index()
        idx2.load(str(tmp_path))

        assert idx2.size == 2
        query = np.array([[1, 0, 0, 0]], dtype=np.float32)
        results = idx2.search(query, top_k=1)
        assert results[0]["metadata"]["page_number"] == 1

    def test_dimension_mismatch_rejected(self):
        idx = self._make_index(dim=4)
        wrong_dim = np.array([[1, 0, 0, 0, 0]], dtype=np.float32)  # 5-dim
        with pytest.raises(AssertionError):
            idx.add(wrong_dim, [self._make_chunk()])

    def test_count_mismatch_rejected(self):
        idx = self._make_index()
        vectors = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        with pytest.raises(AssertionError):
            idx.add(vectors, [self._make_chunk()])  # 2 vectors, 1 chunk


# ══════════════════════════════════════════════════
# PeerGroupRetriever Tests
# ══════════════════════════════════════════════════


class TestPeerGroupRetriever:
    def _build_retriever(self):
        idx1 = BankIndex("bank1", dimension=4)
        idx2 = BankIndex("bank2", dimension=4)

        v1 = np.array([[1, 0, 0, 0]], dtype=np.float32)
        v2 = np.array([[0, 1, 0, 0]], dtype=np.float32)

        chunk1 = TextChunk(
            content="Credit risk disclosure for BANK_A.",
            metadata=ChunkMetadata(
                chunk_id="c1", bank_name="BANK_A",
                document_type="annual_report", fiscal_year="FY25",
                page_number=1, page_type=PageType.NARRATIVE,
                chunk_index=0, token_count=50, parse_version="etl_v1.0.0",
            ),
            content_type="narrative",
        )
        chunk2 = TextChunk(
            content="Capital adequacy table for BANK_B.",
            metadata=ChunkMetadata(
                chunk_id="c2", bank_name="BANK_B",
                document_type="annual_report", fiscal_year="FY25",
                page_number=10, page_type=PageType.TABULAR,
                chunk_index=0, token_count=80, parse_version="etl_v1.0.0",
            ),
            content_type="table_structured",
        )

        idx1.add(v1, [chunk1])
        idx2.add(v2, [chunk2])

        return PeerGroupRetriever({"idx1": idx1, "idx2": idx2})

    def test_fan_out_returns_both_banks(self):
        retriever = self._build_retriever()
        query = np.array([[1, 0, 0, 0]], dtype=np.float32)
        results = retriever.retrieve(query, ["BANK_A", "BANK_B"])
        assert "BANK_A" in results
        assert "BANK_B" in results

    def test_fan_out_single_bank(self):
        retriever = self._build_retriever()
        query = np.array([[1, 0, 0, 0]], dtype=np.float32)
        results = retriever.retrieve(query, ["BANK_A"])
        assert "BANK_A" in results
        assert "BANK_B" not in results

    def test_unknown_bank_excluded(self):
        retriever = self._build_retriever()
        query = np.array([[1, 0, 0, 0]], dtype=np.float32)
        results = retriever.retrieve(query, ["BANK_C"])
        assert "BANK_C" not in results

    def test_results_have_score(self):
        retriever = self._build_retriever()
        query = np.array([[1, 0, 0, 0]], dtype=np.float32)
        results = retriever.retrieve(query, ["BANK_A"])
        assert results["BANK_A"][0]["score"] > 0

    def test_list_available_banks(self):
        retriever = self._build_retriever()
        banks = retriever.list_available_banks()
        assert "BANK_A" in banks
        assert "BANK_B" in banks


# ══════════════════════════════════════════════════
# Numerical Evaluation Tests
# ══════════════════════════════════════════════════


class TestNumericalEvaluation:
    def _make_gold(self, **metric_overrides):
        metrics = {"gross_npa_ratio": 1.33, "net_npa_ratio": 0.43}
        metrics.update(metric_overrides)
        return EvalTestCase(
            test_id="TEST_001",
            query_theme="credit_risk",
            bank_name="HDFC_BANK",
            fiscal_year="FY25",
            expected_metrics=metrics,
            expected_pages=[9],
            source_document="test.pdf",
        )

    def test_exact_match_passes(self):
        gold = self._make_gold()
        extracted = _make_extraction(extracted_metrics=[
            _make_metric(metric_name="gross_npa_ratio", metric_value=1.33),
            _make_metric(metric_name="net_npa_ratio", metric_value=0.43),
        ])
        results = evaluate_numerical_accuracy(extracted, gold)
        assert results["gross_npa_ratio"] is True
        assert results["net_npa_ratio"] is True

    def test_within_tolerance_passes(self):
        gold = self._make_gold()
        # 0.5% of 1.33 = 0.00665, so 1.336 is within tolerance
        extracted = _make_extraction(extracted_metrics=[
            _make_metric(metric_name="gross_npa_ratio", metric_value=1.336),
            _make_metric(metric_name="net_npa_ratio", metric_value=0.43),
        ])
        results = evaluate_numerical_accuracy(extracted, gold)
        assert results["gross_npa_ratio"] is True

    def test_outside_tolerance_fails(self):
        gold = self._make_gold()
        # 1.40 is well outside 0.5% of 1.33
        extracted = _make_extraction(extracted_metrics=[
            _make_metric(metric_name="gross_npa_ratio", metric_value=1.40),
            _make_metric(metric_name="net_npa_ratio", metric_value=0.43),
        ])
        results = evaluate_numerical_accuracy(extracted, gold)
        assert results["gross_npa_ratio"] is False

    def test_missing_metric_fails(self):
        gold = self._make_gold()
        # Only extract one of two expected metrics
        extracted = _make_extraction(extracted_metrics=[
            _make_metric(metric_name="gross_npa_ratio", metric_value=1.33),
        ])
        results = evaluate_numerical_accuracy(extracted, gold)
        assert results["gross_npa_ratio"] is True
        assert results["net_npa_ratio"] is False

    def test_zero_expected_value(self):
        gold = self._make_gold(gross_npa_ratio=0.0)
        extracted = _make_extraction(extracted_metrics=[
            _make_metric(metric_name="gross_npa_ratio", metric_value=0.0),
            _make_metric(metric_name="net_npa_ratio", metric_value=0.43),
        ])
        results = evaluate_numerical_accuracy(extracted, gold)
        assert results["gross_npa_ratio"] is True

    def test_none_metric_value_fails(self):
        gold = self._make_gold()
        extracted = _make_extraction(extracted_metrics=[
            _make_metric(metric_name="gross_npa_ratio", metric_value=None, qualitative_value="high"),
            _make_metric(metric_name="net_npa_ratio", metric_value=0.43),
        ])
        results = evaluate_numerical_accuracy(extracted, gold)
        assert results["gross_npa_ratio"] is False


# ══════════════════════════════════════════════════
# Citation Validation Tests (from extraction_chain)
# ══════════════════════════════════════════════════


class TestCitationValidation:
    def test_valid_citation_passes(self):
        from quantscribe.llm.extraction_chain import _validate_citations

        context = "The bank's gross NPA ratio stood at 1.33% as at March 31, 2025."
        extraction = _make_extraction(extracted_metrics=[
            _make_metric(citation=_make_citation(
                source_excerpt="gross NPA ratio stood at 1.33%"
            )),
        ])
        # Should not raise
        _validate_citations(extraction, context)

    def test_fabricated_citation_fails(self):
        from quantscribe.llm.extraction_chain import _validate_citations

        context = "The bank's gross NPA ratio stood at 1.33%."
        extraction = _make_extraction(extracted_metrics=[
            _make_metric(citation=_make_citation(
                source_excerpt="completely fabricated text not in document anywhere"
            )),
        ])
        with pytest.raises(ValueError, match="Citation validation failed"):
            _validate_citations(extraction, context)

    def test_exact_substring_match_passes(self):
        from quantscribe.llm.extraction_chain import _validate_citations

        context = "The provision coverage ratio was 67.86% for FY25."
        extraction = _make_extraction(extracted_metrics=[
            _make_metric(citation=_make_citation(
                source_excerpt="provision coverage ratio was 67.86%"
            )),
        ])
        _validate_citations(extraction, context)

    def test_partial_overlap_above_threshold_passes(self):
        from quantscribe.llm.extraction_chain import _validate_citations

        context = "The gross NPA ratio of the bank stood at 1.33 percent as at March end."
        extraction = _make_extraction(extracted_metrics=[
            _make_metric(citation=_make_citation(
                source_excerpt="gross NPA ratio stood at 1.33 percent March"
            )),
        ])
        # Most words match, should pass 60% threshold
        _validate_citations(extraction, context)


# ══════════════════════════════════════════════════
# Query Anchor Tests (from peer_comparison)
# ══════════════════════════════════════════════════


class TestQueryAnchors:
    def test_credit_risk_anchor(self):
        from quantscribe.llm.peer_comparison import _build_query_text
        q = _build_query_text("credit_risk")
        assert "NPA" in q
        assert "provision" in q.lower()

    def test_capital_adequacy_anchor(self):
        from quantscribe.llm.peer_comparison import _build_query_text
        q = _build_query_text("capital_adequacy")
        assert "CET1" in q
        assert "CAR" in q

    def test_liquidity_risk_anchor(self):
        from quantscribe.llm.peer_comparison import _build_query_text
        q = _build_query_text("liquidity_risk")
        assert "LCR" in q

    def test_unsecured_lending_anchor(self):
        from quantscribe.llm.peer_comparison import _build_query_text
        q = _build_query_text("unsecured_lending")
        assert "personal loan" in q.lower() or "credit card" in q.lower()

    def test_unknown_theme_fallback(self):
        from quantscribe.llm.peer_comparison import _build_query_text
        q = _build_query_text("some_new_theme")
        assert "some new theme" in q


# ══════════════════════════════════════════════════
# Context Formatting Tests (from peer_comparison)
# ══════════════════════════════════════════════════


class TestContextFormatting:
    def test_bank_context_has_delimiters(self):
        from quantscribe.llm.peer_comparison import _format_bank_context

        results = [{
            "metadata": {
                "page_number": 9,
                "section_header": "Asset Quality",
                "content": "Gross NPA was 1.33%.",
                "fiscal_year": "FY25",
                "document_type": "annual_report",
                "chunk_id": "abc",
            },
            "score": 0.9,
        }]

        ctx = _format_bank_context("HDFC_BANK", results)
        assert "[BEGIN HDFC_BANK CONTEXT" in ctx
        assert "[END HDFC_BANK CONTEXT]" in ctx
        assert "FY25" in ctx
        assert "Page 9" in ctx
        assert "Asset Quality" in ctx
        assert "1.33%" in ctx

    def test_multiple_chunks_separated(self):
        from quantscribe.llm.peer_comparison import _format_bank_context

        results = [
            {"metadata": {"page_number": 9, "section_header": "S1", "content": "chunk1", "fiscal_year": "FY25", "document_type": "annual_report", "chunk_id": "a"}, "score": 0.9},
            {"metadata": {"page_number": 54, "section_header": "S2", "content": "chunk2", "fiscal_year": "FY25", "document_type": "annual_report", "chunk_id": "b"}, "score": 0.8},
        ]

        ctx = _format_bank_context("SBI", results)
        assert "Page 9" in ctx
        assert "Page 54" in ctx
        assert "---" in ctx  # Separator between chunks
