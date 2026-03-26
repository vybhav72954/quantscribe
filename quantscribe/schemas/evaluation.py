"""
Evaluation schemas.

These define the ground-truth test cases and the results of evaluation runs.
Gold-standard test cases are stored as JSON in eval/gold_standard/.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvalTestCase(BaseModel):
    """
    A single evaluation test case with ground-truth annotations.

    Created manually by reading the source PDF and recording
    the correct metric values and their page locations.
    """

    test_id: str = Field(
        description="Unique test ID",
        examples=["HDFC_FY24_credit_risk_001"],
    )
    query_theme: str = Field(examples=["credit_risk", "liquidity_risk"])
    bank_name: str
    fiscal_year: str = Field(pattern=r"^FY\d{2}$")
    expected_metrics: dict[str, float] = Field(
        description="Ground-truth metric name -> value pairs",
        examples=[{"gross_npa_ratio": 1.12, "net_npa_ratio": 0.27}],
    )
    expected_pages: list[int] = Field(
        description="Page numbers where this data should be found",
    )
    source_document: str = Field(
        description="Filename of the source PDF",
        examples=["HDFC_Bank_Annual_Report_FY24.pdf"],
    )


class EvalResult(BaseModel):
    """Result of running evaluation on a single test case."""

    test_id: str
    numerical_accuracy: dict[str, bool] = Field(
        description="Per-metric exact match (within tolerance)",
    )
    schema_valid: bool = Field(
        description="Did the LLM output pass Pydantic validation?",
    )
    context_precision: float = Field(
        ge=0.0,
        le=1.0,
        description="RAGAS context precision score",
    )
    faithfulness: float = Field(
        ge=0.0,
        le=1.0,
        description="DeepEval faithfulness score",
    )
    retrieval_hit: bool = Field(
        description="Did at least one retrieved chunk come from an expected_page?",
    )
    overall_pass: bool = Field(
        description="True if numerical_accuracy >= 90% AND schema_valid AND faithfulness >= 0.85",
    )
