"""
Retrieval & LLM output schemas.

These models define the structured output the LLM MUST produce.
The LangChain PydanticOutputParser enforces these schemas at runtime.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class CitationTrace(BaseModel):
    """
    Every extracted metric MUST have one of these.
    This is what makes the system auditable.

    Design notes:
    - chunk_id defaults to "" — the LLM has no knowledge of SHA-256 chunk IDs.
      It is back-filled from the metadata_store after retrieval if needed.
    - relevance_score defaults to 0.0 — the LLM has no knowledge of FAISS
      cosine similarity scores. It is back-filled in peer_comparison.py.
    """

    chunk_id: str = Field(
        default="",
        description="ID of the source chunk. Do not set — back-filled from retrieval.",
    )
    bank_name: str
    document_type: str
    fiscal_year: str
    page_number: int = Field(ge=1)
    section_header: Optional[str] = None
    relevance_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Cosine similarity score. Do not set — back-filled from retrieval.",
    )
    source_excerpt: str = Field(
        max_length=500,
        description="The exact text span from the chunk that supports this metric.",
    )


class ExtractedMetric(BaseModel):
    """
    A single quantitative or qualitative metric extracted by the LLM.

    At least one of metric_value or qualitative_value must be set.
    If neither is provided by the LLM, qualitative_value is auto-set
    to "not_disclosed" rather than raising a hard validation error.
    """

    metric_name: str = Field(
        examples=["gross_npa_ratio", "capital_adequacy_ratio", "lcr_percent"],
    )
    metric_value: Optional[float] = None
    metric_unit: Optional[str] = Field(
        default=None,
        examples=["%", "INR_crore", "basis_points", "ratio"],
    )
    qualitative_value: Optional[str] = Field(
        default=None,
        description=(
            "For non-numeric assessments: "
            "'high', 'moderate', 'low', 'stable', 'deteriorating', 'not_disclosed'"
        ),
    )
    confidence: Literal["high", "medium", "low"]
    citation: CitationTrace

    @model_validator(mode="after")
    def ensure_at_least_one_value(self) -> "ExtractedMetric":
        """
        Ensure at least one of metric_value or qualitative_value is set.

        Instead of raising, auto-fills qualitative_value with "not_disclosed"
        so the LLM response is never hard-rejected for this constraint alone.
        """
        if self.metric_value is None and self.qualitative_value is None:
            self.qualitative_value = "not_disclosed"
        return self


class ThematicExtraction(BaseModel):
    """
    LLM output for a single bank on a single macro theme.
    This is the atomic unit of analysis.

    The LangChain extraction chain MUST produce one of these per bank per query.
    """

    bank_name: str
    fiscal_year: str
    theme: str = Field(
        description="The macro theme being analyzed",
        examples=["credit_risk", "liquidity_risk", "unsecured_lending"],
    )
    risk_score: float = Field(
        ge=0.0,
        le=10.0,
        description="Computed risk score (0=lowest risk, 10=highest risk)",
    )
    risk_rating: Literal["very_low", "low", "moderate", "high", "critical"]
    summary: str = Field(
        max_length=1000,
        description="Qualitative summary grounded entirely in retrieved text",
    )
    extracted_metrics: list[ExtractedMetric] = Field(min_length=1)
    sentiment_score: float = Field(
        ge=-1.0,
        le=1.0,
        description="Sentiment of disclosure language (-1=very negative, +1=very positive)",
    )